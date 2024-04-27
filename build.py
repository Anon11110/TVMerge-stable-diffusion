import argparse
import os
import pickle
from typing import Dict, List, Tuple

import torch
import tomesd
import tvm
from tvm import relax

import schedulers
from models import model_ir_transforms
from utils import utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="cuda", help="Target device for compilation, GPU or CPU")
    parser.add_argument("--database-dir", type=str, default="scheduler_db/", help="Path to the schedule database")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Path to save artifacts")
    parser.add_argument("--enable-cache", type=int, default=1, help="Use cached IRModule if available")
    parser.add_argument("--enable-debug", action="store_true", default=False, help="Enable debug dump mode")
    parser.add_argument("--apply-tome", action="store_true", default=False, help="Apply token merging")
    return parser.parse_args()


def setup_environment(args):
    print(f"Creating necessary directories...")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)
    target = tvm.target.Target("cuda" if tvm.cuda().exist else "llvm")
    args.target = tvm.target.Target(target, host="llvm")
    print(f"Target configured: {args.target}")


def load_and_trace_models(args):
    print(f"Loading Stable Diffusion pipeline...")
    pipeline = load_stable_diffusion_pipeline(args)
    torch_dev_key = utils.detect_available_torch_device()
    print(f"Tracing models...")
    mod, params = trace_models(pipeline, torch_dev_key)
    return mod, params


def load_stable_diffusion_pipeline(args):
    from diffusers import StableDiffusionPipeline
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    if args.apply_tome:
        tomesd.apply_patch(pipeline, ratio=0.6)
        print(f"Token merging applied")
    return pipeline


def trace_models(pipeline, device_str) -> Tuple[tvm.IRModule, Dict[str, List[tvm.nd.NDArray]]]:
    print(f"Tracing CLIP model...")
    clip_trace = model_ir_transforms.convert_clip_to_embeddings(pipeline)
    print(f"Tracing UNet model...")
    unet_trace = model_ir_transforms.generate_noise_predictions(pipeline, device_str)
    print(f"Tracing VAE model...")
    vae_trace = model_ir_transforms.vae_to_image(pipeline)
    print(f"Tracing concat embeddings...")
    combine_embeddings_trace = model_ir_transforms.combine_embeddings()
    print(f"Tracing image to RGBA...")
    image_to_rgba_trace = model_ir_transforms.image_to_rgba()
    print(f"Tracing schedulers...")
    scheduler_traces = [scheduler.scheduler_steps() for scheduler in schedulers.schedulers_build]
    print(f"Merging IRModules...")
    merged_mod = utils.merge_irmodules(
        clip_trace,
        unet_trace,
        vae_trace,
        combine_embeddings_trace,
        image_to_rgba_trace,
        *scheduler_traces,
    )
    print(f"Detaching parameters...")
    return relax.frontend.detach_params(merged_mod)


def preprocess_module(mod, params, args):
    print(f"Preprocessing module...")
    model_names = ["clip", "unet", "vae"]
    scheduler_func_names = [name for scheduler in schedulers.schedulers_build for name in scheduler.list_step_functions()]
    entry_funcs = model_names + scheduler_func_names + ["image_to_rgba", "combine_embeddings"]

    print(f"Applying Relax transformations...")
    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.DeadCodeElimination(entry_funcs)(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod = relax.transform.BundleModelParams()(mod)
    mod_transform, mod_inference = utils.split_irmodules(mod, model_names, entry_funcs)

    dump_debug_script(mod_transform, "mod_lift_params.py", args)

    print(f"Computing and saving scheduler constants...")
    schedulers.compute_save_scheduler_consts(args.output_dir)
    print(f"Transforming parameters...")
    new_params = utils.transform_params(mod_transform, params)
    print(f"Saving parameters...")
    utils.save_params(new_params, args.output_dir)
    return mod_inference


def dump_debug_script(mod, file_name, args):
    if args.enable_debug:
        dump_path = os.path.join(args.output_dir, "debug", file_name)
        with open(dump_path, "w") as outfile:
            outfile.write(mod.script(show_meta=True))
        print(f"Debug script dumped to {dump_path}")


def apply_meta_schedule_databases(mod, args):
    from tvm import meta_schedule as ms
    databases = [
        (ms.database.create(work_dir=args.database_dir), "softmax2 op"),
        (ms.database.create(work_dir=args.database_dir), "clip and unet part"),
        (ms.database.create(work_dir=args.database_dir), "vae part")
    ]

    for db, db_name in databases:
        print(f"Applying meta-schedule database for {db_name}...")
        with args.target, db, tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(mod)

    print(f"Generating missing schedules...")
    with tvm.target.Target("cuda"):
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)

    return mod


def adjust_function_attributes(mod):
    print(f"Adjusting function attributes...")
    for gv, func in mod.functions.items():
        try:
            print(f"  Processing function {gv}: global_symbol={func.attrs['global_symbol']}, num_inputs={func.attrs['num_input']}")
            if func.attrs["global_symbol"] == "main" and func.attrs["num_input"] == 3:
                mod[gv] = func.with_attr("global_symbol", "unet")
                print(f"    Updated to 'unet'")
            if func.attrs["global_symbol"] == "main" and func.attrs["num_input"] == 1:
                mod[gv] = func.with_attr("global_symbol", "vae")
                print(f"    Updated to 'vae'")
            if func.attrs["global_symbol"] == "subgraph_0":
                mod[gv] = func.with_attr("global_symbol", "clip")
                print(f"    Updated to 'clip'")
        except:
            pass
    return mod


def build_model(mod, args):
    dump_debug_script(mod, "mod_build_stage.py", args)
    print(f"Building the inference model...")
    ex = relax.build(mod, args.target)
    lib_path = f"artifacts/stable_diffusion_{args.target.kind.default_keys[0]}.so"
    ex.export_library(lib_path)
    print(f"Build complete. Library exported to {lib_path}")


def main():
    print(f"Parsing command-line arguments...")
    args = parse_arguments()
    setup_environment(args)

    cache_path = os.path.join(args.output_dir, "mod_cache_before_build.pkl")
    if not args.enable_cache or not os.path.isfile(cache_path):
        mod, params = load_and_trace_models(args)
        mod = preprocess_module(mod, params, args)
        print(f"Saving cached module to {cache_path}...")
        with open(cache_path, "wb") as outfile:
            pickle.dump(mod, outfile)
    else:
        print(f"Loading cached module from {cache_path}...")
        with open(cache_path, "rb") as infile:
            mod = pickle.load(infile)
        print(f"Cached module loaded. Skipping tracing and preprocessing.")

    mod = apply_meta_schedule_databases(mod, args)
    mod = adjust_function_attributes(mod)
    build_model(mod, args)
    print(f"Stable Diffusion inference pipeline completed.")


if __name__ == "__main__":
    main()