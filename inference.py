from typing import Type
import os
import time
import argparse
import torch
import tvm
from tvm import relax
from transformers import CLIPTokenizer
from PIL import Image
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

from utils import utils
import schedulers
import datetime

class StableDiffusionPipeline:
    """
    A class to encapsulate the Stable Diffusion model along with associated methods
    to process inputs and generate outputs using a specific scheduler.

    Attributes:
        vm (relax.VirtualMachine): The TVM virtual machine for executing compiled models.
        tokenizer (CLIPTokenizer): Tokenizer for processing input text.
        scheduler (BaseScheduler): Scheduler object to manage the diffusion steps.
        device: The device on which computations will be performed.
        params (dict): Dictionary containing parameters for the models.
        debug_dir (str): Directory path to save debug outputs.
    """
    def __init__(self, vm, tokenizer, scheduler, device, params, debug_dir):
        self.vm = vm
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        self.params = params
        self.debug_dir = debug_dir

        self.clip_embed = lambda *args: vm["clip"](*args, params["clip"])
        self.unet_pred = lambda *args: vm["unet"](*args, params["unet"])
        self.vae_decode = lambda *args: vm["vae"](*args, params["vae"])
        self.concat_embed = vm["combine_embeddings"]
        self.rgba_convert = vm["image_to_rgba"]

    def _debug_save(self, name, arr):
        if self.debug_dir:
            import numpy as np
            np.save(f"{self.debug_dir}/{name}.npy", arr.numpy())

    def __call__(self, prompt, negative_prompt=""):
        """
        Generate an image based on the provided prompt and optional negative prompt.
        """
        text_embeddings = []
        for text in [negative_prompt, prompt]:
            input_ids = self.tokenizer(
                [text],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(torch.int32)[:, : self.tokenizer.model_max_length]

            input_ids_tvm = tvm.nd.array(input_ids.cpu().numpy(), self.device)
            text_embeddings.append(self.clip_embed(input_ids_tvm))

        concatenated_embeddings = self.concat_embed(*text_embeddings)
        self._debug_save("text_embeddings", concatenated_embeddings)

        latents = tvm.nd.array(
            torch.randn((1, 4, 64, 64), device="cpu", dtype=torch.float32).numpy(),
            self.device,
        )

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            self._debug_save(f"unet_input_{i}", latents)
            self._debug_save(f"timestep_{i}", t)
            noise_pred = self.unet_pred(latents, t, concatenated_embeddings)
            self._debug_save(f"unet_output_{i}", noise_pred)
            latents = self.scheduler.step(self.vm, noise_pred, latents, i)

        self._debug_save("vae_input", latents)
        decoded_image = self.vae_decode(latents)
        self._debug_save("vae_output", decoded_image)
        rgba_image = self.rgba_convert(decoded_image)
        return Image.fromarray(rgba_image.numpy().view("uint8").reshape(512, 512, 4))


def get_scheduler(scheduler_label: str) -> Type[schedulers.BaseScheduler]:
    return next((s for s in schedulers.schedulers_inference if s.scheduler_label == scheduler_label), None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="Target device (cuda, auto)")
    parser.add_argument("--enable-debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Building output and logging directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--prompt", type=str, default="A photo of an astronaut riding a horse on mars.", help="Prompt text")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--scheduler-label", type=str, choices=[s.scheduler_label for s in schedulers.schedulers_inference], default=schedulers.DPMSolverMultistepScheduler.scheduler_label, help="Scheduler name")
    parser.add_argument("--enable-profiling", action="store_true", default=False, help="Enable profiling with TensorBoard")
    args = parser.parse_args()

    if args.device == "auto":
        if not tvm.cuda().exist:
            raise ValueError("No CUDA device found. Please specify the device manually.")
        args.device = "cuda"

    return args


def setup_pipeline(args):
    device = tvm.device(args.device)
    params_dict = utils.load_params(args.artifacts_dir, device)
    vm_exe = tvm.runtime.load_module(f"{args.artifacts_dir}/stable_diffusion_{args.device}.so")
    vm = relax.VirtualMachine(vm_exe, device)

    debug_dump_dir = f"{args.artifacts_dir}/debug/" if args.enable_debug else ""
    if debug_dump_dir:
        os.makedirs(debug_dump_dir, exist_ok=True)
    
    os.makedirs(args.output_dir, exist_ok=True)

    return StableDiffusionPipeline(
        vm=vm,
        tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        scheduler=get_scheduler(args.scheduler_label)(args.artifacts_dir, device),
        device=device,
        params=params_dict,
        debug_dir=debug_dump_dir,
    )


def run_inference(args, pipeline):
    if args.enable_profiling:
        log_dir = args.artifacts_dir + "/logs_temp"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            use_cuda=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(writer.log_dir),
        )
        profiler.__enter__()

    start_time = time.time()
    generated_image = pipeline(args.prompt, args.negative_prompt)
    end_time = time.time()

    if args.enable_profiling:
        for _ in range(100):
            profiler.step()
        profiler.__exit__(None, None, None)
        writer.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/generated_image_{timestamp}.png"
    generated_image.save(output_path)
    print(f"Image generation took {end_time - start_time:.2f} seconds. Output saved to {output_path}")


def main():
    args = parse_args()
    pipeline = setup_pipeline(args)
    run_inference(args, pipeline)


if __name__ == "__main__":
    main()