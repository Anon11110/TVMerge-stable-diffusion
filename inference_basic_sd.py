import argparse
import os
import time
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity
import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion with optional profiling and token merging.")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda, cpu, auto)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for generated images")
    parser.add_argument("--prompt", type=str, default="A photo of an astronaut riding a horse on Mars.",
                        help="Prompt text")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt text")
    parser.add_argument("--enable-profiling", action="store_true", default=False,
                        help="Enable profiling with TensorBoard")
    parser.add_argument("--apply-tome", action="store_true", default=False, help="Apply token merging optimization with TomeSD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image generation")
    return parser.parse_args()


def load_stable_diffusion_pipeline(args):
    from diffusers import StableDiffusionPipeline
    import tomesd

    device = args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

    if args.apply_tome:
        tomesd.apply_patch(pipeline, ratio=0.6)
        print("Token merging applied.")

    return pipeline


def run_inference(args, pipeline):
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    if args.enable_profiling:
        log_dir = os.path.join(args.output_dir, "logs_temp")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        profiler = profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            use_cuda=torch.cuda.is_available(),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(writer.log_dir)
        )
        profiler.__enter__()

    start_time = time.time()
    generated_image = pipeline(prompt=args.prompt, negative_prompt=args.negative_prompt, generator=generator).images[0]
    end_time = time.time()

    if args.enable_profiling:
        profiler.__exit__(None, None, None)
        writer.close()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = f"{args.output_dir}/generated_image_{timestamp}.png"
    generated_image.save(output_path)
    print(f"Image generation took {end_time - start_time:.2f} seconds. Output saved to {output_path}")


def main():
    args = parse_args()
    pipeline = load_stable_diffusion_pipeline(args)
    run_inference(args, pipeline)


if __name__ == "__main__":
    main()
