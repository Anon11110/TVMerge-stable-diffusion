import argparse
import datetime
from PIL import Image
from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion inference.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for generated images.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for image generation.")
    return parser.parse_args()

def setup_pipeline():
    print("Loading Stable Diffusion model...")
    return StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

def run_inference(pipeline, prompt, output_dir):
    print("Generating image...")
    image = pipeline(prompt)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/generated_image_{timestamp}.png"
    image.save(output_path)
    print(f"Image successfully generated and saved to {output_path}")

def main():
    args = parse_args()
    pipeline = setup_pipeline()
    run_inference(pipeline, args.prompt, args.output_dir)

if __name__ == "__main__":
    main()
