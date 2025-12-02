import torch
from diffusers import AutoPipelineForText2Image

def load_z_image_turbo(
    model_id="Tongyi-MAI/Z-Image-Turbo",
    dtype=torch.float16
):
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16",
    ).to("cuda")

    return pipe

if __name__ == "__main__":
    pipe = load_z_image_turbo()

    prompt = "a futuristic electric motorcycle parked under neon lights"
    image = pipe(
        prompt,
        num_inference_steps=4,      # Turbo model → very small step count
        guidance_scale=0.0,         # Turbo models often use guidance 0
    ).images[0]

    image.save("z_image_turbo_output.png")
    print("Saved: z_image_turbo_output.png")
