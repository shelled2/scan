from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=init_image).images[0]
