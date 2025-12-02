from diffusers import AutoModel, AutoPipelineForText2Image

unet = AutoModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
pipe = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet)
