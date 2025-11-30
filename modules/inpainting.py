import torch
import gc
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from PIL import Image

class Inpainter:
    def __init__(self, lora_path: str, device="cuda"):
        self.device = device
        self.lora_path = lora_path
        self.lora_scale = 0.4
        self.pipeline = None  

    def _load_pipeline(self):
        if self.pipeline is None:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16
            )
            self.pipeline = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(self.device)
            # Load IP-Adapter
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
                low_cpu_mem_usage=True
            )
            # Load LoRA weights
            self.pipeline.load_lora_weights(self.lora_path)
            self.pipeline.set_ip_adapter_scale(0.9)

    def _unload_pipeline(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # clean memory 
        gc.collect()
        torch.cuda.empty_cache()

    def inpaint(self, input_image: Image.Image, mask_image: Image.Image, ip_image: Image.Image, garment_name: str) -> Image.Image:
        self._load_pipeline()
        
        prompt = (
            f"A realistic {garment_name}, worn naturally by the person. "
            "Match the exact garment style, length, and sleeve type from the reference garment image. "
            "High quality cloth texture, correct folds, natural shadows, photorealistic."
        )
        negative_prompt = (
            "different color, incorrect garment, white clothes, deformed body, extra limbs, "
            "artifacts, distortions, unrealistic texture, blurry"
        )
        
        # image_creation
        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                mask_image=mask_image,
                ip_adapter_image=ip_image,
                strength=1.0,
                guidance_scale=7.5,
                num_inference_steps=70, 
                cross_attention_kwargs={"scale": self.lora_scale}
            ).images[0]
        except Exception as e:
            raise e
        finally:
            #clean Memory in any case
            self._unload_pipeline()
            
        return result
