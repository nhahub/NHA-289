from PIL import Image
from segmentation import get_cloth_mask, prepare_user_image_for_inpainting
from inpainting import Inpainter

class VTOAgentModule:
    def __init__(self, lora_path: str):
        # Initialize Inpainter
        self.inpainter = Inpainter(lora_path=lora_path)

    def process_tryon(self, user_img_path: str, garment_img_path: str):
        """
        Full VTO pipeline: segmentation → mask preparation → inpainting
        Returns inpainted PIL image
        """
        # get cloth mask and user image
        user_img, final_mask, meta = get_cloth_mask(user_img_path, garment_img_path)
        if user_img is None:
            raise ValueError("No garment detected in the reference image.")

        # Prepare mask and input image for inpainting
        mask_image = final_mask.convert("L")
        input_image = prepare_user_image_for_inpainting(user_img, mask_image)

        # Load IP adapter image
        ip_image = Image.open(garment_img_path).convert("RGB")

        #  Run inpainting
        result_image = self.inpainter.inpaint(
            input_image=input_image,
            mask_image=mask_image,
            ip_image=ip_image,
            garment_name=meta["garment_name"]
        )
        return result_image
