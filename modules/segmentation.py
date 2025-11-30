
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, AutoImageProcessor, AutoModelForObjectDetection

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load SegFormer model
# -----------------------------
seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device).eval()

# -----------------------------
# Load YOLO model for garment detection
# -----------------------------
yolo_processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
yolo_model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")

# -----------------------------
# Mapping & labels
# -----------------------------
segformer_map = {
    "shirt, blouse": 4,
    "top, t-shirt, sweatshirt": 4,
    "sweater": 4,
    "cardigan": 4,
    "jacket": 4,
    "vest": 4,
    "coat": 7,
    "dress": 7,
    "skirt": 5,
    "pants": 6,
    "shorts": 6,
}

garment_categories = list(segformer_map.keys())
arm_labels = [14, 15]  # left/right arm
sweatshirt_keywords = ["sweatshirt", "hoodie", "hooded", "pullover"]

# -----------------------------
# Functions
# -----------------------------
def detect_garment_label(garment_img_path, threshold=0.4):
    image = Image.open(garment_img_path).convert("RGB")
    inputs = yolo_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = yolo_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = yolo_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    best_label, best_score = None, 0
    for score, label in zip(results["scores"], results["labels"]):
        class_name = yolo_model.config.id2label[label.item()]
        if class_name in garment_categories and score > best_score:
            best_label = class_name
            best_score = float(score)
    if best_label is None:
        return None, None
    return best_label, segformer_map[best_label]

def segment_user_image(user_img_path):
    img = Image.open(user_img_path).convert("RGB")
    inputs = seg_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    logits = outputs.logits
    seg = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    ).argmax(dim=1)[0].cpu().numpy()
    return img, seg

def refine_subcategory(yolo_label, garment_img_path):
    if yolo_label not in ["top, t-shirt, sweatshirt"]:
        return yolo_label
    name = garment_img_path.lower()
    for k in sweatshirt_keywords:
        if k in name:
            return "sweatshirt"
    if "tshirt" in name or "t-shirt" in name:
        return "t-shirt"
    return "top"

def get_cloth_mask(user_img_path, garment_img_path):
    # 1. التعرف على نوع القطعة
    garment_name_raw, seg_label = detect_garment_label(garment_img_path)
    if garment_name_raw is None:
        print("No garment detected.")
        return None, None, None
        
    garment_name = refine_subcategory(garment_name_raw, garment_img_path)
    print(f"Detected: {garment_name} | Label: {seg_label}")
    
    user_img, seg = segment_user_image(user_img_path)
    
    base_mask = (seg == seg_label).astype(np.uint8) * 255
    final_mask = base_mask.copy()
    
    
    long_sleeve_keywords = ["sweater", "cardigan", "jacket", "coat", "hoodie", "sweatshirt", "pullover"]
    
    is_long_sleeve = any(k in garment_name.lower() or k in garment_name_raw.lower() for k in long_sleeve_keywords)

    if seg_label == 4 and is_long_sleeve:
        print("   -> Long sleeve detected: Masking arms.")
        arms_mask = np.zeros_like(seg, dtype=np.uint8)
        for arm_label in arm_labels: # 14 & 15
            current_arm = (seg == arm_label).astype(np.uint8) * 255
            arms_mask = np.maximum(arms_mask, current_arm)
        
        kernel = np.ones((15, 15), np.uint8)
        dilated_arms = cv2.dilate(arms_mask, kernel, iterations=4)
        final_mask = np.maximum(final_mask, dilated_arms)
    
    else:
        print("   -> Short sleeve/Sleeveless: Keeping arms visible.")

    return user_img, Image.fromarray(final_mask), {"garment_name": garment_name}

def prepare_user_image_for_inpainting(original_image, mask_image):
    img_arr = np.array(original_image)
    mask_arr = np.array(mask_image.convert("L"))
    mask_area = mask_arr > 128
    noise = np.random.randint(0, 255, img_arr.shape, dtype=np.uint8)
    modified_img_arr = img_arr.copy()
    modified_img_arr[mask_area] = noise[mask_area]
    return Image.fromarray(modified_img_arr)
