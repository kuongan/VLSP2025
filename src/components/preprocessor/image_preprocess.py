import os
import re
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel, AutoTokenizer
from typing import List, Union, Dist

class ImagePreprocessor:
    def __init__(self,
                 input_size=448,
                 min_num=1,
                 max_num=6,
                 use_thumbnail=True,
                 load_grounding_dino=True,
                 load_model_captioning=True,
                 grounding_dino_model_id="IDEA-Research/grounding-dino-base",
                 model_captions_id="5CD-AI/Vintern-1B-v3_5"):
        self.input_size = input_size
        self.min_num = min_num
        self.max_num = max_num
        self.use_thumbnail = use_thumbnail
        self.MEAN, self.STD = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if load_grounding_dino:
            self.grounding_processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(self.device)
        if load_model_captioning:
            self.model_captions = AutoModel.from_pretrained(
                model_captions_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
            ).eval().to(self.device)
            self.tokenizer_caption = AutoTokenizer.from_pretrained(model_captions_id, trust_remote_code=True, use_fast=False, truncation=True, max_length=1024)
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        return Image.open(image_path).convert('RGB')
    
    def extract_image_tags(text: str) -> List[str]:
        """Extract all <<IMAGE: ... /IMAGE>> blocks"""
        return re.findall(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>", text)
    
    def build_transform(self):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.MEAN, std=self.STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.input_size * self.input_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self,image, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(self.min_num, self.max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= self.max_num and i * j >= self.min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height)

        # calculate the target width and height
        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.input_size)) * self.input_size,
                (i // (target_width // self.input_size)) * self.input_size,
                ((i % (target_width // self.input_size)) + 1) * self.input_size,
                ((i // (target_width // self.input_size)) + 1) * self.input_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size,self.input_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image_for_captioning(self, image):
        transform = self.build_transform()
        images = self.dynamic_preprocess(image, use_thumbnail=True)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        print(pixel_values.shape)
        return pixel_values 
     
    def extract_traffic_sign(self, image: Image.Image):
        image = image.resize((self.input_size,self.input_size))

        if not self.grounding_model or not self.grounding_processor:
            raise ValueError("GroundingDINO not loaded.")
        text = "traffic sign."
        inputs = self.grounding_processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]]
        )
        return results[0]

    def crop_traffic_sign(self, image: Image.Image, results: dict):
        cropped_images = []
        boxes = results.get("boxes")
        if boxes is None or len(boxes) == 0:
            print("⚠️ Không tìm thấy box nào.")
            return []
        width, height = image.size
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 > x1 and y2 > y1:
                cropped = image.crop((x1, y1, x2, y2))
                cropped_images.append(cropped)
        print(f"✅ Cropped {len(cropped_images)} traffic sign(s).")
        return cropped_images

    def generate_captions(self, cropped_images: list[Image.Image]) -> list[str]:
        if not self.model_captions or not self.tokenizer_caption:
            raise ValueError("Vintern model not loaded!")

        if not cropped_images:
            print("⚠️ No images to caption.")
            return []

        captions = []

        for image in cropped_images:
            pixel_values = self.load_image_for_captioning(image).to(torch.bfloat16).to(self.device)

            question = "<image> Mô tả biển báo đầy đủ nội dung, ý nghĩa, phạm vi áp dụng, nhóm phương tiện, vị trí hoặc tình huống áp dụng."
            generation_config = dict(
                max_new_tokens= 256, do_sample=False, num_beams = 3, repetition_penalty=2.0
            )
            with torch.no_grad():
                response = self.model_captions.chat(
                    self.tokenizer_caption,
                    pixel_values,
                    question,
                    generation_config,
                )
            captions.append(response)
        torch.cuda.empty_cache()
        return captions


