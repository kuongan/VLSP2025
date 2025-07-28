from base import BaseEmbedder
from typing import List, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.nn import functional as F
import timm
import sys
sys.path.append("src/external/unilm")
import beit3.modeling_finetune  # type: ignore
from transformers import XLMRobertaTokenizer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

class BeIT3(BaseEmbedder):
    """BeIT3 Embedder for text, image, and multimodal embeddings."""

    def __init__(self,
                 model_name: str = "beit3_base_patch16_224_fused",
                 pretrained: bool = False,
                 image_size = 224,
                 tokenizer_path: str = "checkpoint/beit3.spm"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=pretrained).to(self.device)
        self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
        self.model.eval()
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        method = model_name.split('_')[-1]
        if method == 'fused':
            self.method = 'fused'
        else:
            self.method = 'dual'
            
    def embed_query(self, text: Union[str, List[str]]) -> np.ndarray:
        """Embed text into vector space"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            output = self.model(
                image=None,
                text_description=inputs['input_ids'],
                attn_mask=inputs['attention_mask'] if 'attention_mask' in inputs else None,
                only_infer=True
            )
        if self.method != 'fused':
            _, language_cls = output
        else:
            language_cls = output
        language_cls = F.normalize(language_cls, dim=-1)
        return language_cls
    def embed_image(self, image_path: Union[str, List[str]]) -> np.ndarray:
        """Embed image into vector space"""
        if isinstance(image_path, str):
            image_path = [image_path]

        images = []
        for path in image_path:
            img = Image.open(path).convert("RGB")
            img = self.image_transform(img)
            images.append(img)

        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            output= self.model(
                image=images,
                text_description=None,
                only_infer=True
            )
        if self.method != 'fused':
            vision_cls, _ = output
        else:
            vision_cls = output
        vision_cls = F.normalize(vision_cls, dim=-1)
        return vision_cls

    def embed_multimodal(self, text: str, image_path: str) -> np.ndarray:
        """Embed text + image into joint vector space"""
        # Text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        # Image
        img = Image.open(image_path).convert("RGB")
        img = self.image_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            joint_cls = self.model(
                image=img,
                text_description=inputs['input_ids'],
                attn_mask=inputs['attention_mask'] if 'attention_mask' in inputs else None,
                only_infer=True
            )
        if self.method != 'fused':
            vision_cls, language_cls = joint_cls
            joint_cls = (vision_cls + language_cls) / 2.0
        joint_cls = F.normalize(joint_cls, dim=-1)

        return joint_cls


model = BeIT3(model_name='beit3_base_patch16_224_fused',pretrained=True)
print(model)
image_url = r"tech.jpg"
image_emb = model.embed_image(image_url)
query_emb = model.embed_query("goldfish")
multi_emb= model.embed_multimodal(
    "goldfish",
    image_url
)
print("Image Embedding:", image_emb.shape)
print("Query Embedding:", query_emb.shape)
print("Multimodal Embedding:", multi_emb.shape) 
cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
print("Cosine Similarity (Image, Query):", cosine(image_emb, query_emb))
print("Cosine Similarity (Image, Multimodal):", cosine(image_emb, multi_emb))
print("Cosine Similarity (Query, Multimodal):", cosine(query_emb, multi_emb))
