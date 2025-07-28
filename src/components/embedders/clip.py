import torch
from PIL import Image
from typing import Union, List
from transformers import AutoProcessor, AutoModel
from base import BaseEmbedder

class CLIPEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", pretrained: bool = True, image_size = 224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.image_size = image_size
        # Load processor (thay cho tokenizer) & model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed_query(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]

        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings  

    def embed_image(self, image: Union[str, Image.Image]):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings 

    def embed_multimodal(self, text: str, image: Union[str, Image.Image]):
        text_emb = self.embed_query(text)
        image_emb = self.embed_image(image)
        fused_emb = (image_emb + text_emb) / 2.0
        fused_emb = fused_emb / fused_emb.norm(dim=-1, keepdim=True)

        return fused_emb  

# # Example usage:
# image_path = 'data/VLSP2025/law_db/images.fld/image231.png'
# model = CLIPEmbedder(model_name="google/siglip-base-patch16-224")
# image_emb = model.embed_image(image_path)
# query_emb = model.embed_query("traffic sign")   
# multi_emb = model.embed_multimodal("people", image_path)
# print("Image Embedding:", image_emb.shape)
# print("Query Embedding:", query_emb.shape)
# print("Multimodal Embedding:", multi_emb.shape) 
# cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
# print("Cosine Similarity (Image, Query):", cosine(image_emb, query_emb))
# print("Cosine Similarity (Image, Multimodal):", cosine(image_emb, multi_emb))
# print("Cosine Similarity (Query, Multimodal):", cosine(query_emb, multi_emb))