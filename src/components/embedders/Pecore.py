import torch
from PIL import Image
from .base import BaseEmbedder
from typing import List, Union
import sys
sys.path.append("src/external/pecore") 
import core.vision_encoder.pe as pe # type: ignore
import core.vision_encoder.transforms as transforms # type: ignore

print("CLIP configs:", pe.CLIP.available_configs())
# CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224', 'PE-Core-S16-384', 'PE-Core-T16-384']

class PEcore(BaseEmbedder):
    def __init__(self, model_name: str = "PE-Core-L14-336", pretrained: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pe.CLIP.from_config(model_name, pretrained=pretrained).to(self.device)
        self.model.eval()
        self.image_size = self.model.image_size
        self.tokenizer =  transforms.get_text_tokenizer(self.model.context_length)
        self.preprocess = transforms.get_image_transform(self.model.image_size)
    def embed_query(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Embed text into vector space"""
        inputs = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            output = self.model.encode_text(inputs)
        output = torch.nn.functional.normalize(output, dim=-1)
        return output
    def embed_image(self, image_path: Union[str, List[str]]) -> torch.Tensor:
        """Embed image into vector space"""
        if isinstance(image_path, str):
            image_path = [image_path]

        images = []
        for path in image_path:
            image = Image.open(path).convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            images.append(image)

        images = torch.cat(images, dim=0)
        with torch.no_grad():
            output = self.model.encode_image(images)
        output = torch.nn.functional.normalize(output, dim=-1)
        return output
    def embed_multimodal(self, text: Union[str, List[str]], image_path: Union[str, List[str]]) -> torch.Tensor:
        """Embed both text and image into vector space"""
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        inputs = self.tokenizer(text).to(self.device)
        image_features, text_features, _ = self.model(image, inputs)
        joint_embedding = (image_features + text_features) / 2.0
        joint_embedding = torch.nn.functional.normalize(joint_embedding, dim=-1)
        return joint_embedding
    
    
# image_path = '000000039769.jpg'
# model = PEcore(model_name='PE-Core-L14-336', pretrained=True)
# print(model)
# image_emb = model.embed_image(image_path)
# query_emb = model.embed_query("cat")   
# multi_emb = model.embed_multimodal("cat", image_path)
# print("Image Embedding:", image_emb.shape)
# print("Query Embedding:", query_emb.shape)
# print("Multimodal Embedding:", multi_emb.shape) 
# cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
# print("Cosine Similarity (Image, Query):", cosine(image_emb, query_emb))
# print("Cosine Similarity (Image, Multimodal):", cosine(image_emb, multi_emb))
# print("Cosine Similarity (Query, Multimodal):", cosine(query_emb, multi_emb))



