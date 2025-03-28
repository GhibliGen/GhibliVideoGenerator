import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
    def process(self, image_path: str) -> torch.Tensor:
        """Preprocess input image"""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image)
            return tensor.unsqueeze(0)
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
