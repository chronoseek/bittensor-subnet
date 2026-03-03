import os
import torch
import numpy as np
import bittensor as bt
from typing import List, Tuple
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPProcessorEngine:
    """
    Encapsulates CLIP model loading and inference logic.
    """
    
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bt.logging.info(f"Loading CLIP model '{model_id}' on {self.device}...")
        
        token = os.getenv("HF_TOKEN")
        if not token:
            bt.logging.warning("HF_TOKEN not found in environment. Public models may be rate-limited.")

        try:
            self.model = CLIPModel.from_pretrained(model_id, token=token).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_id, token=token)
            bt.logging.success("CLIP model loaded successfully.")
        except Exception as e:
            bt.logging.error(f"Failed to load CLIP model: {e}")
            raise e

    def compute_similarity(self, query: str, images: List[Image.Image]) -> np.ndarray:
        """
        Compute similarity scores between a text query and a list of images.
        Returns: 1D numpy array of probabilities/scores.
        """
        if not images:
            return np.array([])
            
        try:
            inputs = self.processor(
                text=[query],
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # logits_per_image: [num_images, num_text] -> [num_images, 1]
                logits_per_image = outputs.logits_per_image
                # Softmax across the batch dimension isn't quite right for "independent" relevance,
                # but standard CLIP usage often does softmax(dim=1) for text retrieval or dim=0 for image retrieval.
                # Here we want raw similarity or normalized scores.
                # Let's use raw probabilities from softmax for now as per original logic.
                probs = logits_per_image.softmax(dim=0).cpu().numpy().flatten()
                
            return probs
            
        except Exception as e:
            bt.logging.error(f"Inference failed: {e}")
            return np.array([])
