import os
import torch
import numpy as np
import bittensor as bt
from typing import List
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
            bt.logging.warning(
                "HF_TOKEN not found in environment. Public models may be rate-limited."
            )

        try:
            self.model = CLIPModel.from_pretrained(model_id, token=token).to(
                self.device
            )
            self.processor = CLIPProcessor.from_pretrained(model_id, token=token)
            bt.logging.success("CLIP model loaded successfully.")
        except Exception as e:
            bt.logging.error(f"Failed to load CLIP model: {e}")
            raise e

    def compute_similarity(self, query: str, images: List[Image.Image]) -> np.ndarray:
        """
        Compute cosine similarity scores between a text query and a list of images.
        Returns a 1D numpy array normalized to [0, 1].
        """
        if not images:
            return np.array([])

        try:
            text_inputs = self.processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                if not isinstance(text_features, torch.Tensor):
                    text_features = self._extract_feature_tensor(text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                batch_scores: List[np.ndarray] = []
                batch_size = 32

                for start in range(0, len(images), batch_size):
                    image_batch = images[start : start + batch_size]
                    image_inputs = self.processor(
                        images=image_batch,
                        return_tensors="pt",
                    ).to(self.device)
                    image_features = self.model.get_image_features(**image_inputs)
                    if not isinstance(image_features, torch.Tensor):
                        image_features = self._extract_feature_tensor(image_features)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    similarities = torch.matmul(
                        image_features, text_features.T
                    ).squeeze(-1)
                    batch_scores.append(similarities.cpu().numpy())

            scores = np.concatenate(batch_scores).astype(np.float32)
            if scores.size == 0:
                return np.array([])

            score_min = float(np.min(scores))
            score_max = float(np.max(scores))
            if score_max - score_min < 1e-8:
                return np.full_like(scores, 0.5, dtype=np.float32)

            return (scores - score_min) / (score_max - score_min)

        except Exception as e:
            bt.logging.error(f"Inference failed: {e}")
            return np.array([])

    @staticmethod
    def _extract_feature_tensor(output) -> torch.Tensor:
        """
        Handle transformers return-type differences across versions.
        """
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if (
            hasattr(output, "last_hidden_state")
            and output.last_hidden_state is not None
        ):
            return output.last_hidden_state[:, 0, :]
        raise TypeError(f"Unsupported feature output type: {type(output)}")
