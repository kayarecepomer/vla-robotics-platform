"""Text-prompted object masks: OWLv2 finds the best box for a query, SAM turns it into a mask.

Both models come from Hugging Face ``transformers``. Masks are computed on the same square
crop the encoders see, so they line up with the patch grid.
"""

from __future__ import annotations

import numpy as np

from vla.encoders.registry import square_crop

DETECTOR_ID = "google/owlv2-base-patch16-ensemble"
SEGMENTER_ID = "facebook/sam-vit-base"


class GroundedMasker:
    def __init__(self, query: str, device: str = "cuda", score_threshold: float = 0.15):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

        self.torch = torch
        self.query = query
        self.device = device
        self.score_threshold = score_threshold
        self.detector = Owlv2ForObjectDetection.from_pretrained(DETECTOR_ID).to(device).eval()
        self.detector_processor = Owlv2Processor.from_pretrained(DETECTOR_ID)
        self.segmenter = SamModel.from_pretrained(SEGMENTER_ID).to(device).eval()
        self.segmenter_processor = SamProcessor.from_pretrained(SEGMENTER_ID)

    def best_box(self, image: np.ndarray) -> list[float] | None:
        torch = self.torch
        inputs = self.detector_processor(text=[[self.query]], images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        side = image.shape[0]
        post_process = getattr(
            self.detector_processor,
            "post_process_grounded_object_detection",
            self.detector_processor.post_process_object_detection,
        )
        result = post_process(outputs, threshold=self.score_threshold, target_sizes=torch.tensor([[side, side]]))[0]
        if len(result["scores"]) == 0:
            return None
        return result["boxes"][result["scores"].argmax()].tolist()

    def mask(self, image: np.ndarray) -> np.ndarray:
        """Boolean mask of the queried object on the square-cropped image (all False if not found)."""
        torch = self.torch
        image = square_crop(image)
        box = self.best_box(image)
        if box is None:
            return np.zeros(image.shape[:2], dtype=bool)
        inputs = self.segmenter_processor(images=image, input_boxes=[[box]], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.segmenter(**inputs, multimask_output=False)
        masks = self.segmenter_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        return masks[0][0, 0].numpy().astype(bool)
