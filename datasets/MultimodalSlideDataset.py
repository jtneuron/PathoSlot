import os

import torch

from datasets.BaseDataset import BaseDataset


class MultimodalSlideDataset(BaseDataset):
    """Loads precomputed multimodal h5 embeddings for MultimodalSlotModel."""

    def __init__(self, split, load_from: str):
        super().__init__(split)
        self.load_from = load_from
        if not os.path.isdir(self.load_from):
            raise FileNotFoundError(f"Embedding directory does not exist: {self.load_from}")

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        load_path = self._embedding_path(sample_id)
        assets, _ = self.load_h5(
            load_path,
            keys=["vis_features", "text_features", "name_embedding"],
        )

        return {
            "id": sample_id,
            "vis_features": torch.as_tensor(assets["vis_features"], dtype=torch.float32),
            "text_features": torch.as_tensor(assets["text_features"], dtype=torch.float32),
            "name_embedding": torch.as_tensor(assets["name_embedding"], dtype=torch.float32),
        }

    def _embedding_path(self, sample_id: str) -> str:
        filename = sample_id if str(sample_id).endswith(".h5") else f"{sample_id}.h5"
        path = os.path.join(self.load_from, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found for sample {sample_id}: {path}")
        return path
