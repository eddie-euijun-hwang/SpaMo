import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import models and dataset
from dataset.pjm import PJMDataModule
from dataset.T5 import T5DataModule
from dataset.fusion import FusionDataModule
from models.fusion import LightningFusion
from spamo.t5_model import T5SLT
import h5py
import numpy as np

# Mock data paths
MOCK_DIR = Path(__file__).resolve().parent.parent / "tests" / "mock_data"
VIS_PATH = MOCK_DIR / "mock_visual.h5"
MOT_PATH = MOCK_DIR / "mock_motion.h5"
TXT_PATH = MOCK_DIR / "mock_text.h5"
FUSED_PATH = MOCK_DIR / "mock_fused.h5"


def collate_fn_t5(batch):
    """
    Collate function for T5SLT.
    Returns a list of samples (dicts), as T5SLT.get_inputs expects a list.
    """
    # Decode text if bytes and adjust feature dimension for T5 small (512)
    processed_batch = []
    for item in batch:
        new_item = item.copy()
        t = item["text"]
        if isinstance(t, bytes):
            new_item["text"] = t.decode("utf-8")

        # Mock data has 2048 dim, but T5-small needs 512.
        # In real pipeline, FusionModel would output correct dim.
        # Here we slice it.
        if new_item["pixel_value"].shape[1] > 512:
            new_item["pixel_value"] = new_item["pixel_value"][:, :512]

        processed_batch.append(new_item)

    return processed_batch


def collate_fn_fusion(batch):
    """
    Collate function for FusionModel.
    Pads sequences and returns a dict of tensors.
    """
    vision_features = [item["pixel_value"] for item in batch]
    motion_features = [item["glor_value"] for item in batch]

    # Decode text if bytes
    texts = []
    for item in batch:
        t = item["text"]
        if isinstance(t, bytes):
            t = t.decode("utf-8")
        texts.append(t)

    vision_lens = [v.shape[0] for v in vision_features]
    motion_lens = [m.shape[0] for m in motion_features]

    # Pad sequences
    vision_padded = torch.nn.utils.rnn.pad_sequence(vision_features, batch_first=True)
    motion_padded = torch.nn.utils.rnn.pad_sequence(motion_features, batch_first=True)

    return {
        "vision_features": vision_padded,
        "motion_features": motion_padded,
        "vision_features_seq_lengths": torch.tensor(vision_lens),
        "motion_features_seq_lengths": torch.tensor(motion_lens),
        "texts": texts,
    }


class MockPJMDatamodule(PJMDataModule):
    def __init__(self, collate_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn,
        )


def test_fusion_model():
    print("\n=== Testing FusionModel ===")

    # Setup Datamodule using the new FusionDataModule
    datamodule = FusionDataModule(
        visual_features_path=str(VIS_PATH),
        motion_features_path=str(MOT_PATH),
        texts_path=str(TXT_PATH),
        device="cpu",
        batch_size=4,
        num_workers=0,
    )

    # Create dummy weights for testing
    dummy_weights_path = os.path.join(MOCK_DIR, "dummy_embeddings.pt")
    # T5-small vocab size is approx 32128, hidden dim 512 (but we used 2048 in fusion model target size)
    # Let's match the fusion model target size for the embedding dim to avoid shape mismatch in loss
    torch.save(torch.randn(32128, 2048), dummy_weights_path)

    model = LightningFusion(
        vision_input_dim=2048,
        motion_input_dim=1024,
        hidden_dim=768,
        target_size=2048,
        device="cpu",
        t5_checkpoint="google/flan-t5-small",  # Use small for test
        target_embedding_weights_path=dummy_weights_path,
        max_txt_len=32,
    )

    # Mock target embeddings since we don't have the file
    # We replace the embedding layer with a random one for testing
    # model.target_embedding = torch.nn.Embedding(32128, 2048) # vocab size of t5-small approx, dim 2048

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=False,  # Run 1 batch of train/val/test
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=datamodule)
    print("FusionModel test passed!")


def test_t5_model():
    print("\n=== Testing T5SLT ===")

    # Generate mock fused data (dim 512 for T5-small)
    if not FUSED_PATH.exists():
        print("Generating mock fused data...")
        with h5py.File(FUSED_PATH, "w") as f:
            for i in range(10):  # 10 samples
                key = f"video_{i:04d}"
                # Random frames, 512 dim
                data = np.random.randn(20, 512).astype(np.float32)
                f.create_dataset(key, data=data)

    # Setup Datamodule using the new T5DataModule
    datamodule = T5DataModule(
        fused_features_path=str(FUSED_PATH),
        texts_path=str(TXT_PATH),
        batch_size=4,
        num_workers=0,
        val_split=0.2,
    )

    # Setup Model
    model = T5SLT(
        model_name="google/flan-t5-small",
        tuning_type="lora",
        max_frame_len=128,
        max_txt_len=32,
        lora_r=4,
        cache_dir="./cache",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=False,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=datamodule)
    print("T5SLT test passed!")


if __name__ == "__main__":
    # Ensure mock data exists
    if not os.path.exists(VIS_PATH):
        print("Mock data not found. Please run generate_mock_data.py first.")
        exit(1)

    test_fusion_model()
    test_t5_model()
