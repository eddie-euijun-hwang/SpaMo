import random
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class T5Dataset(Dataset):
    def __init__(
        self,
        fused_features_path: str,
        texts_path: str,
        max_frame_length: int = 128,
        sample_rate: int = 1,
    ):
        super().__init__()
        self.fused_features_path = fused_features_path
        self.max_frame_length = max_frame_length
        self.sample_rate = sample_rate

        self.fused_features = h5py.File(self.fused_features_path, mode="r")
        self.texts = h5py.File(texts_path, mode="r")

        self.keys = sorted(self.fused_features.keys())

    def __getitem__(self, index: int) -> dict[str, any]:
        key = self.keys[index]

        feat = torch.from_numpy(self.fused_features[key][()])

        if self.sample_rate > 1:
            feat = feat[:: self.sample_rate]

        num_frames = feat.shape[0]
        if num_frames > self.max_frame_length:
            start = random.randint(0, num_frames - self.max_frame_length)
            feat = feat[start : start + self.max_frame_length]

        text = self.texts[key][()]
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        return {
            "pixel_value": feat,
            "glor_value": None,
            "text": text,
            "id": key,
            "lang": "pl",
            "num_frames": len(feat),
        }

    def __len__(self) -> int:
        return len(self.keys)

    def close(self):
        self.fused_features.close()
        self.texts.close()


def collate_fn_t5(batch):
    return batch


class T5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        fused_features_path,
        texts_path,
        batch_size=32,
        num_workers=4,
        val_split=0.1,
        test_split=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        dataset = T5Dataset(self.hparams.fused_features_path, self.hparams.texts_path)

        total_len = len(dataset)
        val_len = int(total_len * self.hparams.val_split)

        # test_len = int(total_len * self.hparams.test_split)
        # train_len = total_len - val_len - test_len
        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(
        #     dataset,
        #     [train_len, val_len, test_len],
        #     generator=torch.Generator().manual_seed(42)
        # )
        # self.predict_dataset = self.test_dataset

        train_len = total_len - val_len
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn_t5,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn_t5,
            num_workers=self.hparams.num_workers,
        )

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
    #                       collate_fn=collate_fn_t5, num_workers=self.hparams.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size,
    #                       collate_fn=collate_fn_t5, num_workers=self.hparams.num_workers)
