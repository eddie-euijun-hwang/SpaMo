import torch
import h5py
from torch.nn.utils.rnn import pad_sequence
import random
import pytorch_lightning as pl


class FusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        visual_features_path: str,
        motion_features_path: str,
        texts_path: str,
        device: str,
        max_frame_length: int = 512,
    ):
        super().__init__()

        visual_features = h5py.File(visual_features_path, mode="r")
        motion_features = h5py.File(motion_features_path, mode="r")
        texts = h5py.File(texts_path, mode="r")
        self.device = device
        self.data = []
        self.max_num_of_frames = max_frame_length

        idx_to_key = sorted(visual_features.keys())
        for key in idx_to_key:
            vf = visual_features[key][()]
            mf = motion_features[key][()]
            text = texts[key][()].decode()
            self.data.append((vf, mf, text))

        visual_features.close()
        motion_features.close()
        texts.close()

    def __getitem__(self, index: int) -> tuple:
        (item_visual_features, item_motion_features, text) = self.data[index]
        item_visual_features = torch.from_numpy(item_visual_features)
        item_motion_features = torch.from_numpy(item_motion_features)
        num_of_frames = item_visual_features.size(dim=0)
        if num_of_frames > self.max_num_of_frames:
            start_index = random.randint(0, num_of_frames - self.max_num_of_frames)
            item_visual_features = item_visual_features[
                start_index : start_index + self.max_num_of_frames
            ]

        return (
            item_visual_features,
            item_visual_features.size(dim=0),
            item_motion_features,
            item_motion_features.size(dim=0),
            text,
        )

    def __len__(self) -> int:
        return len(self.data)


def collate_data(batch):
    vfs, vf_lengths, mfs, mf_lengths, texts = zip(*batch)
    return {
        "vision_features": pad_sequence(vfs, batch_first=True),
        "vision_features_seq_lengths": vf_lengths,
        "motion_features": pad_sequence(mfs, batch_first=True),
        "motion_features_seq_lengths": mf_lengths,
        "texts": texts,
    }


class FusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        visual_features_path: str,
        motion_features_path: str,
        texts_path: str,
        device: str,
        max_frame_length: int = 512,
        batch_size=64,
        validation_proportion=0.1,
        num_workers=4,
    ):
        super().__init__()
        self.visual_features_path = visual_features_path
        self.motion_features_path = motion_features_path
        self.texts_path = texts_path
        self.device = device
        self.max_frame_length = max_frame_length
        self.batch_size = batch_size
        self.validation_proportion = validation_proportion
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = FusionDataset(
            self.visual_features_path,
            self.motion_features_path,
            self.texts_path,
            device=self.device,
            max_frame_length=self.max_frame_length,
        )
        train_length = int(len(dataset) * (1.0 - self.validation_proportion))
        val_length = len(dataset) - train_length
        self.train_set, self.val_set = torch.utils.data.random_split(
            dataset, (train_length, val_length)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_data,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data,
        )


if __name__ == "__main__":
    dataset = FusionDataset(
        "features/vit_feat_pjm.h5",
        "features/mae_feat_pjm.h5",
        "features/texts.h5",
        device="cuda",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, collate_fn=collate_data
    )
