import torch
import os
import numpy as np
import glob

from PIL import Image


class How2Sign(torch.utils.data.Dataset):
    def __init__(
        self,
        anno_root,
        vid_root,
        feat_root,
        mae_feat_root,
        mode,
        spatial,
        spatiotemporal,
        spatial_postfix,
        spatiotemporal_postfix
    ):
        super().__init__()

        self.anno_root = anno_root
        self.vid_root = vid_root
        self.feat_root = feat_root
        self.mae_feat_root = mae_feat_root
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        
        self.data = np.load(os.path.join(self.anno_root, f'{mode}_info_ml.npy'), allow_pickle=True).item()

    def __getitem__(self, index):
        data = self.data[index]
        start_time, end_time = data['original_info']['START_REALIGNED'], data['original_info']['END_REALIGNED']
        
        pixel_value, glor_value = None, None
        
        if self.spatial:
            feat_path = os.path.join(self.feat_root, self.mode, f"{data['fileid']}_{str(start_time)}{self.spatial_postfix}.npy")
            pixel_value = torch.tensor(np.load(feat_path))
        if self.spatiotemporal:
            glor_path = os.path.join(self.mae_feat_root, self.mode, f"{data['fileid']}_{str(start_time)}{self.spatiotemporal_postfix}.npy")
            glor_value = torch.tensor(np.load(glor_path))

        return {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'bool_mask_pos': None,
            'text': f"{data['text'].lower()}",
            'de_text': data['de_text'].lower(),
            'es_text': data['es_text'].lower(),
            'fr_text': data['fr_text'].lower(),
            'gloss': None,
            'id': data['fileid'],
            'num_frames': len(pixel_value) if pixel_value is not None else 0, 
            'vid_path': os.path.join(self.vid_root, 'CSL-Daily_256x256px', data['folder']),
            'original_info': data,
            'lang': 'English'
        }

    def __len__(self):
        return len(self.data) - 1

    @staticmethod
    def collate_fn(batch):
        return batch