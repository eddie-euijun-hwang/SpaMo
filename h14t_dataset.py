import torch
import os
import numpy as np


class Phoenix14T(torch.utils.data.Dataset):
    def __init__(
        self,
        anno_root,
        vid_root,
        feat_root,
        mae_feat_root,
        mode='dev',
        spatial=False,
        spatiotemporal=False,
        spatial_postfix='',
        spatiotemporal_postfix=''
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
        pixel_value, glor_value = None, None
        
        if self.spatial:
            feat_path = os.path.join(self.feat_root, self.mode, f"{data['fileid']}{self.spatial_postfix}.npy")
            pixel_value = torch.tensor(np.load(feat_path))
        
        if self.spatiotemporal:
            if isinstance(self.spatiotemporal_postfix, str):
                glor_path = os.path.join(self.mae_feat_root, self.mode, f"{data['fileid']}{self.spatiotemporal_postfix}.npy")
                glor_value = torch.tensor(np.load(glor_path))
            else:
                glor_value = []
                for _spatiotemporal_postfix in self.spatiotemporal_postfix:
                    _glor_path = os.path.join(self.mae_feat_root, self.mode, f"{data['fileid']}{_spatiotemporal_postfix}.npy")
                    _glor_value = torch.tensor(np.load(_glor_path))
                    glor_value.append(_glor_value)
        
        return {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'bool_mask_pos': None,
            'text': f"{data['text']}.",  # Add full stop
            'en_text': data['en_text'],
            'es_text': data['es_text'],
            'fr_text': data['fr_text'],
            'gloss': data['gloss'],
            'id': data['fileid'],
            'num_frames': len(pixel_value) if pixel_value is not None else 0,
            'vid_path': os.path.join(self.vid_root, 'features', 'fullFrame-256x256px', data['folder']),
            'original_info': data,
            'lang': 'German'
        }

    def __len__(self):
        return len(self.data) - 1

    @staticmethod
    def collate_fn(batch):
        return batch





