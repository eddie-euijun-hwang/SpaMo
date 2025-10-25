import os
import numpy as np
import torch
import argparse
import tqdm
import os.path as osp
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor

import sys
# Get the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print(f"Added to path: {parent_dir}")  # Debug line

from utils.helpers import sliding_window_for_list, read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


class VideoMAEFeatureReader(object):
    def __init__(
        self, 
        model_name='MCG-NJU/videomae-large', 
        cache_dir=None,
        device='cuda:0',
        overlap_size=0,
        nth_layer=-1
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()
        
    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(images=video, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        
        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]
        
        return outputs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='location of tsv files', required=True)
    parser.add_argument('--video_root', help='location of tsv files', required=True)
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--model_name', help='ViT model name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', help='device to use', default='cpu')
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--mode', nargs='+', type=str)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    return parser


def get_iterator(args, mode):
    batch_size = args.batch_size

    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    reader = VideoMAEFeatureReader(
        args.model_name, 
        device=args.device, 
        overlap_size=args.overlap_size, 
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )
    
    def iterate():
        for i in range(num):
            fname = data[i]['folder']
            
            if ds_name == 'Phoenix14T' or ds_name == 'CSL-Daily':
                image_list = get_img_list(ds_name, args.video_root, fname)
                
                if len(image_list) < 16:
                    len_diff = 16 - len(image_list)
                    image_list.extend([image_list[-1]] * (16 - len(image_list)))
                image_list_chunks = sliding_window_for_list(image_list, window_size=16, overlap_size=args.overlap_size)
                
                videos = []
                for image_list in image_list_chunks:
                    videos.append([Image.open(image).convert('RGB') for image in image_list])
                
                video_feats = []
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]
                    feats = reader.get_feats(video_batch).cpu().numpy()
                    video_feats.append(feats)
                    
                yield np.concatenate(video_feats, axis=0), data[i]['fileid'], None
            
            else:
                if ds_name == 'How2Sign':
                    start_time, end_time = data[i]['original_info']['START_REALIGNED'], data[i]['original_info']['END_REALIGNED']
                    videos = read_video(fname, start_time=start_time, end_time=end_time)
                    
                    if len(videos) > 0:
                        if len(videos) < 16:
                            len_diff = 16 - len(videos)
                            videos.extend([videos[-1]] * (16 - len(videos)))
                        
                        videos = sliding_window_for_list(videos, window_size=16, overlap_size=args.overlap_size)
                        
                        video_feats = []
                        for j in range(0, len(videos), batch_size):
                            video_batch = videos[j:min(j + batch_size, len(videos))]
                            feats = reader.get_feats(video_batch).cpu().numpy()
                            video_feats.append(feats)
                        
                        yield np.concatenate(video_feats, axis=0), data[i]['fileid'], str(start_time)
                    
                    else:
                        yield [], data[i]['fileid'], str(start_time)
    
    return iterate, num

def main():
    parser = get_parser()
    args = parser.parse_args()

    mode = ["dev", "test", "train"]
    for m in mode:
        ds_name = osp.split(args.anno_root)[-1]
        fname = f'mae_feat_{ds_name}'
        os.makedirs(osp.join(args.save_dir, fname, m), exist_ok=True)
    
        if ds_name == 'How2Sign':
            if m == 'dev': _m = 'val'
            else: _m = m
        elif ds_name == 'NIASL2021':
            if m == 'dev': _m = 'validation' 
        else:
            _m = m

        generator, num = get_iterator(args, _m)
        iterator = generator()

        for vit_feat in tqdm.tqdm(iterator, total=num):
            feats, id, st = vit_feat
            save_path = osp.join(args.save_dir, fname, m)
            postfix = f'_overlap-{args.overlap_size}'
            
            if st is not None:
                postfix = f'_{st}{postfix}'
            
            np.save(osp.join(save_path, f'{id}{postfix}.npy'), feats)


if __name__ == "__main__":
    main()
