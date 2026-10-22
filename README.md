# SpaMo-PJM

Fork of [SpaMo](https://github.com/eddie-euijun-hwang/SpaMo) (Hwang et al., NAACL 2025, [paper](https://aclanthology.org/2025.naacl-long.197.pdf)), a gloss-free sign language translation model that feeds spatial features (CLIP ViT) and motion features (VideoMAE) into an LLM (Flan-T5-XL with LoRA). We adapted it to translate Polish Sign Language (PJM) into English and added a four-stream variant, Mod-SpaMo. The method and original code are the work of the SpaMo authors; please cite their paper (see Citation).

Part of CrocoSign, our PJATK engineering thesis. All repositories are under [github.com/crocosign](https://github.com/crocosign), project site https://crocosign.github.io:

CrocoSign (data preparation) -> SpaMo-PJM / Uni-Sign-PJM (training) -> pjm-models (inference) -> croco-sign-backend / croco-sign-frontend (app).

## PJM adaptation


What we changed relative to upstream:

- **PJM data**: `dataset/pjm.py` loads pre-extracted features from HDF5 files and the train/val/test CSV splits (`train_split_path`, `val_split_path`, `test_split_path`). Clips shorter than 16 feature frames are dropped. `target_lang` selects the output language.
- **Feature extraction from the Hugging Face dataset** (`scripts/`): `pjm_vit_extract.py` (CLIP ViT-L/14, multi-scale), `mae_features_for_pjm.py` (VideoMAE-large, 16-frame chunks, overlap 8, resumable), both using the signer crop parameters in `crop_params/crop_params.lmdb`. For Mod-SpaMo: `hand_crop_vit_extract.py` (ViT on hand crops) and `mediapipe_extract.py` (body and hand keypoints, 258 values per frame).
- **Output language**: English. Flan-T5 turns Polish diacritics into unknown tokens, so Polish transcripts were machine-translated to English (DeepL) and used as targets. Optional in-context examples in Polish, French and Spanish (`scripts/translate_pjm_multilang.py`, NLLB-200 for French and Spanish).
- **Mod-SpaMo** (`spamo/t5_slt.py`): extra hand-ViT and MediaPipe streams fused with the base ones by cross-attention with a zero-initialised output projection, so a model trained with two streams can be extended without changing the prefix length. Design notes: `docs/quad_fusion_xattn/`.
- **Training/eval additions**: dropout (0.3) in the fusion module, ReduceLROnPlateau, optional frozen RoBERTa text encoder, ChrF and ROUGE-L on validation, test outputs dumped to a file and to W&B, `--tags` for W&B runs, Phoenix-2014T Lightning data module (`dataset/p14t_datamodule.py`) for pretraining on German Sign Language.
- `configs/`: one YAML per experiment (v3 to v33), each with a header comment saying what it tests. `docs/runs_report_v3-v21.md` is our run log (in Polish).

### Requirements

Python 3.10 or newer and a CUDA GPU. We trained and evaluated on one RTX 3090 (24 GB) with Flan-T5-XL and LoRA in bf16.

```bash
uv sync     # or: uv pip install -e .
```

### Data

[croco-corp/pjm-data](https://huggingface.co/datasets/croco-corp/pjm-data) holds the features, splits, translations and videos of the filtered set (29,932 pairs); its folders are `checkpoints`, `splits`, `translations`, `vectors` and `videos`, plus `dialogs_data.csv`. [croco-corp/pjm-segments](https://huggingface.co/datasets/croco-corp/pjm-segments) holds the unfiltered segments (about 40k) as WebDataset shards. This repository uses the `vectors/` and `splits/` folders of pjm-data (as in `scripts/setup_remote_v27.sh`):

- `vectors/vit_feat_pjm.h5`, `mae_feat_pjm.h5`, `hand_vit_feat_pjm.h5`, `mediapipe_feat_pjm.h5`, `texts_eng.h5`, `texts_multilang_pjm.h5` -> put in `features/`
- `splits/split_{train,val,test}.csv` (SI) and `split_{train,val,test}_ms.csv` (MS) -> put in `data/splits/`

To rebuild the features from the raw clips (mp4 files in the pjm-segments shards, or the `videos/` folder of pjm-data for the filtered set) use the extraction scripts above, for example:

```bash
python scripts/pjm_vit_extract.py --save_path features/vit_feat_pjm.h5 --s2_mode s2wrapping --scales 1 2 --device cuda:0
python scripts/mae_features_for_pjm.py --save-dir features --overlap-size 8 --device cuda:0 --split train
```

(`--help` lists the remaining options; the VideoMAE script writes `mae_feat_pjm_{split}.h5`.)

### Train and evaluate

`main.py` takes one or more YAML configs. `-e bleu` is the evaluation metric argument, as in upstream.

```bash
python main.py -c configs/finetune_v31_quad_xattn_ms_scratch.yaml -e bleu --tags ms-split 4-stream
python main.py -c configs/finetune_v31_quad_xattn_ms_scratch.yaml -e bleu --train False --test True --ckpt /path/to/checkpoint.ckpt
```

Config families (see each file's header):

- `finetune_v30..v33_*_scratch.yaml`: trained from scratch, MS and SI, two-stream ("dual", SpaMo) and four-stream cross-attention ("quad xattn", Mod-SpaMo).
- `pretrain_v22*_phoenix*.yaml`, `finetune_v23*.yaml`: pretraining on Phoenix-2014T and fine-tuning on PJM (German Sign Language transfer).
- `finetune_v26..v29_*_from_upstream.yaml`: continue from an earlier PJM-MS checkpoint trained with the upstream code.
- `finetune_v3..v21_*.yaml`: earlier experiments (contrastive-only, hand-ViT, MediaPipe, plateau scheduler, generation loss, ...).

The configs contain absolute paths from our training machine (`/home/croco/...` for the Hugging Face cache and split files). Change them to your paths, or read `scripts/setup_remote_v27.sh`, which downloads the features and rewrites these paths for the v26 to v29 configs (it also installs uv and downloads Flan-T5-XL, about 11 GB).

## Results

Test set, one run per row, beam 5. Data: 29,932 video-sentence pairs, 150 signers. **MS** (multi-signer): test signers also appear in training. **SI** (signer-independent): test signers are never seen in training. Transfer source is the sign language used before PJM.

| Model | Transfer source | Split | BLEU-4 | ROUGE-L | chrF | BLEURT |
| --- | --- | --- | --- | --- | --- | --- |
| SpaMo | PHOENIX-2014T | MS | 5.54 | 19.66 | 21.06 | 35.12 |
| SpaMo | PHOENIX-2014T | SI | 0.53 | 6.94 | 10.06 | 29.42 |
| SpaMo | scratch | MS | 2.03 | 12.37 | 15.32 | 30.25 |
| SpaMo | scratch | SI | 0.93 | 9.28 | 13.14 | 27.97 |
| Mod-SpaMo | PHOENIX-2014T | MS | 5.06 | 18.86 | 20.94 | 34.30 |
| Mod-SpaMo | PHOENIX-2014T | SI | 0.47 | 7.17 | 10.34 | 29.96 |
| Mod-SpaMo | scratch | MS | 1.56 | 11.92 | 14.50 | 30.42 |
| Mod-SpaMo | scratch | SI | 1.17 | 10.12 | 14.14 | 27.67 |

References are DeepL translations of the Polish transcripts. Mod-SpaMo did not improve on SpaMo. The Uni-Sign models in [Uni-Sign-PJM](https://github.com/crocosign/Uni-Sign-PJM) scored higher (best 7.61 BLEU-4 on MS), but they also started from ASL training, so the comparison is not architecture against architecture. SpaMo has no public weights usable for our comparison.

## Caveats

- SI generalisation is very weak: SpaMo drops from 5.54 to 0.53 BLEU-4.
- Trained from scratch, all models end near 2 BLEU-4 or below.
- Single runs; the epoch count differs per run and the best checkpoint was chosen on the validation set.
- The mapping from results table rows to config files is not recorded anywhere, so we cannot tell you which YAML produced which cell of the table. The `from_upstream` configs start from an earlier PJM-MS checkpoint (dev BLEU-4 5.36). That 5.36 is a development-set number from the process, not a thesis result.
- Training on TPU did not work for us (no static graph), so we trained on a single GPU.

## Upstream

Original instructions for Phoenix-2014T, CSL-Daily and How2Sign (feature extraction, pre-extracted features, pretrained checkpoint) are in the [upstream README](https://github.com/eddie-euijun-hwang/SpaMo#readme). The upstream script names (`scripts/vit_extract_feature.py`, `scripts/mae_extract_feature.py`) are kept in this fork.

## Citation

```bibtex
@inproceedings{hwang2025efficient,
  title={An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs},
  author={Hwang, Eui Jun and Cho, Sukmin and Lee, Junmyeong and Park, Jong C},
  booktitle={NAACL},
  year={2025}
}
```

## License

Our changes are under the MIT licence; the upstream code is not (see `LICENSE`). The upstream repository has no licence. Upstream terms must be respected: check [eddie-euijun-hwang/SpaMo](https://github.com/eddie-euijun-hwang/SpaMo) before reuse. The PJM Corpus data is non-commercial only, and Flan-T5, CLIP and VideoMAE have their own licences.

## Authors

[Wojciech Sieńko](https://www.linkedin.com/in/wojciech-sie%C5%84ko/) and [Rostyslav Kovalenko](https://www.linkedin.com/in/rost-koff/), PJATK engineering thesis, supervised by dr Albert Śledzianowski, MBA.
