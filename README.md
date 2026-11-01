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
- `configs/`: the configs behind the paper rows are named like the checkpoint folders on Hugging Face (`spamo-pjm-scratch-ms.yaml`, `mod-spamo-pjm-phoenix-si.yaml`, ...; see Reproduce). `finetune.yaml` is the upstream PHOENIX-2014T config. Our earlier experiments (configs v3 to v29, their queue scripts and our run log) are not part of this branch; they are in the history at commit `05e8001`.

### Requirements

Python 3.10 or newer and a CUDA GPU. We trained and evaluated on one RTX 3090 (24 GB) with Flan-T5-XL and LoRA in bf16.

```bash
uv sync     # or: uv pip install -e .
```

### Data

[CrocoSign/pjm-data](https://huggingface.co/datasets/CrocoSign/pjm-data) holds the features, splits, translations and videos of the filtered set (29,928 pairs); its folders are `checkpoints`, `splits`, `translations`, `vectors` and `videos`, plus `dialogs_data.csv`. [CrocoSign/pjm-segments](https://huggingface.co/datasets/CrocoSign/pjm-segments) holds the unfiltered segments (about 40k) as WebDataset shards. This repository uses the `vectors/` and `splits/` folders of pjm-data (as in `scripts/setup_remote_v27.sh`):

- `vectors/vit_feat_pjm.h5`, `mae_feat_pjm.h5`, `hand_vit_feat_pjm.h5`, `mediapipe_feat_pjm.h5`, `texts_eng.h5`, `texts_multilang_pjm.h5` -> put in `features/`
- `splits/split_{train,val,test}_ms.csv` (MS), `split_{train,val,test}_filtered.csv` (SI, used by the PHOENIX-2014T SI rows) and `split_{train,val,test}.csv` (SI before the quality filter, used by the SI rows from scratch) -> put in `data/splits/`

To rebuild the features from the raw clips (mp4 files in the pjm-segments shards, or the `videos/` folder of pjm-data for the filtered set) use the extraction scripts above, for example:

```bash
python scripts/pjm_vit_extract.py --save_path features/vit_feat_pjm.h5 --s2_mode s2wrapping --scales 1 2 --device cuda:0
python scripts/mae_features_for_pjm.py --save-dir features --overlap-size 8 --device cuda:0 --split train
```

(`--help` lists the remaining options; the VideoMAE script writes `mae_feat_pjm_{split}.h5`.)

### Train and evaluate

`main.py` takes one or more YAML configs. `-e bleu` is the evaluation metric argument, as in upstream.

```bash
python main.py -c configs/mod-spamo-pjm-scratch-ms.yaml -e bleu --tags ms-split 4-stream
python main.py -c configs/mod-spamo-pjm-scratch-ms.yaml -e bleu --train False --test True --ckpt /path/to/checkpoint.ckpt
```

Config families (see each file's header):

- `{spamo,mod-spamo}-pjm-scratch-{ms,si}.yaml` (formerly `finetune_v30..v33_*_scratch.yaml`): trained from scratch, MS and SI, two-stream ("dual", SpaMo) and four-stream cross-attention ("quad xattn", Mod-SpaMo).
- `mod-spamo-pjm-phoenix-{ms,si}.yaml` (formerly v27): Mod-SpaMo continued from the SpaMo PHOENIX-2014T checkpoints; `spamo-pjm-phoenix-{ms,si}-test.yaml` (formerly v26): this repository's two-stream model, used to test the SpaMo checkpoints trained with the upstream code.

The configs read features from `features/` and splits from `data/splits/`, relative to the repository root, and use the default Hugging Face cache (`cache_dir: null`). The run scripts in `scripts/` (`queue_*.sh`, `setup_remote_*.sh`) are kept as we ran them: they contain the paths of our training machine, and two of them also start the discarded v28/v29 runs, whose configs are only in the history; `scripts/setup_remote_v27.sh` downloads the features and Flan-T5-XL (about 11 GB).

## Results

Test set, one run per row, beam 5. Data: 29,928 video-sentence pairs, 151 signers. **MS** (multi-signer): test signers also appear in training. **SI** (signer-independent): test signers are never seen in training. Transfer source is the sign language used before PJM.

| Model | Transfer source | Split | BLEU-4 | ROUGE-L | chrF | BLEURT |
| --- | --- | --- | --- | --- | --- | --- |
| SpaMo | PHOENIX-2014T | MS | 5.54 | 19.66 | 21.06 | 35.12 |
| SpaMo | PHOENIX-2014T | SI | 2.09 | 12.42 | 16.02 | 28.84 |
| SpaMo | scratch | MS | 2.03 | 12.37 | 15.32 | 30.25 |
| SpaMo | scratch | SI | 0.93 | 9.28 | 13.14 | 27.97 |
| Mod-SpaMo | PHOENIX-2014T | MS | 5.06 | 18.86 | 20.94 | 34.30 |
| Mod-SpaMo | PHOENIX-2014T | SI | 2.06 | 12.26 | 16.17 | 27.86 |
| Mod-SpaMo | scratch | MS | 1.56 | 11.92 | 14.50 | 30.42 |
| Mod-SpaMo | scratch | SI | 1.17 | 10.12 | 14.14 | 27.67 |

References are DeepL translations of the Polish transcripts. The PHOENIX-2014T SI rows use the filtered SI split (9,776 of its 9,910 test clips; clips shorter than 16 feature frames are dropped), the SI rows from scratch the SI split before the quality filter. Mod-SpaMo did not improve on SpaMo. The Uni-Sign models in [Uni-Sign-PJM](https://github.com/crocosign/Uni-Sign-PJM) scored higher (best 8.61 BLEU-4 on MS), but they also started from ASL training, so the comparison is not architecture against architecture. SpaMo has no public weights usable for our comparison.

## Caveats

- SI generalisation is weak: SpaMo drops from 5.54 to 2.09 BLEU-4.
- Trained from scratch, all models end near 2 BLEU-4 or below.
- Single runs; the epoch count differs per run and the best checkpoint was chosen on the validation set.
- The Mod-SpaMo PHOENIX-2014T MS config and the `from_upstream` configs start from the SpaMo PHOENIX-2014T MS checkpoint (dev BLEU-4 5.36).
- SpaMo and Mod-SpaMo decode with sampling (`do_sample=True`, `top_p=0.9`, beam 5), so a second test run gives slightly different numbers. An audit after the experiments found that LoRA ran in bf16; the rows were trained before that fix.
- Training on TPU did not work for us (no static graph), so we trained on a single GPU.

## Reproduce

Checkpoints are in [CrocoSign/pjm-models](https://huggingface.co/CrocoSign/pjm-models) under `checkpoints/spamo/<folder>/`, and the test outputs and logs under `results/spamo/`. The SpaMo PHOENIX-2014T rows were trained with the upstream SpaMo code plus our changes in `upstream/spamo-pjm.patch` (apply to [eddie-euijun-hwang/SpaMo](https://github.com/eddie-euijun-hwang/SpaMo) at `aee7f8b`, with `git apply`), starting from the upstream PHOENIX-2014T checkpoint `spamo.ckpt`. Everything else uses this repository. Pass `-e bleu` when training: the default metric selects checkpoints and stops early on the wrong direction. Tests take `-c <test config> --train False --test True --ckpt <checkpoint> -e bleu`.

| Row | Train | Test config | Checkpoint folder |
| --- | --- | --- | --- |
| SpaMo PHOENIX-2014T MS | upstream: `python main.py -c configs/spamo-pjm-phoenix-ms.yaml --ckpt spamo.ckpt -e bleu` | `spamo-pjm-phoenix-ms-test.yaml` | `spamo-pjm-phoenix-ms` |
| SpaMo PHOENIX-2014T SI | upstream: `python main.py -c configs/spamo-pjm-phoenix-si.yaml --ckpt spamo.ckpt -e bleu --patience 5` | `spamo-pjm-phoenix-si-test.yaml` | `spamo-pjm-phoenix-si` |
| Mod-SpaMo PHOENIX-2014T MS | `python main.py -c configs/mod-spamo-pjm-phoenix-ms.yaml --ckpt <SpaMo MS checkpoint> -e bleu` | same | `mod-spamo-pjm-phoenix-ms` |
| Mod-SpaMo PHOENIX-2014T SI | `python main.py -c configs/mod-spamo-pjm-phoenix-si.yaml --ckpt <SpaMo SI checkpoint> -e bleu` | same | `mod-spamo-pjm-phoenix-si` |
| SpaMo / Mod-SpaMo scratch MS | `python main.py -c configs/spamo-pjm-scratch-ms.yaml -e bleu` (Mod-SpaMo: `mod-spamo-pjm-scratch-ms.yaml`) | same | `spamo-pjm-scratch-ms`, `mod-spamo-pjm-scratch-ms` |
| SpaMo / Mod-SpaMo scratch SI | `python main.py -c configs/spamo-pjm-scratch-si.yaml -e bleu` (Mod-SpaMo: `mod-spamo-pjm-scratch-si.yaml`) | same | `spamo-pjm-scratch-si`, `mod-spamo-pjm-scratch-si` |

`run_phoenix_si.sh` in the patch is the script that ran the SI chain (SpaMo SI training, its test, Mod-SpaMo SI training and test) on our machine; it uses our directory layout (the run names match the log folders in `results/spamo/` on Hugging Face), so read it as a record of the four commands above. BLEURT-20 comes from `scripts/compute_bleurt.py <test_outputs.txt>`.

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

Our changes are under the MIT licence; the upstream code is not (see `LICENSE`). The upstream repository has no licence. Upstream terms must be respected: check [eddie-euijun-hwang/SpaMo](https://github.com/eddie-euijun-hwang/SpaMo) before reuse. The PJM Corpus data is licensed under CC BY-SA 4.0, and Flan-T5, CLIP and VideoMAE have their own licences.

## Authors

[Wojciech Sieńko](https://www.linkedin.com/in/wojciech-sie%C5%84ko/) and [Rostyslav Kovalenko](https://www.linkedin.com/in/rost-koff/), PJATK engineering thesis, supervised by dr Albert Śledzianowski, MBA.
