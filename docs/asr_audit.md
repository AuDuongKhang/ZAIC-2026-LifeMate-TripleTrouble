# ASR model and dataset audit

Task 1 of `docs/superpowers/plans/2026-10-28-asr-models-and-data.md`. Checked on 29 Oct 2026 against the Hugging Face
API and model/dataset cards. "Released" is the date the HF repo was created. Parameter counts come from the checkpoint
(safetensors metadata, or fp32 weight file size / 4), not from the model name.

Rules that decide "usable" (`context/RULE.md`): open-source models and public data released **before 26 Oct 2026** whose
licences allow it; whole system < 6B parameters. Next to Qwen3-4B-Instruct-2507 (4.02B) the ASR gets **≤ ~1.9B**.

## Models

| Model (HF id) | Params | Licence | Released | Package | Usable? |
|---|---|---|---|---|---|
| `vinai/PhoWhisper-large` | 1.55B | BSD-3-Clause | 2023-12-19 | transformers (existing `whisper` backend) | **yes**, current baseline |
| `vinai/PhoWhisper-medium` | 0.77B | BSD-3-Clause | 2024-02-18 | transformers | **yes** |
| `vinai/PhoWhisper-small` | 0.24B | BSD-3-Clause | 2024-02-18 | transformers | **yes** |
| `Qualcomm-AI-Research/PhoASR-whisper-small` | 0.24B | BSD-3-Clause-Clear + Qualcomm Responsible AI Licence | 2026-07-06 | transformers (Whisper, no new code) | **yes**, after reading the responsible-use terms |
| `Qwen/Qwen3-ASR-0.6B` | **0.94B** | Apache-2.0 | 2026-01-28 | `qwen-asr` | **yes**; lists Vietnamese among its 30 languages |
| `Qwen/Qwen3-ASR-1.7B` | **2.35B** | Apache-2.0 | 2026-01-28 | `qwen-asr` | **no** with a 4B LLM (4.02 + 2.35 > 6B); only with an LLM ≤ ~3.6B |
| `nvidia/parakeet-ctc-0.6b-Vietnamese` | 0.6B | NVIDIA Open Model License (commercial use allowed) | 2026-01-15 | NeMo | **yes**; ~2,000 h Vietnamese + Vietnamese–English code-switching. NeMo is a heavy dependency for Docker |
| `khanhld/chunkformer-rnnt-large-vie` | ~0.15B (to confirm) | CC-BY-4.0 | 2025-10-07 | `chunkformer` | **yes**; RNN-T variant, the licence-clean ChunkFormer |
| `khanhld/chunkformer-ctc-large-vie` | ~0.15B (614 MB fp32; card says ~110M) | **CC-BY-NC-4.0** | 2025-02-01 | `chunkformer` | ask organisers (non-commercial) |
| `nguyenvulebinh/wav2vec2-base-vietnamese-250h` | ~0.09B | **CC-BY-NC-4.0** | 2022-03-02 | transformers (needs a CTC backend) | ask organisers (non-commercial) |
| `5CD-AI/VieGem-ASR` | ? | **CC-BY-NC-SA-4.0** | 2026-10-19 | pytorch | ask organisers; very new, unclear architecture |
| `nmcuong/parakeet-tdt-0.6b-v3-vietnamese` | 0.6B | CC-BY-4.0 | 2026-05-18 | NeMo | optional; community fine-tune, gated (manual approval) |
| `whisper-small-vi` (e.g. `Edkl/whisper-small-vi`) | 0.24B | not stated | 2022-11 | transformers | **no**: no licence on the card |

Corrections to the plan's "known so far" column:
- Qwen3-ASR checkpoints are larger than their names: 1.7B → 2.35B, 0.6B → 0.94B. The 1.7B no longer fits next to a 4B LLM.
- Parakeet-vi exists as `nvidia/parakeet-ctc-0.6b-Vietnamese`. Its card says ~2,000 h, not 4,000+ h.
- ChunkFormer has a CC-BY-4.0 RNN-T release, so the licence question only concerns the CTC checkpoint.

## Datasets

| Dataset (HF id) | Hours | Regions | Licence | Size on disk | Usable? |
|---|---|---|---|---|---|
| Our train, pool clips only | ≤ 6.8 h (5,036 clips in all of train) | North, Central, South; real speakers | competition data, licensed to registered teams | in `data/` | **yes**; in-domain vocabulary |
| `NhutP/VietSpeech` | 1,100+ h (from the plan; card is gated, not re-checked) | North, Central, South | Apache-2.0 | 130.5 GB | **yes**; gated (automatic approval after login) |
| `nguyendv02/ViMD_Dataset` | 102.56 h | 63 provinces, region + province labels; news broadcasts | **CC-BY-NC-ND-4.0** | 74.2 GB | ask organisers (non-commercial, no derivatives) |
| `linhtran92/viet_bud500` | ~500 h (from the plan) | North, Central, South | **CC-BY-NC-SA-4.0** | 59.5 GB | ask organisers (non-commercial) |

## Questions for the organisers

Answers to be recorded here.

1. The challenge has prizes. Do non-commercial licences (CC-BY-NC, CC-BY-NC-SA, CC-BY-NC-ND) allow a model or dataset to be
   used in it? This affects ChunkFormer-CTC, wav2vec2-base-vi-250h, VieGem-ASR, ViMD and Bud500.
   - Answer: _pending_
2. Can the Docker package ship download scripts and clip manifests for public training audio, instead of hundreds of
   hours of audio?
   - Answer: _pending_

## Recommendation

- **B1 order**: the three PhoWhisper runs (in progress), then PhoASR-whisper-small (no new code), then Qwen3-ASR-0.6B,
  Parakeet-vi and ChunkFormer-RNNT-large (new backends, Task 9). Skip Qwen3-ASR-1.7B unless the LLM shrinks.
- **B2 data**: start from the train pool plus VietSpeech, the only large dataset with a clean licence. Add ViMD or Bud500
  only if the organisers allow non-commercial data.
