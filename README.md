# Google WAXAL ASR Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashuza11/google-waxal-asr/blob/main/notebooks/01_zeroshot_mms_baseline.ipynb)

Multilingual ASR for **Lingala (`lin`)**, **Shona (`sna`)**, and **Luganda (`lug`)**,
built on the [WAXAL dataset](https://huggingface.co/datasets/google/WaxalNLP)
(Google Research + Makerere University, University of Ghana, Digital Umuganda).
Zindi challenge: https://zindi.africa/competitions/google-waxal-asr-challenge

Evaluation: `0.5 * WER + 0.5 * CER` (lower is better). Phase 1 = HF train/val/test
splits, leaderboard-visible. Phase 2 = a held-out unseen audio set released ~1
week before close — no labels, no language metadata, final ranking is Phase 2 only.

Inspired by the modeling approach from a past project,
[afrivoices-asr-hack](https://github.com/Ashuza11/afrivoices-asr-hack)
(Whisper-small fine-tuning for 6 East African languages) — same eval-harness
shape, same "prepare data / train / infer" separation, adapted here for
WAXAL's languages and Zindi's submission format.

## Why this repo runs on Colab, not locally

This machine has **no GPU and 3.8GB RAM** — not enough to load even a
1B-parameter ASR model comfortably, let alone fine-tune one. So:

- **Local (this repo):** eval harness, CSV/text-only data exploration
  (`notebooks/00_data_exploration.py` — no audio, no GPU needed), scaffolding.
- **Colab (`notebooks/01_...ipynb`):** everything that needs a GPU or audio —
  loading WaxalNLP audio, model inference, submission generation.
- **Zindi**: the community `zindi` pip package needs interactive
  username/password login, not a real token/API — Zindi doesn't currently
  expose a public developer API. Given that, submission is a deliberate
  **manual step**: the notebook downloads `submission.csv` to your machine
  and you upload it by hand on the Zindi Submissions tab. No third-party
  package touches your Zindi credentials anywhere in this pipeline.
- **Hugging Face**: `google/WaxalNLP` isn't gated, but the dataset card
  recommends logging in (`huggingface_hub.login()`) — that prompt runs
  inside Colab too, same reasoning.

## Data — verified against the actual downloaded files

Files come from the Zindi "Data" tab (`Train.csv`, `Test.csv`,
`SampleSubmission.csv`, `Test_phase2.csv`, `Waxal_Challenge_Starter_Code.ipynb`),
placed in `data/` (gitignored — not committed, per Zindi's code-sharing rules
and to keep the repo light).

**`Train.csv`** — 38,199 rows, columns `id, transcription, language, original_split`.
This is the HF `train`+`validation` splits merged (`original_split` is only
ever `train` or `validation`) — text only, no audio.
| language | rows | train | validation |
|---|---|---|---|
| `lin` (Lingala) | 16,244 | 14,400 | 1,844 |
| `sna` (Shona) | 15,836 | 14,109 | 1,727 |
| `lug` (Luganda) | 6,119 | 5,455 | 664 |

Luganda has under half the training rows of the other two — matches the
WAXAL paper's transcribed-hours numbers below.

**Parser quirk:** 23 rows contain literal backslash-escaped quotes inside
the transcription (people quoting book/film titles, e.g. `"L'oeil sur la
nature"`). Default `pd.read_csv` throws `Expected 4 fields, saw 5` on line
9570. Fix: `pd.read_csv(..., engine="python", escapechar="\\")`. Both
`notebooks/00_data_exploration.py` and `01_zeroshot_mms_baseline.ipynb` do
this already.

**`Test.csv`** — 4,253 rows, **one column, `ID`** — no language column.
Language is recovered from the id prefix (`lin_9470` → `lin`); verified
against `Train.csv` that id-prefix always equals the language column, so
this is safe. 0 overlap with `Train.csv` ids. Distribution: `lin` 1,866,
`sna` 1,749, `lug` 638.

**`SampleSubmission.csv`** — columns `ID, Target`; ids in exactly the same
order as `Test.csv`; `Target` is the placeholder string `"XXX"`.

**`Test_phase2.csv`** — 1,500 rows, columns `ID, Target`, ids like
`ID_TBDTM` — anonymized, no language-revealing prefix (matches the rules:
Phase 2 gives audio only, no metadata). Phase 2 audio itself isn't released
until ~1 week before close, **but the submission validator already requires
predictions for these 1,500 ids too** — found the hard way, from a real
failed submission: `Wer error: Missing entries for IDs ID_TBDTM, ID_JZFXM,
...`. `SampleSubmission.csv` only lists the 4,253 Phase 1 ids, which is
misleading; the validator actually checks against the combined Phase 1 +
Phase 2 set (5,753 ids). `01_zeroshot_mms_baseline.ipynb` handles this by
filling all 1,500 Phase 2 rows with a `"."` placeholder (unscorable until
Phase 2 opens, but passes the format check) and concatenating them onto the
4,253 real Phase 1 predictions.

**`google/WaxalNLP` schema** (HF dataset card), one config per language
(`lin_asr`, `lug_asr`, `sna_asr`), splits `train`/`validation`/`test`/`unlabeled`:
```python
{
  'id': 'sna_0',
  'speaker_id': '...',
  'audio': {'array': [...], 'sampling_rate': 16_000},
  'transcription': '...',
  'language': 'sna',
  'gender': 'Female',
}
```
`Train.csv`/`Test.csv` ids join directly against this `id` field.

**Orthography / code-switching, read off the actual transcriptions:**
- **Lingala** carries real French code-switching and French loanwords/diacritics
  (`é è î ô â à ê ï ç` — e.g. *"on dirait"*, *"carburant"*, *"farine"*). A
  model with some French exposure may help here.
- **Luganda** uses `ŋ`/`Ŋ` (eng) as a real letter (466 occurrences) plus
  apostrophes for elision (`g'ennyanja`) — standard orthography, not noise.
- **Shona** shows inconsistent tone/vowel marking (`í ú à ñ ó á é ò`) across
  transcribers — expect some orthographic variance even within-language.
- One near-empty row: `lin_9193` transcription is literally `"\n"` — a data
  artifact, not evidence of others like it (only 1 found).

**From the WAXAL paper** (arXiv:2602.02734 — a data-descriptor paper, no
baselines/models/splits reported):
- Transcribed hours: **Lingala 101.5h, Shona 99.2h, Luganda 46.0h**.
- Speech is **unscripted, image-prompted** description (not read scripts) —
  expect natural disfluency, off-topic drift, background noise.
- Only **10% of collected audio was transcribed**, by paid local linguists,
  in local script where one exists, else transliterated to Latin script.
- No documented train/test split methodology (e.g. speaker-disjoint) — treat
  local validation scores as directional, not a leaderboard guarantee.

## First submission: zero-shot MMS baseline

`notebooks/01_zeroshot_mms_baseline.ipynb` — no training required. Uses
`facebook/mms-1b-all`, which ships dedicated CTC adapters for `lin`, `lug`,
and `sna` (verified against the model repo file list), so it can transcribe
all three challenge languages out of the box. Goal: get a real number on the
board fast, then iterate.

**Run it:**
1. Open in Colab, set runtime to GPU (T4 is enough).
2. Run cells top to bottom. Upload `Train.csv`, `Test.csv`,
   `SampleSubmission.csv` when prompted (from your local `data/` folder).
3. It loads only the WaxalNLP **test**-split audio (~1.3GB across all 3
   languages) to build predictions, and streams a small train+validation
   slice (no full ~10GB download) for a WER/CER sanity check.
4. Review the sanity-check numbers, then check `submission_mms_zeroshot.csv`
   once it auto-downloads.
5. Upload it by hand on Zindi's Submissions tab (costs one of your 5
   daily / 200 total submissions).

## Phase 2 correction (2 August 2026)

The organisers announced that the first Phase 2 release was incorrect and
replaced it. **Do not use the earlier Phase 2 audio.** Download the corrected
Phase 2 CSV/audio from Zindi before generating any further submission. The
copy currently in this repository's ignored `data/` directory predates that
announcement and is not treated as authoritative.

`train_mms_adapters.py` is the first trainable pipeline. It fine-tunes the
language-specific CTC adapter/head from `facebook/mms-1b-all` on each WAXAL
language, selects checkpoints using validation `0.5*WER + 0.5*CER`, identifies
the language of anonymous Phase 2 audio with `facebook/mms-lid-126`, and
generates predictions with the matching ASR adapter. It intentionally requires
the corrected Phase 2 audio directory rather than emitting placeholder text.

Two data/decoding improvements are on by default, following
[WAXAL-NET](https://arxiv.org/abs/2606.02375) (fine-tuning ablations on this
same corpus) and the standard HF wav2vec2/MMS n-gram recipe:
- **Audio filtering** — clips shorter than 1.5s or faster than 4 words/sec are
  dropped before training (`--min-audio-seconds`/`--max-words-per-second`,
  `<=0` disables). WAXAL-NET's own ablation found this alone cut Lingala WER
  from 113.5% to 49.0%.
- **KenLM beam-search rescoring** — a per-language n-gram model is built from
  the training transcriptions and used for CTC beam search at inference
  instead of greedy argmax decoding (`--build-lm`/`--no-lm`, `--lm-ngram`,
  predict's `--beam-width`). Best-effort: if the `lmplz`/`build_binary` tools
  aren't available, training logs a warning and inference falls back to
  greedy decoding automatically — nothing breaks.

The recommended execution environment is Modal. It keeps checkpoints in a
persistent Volume, builds the KenLM CLI tools into the training image, and
runs training on an A100 80GB:

```bash
pip install modal
modal setup
modal volume create waxal-data
modal volume create waxal-models
modal volume put waxal-data data/Test.csv /Test.csv
modal volume put waxal-data data/Test_Phase2.csv /Test_Phase2.csv
modal run modal_app.py --action fetch-phase2

modal run modal_app.py --action train
modal run modal_app.py --action predict
modal volume get waxal-models \
  mms_adapters/submission_mms_adapters.csv submissions/
```

The three language jobs run **concurrently** by default (each writes to its
own Volume subdirectory, so parallel commits don't collide) — trades GPU spend
(up to 3 A100s at once) for wall-clock time. Pass `--sequential` to go back to
one job at a time if you'd rather control spend than save time. You can also
train a single language using `--languages lug`, for example. No leaderboard
position can be guaranteed: retain the two strongest validation/leaderboard
submissions for private evaluation.

## Next steps after the baseline lands


- **Second part**: fine-tune. Google's own starter notebook
  (`data/Waxal_Challenge_Starter_Code.ipynb`) LoRA-fine-tunes
  `google/gemma-3n-E2B-it` (gated model, needs HF license acceptance) per
  language via PEFT + TRL SFTTrainer — one language at a time, A100
  recommended (T4 works with small batches + gradient checkpointing). It
  trains and evaluates WER/CER on the HF test split, but does **not**
  generate a `Test.csv`-shaped submission — that join-by-id step from
  `01_zeroshot_mms_baseline.ipynb` still needs to be reused on top of it.
- Alternative: fine-tune MMS's CTC adapters or Whisper-small on the WAXAL
  train split, following the multilingual fine-tuning pattern from
  `afrivoices-asr-hack` (temperature-sampled language balance, since Luganda
  is ~2.6x smaller than the other two here as well).
- Weight training/augmentation toward Luganda given its smaller pool.
- Consider the `unlabeled` split per language for semi-supervised / self-training.
- Watch generalization, not just the Phase-1 score — Phase 2's unseen set
  with no language metadata is what actually determines prizes.

## Repo structure

```
eval/
  compute_wer.py            WER/CER + leaderboard-weighted score harness (jiwer-based)
notebooks/
  00_data_exploration.py    Local, no-GPU pandas EDA on the CSVs (schema, quirks, stats)
  01_zeroshot_mms_baseline.ipynb   Colab: WaxalNLP audio, zero-shot MMS, submission.csv
data/                        (gitignored) Zindi CSVs + starter notebook live here
submissions/                 (gitignored) generated submission CSVs
```
