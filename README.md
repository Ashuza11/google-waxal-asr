# Google WAXAL ASR Challenge

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
Phase 2 gives audio only, no metadata). Nothing to do with this yet — Phase 2
audio isn't released until ~1 week before close.

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
