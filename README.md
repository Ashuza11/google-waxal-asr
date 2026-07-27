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

- **Local (this repo):** eval harness, project scaffolding, notes.
- **Colab (`notebooks/`):** everything that needs a GPU — data download, model
  inference/training, submission generation.
- **Zindi auth** (username + password) happens *inside* the Colab notebook via
  a masked `getpass()` prompt. It never passes through this chat or any tool
  call — that's intentional, not an oversight. The `zindi` pip package
  (`pip install zindi`, `from zindi.user import Zindian`) needs a
  username/password login, not a bare API token, despite how "API token" is
  sometimes used informally on Zindi's docs.

## Data

Zindi's `Train.csv` / `Test.csv` / `SampleSubmission.csv` are (presumed, pending
your download) manifests over the HF dataset — matched to actual audio by `id`.
The notebook prints the real columns and validates the id match before doing
anything else; don't assume the schema below is exact until that cell runs.

**`google/WaxalNLP` schema** (confirmed via the HF dataset card), one config
per language (`lin_asr`, `lug_asr`, `sna_asr`), splits `train`/`validation`/
`test`/`unlabeled`:
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

**From the WAXAL paper** (arXiv:2602.02734 — a data-descriptor paper, no
baselines/models/splits reported):
- Transcribed hours: **Lingala 101.5h, Shona 99.2h, Luganda 46.0h** — Luganda
  is the lowest-resource of the three and likely the hardest to model.
- Speech is **unscripted, image-prompted** description (not read scripts) —
  expect natural disfluency, off-topic drift, background noise.
- Only **10% of collected audio was transcribed**, by paid local linguists,
  in local script where one exists, else transliterated to Latin script —
  expect orthographic inconsistency, especially for Luganda tone/vowel length
  marking.
- No documented train/test split methodology (e.g. speaker-disjoint) — treat
  with caution when judging local validation scores vs. leaderboard.

## First submission: zero-shot MMS baseline

`notebooks/01_zeroshot_mms_baseline.ipynb` — no training required. Uses
`facebook/mms-1b-all`, which ships dedicated CTC adapters for `lin`, `lug`,
and `sna` (verified against the model repo file list), so it can transcribe
all three challenge languages out of the box. Goal: get a real number on the
board fast, then iterate.

**Run it:**
1. Open in Colab, set runtime to GPU (T4 is enough).
2. Run cells top to bottom. You'll be prompted for your Zindi username/password
   (masked) to download `Train.csv`/`Test.csv`/`SampleSubmission.csv` and the
   official starter notebook.
3. Check the printed column names / id-match counts in cells 3–4 — if `0`
   ids match, the id format differs from raw WaxalNLP ids and the matching
   logic needs a tweak (e.g. stripped prefix) before continuing.
4. Review the sanity-check WER/CER printed against `Train.csv` ground truth.
5. `submission_mms_zeroshot.csv` is written to `./dataset/`. Inspect it.
6. Flip `DO_SUBMIT = True` in the last cell to submit via the `zindi` package
   (uses one of your 5 daily / 200 total submissions) — or just download the
   CSV and upload it by hand on the Zindi submission page.

## Next steps after the baseline lands

- Fine-tune (Whisper-small or MMS adapters) on the WAXAL train split proper —
  zero-shot MMS is a floor, not a ceiling.
- Weight training toward Luganda given its smaller transcribed pool.
- Consider the `unlabeled` split per language for semi-supervised / self-training,
  same pattern as the `afrivoices-asr-hack` project's later rounds.
- Watch generalization, not just Phase-1 leaderboard score — Phase 2's unseen
  set with no language metadata is what actually determines prizes.

## Repo structure

```
eval/
  compute_wer.py     WER/CER + leaderboard-weighted score harness (jiwer-based)
notebooks/
  01_zeroshot_mms_baseline.ipynb   Colab: data download, zero-shot MMS, submission.csv
data/                 (gitignored) Zindi CSVs land here via the notebook
submissions/           (gitignored) generated submission CSVs
```
