"""Train language-specific MMS CTC adapters and create a WAXAL submission.

Designed for a Colab/Kaggle A100 (T4 works with smaller batches). The script
never trains on the Hugging Face test split: it is used only to resolve Phase 1
audio IDs. Corrected Phase 2 audio must be downloaded from Zindi by the user.

Training also, by default: (1) drops clips shorter than 1.5s or faster than
4 words/sec before fine-tuning (`--min-audio-seconds`/`--max-words-per-second`,
see WAXAL-NET arXiv:2606.02375), and (2) builds a per-language KenLM n-gram
model from the training transcriptions for CTC beam-search rescoring at
inference (`--build-lm`/`--no-lm`, `--lm-ngram`). The LM build is best-effort:
if `lmplz`/`build_binary` aren't on PATH, it's skipped and inference falls
back to greedy decoding automatically.

Example:
    python train_mms_adapters.py train --output-dir outputs/mms
    python train_mms_adapters.py predict \
      --output-dir outputs/mms --phase2-csv data/Test_phase2.csv \
      --phase2-audio-dir data/phase2_audio --submission submissions/mms.csv
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Audio, concatenate_datasets, load_dataset
from jiwer import cer, wer
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2ProcessorWithLM,
)

LANGS = ("lin", "lug", "sna")
BASE_MODEL = "facebook/mms-1b-all"
LID_MODEL = "facebook/mms-lid-126"
DEFAULT_BEAM_WIDTH = 100


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


@dataclass
class CTCDataCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(audio, padding=True, return_tensors="pt")
        label_batch = self.processor.pad(labels=labels, padding=True, return_tensors="pt")
        batch["labels"] = label_batch.input_ids.masked_fill(
            label_batch.attention_mask.ne(1), -100
        )
        return batch


def keep_by_duration_and_rate(
    row: dict[str, Any], min_seconds: float, max_words_per_second: float
) -> bool:
    """Drop clips shorter than min_seconds or faster than max_words_per_second.

    WAXAL-NET (arXiv:2606.02375) found this single filter cut Lingala WER from
    113.5% to 49.0% before any fine-tuning — short clips and unrealistically
    fast "speech" are mostly misaligned transcriptions, not hard examples.
    A threshold <= 0 disables that half of the check.
    """
    audio = row["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    if min_seconds > 0 and duration < min_seconds:
        return False
    if max_words_per_second > 0 and duration > 0:
        n_words = len(str(row["transcription"]).split())
        if n_words / duration > max_words_per_second:
            return False
    return True


def build_language_model(texts: list[str], destination: Path, ngram: int) -> Path | None:
    """Build a KenLM n-gram model from training transcriptions for CTC beam-search rescoring.

    Requires the `lmplz`/`build_binary` command-line tools on PATH (built into
    the Modal training image). Returns None on any failure so callers fall
    back to greedy decoding instead of breaking the pipeline.
    """
    if shutil.which("lmplz") is None or shutil.which("build_binary") is None:
        print("kenlm tools not found on PATH; skipping LM build, will decode greedily")
        return None
    corpus = "\n".join(t for t in (normalise(x) for x in texts) if t)
    if not corpus:
        return None
    corpus_path = destination / "lm_corpus.txt"
    arpa_path = destination / f"{ngram}gram.arpa"
    binary_path = destination / f"{ngram}gram.bin"
    corpus_path.write_text(corpus, encoding="utf-8")
    try:
        with corpus_path.open("rb") as source, arpa_path.open("wb") as sink:
            subprocess.run(
                ["lmplz", "-o", str(ngram), "--discount_fallback", "-S", "20%"],
                stdin=source,
                stdout=sink,
                check=True,
            )
        subprocess.run(["build_binary", str(arpa_path), str(binary_path)], check=True)
    except subprocess.CalledProcessError as error:
        print(f"kenlm build failed ({error}); skipping LM, will decode greedily")
        return None
    finally:
        corpus_path.unlink(missing_ok=True)
        arpa_path.unlink(missing_ok=True)
    return binary_path


def build_processor_with_lm(processor: Any, lm_binary_path: Path) -> Wav2Vec2ProcessorWithLM:
    from pyctcdecode import build_ctcdecoder

    vocab_dict = processor.tokenizer.get_vocab()
    labels = [item[0].lower() for item in sorted(vocab_dict.items(), key=lambda item: item[1])]
    decoder = build_ctcdecoder(labels=labels, kenlm_model_path=str(lm_binary_path))
    return Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder,
    )


def metric_fn(processor: Any):
    def compute(prediction: Any) -> dict[str, float]:
        pred_ids = np.argmax(prediction.predictions, axis=-1)
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hypotheses = [normalise(x) for x in processor.batch_decode(pred_ids)]
        references = [
            normalise(x) for x in processor.batch_decode(label_ids, group_tokens=False)
        ]
        w, c = wer(references, hypotheses), cer(references, hypotheses)
        return {"wer": w, "cer": c, "score": 0.5 * w + 0.5 * c}

    return compute


def train_language(lang: str, args: argparse.Namespace) -> None:
    destination = Path(args.output_dir) / lang
    destination.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, target_lang=lang)
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL, target_lang=lang, ignore_mismatched_sizes=True
    )
    model.freeze_base_model()  # train the small language adapter/head, not all 1B weights
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = True

    train_ds = load_dataset("google/WaxalNLP", f"{lang}_asr", split="train")
    valid_ds = load_dataset("google/WaxalNLP", f"{lang}_asr", split="validation")
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
    valid_ds = valid_ds.cast_column("audio", Audio(sampling_rate=16_000))

    if args.min_audio_seconds > 0 or args.max_words_per_second > 0:
        fn_kwargs = {
            "min_seconds": args.min_audio_seconds,
            "max_words_per_second": args.max_words_per_second,
        }
        before = len(train_ds)
        train_ds = train_ds.filter(keep_by_duration_and_rate, fn_kwargs=fn_kwargs, num_proc=args.num_proc)
        valid_ds = valid_ds.filter(keep_by_duration_and_rate, fn_kwargs=fn_kwargs, num_proc=args.num_proc)
        print(
            f"[{lang}] audio filter (>= {args.min_audio_seconds}s, "
            f"<= {args.max_words_per_second} words/s): train {before} -> {len(train_ds)}"
        )

    train_texts = list(train_ds["transcription"])

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        audio = row["audio"]
        row["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        row["labels"] = processor(text=normalise(row["transcription"])).input_ids
        return row

    remove = train_ds.column_names
    train_ds = train_ds.map(prepare, remove_columns=remove, num_proc=args.num_proc)
    valid_ds = valid_ds.map(prepare, remove_columns=valid_ds.column_names, num_proc=args.num_proc)

    training_args = TrainingArguments(
        output_dir=str(destination / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.08,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=False,
        group_by_length=True,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=CTCDataCollator(processor),
        compute_metrics=metric_fn(processor),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    metrics = trainer.evaluate()
    trainer.save_model(str(destination / "best"))
    processor.save_pretrained(str(destination / "best"))
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.build_lm:
        lm_path = build_language_model(train_texts, destination, args.lm_ngram)
        if lm_path is not None:
            try:
                processor_with_lm = build_processor_with_lm(processor, lm_path)
                processor_with_lm.save_pretrained(str(destination / "best_lm"))
                print(f"[{lang}] saved {args.lm_ngram}-gram LM decoder to {destination / 'best_lm'}")
            except Exception as error:  # pyctcdecode/vocab mismatch -> fall back to greedy
                print(f"[{lang}] failed to build LM-aware processor ({error}); using greedy decoding")


def audio_array(path: Path) -> np.ndarray:
    import librosa

    values, _ = librosa.load(path, sr=16_000, mono=True)
    return values.astype(np.float32)


def load_inference_processor(model_dir: Path) -> tuple[Any, bool]:
    """Prefer the LM-aware processor saved alongside `best`, if training built one."""
    lm_dir = model_dir.parent / "best_lm"
    if lm_dir.exists():
        return Wav2Vec2ProcessorWithLM.from_pretrained(lm_dir), True
    return AutoProcessor.from_pretrained(model_dir), False


def transcribe(
    paths_or_arrays: list[Any],
    model_dir: Path,
    batch_size: int,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> list[str]:
    processor, use_lm = load_inference_processor(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    result: list[str] = []
    for start in range(0, len(paths_or_arrays), batch_size):
        chunk = paths_or_arrays[start : start + batch_size]
        arrays = [audio_array(x) if isinstance(x, Path) else x for x in chunk]
        inputs = processor(arrays, sampling_rate=16_000, padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(
                input_values=inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device),
            ).logits
        if use_lm:
            texts = processor.batch_decode(logits.cpu().numpy(), beam_width=beam_width).text
        else:
            texts = processor.batch_decode(logits.argmax(-1))
        result.extend(normalise(x) or "." for x in texts)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def identify_languages(paths: list[Path], batch_size: int) -> list[str]:
    extractor = AutoFeatureExtractor.from_pretrained(LID_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(LID_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output: list[str] = []
    for start in range(0, len(paths), batch_size):
        arrays = [audio_array(p) for p in paths[start : start + batch_size]]
        inputs = extractor(arrays, sampling_rate=16_000, padding=True, return_tensors="pt")
        with torch.inference_mode():
            ids = model(**{k: v.to(device) for k, v in inputs.items()}).logits.argmax(-1)
        labels = [model.config.id2label[int(i)] for i in ids]
        # Restrict uncertain global LID output to the three competition languages.
        if any(label not in LANGS for label in labels):
            allowed = torch.tensor(
                [i for i, label in model.config.id2label.items() if label in LANGS], device=device
            )
            with torch.inference_mode():
                logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            ids = allowed[logits[:, allowed].argmax(-1)]
            labels = [model.config.id2label[int(i)] for i in ids]
        output.extend(labels)
    return output


def resolve_audio(audio_dir: Path, sample_id: str) -> Path:
    candidates = [p for ext in ("wav", "mp3", "flac", "ogg", "m4a") for p in audio_dir.rglob(f"{sample_id}.{ext}")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one audio file for {sample_id}, found {len(candidates)}")
    return candidates[0]


def predict(args: argparse.Namespace) -> None:
    test = pd.read_csv(args.test_csv)
    phase2 = pd.read_csv(args.phase2_csv)
    if len(phase2) == 0:
        raise ValueError("Corrected Phase 2 CSV is empty")
    phase2_ids = phase2.iloc[:, 0].astype(str).tolist()
    phase2_paths = [resolve_audio(Path(args.phase2_audio_dir), x) for x in phase2_ids]

    predictions: dict[str, str] = {}
    for lang in LANGS:
        ds = load_dataset("google/WaxalNLP", f"{lang}_asr", split="test").cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
        wanted = set(test.loc[test.ID.str.startswith(f"{lang}_"), "ID"])
        rows = [r for r in ds if r["id"] in wanted]
        predictions.update(
            zip(
                [r["id"] for r in rows],
                transcribe(
                    [r["audio"]["array"] for r in rows],
                    Path(args.output_dir) / lang / "best",
                    args.inference_batch_size,
                    args.beam_width,
                ),
            )
        )

    phase2_langs = identify_languages(phase2_paths, args.inference_batch_size)
    for lang in LANGS:
        indices = [i for i, value in enumerate(phase2_langs) if value == lang]
        hypotheses = transcribe(
            [phase2_paths[i] for i in indices],
            Path(args.output_dir) / lang / "best",
            args.inference_batch_size,
            args.beam_width,
        )
        predictions.update((phase2_ids[i], h) for i, h in zip(indices, hypotheses))

    ids = test.ID.astype(str).tolist() + phase2_ids
    missing = [x for x in ids if x not in predictions]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} predictions; examples: {missing[:5]}")
    submission = pd.DataFrame({"ID": ids, "Target": [predictions[x] for x in ids]})
    Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission, index=False)
    print(f"Saved {len(submission)} predictions to {args.submission}")
    print("Phase 2 language counts:", pd.Series(phase2_langs).value_counts().to_dict())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output-dir", default="outputs/mms_adapters")
    train.add_argument("--languages", nargs="+", choices=LANGS, default=list(LANGS))
    train.add_argument("--epochs", type=float, default=8.0)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--eval-batch-size", type=int, default=8)
    train.add_argument("--gradient-accumulation", type=int, default=4)
    train.add_argument("--eval-steps", type=int, default=500)
    train.add_argument("--num-proc", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--min-audio-seconds",
        type=float,
        default=1.5,
        help="Drop clips shorter than this (<=0 disables). See WAXAL-NET arXiv:2606.02375.",
    )
    train.add_argument(
        "--max-words-per-second",
        type=float,
        default=4.0,
        help="Drop clips faster than this word rate (<=0 disables).",
    )
    train.add_argument(
        "--build-lm",
        dest="build_lm",
        action="store_true",
        default=True,
        help="Build a KenLM n-gram model per language for beam-search rescoring (default on).",
    )
    train.add_argument("--no-lm", dest="build_lm", action="store_false")
    train.add_argument("--lm-ngram", type=int, default=3)

    pred = commands.add_parser("predict")
    pred.add_argument("--output-dir", default="outputs/mms_adapters")
    pred.add_argument("--test-csv", default="data/Test.csv")
    pred.add_argument("--phase2-csv", default="data/Test_phase2.csv")
    pred.add_argument("--phase2-audio-dir", required=True)
    pred.add_argument("--submission", default="submissions/mms_adapters.csv")
    pred.add_argument("--inference-batch-size", type=int, default=8)
    pred.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    pred.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = parser().parse_args()
    seed_everything(args.seed)
    if args.command == "train":
        for language in args.languages:
            train_language(language, args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
