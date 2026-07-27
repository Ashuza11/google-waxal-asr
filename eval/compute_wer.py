"""
eval/compute_wer.py — reusable eval harness for the WAXAL ASR challenge.

Usage:
    from eval.compute_wer import evaluate
    results = evaluate(references, hypotheses)
    print(results)  # {'wer': ..., 'cer': ..., 'score': ...}

`score` is the weighted mean the leaderboard uses: 0.5 * WER + 0.5 * CER.
"""
import re
import unicodedata

from jiwer import cer, wer

WER_WEIGHT = 0.5
CER_WEIGHT = 0.5


def normalise(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.
    Applied identically to reference and hypothesis so WER/CER is fair.
    """
    text = "" if text is None else str(text)
    text = text.lower()
    # decompose unicode (ĩ -> i + combining tilde) then drop combining marks
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # keep only letters, digits, spaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate(references: list[str], hypotheses: list[str],
             normalise_text: bool = True) -> dict:
    """
    Compute WER, CER, and the leaderboard's weighted score over lists of
    reference and hypothesis strings.

    Args:
        references:      ground-truth transcripts
        hypotheses:       model output transcripts
        normalise_text:  apply lowercasing + punctuation removal first

    Returns dict with:
        wer      — word error rate  (0.0 = perfect, 1.0 = 100% errors)
        cer      — character error rate
        score    — 0.5*wer + 0.5*cer (lower is better, matches leaderboard)
        n_refs   — total reference words
        n_hyps   — total hypothesis words
    """
    assert len(references) == len(hypotheses), "lists must be same length"

    if normalise_text:
        references = [normalise(r) for r in references]
        hypotheses = [normalise(h) for h in hypotheses]

    word_error_rate = wer(references, hypotheses)
    char_error_rate = cer(references, hypotheses)

    n_ref_words = sum(len(r.split()) for r in references)
    n_hyp_words = sum(len(h.split()) for h in hypotheses)

    return {
        "wer": round(word_error_rate, 4),
        "cer": round(char_error_rate, 4),
        "score": round(WER_WEIGHT * word_error_rate + CER_WEIGHT * char_error_rate, 4),
        "n_refs": n_ref_words,
        "n_hyps": n_hyp_words,
    }


def print_report(results: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}WER = {results['wer']:.1%}  |  CER = {results['cer']:.1%}  "
          f"|  score = {results['score']:.4f}  "
          f"|  ref words = {results['n_refs']}  hyp words = {results['n_hyps']}")
