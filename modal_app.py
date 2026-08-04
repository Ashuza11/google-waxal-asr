"""Modal GPU launcher for WAXAL MMS adapter training and inference.

Setup once:
    pip install modal
    modal setup
    modal volume create waxal-data
    modal volume create waxal-models

Train all three adapters (concurrently, one A100 job each; pass --sequential
for one at a time):
    modal run modal_app.py --action train

After putting corrected Phase 2 files on the data volume, predict:
    modal run modal_app.py --action predict
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import urllib.request
import zipfile

import modal

APP_NAME = "waxal-mms-training"
MODEL_ROOT = "/models/mms_adapters"
DATA_ROOT = "/data"
PHASE2_URL = "https://storage.googleapis.com/waxalphase2/newaudios.zip"
PHASE2_BYTES = 1_086_719_156
PHASE2_MD5 = "e858c544f010bfa3d8791179f1b6a155"
PHASE2_ROWS = 892

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("waxal-models", create_if_missing=True)
data_volume = modal.Volume.from_name("waxal-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("waxal-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "build-essential",
        "cmake",
        "libboost-all-dev",
        "libeigen3-dev",
        "zlib1g-dev",
        "libbz2-dev",
        "liblzma-dev",
        "wget",
    )
    # KenLM ships no PyPI wheel with the lmplz/build_binary CLI tools, so build
    # them from source once here; train_mms_adapters.py shells out to them to
    # build a per-language n-gram model for CTC beam-search rescoring.
    .run_commands(
        "wget -qO - https://kheafield.com/code/kenlm.tar.gz | tar xz -C /opt",
        "cmake -S /opt/kenlm -B /opt/kenlm/build -DCMAKE_BUILD_TYPE=Release",
        "cmake --build /opt/kenlm/build -j$(nproc) --target lmplz build_binary",
        "install /opt/kenlm/build/bin/lmplz /opt/kenlm/build/bin/build_binary /usr/local/bin/",
    )
    .uv_pip_install(
        "torch>=2.4",
        "transformers>=4.46,<5",
        "datasets[audio]>=2.20",
        "accelerate>=0.34",
        "jiwer>=3.0",
        "pandas>=2.0",
        "librosa>=0.10",
        "soundfile>=0.12",
        "pyctcdecode>=0.5",
        "https://github.com/kpu/kenlm/archive/master.zip",
    )
    .env({"HF_HOME": "/hf-cache", "PYTHONPATH": "/workspace"})
    .add_local_file("train_mms_adapters.py", "/workspace/train_mms_adapters.py")
)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume},
)
def fetch_corrected_phase2() -> dict:
    """Download the organizers' corrected archive directly inside Modal."""
    destination = Path(DATA_ROOT) / "newaudios.zip"
    temporary = destination.with_suffix(".zip.part")
    digest = hashlib.md5()
    size = 0
    with urllib.request.urlopen(PHASE2_URL) as response, temporary.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != PHASE2_BYTES or digest.hexdigest() != PHASE2_MD5:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Corrected archive integrity failure: bytes={size}, md5={digest.hexdigest()}"
        )
    temporary.replace(destination)
    data_volume.commit()
    return {"path": str(destination), "bytes": size, "md5": digest.hexdigest()}


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=24 * 60 * 60,
    volumes={"/models": model_volume, "/hf-cache": hf_cache},
)
def train_one(lang: str, epochs: float = 8.0) -> dict:
    from train_mms_adapters import seed_everything, train_language

    args = SimpleNamespace(
        output_dir=MODEL_ROOT,
        epochs=epochs,
        learning_rate=1e-3,
        batch_size=8,
        eval_batch_size=12,
        gradient_accumulation=2,
        eval_steps=500,
        num_proc=8,
        seed=42,
        resume=False,
        min_audio_seconds=1.5,
        max_words_per_second=4.0,
        build_lm=True,
        lm_ngram=3,
    )
    seed_everything(args.seed)
    train_language(lang, args)
    model_volume.commit()
    metrics_path = Path(MODEL_ROOT) / lang / "metrics.json"
    return {"language": lang, "metrics": metrics_path.read_text()}


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=24 * 60 * 60,
    volumes={
        "/models": model_volume,
        "/data": data_volume,
        "/hf-cache": hf_cache,
    },
)
def make_submission() -> str:
    import pandas as pd

    from train_mms_adapters import predict, seed_everything

    model_volume.reload()
    data_volume.reload()
    csv_candidates = [
        Path(DATA_ROOT) / "Test_Phase2.csv",
        Path(DATA_ROOT) / "Test_phase2.csv",
    ]
    phase2_csv = next((p for p in csv_candidates if p.exists()), csv_candidates[0])
    phase2_zip = Path(DATA_ROOT) / "newaudios.zip"
    phase2_audio = Path("/tmp/corrected_phase2_audio")
    if not phase2_csv.exists() or not phase2_zip.is_file():
        raise FileNotFoundError(
            "Upload corrected Test_Phase2.csv and newaudios.zip to the "
            "waxal-data Modal Volume before prediction."
        )
    phase2_audio.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(phase2_zip) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt Phase 2 archive member: {bad}")
        audio_ids = {
            Path(name).stem
            for name in archive.namelist()
            if name.startswith("newaudios/") and name.lower().endswith(".wav")
        }
        csv_ids = set(pd.read_csv(phase2_csv).iloc[:, 0].astype(str))
        if len(audio_ids) != PHASE2_ROWS or audio_ids != csv_ids:
            raise ValueError(
                f"Corrected Phase 2 mismatch: {len(audio_ids)} audio IDs, "
                f"{len(csv_ids)} CSV IDs, overlap={len(audio_ids & csv_ids)}"
            )
        archive.extractall(phase2_audio)
    args = SimpleNamespace(
        output_dir=MODEL_ROOT,
        test_csv=str(Path(DATA_ROOT) / "Test.csv"),
        phase2_csv=str(phase2_csv),
        phase2_audio_dir=str(phase2_audio),
        submission=str(Path(MODEL_ROOT) / "submission_mms_adapters.csv"),
        inference_batch_size=12,
        beam_width=100,
        seed=42,
    )
    seed_everything(args.seed)
    predict(args)
    model_volume.commit()
    return args.submission


@app.local_entrypoint()
def main(
    action: str = "train",
    languages: str = "lin,lug,sna",
    epochs: float = 8.0,
    sequential: bool = False,
):
    selected = [x.strip() for x in languages.split(",") if x.strip()]
    invalid = set(selected) - {"lin", "lug", "sna"}
    if invalid:
        raise ValueError(f"Unsupported languages: {sorted(invalid)}")
    if action == "train":
        if sequential:
            for lang in selected:
                print(train_one.remote(lang, epochs))
        else:
            # Concurrent by default: each language writes to its own volume
            # subdirectory (lang/checkpoints, lang/best), so parallel commits
            # don't collide. Trades GPU spend (up to 3 A100s at once) for
            # wall-clock time; pass --sequential to go back to one at a time.
            calls = [train_one.spawn(lang, epochs) for lang in selected]
            for call in calls:
                print(call.get())
    elif action == "fetch-phase2":
        print(fetch_corrected_phase2.remote())
    elif action == "predict":
        print("Submission saved on Modal Volume:", make_submission.remote())
        print(
            "Download with: modal volume get waxal-models "
            "mms_adapters/submission_mms_adapters.csv submissions/"
        )
    else:
        raise ValueError("action must be 'train', 'fetch-phase2', or 'predict'")
