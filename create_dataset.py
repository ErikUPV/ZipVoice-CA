#!/usr/bin/env python3
"""
ZipVoice TSV builder (Catalan / CA) — HF-only, leakage-safe.

CV17 is split ONCE (seeded) into:
  - train
  - dev (n_dev)
  - test_pool (n_test)  -> used ONLY for building test.tsv (CV17 portion)

TRAIN mode:
  - uses cv17_splits/train + cv17_splits/dev
  - materializes audio
  - writes:
      custom_train.tsv  (path, id, text)
      custom_dev.tsv    (path, id, text)

TEST mode:
  - uses cv17_splits/test_pool for CV17 prompts (no overlap with train/dev)
  - also uses FestCat + LaFrescat prompts
  - materializes prompt audios
  - writes:
      test.tsv (names, prompt_texts, prompt_wavs, texts)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Audio


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def write_tsv_no_header(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, sep="\t", header=None, index=False)


def pick_first_existing(colnames: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in colnames:
            return c
    return None


def load_hf_split(dataset_id: str, split: Optional[str] = None, config: Optional[str] = None) -> Dataset:
    obj = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)
    if isinstance(obj, DatasetDict):
        if split is None:
            split = "train" if "train" in obj else next(iter(obj.keys()))
        return obj[split]
    return obj

def infer_text_col(ds: Dataset) -> str:
    candidates = [
        "sentence", "text", "transcription",
        "raw_transcription", "raw_text",
        "normalized_text", "transcript", "transcript_1",
    ]
    col = pick_first_existing(list(ds.column_names), candidates)
    if col is None:
        raise ValueError(f"Could not infer text column. Columns: {ds.column_names}")
    return col


def infer_id_cols(ds: Dataset) -> List[str]:
    # Prefer stable identifiers if present (Common Voice often has 'path', 'sentence_id', 'client_id')
    candidates = ["sentence_id", "id", "utt_id", "segment", "path", "client_id"]
    return [c for c in candidates if c in ds.column_names]


def materialize_audio_and_rows(
    ds: Dataset,
    audio_col: str,
    text_col: str,
    out_audio_dir: Path,
    dataset_tag: str,   # "cv17", "festcat", "frescat"
) -> pd.DataFrame:
    """
    Save audio as .wav into out_audio_dir and return DF with:
      path, id, text, dur

    ID format: {dataset_tag}_{i:08d}
    """
    ensure_dir(out_audio_dir)

    paths, ids, texts, durs = [], [], [], []

    for i, item in enumerate(tqdm(ds, desc=f"Saving audio -> {out_audio_dir}", total=len(ds))):
        a = item[audio_col]
        arr = a["array"]
        sr = int(a["sampling_rate"])

        uid = f"{dataset_tag}_{i:08d}"
        wav_path = out_audio_dir / f"{uid}.wav"
        sf.write(wav_path, arr, sr)

        frames = arr.shape[0] if hasattr(arr, "shape") else len(arr)
        dur = float(frames) / float(sr)

        paths.append(str(wav_path))
        ids.append(uid)
        texts.append(str(item[text_col]))
        durs.append(dur)

    df = pd.DataFrame(
        {
            "path": paths,
            "id": ids,
            "text": texts,
            "dur": durs,
        }
    )

    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df



# -----------------------------------------------------------------------------
# CV17 global split (persisted) to prevent leakage
# -----------------------------------------------------------------------------

def get_cv17_splits_cache_dir(out_dir: Path) -> Path:
    return out_dir / "cv17_splits"


def split_cv17_global(ds: Dataset, seed: int, n_dev: int, n_test: int) -> DatasetDict:
    """
    Deterministic split:
      train = all - n_dev - n_test
      dev = next n_dev
      test_pool = last n_test
    """
    ds = ds.shuffle(seed=seed)

    n = len(ds)
    if n_dev + n_test >= n:
        raise ValueError(f"n_dev + n_test must be < dataset size. Got {n_dev}+{n_test} >= {n}")

    n_train = n - n_dev - n_test
    return DatasetDict({
        "train": ds.select(range(0, n_train)),
        "dev": ds.select(range(n_train, n_train + n_dev)),
        "test_pool": ds.select(range(n_train + n_dev, n_train + n_dev + n_test)),
    })


def load_or_create_cv17_splits(
    dataset_id: str,
    split: Optional[str],
    config: Optional[str],
    out_dir: Path,
    seed: int,
    n_dev: int,
    n_test: int,
    max_total_items: Optional[int] = None,
) -> DatasetDict:
    """
    Creates and saves cv17_splits/ (train/dev/test_pool) once.
    On subsequent runs, loads the cached split to guarantee consistency.
    """
    ensure_dir(out_dir)
    cache_dir = get_cv17_splits_cache_dir(out_dir)
    meta_path = cache_dir / "meta.json"

    # If cache exists, validate meta and load
    if cache_dir.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {
            "dataset_id": dataset_id,
            "split": split,
            "config": config,
            "seed": seed,
            "n_dev": n_dev,
            "n_test": n_test,
            "max_total_items": max_total_items,
        }
        if meta != expected:
            # Regenerate automatically (safer than silently mixing)
            print("[WARN] Existing cv17_splits meta differs from current args. Regenerating splits...")
            shutil.rmtree(cache_dir)

        else:
            return DatasetDict({
                "train": load_from_disk(str(cache_dir / "train")),
                "dev": load_from_disk(str(cache_dir / "dev")),
                "test_pool": load_from_disk(str(cache_dir / "test_pool")),
            })

    # Create cache
    ensure_dir(cache_dir)
    ds = load_hf_split(dataset_id, split=split, config=config)

    if max_total_items is not None:
        ds = ds.select(range(min(max_total_items, len(ds))))

    splits = split_cv17_global(ds, seed=seed, n_dev=n_dev, n_test=n_test)

    splits["train"].save_to_disk(str(cache_dir / "train"))
    splits["dev"].save_to_disk(str(cache_dir / "dev"))
    splits["test_pool"].save_to_disk(str(cache_dir / "test_pool"))

    meta = {
        "dataset_id": dataset_id,
        "split": split,
        "config": config,
        "seed": seed,
        "n_dev": n_dev,
        "n_test": n_test,
        "max_total_items": max_total_items,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Created CV17 global splits under {cache_dir}")
    return splits


# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------

def build_train_tsvs_from_cv17_splits(
    splits: DatasetDict,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    train_ds = splits["train"]
    dev_ds = splits["dev"]

    # train
    train_df = materialize_audio_and_rows(
        ds=train_ds,
        audio_col="audio",
        text_col="sentence",
        out_audio_dir=out_dir / "train_audios",
        dataset_tag="cv17_train",
    )

    write_tsv_no_header(train_df[["path", "id", "text"]], out_dir / "custom_train.tsv")

    # dev
    dev_df = materialize_audio_and_rows(
        ds=dev_ds,
        audio_col="audio",
        text_col="sentence",
        out_audio_dir=out_dir / "dev_audios",
        dataset_tag="cv17_dev",
    )

    write_tsv_no_header(dev_df[["path", "id", "text"]], out_dir / "custom_dev.tsv")

    print(f"[OK] TRAIN: wrote {out_dir / 'custom_train.tsv'}")
    print(f"[OK] TRAIN: wrote {out_dir / 'custom_dev.tsv'}")


# -----------------------------------------------------------------------------
# TEST (test.tsv from CV17 test_pool + FestCat + LaFrescat)
# -----------------------------------------------------------------------------

def load_prompt_dataset(name: str, cv17_test_pool: Optional[Dataset], max_items: int, seed: int) -> Dataset:
    if name == "cv17":
        if cv17_test_pool is None:
            raise ValueError("cv17_test_pool is required for CV17 prompts.")
        ds = cv17_test_pool
    elif name == "festcat":
        ds = load_dataset("projecte-aina/festcat_trimmed_denoised", split="train")
    elif name == "frescat":
        ds = load_dataset("projecte-aina/LaFrescat", split="train")
    else:
        raise ValueError(f"Unknown prompt dataset: {name}")

    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(max_items, len(ds))))
    return ds


def build_test_tsv(
    cv17_test_pool: Dataset,
    out_dir: Path,
    max_items: int,
    max_prompt_sec: float,
    seed: int,
) -> None:
    ensure_dir(out_dir)

    prompt_sets = ["cv17", "festcat", "frescat"]
    out_dfs = []

    text_cols = {
        "cv17" : "sentence",
        "festcat": "transcription",
        "frescat": "transcription",
    }

    for name in prompt_sets:
        ds = load_prompt_dataset(name, cv17_test_pool=cv17_test_pool if name == "cv17" else None,
                                 max_items=max_items, seed=seed)



        audio_dir = out_dir / "prompt_audios" / name
        df = materialize_audio_and_rows(
            ds=ds,
            audio_col="audio",
            text_col=text_cols[name],
            out_audio_dir=audio_dir,
            dataset_tag=f"{name}_test",
        )

        ok_idx = df.index[df["dur"] <= max_prompt_sec].tolist()
        if not ok_idx:
            raise ValueError(f"No prompt audio <= {max_prompt_sec}s found in {name}")

        rng = random.Random(seed)

        names, prompt_texts, prompt_wavs, texts = [], [], [], []
        for i in range(len(df)):
            # avoid self-prompt if possible
            candidate_ok = ok_idx
            if len(ok_idx) > 1 and i in ok_idx:
                candidate_ok = [j for j in ok_idx if j != i]

            chosen_i = rng.choice(candidate_ok)
            chosen = df.loc[chosen_i]

            out_name = f"{name}/{name}_test_{i:08d}.wav"
            
            names.append(out_name)
            prompt_texts.append(chosen["text"])
            prompt_wavs.append(os.path.abspath(chosen["path"]))
            texts.append(df.loc[i, "text"])

        out_df = pd.DataFrame({
            "names": names,
            "prompt_texts": prompt_texts,
            "prompt_wavs": prompt_wavs,
            "texts": texts,
        })
        out_dfs.append(out_df)

    final_df = pd.concat(out_dfs, ignore_index=True)
    write_tsv_no_header(final_df, out_dir / "test.tsv")

    print(f"[OK] TEST: wrote {out_dir / 'test.tsv'}")
    print(f"[OK] TEST: prompt audios saved under {out_dir / 'prompt_audios'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZipVoice TSV builder.")

    p.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Generate train/dev TSVs or test.tsv.")
    p.add_argument("--out_dir", type=str, default="data/raw_ca", help="Output directory for TSVs and saved audios.")
    p.add_argument("--seed", type=int, default=42, help="Seed for deterministic split + shuffles.")

    # CV17 (shared)
    p.add_argument("--cv17_dataset_id", type=str, default="ebellob/annotated_catalan_common_voice_v17_cleaned_enhanced",
                   help="HF dataset id for CV17 (train/dev/test_pool split source).")
    p.add_argument("--cv17_split", type=str, default=None,
                   help="Which HF split to use as the source pool (default: 'train' if exists else first).")
    p.add_argument("--cv17_config", type=str, default=None, help="Optional HF config name for CV17.")
    p.add_argument("--max_cv17_items", type=int, default=None,
                   help="Optional cap on total CV17 items BEFORE splitting (mainly for quick experiments).")

    # split sizes
    p.add_argument("--n_dev", type=int, default=1000, help="CV17 dev size (held out from training).")
    p.add_argument("--n_test", type=int, default=1000, help="CV17 test_pool size (used ONLY for test.tsv prompts).")

    # test.tsv params
    p.add_argument("--max_items", type=int, default=1000, help="Max items per prompt dataset (cv17/festcat/frescat).")
    p.add_argument("--max_prompt_sec", type=float, default=10.0, help="Max duration allowed for prompt audio.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Always load/create the global CV17 split so both modes share it
    splits = load_or_create_cv17_splits(
        dataset_id=args.cv17_dataset_id,
        split=args.cv17_split,
        config=args.cv17_config,
        out_dir=out_dir,
        seed=args.seed,
        n_dev=args.n_dev,
        n_test=args.n_test,
        max_total_items=args.max_cv17_items,
    )

    if args.mode == "train":
        build_train_tsvs_from_cv17_splits(splits=splits, out_dir=out_dir)

    elif args.mode == "test":
        # CV17 prompts MUST come from test_pool to avoid leakage
        cv17_test_pool = splits["test_pool"]
        build_test_tsv(
            cv17_test_pool=cv17_test_pool,
            out_dir=out_dir,
            max_items=args.max_items,
            max_prompt_sec=args.max_prompt_sec,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
