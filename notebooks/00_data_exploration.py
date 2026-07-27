"""
00_data_exploration.py — pure pandas EDA on the Zindi CSVs.

No GPU/audio needed — this only looks at the text/metadata files, so it runs
locally. Run after placing Train.csv / Test.csv / SampleSubmission.csv /
Test_phase2.csv (from the Zindi "Data" tab) in data/.

Key quirk documented here so it doesn't bite you again: Train.csv has 23
rows with literal, backslash-escaped quote characters inside the
transcription field (people quoting book/film titles, e.g. \"L'oeil sur la
nature\"). The default C parser chokes on line 9570 with
`Expected 4 fields, saw 5`. Fix: `engine="python", escapechar="\\\\"`.
"""
import collections

import pandas as pd

DATA_DIR = "data"


def load_train() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/Train.csv", engine="python", escapechar="\\")


def main() -> None:
    train = load_train()
    test = pd.read_csv(f"{DATA_DIR}/Test.csv")
    sub = pd.read_csv(f"{DATA_DIR}/SampleSubmission.csv")

    print("=== shapes ===")
    print(f"Train: {train.shape}  Test: {test.shape}  SampleSubmission: {sub.shape}")

    print("\n=== Train: rows per language ===")
    print(train["language"].value_counts())

    print("\n=== Train: language x original_split ===")
    print(pd.crosstab(train["language"], train["original_split"]))

    test["lang"] = test["ID"].str.split("_").str[0]
    print("\n=== Test: rows per language (derived from id prefix — Test.csv has no language column) ===")
    print(test["lang"].value_counts())

    print("\n=== sanity checks ===")
    print("duplicate train ids:", train["id"].duplicated().sum())
    print("train id-prefix vs language column mismatch:",
          (train["id"].str.split("_").str[0] != train["language"]).sum())
    print("test ids overlapping train ids (should be 0 — no leakage):",
          test["ID"].isin(train["id"]).sum())
    print("SampleSubmission ids match Test ids in order:",
          sub["ID"].tolist() == test["ID"].tolist())

    empty_rows = train[train["transcription"].str.strip().str.len() <= 2]
    print(f"\nnear-empty transcriptions in Train: {len(empty_rows)}")
    if len(empty_rows):
        print(empty_rows[["id", "transcription", "language"]].to_string())

    print("\n=== transcription length (words) by language ===")
    train["n_words"] = train["transcription"].str.split().str.len()
    print(train.groupby("language")["n_words"].describe())

    print("\n=== non-ASCII characters in use per language (orthography/code-switching signal) ===")
    for lang in sorted(train["language"].unique()):
        chars = collections.Counter()
        for t in train.loc[train.language == lang, "transcription"]:
            for ch in t:
                if ord(ch) > 127:
                    chars[ch] += 1
        print(f"  {lang}: {dict(chars.most_common(12))}")


if __name__ == "__main__":
    main()
