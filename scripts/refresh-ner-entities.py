#!/usr/bin/env python3
"""Refresh the deterministic Indonesian NER cache used by the network notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

MODEL_ID = "cahya/bert-base-indonesian-NER"
MODEL_REVISION = "a3a3fa494cf7555ef87f446af5e826de3ed181c0"
MODEL_LICENSE = "MIT"
AGGREGATION_STRATEGY = "simple"
MAX_LENGTH = 512
STRIDE = 64
BATCH_SIZE = 16
SEED = 42
INFERENCE_SCHEMA_VERSION = 3

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "data" / "google_fact_check_tool_id.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "google_fact_check_ner_entities_id.parquet"
TMP_DIR = REPO_ROOT / "tmp"
HF_HOME = TMP_DIR / "huggingface"
MODEL_SNAPSHOT_DIR = HF_HOME / f"{MODEL_ID.replace('/', '--')}--{MODEL_REVISION}"
MODEL_FILE_SHA256 = {
    "README.md": "993107ecdd3abe478ebf96e5b9b261be31026ea28ba97331dacf5a94522a8e89",
    "config.json": "541dcf6350f78fbb1c45fba6fd4de993951abd80732a2cd2230e18b90e869da1",
    "pytorch_model.bin": "dc1b3af2f0327f4c2a16b0dc8630e5e19095493ec1bf61bd78495924187295a5",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer_config.json": "1fa41aa86183c3b5f1a1aff11269a7edcb4a2722d754df7ffee8546cd371731f",
    "vocab.txt": "fd533992b2175d1b86180f830c64db6adea7247138ad9056a2b0378af098af69",
}
MODEL_FILES = list(MODEL_FILE_SHA256)

ENTITY_STRUCT = pl.Struct(
    [
        pl.Field("mention_text", pl.Utf8),
        pl.Field("start_char", pl.Int64),
        pl.Field("end_char", pl.Int64),
        pl.Field("raw_label", pl.Utf8),
        pl.Field("entity_type", pl.Utf8),
        pl.Field("score", pl.Float64),
        pl.Field("normalized_text", pl.Utf8),
    ]
)
CACHE_SCHEMA = {
    "claim_text": pl.Utf8,
    "claim_text_sha256": pl.Utf8,
    "token_count": pl.Int64,
    "window_count": pl.Int64,
    "entities": pl.List(ENTITY_STRUCT),
    "model_id": pl.Utf8,
    "model_revision": pl.Utf8,
    "model_license": pl.Utf8,
    "model_file_sha256_json": pl.Utf8,
    "model_label_mapping_json": pl.Utf8,
    "tokenizer_class": pl.Utf8,
    "aggregation_strategy": pl.Utf8,
    "max_length": pl.Int64,
    "stride": pl.Int64,
    "batch_size": pl.Int64,
    "seed": pl.Int64,
    "inference_schema_version": pl.Int64,
    "source_text_hash": pl.Utf8,
}


def normalize_entity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def corpus_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def source_texts() -> list[str]:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_PATH}; run notebooks/00_google_fact_check_tool_id.py first."
        )
    texts = (
        pl.read_parquet(SOURCE_PATH, columns=["claim_text"])
        .filter(pl.col("claim_text").is_not_null() & (pl.col("claim_text").str.len_chars() > 0))
        .select("claim_text")
        .unique()
        .sort("claim_text")
        .get_column("claim_text")
        .to_list()
    )
    if not texts:
        raise ValueError("The source Parquet contains no non-empty claim text.")
    return texts


def cache_matches(texts: list[str], source_hash: str) -> bool:
    if not OUTPUT_PATH.exists():
        return False
    cache = pl.read_parquet(OUTPUT_PATH)
    expected_metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "model_file_sha256_json": json.dumps(MODEL_FILE_SHA256, sort_keys=True, separators=(",", ":")),
        "tokenizer_class": "BertTokenizerFast",
        "aggregation_strategy": AGGREGATION_STRATEGY,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "inference_schema_version": INFERENCE_SCHEMA_VERSION,
        "source_text_hash": source_hash,
    }
    if cache.schema != pl.Schema(CACHE_SCHEMA):
        return False
    if cache.height != len(texts) or cache.get_column("claim_text").n_unique() != len(texts):
        return False
    if cache.get_column("claim_text").to_list() != texts:
        return False
    return all(
        cache.get_column(column).n_unique() == 1
        and cache.get_column(column)[0] == expected
        for column, expected in expected_metadata.items()
    )


def deduplicate_entities(text: str, predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    by_span: dict[tuple[int, int, str, str], dict[str, object]] = {}
    for prediction in predictions:
        start = int(prediction["start"])
        end = int(prediction["end"])
        raw_label = str(prediction.get("entity_group") or prediction.get("entity"))
        mention_text = text[start:end]
        normalized_text = normalize_entity(mention_text)
        if not (0 <= start < end <= len(text)) or not normalized_text:
            continue
        entity_type = {"PER": "PERSON", "ORG": "ORGANIZATION"}.get(raw_label, raw_label)
        key = (start, end, raw_label, normalized_text)
        candidate = {
            "mention_text": mention_text,
            "start_char": start,
            "end_char": end,
            "raw_label": raw_label,
            "entity_type": entity_type,
            "score": float(prediction["score"]),
            "normalized_text": normalized_text,
        }
        previous = by_span.get(key)
        if previous is None or float(candidate["score"]) > float(previous["score"]):
            by_span[key] = candidate
    return [by_span[key] for key in sorted(by_span)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model_snapshot() -> Path:
    MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        destination = MODEL_SNAPSHOT_DIR / filename
        expected_hash = MODEL_FILE_SHA256[filename]
        if destination.is_file() and file_sha256(destination) == expected_hash:
            continue
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.unlink(missing_ok=True)
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{filename}"
        print(f"Downloading pinned model file {filename}…", flush=True)
        subprocess.run(
            [
                "curl", "--fail", "--location", "--retry", "3",
                "--output", str(temporary), url,
            ],
            check=True,
        )
        downloaded_hash = file_sha256(temporary)
        if downloaded_hash != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {filename}: {downloaded_hash}, expected {expected_hash}."
            )
        os.replace(temporary, destination)
    if (MODEL_SNAPSHOT_DIR / "pytorch_model.bin").stat().st_size < 100_000_000:
        raise RuntimeError("The downloaded model weight file is unexpectedly small.")
    provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "file_sha256": MODEL_FILE_SHA256,
    }
    (MODEL_SNAPSHOT_DIR / "MODEL_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MODEL_SNAPSHOT_DIR


def refresh(force: bool) -> None:
    texts = source_texts()
    source_hash = corpus_hash(texts)
    if not force and cache_matches(texts, source_hash):
        print(
            f"NER cache is current: {OUTPUT_PATH.relative_to(REPO_ROOT)} "
            f"({len(texts):,} claim texts, source hash {source_hash[:12]}…)"
        )
        return

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    HF_HOME.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    model_snapshot = ensure_model_snapshot()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Overlapping-window inference requires a fast tokenizer.")
    tokenizer.model_max_length = MAX_LENGTH
    model = AutoModelForTokenClassification.from_pretrained(
        model_snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    label_mapping_json = json.dumps(
        {str(key): value for key, value in sorted(model.config.id2label.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    ner = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=AGGREGATION_STRATEGY,
        stride=STRIDE,
        device=-1,
    )

    token_lengths = [
        len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        for text in texts
    ]
    window_capacity = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)
    window_step = window_capacity - STRIDE
    window_counts = [
        1 if token_count <= window_capacity
        else 1 + (token_count - window_capacity + window_step - 1) // window_step
        for token_count in token_lengths
    ]

    predictions = ner(texts, batch_size=BATCH_SIZE)
    rows = []
    for text, token_count, window_count, text_predictions in zip(
        texts, token_lengths, window_counts, predictions, strict=True
    ):
        rows.append(
            {
                "claim_text": text,
                "claim_text_sha256": text_sha256(text),
                "token_count": token_count,
                "window_count": window_count,
                "entities": deduplicate_entities(text, text_predictions),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_license": MODEL_LICENSE,
                "model_file_sha256_json": json.dumps(MODEL_FILE_SHA256, sort_keys=True, separators=(",", ":")),
                "model_label_mapping_json": label_mapping_json,
                "tokenizer_class": tokenizer.__class__.__name__,
                "aggregation_strategy": AGGREGATION_STRATEGY,
                "max_length": MAX_LENGTH,
                "stride": STRIDE,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "inference_schema_version": INFERENCE_SCHEMA_VERSION,
                "source_text_hash": source_hash,
            }
        )

    cache = pl.DataFrame(rows, schema=CACHE_SCHEMA, strict=True).sort("claim_text")
    assert cache.height == len(texts)
    assert cache.get_column("claim_text").n_unique() == len(texts)
    assert cache.get_column("source_text_hash").unique().to_list() == [source_hash]

    temporary_path = TMP_DIR / f"{OUTPUT_PATH.name}.tmp"
    cache.write_parquet(temporary_path, compression="zstd")
    os.replace(temporary_path, OUTPUT_PATH)
    print(
        f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} with {cache.height:,} claim texts, "
        f"{sum(len(entities) for entities in cache['entities']):,} entity spans, "
        f"source hash {source_hash[:12]}…, model revision {MODEL_REVISION[:12]}…."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Run inference even when the cache matches.")
    arguments = parser.parse_args()
    refresh(force=arguments.force)


if __name__ == "__main__":
    main()
