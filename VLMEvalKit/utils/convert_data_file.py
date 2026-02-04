#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsv_fix_paths_and_options.py

- 自动探测编码 -> 转 UTF-8
- 规范路径分隔符
- 重命名 options_A/B/C/D -> A/B/C/D
- image_path / question_image_path:
  * 去掉首段 strip_token（默认 MMBench_EN_Arial_22）
  * 相对路径加上 --image-prefix
  * question_image_path 支持中英文逗号/分号分隔，逐个处理、去重、并最终写成
    [abs1, abs2] 这样的方括号字符串
- 选行：--limit START END  (0 基, [START, END), END<0 表示到末尾)
"""

import os
import re
import csv
import argparse
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd

CANDIDATES = [
    "utf-8-sig", "utf-8",
    "gb18030", "gbk",
    "big5", "cp950",
    "cp1252", "latin-1",
]

DEFAULT_PATH_COLS = {
    "image", "images", "img", "img_path", "image_path", "imagefile", "filepath",
    "file_path", "file", "filename", "relative_path", "path", "video", "video_path",
    "question_image", "question_img", "question_image_path"
}

EXT_RE = re.compile(r"\.(png|jpg|jpeg|bmp|gif|webp|tif|tiff|svg|mp4|avi|mov|mkv|json|txt|csv|tsv)$", re.I)
OPTIONS_COL_RE = re.compile(r"^options?_([A-Za-z])$", re.I)
PROTOCOL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
# 支持英文/中文逗号、分号
MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")


# ---------------- encoding helpers ----------------
def choose_strict_encoding(path: str, candidates: List[str]) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in candidates:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def decode_clean_to_utf8(in_path: str, out_path: str, enc: Optional[str] = None) -> str:
    if enc is None:
        enc = choose_strict_encoding(in_path, CANDIDATES)
    try:
        with open(in_path, "r", encoding=enc, errors="strict", newline="") as fin, \
             open(out_path, "w", encoding="utf-8", errors="strict", newline="\n") as fout:
            for line in fin:
                line = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", line)
                fout.write(line)
    except UnicodeDecodeError:
        with open(in_path, "r", encoding=enc, errors="replace", newline="") as fin, \
             open(out_path, "w", encoding="utf-8", errors="strict", newline="\n") as fout:
            for line in fin:
                line = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", line)
                fout.write(line)
    return enc


# ---------------- path helpers ----------------
def looks_like_path(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if ("\\" in value) or ("/" in value):
        return True
    return EXT_RE.search(value) is not None


def detect_path_columns(df: pd.DataFrame, default_names: set) -> List[str]:
    cols = []
    for col in df.columns:
        name_l = str(col).strip().lower()
        if name_l in default_names:
            cols.append(col)
            continue
        s = df[col].astype(str)
        sample = s.head(200)
        total = len(sample)
        if total == 0:
            continue
        pathish = sum(looks_like_path(x) for x in sample)
        if pathish / total >= 0.10:
            cols.append(col)
    return cols


def normalize_to_unix_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    q = p.strip().replace("\\", "/")
    q = re.sub(r"(?<!:)/{2,}", "/", q)
    return q


def strip_first_component(p: str, comp: str) -> str:
    q = normalize_to_unix_path(p)
    if not comp:
        return q
    comp_esc = re.escape(comp)
    q = re.sub(rf"^(?:\./)?{comp_esc}(?=/|$)", "", q, count=1)
    q = re.sub(r"^/+", "", q)
    return q


def add_prefix_if_relative(p: str, prefix: str) -> str:
    if not isinstance(p, str) or p == "":
        return p
    if PROTOCOL_RE.match(p) or p.startswith("/"):
        return normalize_to_unix_path(p)
    prefix = normalize_to_unix_path(prefix)
    p_norm = normalize_to_unix_path(p)
    if p_norm.startswith(prefix.rstrip("/") + "/"):
        return p_norm
    return prefix.rstrip("/") + "/" + p_norm.lstrip("/")


def process_multi_paths_field_to_bracket_string(field: str, prefix: str, strip_token: str) -> str:
    """
    输入可为 "a; b" / "a，b" / "[a, b]" 等。
    输出严格为: "[abs1, abs2]"；去重且保序；空 -> "[]"
    """
    if not isinstance(field, str):
        return "[]"
    raw = field.strip()
    if raw == "" or raw.lower() in {"nan", "none", "null"}:
        return "[]"
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()

    parts = MULTI_SPLIT_RE.split(raw)
    seen, uniq = set(), []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p2 = strip_first_component(p, strip_token)
        p2 = add_prefix_if_relative(p2, prefix)
        p2 = normalize_to_unix_path(p2)
        if p2 not in seen:
            seen.add(p2)
            uniq.append(f'"{p2}"')
    return "[" + ", ".join(uniq) + "]" if uniq else "[]"


# ---------------- rename options_* ----------------
def rename_options_columns(df: pd.DataFrame, overwrite: bool = False) -> Dict[str, str]:
    applied: Dict[str, str] = {}
    for col in list(df.columns):
        m = OPTIONS_COL_RE.match(str(col).strip())
        if not m:
            continue
        letter = m.group(1).upper()
        target = letter
        if target in df.columns:
            if not overwrite:
                continue
            df.drop(columns=[target], inplace=True)
        df.rename(columns={col: target}, inplace=True)
        applied[col] = target
    return applied


# ---------------- main pipeline ----------------
def convert_and_normalize_paths_and_rename(
    in_path: str,
    out_path: str,
    path_cols: Optional[List[str]] = None,
    force_rename: bool = False,
    image_prefix: Optional[str] = None,
    strip_token: str = "MMMU_val_Arial_16",
    rows_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    tmp_utf8_path = out_path + ".__tmp__"

    enc = decode_clean_to_utf8(in_path, tmp_utf8_path, enc=None)
    df = pd.read_csv(tmp_utf8_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)

    if rows_range is not None:
        start, end = rows_range
        if start is None or start < 0:
            start = 0
        if end is not None and end < 0:
            end = None
        df = df.iloc[start:end]

    # 规范所有可疑路径列
    target_cols = detect_path_columns(df, DEFAULT_PATH_COLS) if not path_cols else path_cols
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].map(normalize_to_unix_path)

    # 选项列重命名
    renames = rename_options_columns(df, overwrite=force_rename)

    # 处理 image_path / question_image_path
    if image_prefix:
        if "image_path" in df.columns:
            df["image_path"] = df["image_path"].map(lambda x: process_multi_paths_field_to_bracket_string(x, image_prefix, strip_token))
        elif "image" in df.columns:
            df["image_path"] = df["image"].map(lambda x: process_multi_paths_field_to_bracket_string(x, image_prefix, strip_token))

        # question_image_path：多路径 -> 方括号字符串
        if "question_image_path" in df.columns:
            df["question_image_path"] = df["question_image_path"].map(
                lambda x: process_multi_paths_field_to_bracket_string(x, image_prefix, strip_token)
            )
        elif "question_image" in df.columns:
            df["question_image_path"] = df["question_image"].map(
                lambda x: process_multi_paths_field_to_bracket_string(x, image_prefix, strip_token)
            )

    # 写 TSV：禁用自动加引号，防止把逗号字段包成 "..."
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",  # 防守性设置
    )

    try:
        os.remove(tmp_utf8_path)
    except Exception:
        pass

    return {
        "source_encoding": enc,
        "path_columns": target_cols,
        "renamed": renames,
        "shape": df.shape,
        "rows_range": rows_range,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Fix TSV paths & options; build image_path/question_image_path (multi-path -> bracket string); slice rows."
    )
    ap.add_argument("--in", dest="in_path",
                    default="/VISTA-Bench/VISTA-Bench.tsv")
    ap.add_argument("--out", dest="out_path",
                    default="/VISTA-Bench/VISTA-Bench_norm.tsv")
    ap.add_argument("--cols", nargs="*", default=None)
    ap.add_argument("--force-rename", action="store_true")
    ap.add_argument("--image-prefix",
                    default="/root/VISTA-Bench")
    ap.add_argument("--strip-token", default="")
    ap.add_argument("--limit", nargs=2, type=int, metavar=("START", "END"), default=None)
    args = ap.parse_args()

    rows_range = tuple(args.limit) if args.limit is not None else None

    info = convert_and_normalize_paths_and_rename(
        in_path=args.in_path,
        out_path=args.out_path,
        path_cols=args.cols,
        force_rename=args.force_rename,
        image_prefix=args.image_prefix,
        strip_token=args.strip_token,
        rows_range=rows_range,
    )
    print("Done.")
    print(f"Source encoding guessed: {info['source_encoding']}")
    print(f"Path columns normalized: {info['path_columns']}")
    print(f"Renamed columns: {info['renamed']}")
    print(f"Output shape: {info['shape']}")
    if info["rows_range"] is not None:
        print(f"Rows sliced to [start, end) = {info['rows_range']}")
    else:
        print("Rows range: None (no slicing)")


if __name__ == "__main__":
    main()
