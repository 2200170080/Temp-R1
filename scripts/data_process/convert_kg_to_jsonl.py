#!/usr/bin/env python3
"""Convert temporal KG facts without changing their order or retrieval text.

Input: subject, predicate, object, YYYY-MM-DD (tab, pipe, or whitespace).
Output: id, contents, subject, predicate, object, date.

Defaults are relative to this project's root, regardless of the working directory.
Existing outputs are never overwritten. Use --verify-against to check the deployed
corpus without creating a file, touching an index, importing Torch, or using a GPU.
Python 3.9+, standard library only.
"""

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import BinaryIO, Dict, Iterator, Optional


DELIMITERS = ("auto", "tab", "pipe", "whitespace")
FIELDS = ("id", "contents", "subject", "predicate", "object", "date")
DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")


class ConversionError(ValueError):
    """Invalid input or a mismatch with the existing corpus."""


@dataclass
class ConversionStats:
    records: int = 0
    delimiter: str = "auto"
    input_sha256: str = ""
    contents_sha256: str = ""


def project_root() -> Path:
    """Canonical installation: project/scripts/data_process/this_file.py."""
    return Path(__file__).resolve().parents[2]


def file_signature(stream: BinaryIO):
    info = os.fstat(stream.fileno())
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def check_unchanged(path: Path, stream: BinaryIO, initial):
    info = path.stat()
    current = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if current != initial or file_signature(stream) != initial:
        raise ConversionError("File changed while it was being read: " + str(path))


def decode_line(raw: bytes, line_number: int, path: Path) -> str:
    try:
        text = raw.decode("utf-8-sig" if line_number == 1 else "utf-8")
    except UnicodeError as error:
        raise ConversionError("{}:{}: invalid UTF-8: {}".format(path, line_number, error)) from error
    # Remove one valid line terminator, not arbitrary trailing control bytes.
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


def detect_delimiter(line: str) -> str:
    # Never fall back from malformed tab/pipe records to whitespace parsing.
    if "\t" in line:
        return "tab"
    if "|" in line:
        return "pipe"
    return "whitespace"


def parse_fact(line: str, row_index: int, delimiter: str, source: Path) -> Dict[str, str]:
    label = "{}:{}".format(source, row_index + 1)
    if not line.strip():
        raise ConversionError(label + ": empty record; no rows were skipped")
    if any((ord(char) < 32 and char != "\t") or ord(char) == 127 for char in line):
        raise ConversionError(label + ": control character in record")
    if delimiter == "tab":
        parts = line.split("\t")
    elif delimiter == "pipe":
        parts = line.split("|")
    elif delimiter == "whitespace":
        parts = line.split()
    else:
        raise ConversionError("Unknown delimiter: " + delimiter)
    if len(parts) != 4:
        raise ConversionError(
            "{}: expected 4 columns (subject, predicate, object, date), got {}".format(label, len(parts))
        )
    subject, predicate, obj, day = (part.strip() for part in parts)
    if not all((subject, predicate, obj, day)):
        raise ConversionError(label + ": subject, predicate, object and date must be nonempty")
    if DATE_PATTERN.fullmatch(day) is None:
        raise ConversionError(label + ": date must be YYYY-MM-DD, got " + repr(day))
    try:
        date.fromisoformat(day)
    except ValueError as error:
        raise ConversionError(label + ": invalid calendar date " + repr(day)) from error
    # Match the deployed corpus exactly: preserve underscores and include the date
    # in contents, because both the index builder and GRPO consume this string.
    return {
        "id": str(row_index),
        "contents": "{} {} {} on {}".format(subject, predicate, obj, day),
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "date": day,
    }


def iter_records(
    stream: BinaryIO, source: Path, delimiter: str, stats: ConversionStats
) -> Iterator[Dict[str, str]]:
    if delimiter not in DELIMITERS:
        raise ConversionError("Unknown delimiter: " + delimiter)
    selected = None if delimiter == "auto" else delimiter
    input_hash, contents_hash = hashlib.sha256(), hashlib.sha256()
    for row_index, raw in enumerate(stream):
        input_hash.update(raw)
        line = decode_line(raw, row_index + 1, source)
        if selected is None:
            selected = detect_delimiter(line)
        record = parse_fact(line, row_index, selected, source)
        encoded = record["contents"].encode("utf-8")
        contents_hash.update(struct.pack(">Q", len(encoded)))
        contents_hash.update(encoded)
        stats.records += 1
        yield record
    if stats.records == 0:
        raise ConversionError("Input has no facts: " + str(source))
    stats.delimiter = selected or delimiter
    stats.input_sha256 = input_hash.hexdigest()
    stats.contents_sha256 = contents_hash.hexdigest()


def input_file(path: Path) -> Path:
    result = Path(path).expanduser().resolve(strict=True)
    if not result.is_file():
        raise ConversionError("Input is not a regular file: " + str(result))
    return result


def output_file(path: Path, source: Path) -> Path:
    raw = Path(path).expanduser()
    # Resolve parents, but do not follow a destination symlink.
    target = raw.parent.resolve(strict=True) / raw.name
    if target.resolve() == source:
        raise ConversionError("Input and output refer to the same file: " + str(target))
    if os.path.lexists(str(target)):
        raise FileExistsError(
            "Output already exists; use --verify-against to check it, or choose a new --output: "
            + str(target)
        )
    if not target.parent.is_dir():
        raise ConversionError("Output parent is not a directory: " + str(target.parent))
    return target


def convert_corpus(input_path: Path, output_path: Path, delimiter: str = "auto") -> ConversionStats:
    """Stream into a temporary file, then publish atomically without overwriting."""
    source = input_file(input_path)
    target = output_file(output_path, source)
    stats = ConversionStats()
    temporary: Optional[Path] = None
    try:
        with source.open("rb") as fin:
            initial = file_signature(fin)
            descriptor, name = tempfile.mkstemp(
                prefix="." + target.name + ".", suffix=".tmp", dir=str(target.parent)
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as fout:
                for record in iter_records(fin, source, delimiter, stats):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
            check_unchanged(source, fin, initial)
            # Same-directory hard linking is atomic and fails if another writer
            # created target meanwhile. Never fall back to an overwriting rename.
            os.link(str(temporary), str(target))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return stats


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError("duplicate JSON key " + repr(key))
        result[key] = value
    return result


def verify_corpus(input_path: Path, reference_path: Path, delimiter: str = "auto") -> ConversionStats:
    """Compare every generated record with an existing JSONL; write nothing."""
    source = input_file(input_path)
    reference = input_file(reference_path)
    if os.path.samefile(str(source), str(reference)):
        raise ConversionError("Input and reference refer to the same file")
    stats = ConversionStats()
    with source.open("rb") as fin, reference.open("rb") as actual:
        initial_source, initial_reference = file_signature(fin), file_signature(actual)
        for expected in iter_records(fin, source, delimiter, stats):
            line_number = stats.records
            raw = actual.readline()
            if not raw:
                raise ConversionError(
                    "{}:{}: reference ended before the input".format(reference, line_number)
                )
            text = decode_line(raw, line_number, reference)
            try:
                record = json.loads(text, object_pairs_hook=unique_object)
            except (ValueError, TypeError) as error:
                raise ConversionError(
                    "{}:{}: invalid JSON: {}".format(reference, line_number, error)
                ) from error
            if not isinstance(record, dict) or set(record) != set(FIELDS):
                raise ConversionError(
                    "{}:{}: expected exactly these fields: {}".format(reference, line_number, ", ".join(FIELDS))
                )
            for key in FIELDS:
                if not isinstance(record[key], str) or record[key] != expected[key]:
                    raise ConversionError(
                        "{}:{}: field {!r} differs: expected {!r}, got {!r}".format(
                            reference, line_number, key, expected[key][:180], str(record[key])[:180]
                        )
                    )
        if actual.readline():
            raise ConversionError("{}: reference has extra records after row {}".format(reference, stats.records))
        check_unchanged(source, fin, initial_source)
        check_unchanged(reference, actual, initial_reference)
    return stats


def main(argv=None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, default=root / "corpus/full.txt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", "-o", type=Path, help="New output file; its parent directory must exist")
    mode.add_argument("--verify-against", type=Path, help="Read-only comparison with an existing JSONL corpus")
    parser.add_argument("--delimiter", choices=DELIMITERS, default="auto")
    args = parser.parse_args(argv)
    try:
        if args.verify_against is not None:
            stats = verify_corpus(args.input, args.verify_against, args.delimiter)
            action = "VERIFIED"
        else:
            target = args.output if args.output is not None else root / "corpus/full_kg.jsonl"
            stats = convert_corpus(args.input, target, args.delimiter)
            action = "CONVERTED"
    except (OSError, ConversionError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2
    print("{} records={} delimiter={}".format(action, stats.records, stats.delimiter))
    print("input_sha256=" + stats.input_sha256)
    print("ordered_contents_sha256=" + stats.contents_sha256)
    print("Existing corpus, embeddings and indexes were not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
