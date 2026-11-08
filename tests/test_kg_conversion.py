"""Independent conversion regressions; every data fixture is temporary.

Run with: python -B -m unittest discover -s tests -p test_kg_conversion.py -v
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = PACKAGE_ROOT / "scripts/data_process/convert_kg_to_jsonl.py"
SPEC = importlib.util.spec_from_file_location("kg_converter_under_test", CONVERTER_PATH)
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)

FIELDS = ["id", "contents", "subject", "predicate", "object", "date"]
RAW = (
    "Abdul_Kalam\tMake_statement\tSwraj_Paul\t2005-01-01\n"
    "Peter_Pace\tMake_a_visit\tChina\t2007-03-22\n"
)
EXPECTED = [
    {
        "id": "0",
        "contents": "Abdul_Kalam Make_statement Swraj_Paul on 2005-01-01",
        "subject": "Abdul_Kalam",
        "predicate": "Make_statement",
        "object": "Swraj_Paul",
        "date": "2005-01-01",
    },
    {
        "id": "1",
        "contents": "Peter_Pace Make_a_visit China on 2007-03-22",
        "subject": "Peter_Pace",
        "predicate": "Make_a_visit",
        "object": "China",
        "date": "2007-03-22",
    },
]


class CorpusCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="kgtest_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "full.txt"
        self.target = self.root / "full_kg.jsonl"
        self.source.write_bytes(RAW.encode("utf-8"))

    def snapshot(self):
        found = {}
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root).as_posix()
            if path.is_symlink():
                found[relative] = ("symlink", os.readlink(path))
            elif path.is_file():
                info = path.stat()
                found[relative] = (
                    info.st_size,
                    info.st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            else:
                found[relative] = ("directory",)
        return found

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def reference(self, records=None, name="reference.jsonl"):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in EXPECTED if records is None else records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def invalid_conversion(self, raw, line=None, delimiter="auto"):
        self.source.write_bytes(raw.encode("utf-8") if isinstance(raw, str) else raw)
        before = self.snapshot()
        with self.assertRaises(converter.ConversionError) as caught:
            converter.convert_corpus(self.source, self.target, delimiter)
        if line is not None:
            self.assertIn("{}:{}".format(self.source, line), str(caught.exception))
        self.assertFalse(self.target.exists())
        self.assertEqual(before, self.snapshot())
        self.assert_no_temporary_files()
        return str(caught.exception)


class ConversionTests(CorpusCase):
    def test_real_sample_schema_text_order_and_stats(self):
        stats = converter.convert_corpus(self.source, self.target)
        rows = [json.loads(line) for line in self.target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, EXPECTED)
        self.assertEqual([list(row) for row in rows], [FIELDS, FIELDS])
        self.assertTrue(all(isinstance(value, str) for row in rows for value in row.values()))
        self.assertEqual(stats.records, 2)
        self.assertEqual(stats.delimiter, "tab")
        self.assertEqual(stats.input_sha256, hashlib.sha256(RAW.encode()).hexdigest())
        framed = b"".join(
            len(row["contents"].encode()).to_bytes(8, "big") + row["contents"].encode()
            for row in EXPECTED
        )
        self.assertEqual(stats.contents_sha256, hashlib.sha256(framed).hexdigest())
        self.assertTrue(self.target.read_bytes().endswith(b"\n"))
        self.assert_no_temporary_files()

    def test_explicit_and_auto_delimiters(self):
        for delimiter, separator in [("tab", "\t"), ("pipe", "|"), ("whitespace", " ")]:
            for choice in [delimiter, "auto"]:
                with self.subTest(delimiter=delimiter, choice=choice):
                    self.source.write_text(separator.join(["A", "Make_statement", "B", "2004-02-29"]) + "\n", encoding="utf-8")
                    target = self.root / (delimiter + "_" + choice + ".jsonl")
                    stats = converter.convert_corpus(self.source, target, choice)
                    self.assertEqual(stats.delimiter, delimiter)
                    self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["contents"], "A Make_statement B on 2004-02-29")

    def test_auto_delimiter_is_fixed_after_first_record(self):
        self.invalid_conversion("A\tP\tB\t2005-01-01\nC|P|D|2005-01-02\n", line=2)
        self.invalid_conversion("A|P|B|2005-01-01\nC P D 2005-01-02\n", line=2)
        self.invalid_conversion("A\tP B\t\t2005-01-01\n", line=1)

    def test_empty_columns_keep_their_positions_and_fail(self):
        for separator, choice in [("\t", "auto"), ("|", "auto")]:
            for position in range(4):
                fields = ["A", "P", "B", "2005-01-01"]
                fields[position] = ""
                with self.subTest(separator=separator, empty_column=position):
                    self.invalid_conversion(separator.join(fields) + "\n", line=1, delimiter=choice)

    def test_three_and_five_columns_fail(self):
        for raw in ["A\tP\t2005-01-01\n", "A\tP\tB\t2005-01-01\textra\n"]:
            with self.subTest(raw=raw):
                self.invalid_conversion(raw, line=1)

    def test_empty_file_and_blank_records_are_not_skipped(self):
        self.invalid_conversion(b"")
        self.invalid_conversion("\n", line=1)
        self.invalid_conversion(RAW.splitlines()[0] + "\n\n" + RAW.splitlines()[1] + "\n", line=2)

    def test_dates_are_calendar_valid_and_zero_padded_ascii(self):
        for day in ["2005-02-29", "2005-13-01", "2005-04-31", "0000-01-01", "2005-1-01", "2005-01-1", "2005-01-01T00:00:00", "２００５-01-01"]:
            with self.subTest(day=day):
                self.invalid_conversion("A\tP\tB\t" + day + "\n", line=1)

    def test_control_characters_are_not_hidden_by_line_ending_cleanup(self):
        for raw in ["A\tP\x00\tB\t2005-01-01\n", "A\tP\rQ\tB\t2005-01-01\n", "A\tP\tB\t2005-01-01\r\r\n"]:
            with self.subTest(raw=repr(raw)):
                self.invalid_conversion(raw, line=1)

    def test_invalid_utf8_reports_the_physical_line(self):
        self.invalid_conversion(RAW.splitlines()[0].encode() + b"\nA\tP\t\xff\t2005-01-01\n", line=2)

    def test_bom_crlf_last_record_without_newline_and_unicode(self):
        raw = b"\xef\xbb\xbf" + RAW.splitlines()[0].encode() + b"\r\n" + '中国\tMake_statement\t"quoted"\\value\t2004-02-29'.encode()
        self.source.write_bytes(raw)
        stats = converter.convert_corpus(self.source, self.target)
        rows = [json.loads(line) for line in self.target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0], EXPECTED[0])
        self.assertEqual(rows[1]["id"], "1")
        self.assertEqual(rows[1]["object"], '"quoted"\\value')
        self.assertEqual(rows[1]["contents"], '中国 Make_statement "quoted"\\value on 2004-02-29')
        self.assertIn("中国", self.target.read_text(encoding="utf-8"))
        self.assertEqual(stats.input_sha256, hashlib.sha256(raw).hexdigest())

    def test_duplicates_and_temporal_variants_keep_faiss_row_alignment(self):
        first = "A\tP\tB\t2005-01-01\n"
        self.source.write_text(first + first + "A\tP\tB\t2005-01-02\n", encoding="utf-8")
        stats = converter.convert_corpus(self.source, self.target)
        rows = [json.loads(line) for line in self.target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(stats.records, 3)
        self.assertEqual([row["id"] for row in rows], ["0", "1", "2"])
        self.assertEqual(rows[0]["contents"], rows[1]["contents"])
        self.assertNotEqual(rows[1]["contents"], rows[2]["contents"])

    def test_fingerprints_separate_raw_bytes_from_ordered_embedding_text(self):
        hashes = []
        for name, raw in [
            ("tab", RAW),
            ("pipe", RAW.replace("\t", "|")),
            ("date_changed", RAW.replace("2005-01-01", "2005-01-02")),
            ("reordered", "\n".join(reversed(RAW.splitlines())) + "\n"),
        ]:
            self.source.write_bytes(raw.encode("utf-8"))
            stats = converter.convert_corpus(self.source, self.root / (name + ".jsonl"))
            hashes.append((stats.input_sha256, stats.contents_sha256))
        self.assertNotEqual(hashes[0][0], hashes[1][0])
        self.assertEqual(hashes[0][1], hashes[1][1])
        self.assertNotEqual(hashes[0][1], hashes[2][1])
        self.assertNotEqual(hashes[0][1], hashes[3][1])

    def test_existing_output_is_never_overwritten(self):
        self.target.write_bytes(b"existing deployed corpus sentinel\n")
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            converter.convert_corpus(self.source, self.target)
        self.assertEqual(before, self.snapshot())

    def test_same_input_output_and_lexical_alias_are_rejected(self):
        (self.root / "sub").mkdir()
        for target in [self.source, self.root / "sub" / ".." / "full.txt"]:
            with self.subTest(target=target):
                before = self.snapshot()
                with self.assertRaises(converter.ConversionError):
                    converter.convert_corpus(self.source, target)
                self.assertEqual(before, self.snapshot())

    def test_existing_hardlink_alias_preserves_input(self):
        alias = self.root / "input_alias.txt"
        try:
            os.link(self.source, alias)
        except OSError as error:
            self.skipTest("hard links unavailable: " + str(error))
        before = self.snapshot()
        with self.assertRaises((converter.ConversionError, FileExistsError)):
            converter.convert_corpus(self.source, alias)
        self.assertEqual(before, self.snapshot())

    def test_destination_symlink_alias_preserves_input(self):
        alias = self.root / "input_alias.txt"
        try:
            alias.symlink_to(self.source)
        except OSError as error:
            self.skipTest("file symlinks unavailable: " + str(error))
        before = self.snapshot()
        with self.assertRaises(converter.ConversionError):
            converter.convert_corpus(self.source, alias)
        self.assertEqual(before, self.snapshot())

    def test_late_bad_record_does_not_publish_partial_output(self):
        self.invalid_conversion(RAW + "bad\tlast\trow\n", line=3)

    def test_fsync_failure_cleans_temporary_output(self):
        before = self.snapshot()
        with mock.patch.object(converter.os, "fsync", side_effect=OSError("injected fsync failure")):
            with self.assertRaisesRegex(OSError, "injected fsync"):
                converter.convert_corpus(self.source, self.target)
        self.assertEqual(before, self.snapshot())
        self.assert_no_temporary_files()

    def test_link_failure_cleans_temporary_output(self):
        before = self.snapshot()
        with mock.patch.object(converter.os, "link", side_effect=PermissionError("injected link failure")):
            with self.assertRaisesRegex(PermissionError, "injected link"):
                converter.convert_corpus(self.source, self.target)
        self.assertEqual(before, self.snapshot())
        self.assert_no_temporary_files()

    def test_publish_race_preserves_the_competing_writer(self):
        original = self.source.read_bytes()
        real_link = os.link

        def competing_writer(temporary, target):
            Path(target).write_bytes(b"competing writer owns this output\n")
            return real_link(temporary, target)

        with mock.patch.object(converter.os, "link", side_effect=competing_writer):
            with self.assertRaises(FileExistsError):
                converter.convert_corpus(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), b"competing writer owns this output\n")
        self.assertEqual(self.source.read_bytes(), original)
        self.assert_no_temporary_files()

    def test_source_changed_after_read_is_not_published(self):
        real_fsync = os.fsync
        original = self.source.read_bytes()

        def append_to_source(descriptor):
            real_fsync(descriptor)
            with self.source.open("ab") as stream:
                stream.write(b"new concurrent bytes\n")

        with mock.patch.object(converter.os, "fsync", side_effect=append_to_source):
            with self.assertRaisesRegex(converter.ConversionError, "changed"):
                converter.convert_corpus(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source.read_bytes(), original + b"new concurrent bytes\n")
        self.assert_no_temporary_files()

    def test_conversion_does_not_touch_existing_embeddings_or_indexes(self):
        sentinels = {}
        for name in ["full_kg_old.jsonl", "e5_Flat.index", "bge_Flat.index", "embeddings.npy", "model-config.json"]:
            path = self.root / name
            path.write_bytes(("sentinel for " + name).encode())
            sentinels[name] = path.read_bytes()
        converter.convert_corpus(self.source, self.target)
        self.assertEqual(sentinels, {name: (self.root / name).read_bytes() for name in sentinels})


class VerificationTests(CorpusCase):
    def test_verification_is_read_only_and_checks_canonical_records(self):
        reference = self.reference()
        before = self.snapshot()
        with mock.patch.object(converter.tempfile, "mkstemp", side_effect=AssertionError("verification must not write")):
            stats = converter.verify_corpus(self.source, reference)
        self.assertEqual(stats.records, 2)
        self.assertEqual(stats.delimiter, "tab")
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.target.exists())

    def test_each_of_the_six_fields_is_compared(self):
        for key in FIELDS:
            with self.subTest(field=key):
                records = [dict(row) for row in EXPECTED]
                records[1][key] += "_wrong"
                reference = self.reference(records)
                before = self.snapshot()
                with self.assertRaisesRegex(converter.ConversionError, "field"):
                    converter.verify_corpus(self.source, reference)
                self.assertEqual(before, self.snapshot())

    def test_field_set_object_type_and_string_types_are_strict(self):
        bad = []
        missing = dict(EXPECTED[0]); missing.pop("date"); bad.append(missing)
        extra = dict(EXPECTED[0]); extra["extra"] = "not canonical"; bad.append(extra)
        for key, value in [("id", 0), ("date", None), ("contents", ["not", "a string"])]:
            record = dict(EXPECTED[0]); record[key] = value; bad.append(record)
        bad.extend([[], None])
        for first in bad:
            with self.subTest(record=first):
                reference = self.reference([first, EXPECTED[1]])
                before = self.snapshot()
                with self.assertRaises(converter.ConversionError):
                    converter.verify_corpus(self.source, reference)
                self.assertEqual(before, self.snapshot())

    def test_duplicate_json_keys_are_rejected_even_with_equal_values(self):
        reference = self.reference()
        text = reference.read_text(encoding="utf-8").replace('"id": "0"', '"id": "0", "id": "0"', 1)
        reference.write_text(text, encoding="utf-8")
        before = self.snapshot()
        with self.assertRaisesRegex(converter.ConversionError, "duplicate JSON key"):
            converter.verify_corpus(self.source, reference)
        self.assertEqual(before, self.snapshot())

    def test_short_long_and_extra_blank_reference_records_fail(self):
        for records, suffix in [(EXPECTED[:1], ""), (EXPECTED + EXPECTED[:1], ""), (EXPECTED, "\n")]:
            with self.subTest(records=len(records), suffix=repr(suffix)):
                reference = self.reference(records)
                if suffix:
                    with reference.open("a", encoding="utf-8") as stream:
                        stream.write(suffix)
                before = self.snapshot()
                with self.assertRaises(converter.ConversionError):
                    converter.verify_corpus(self.source, reference)
                self.assertEqual(before, self.snapshot())

    def test_malformed_json_blank_record_and_invalid_utf8_fail_read_only(self):
        for payload in [b"{broken json}\n", b"\n", b"\xff\n"]:
            with self.subTest(payload=payload):
                reference = self.root / "reference.jsonl"
                reference.write_bytes(payload)
                before = self.snapshot()
                with self.assertRaises(converter.ConversionError):
                    converter.verify_corpus(self.source, reference)
                self.assertEqual(before, self.snapshot())

    def test_same_source_and_hardlinked_reference_are_rejected(self):
        with self.assertRaises(converter.ConversionError):
            converter.verify_corpus(self.source, self.source)
        alias = self.root / "reference_alias"
        try:
            os.link(self.source, alias)
        except OSError as error:
            self.skipTest("hard links unavailable: " + str(error))
        before = self.snapshot()
        with self.assertRaises(converter.ConversionError):
            converter.verify_corpus(self.source, alias)
        self.assertEqual(before, self.snapshot())

    def test_reference_changed_during_verification_is_rejected(self):
        reference = self.reference()
        original_iter = converter.iter_records

        def alter_reference_after_read(*args, **kwargs):
            yield from original_iter(*args, **kwargs)
            payload = reference.read_bytes()
            reference.write_bytes(payload.replace(b"Abdul_Kalam", b"Abdul_Kalax", 1))

        with mock.patch.object(converter, "iter_records", side_effect=alter_reference_after_read):
            with self.assertRaisesRegex(converter.ConversionError, "changed"):
                converter.verify_corpus(self.source, reference)
        self.assertFalse(self.target.exists())
        self.assert_no_temporary_files()


class CommandLineTests(CorpusCase):
    def test_default_root_is_based_on_script_location_not_cwd(self):
        project = self.root / "project with spaces"
        installed = project / "scripts/data_process/convert_kg_to_jsonl.py"
        installed.parent.mkdir(parents=True)
        shutil.copy2(CONVERTER_PATH, installed)
        corpus = project / "corpus"
        corpus.mkdir()
        (corpus / "full.txt").write_bytes(RAW.encode())
        elsewhere = self.root / "unrelated cwd"
        elsewhere.mkdir()
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", str(installed)],
            cwd=str(elsewhere), capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("CONVERTED records=2 delimiter=tab", process.stdout)
        records = [json.loads(line) for line in (corpus / "full_kg.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records, EXPECTED)
        self.assertEqual(list(elsewhere.iterdir()), [])
        self.assertEqual(list(project.rglob("__pycache__")), [])

    def test_cli_verification_creates_no_output(self):
        reference = self.reference()
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = converter.main(["--input", str(self.source), "--verify-against", str(reference)])
        self.assertEqual(code, 0)
        self.assertIn("VERIFIED records=2", stdout.getvalue())
        self.assertEqual(before, self.snapshot())

    def test_cli_bad_record_returns_nonzero_without_partial_output(self):
        self.source.write_text(RAW + "bad\tlast\trow\n", encoding="utf-8")
        before = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = converter.main(["--input", str(self.source), "--output", str(self.target)])
        self.assertEqual(code, 2)
        self.assertIn(":3:", stderr.getvalue())
        self.assertEqual(before, self.snapshot())

    def test_cli_output_and_verify_are_mutually_exclusive(self):
        reference = self.reference()
        before = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                converter.main(["--input", str(self.source), "--output", str(self.target), "--verify-against", str(reference)])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
