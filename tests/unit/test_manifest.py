import hashlib
import json
from pathlib import Path

from lmr4dvar.workflow.manifest import build_manifest, write_manifest


def test_manifest_records_checksum(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("LMR4DVar\n", encoding="utf-8")

    manifest = build_manifest([source], checksum=True)

    record = manifest["inputs"][0]
    assert record["path"] == str(source.resolve())
    assert record["kind"] == "file"
    assert record["sha256"] == hashlib.sha256(b"LMR4DVar\n").hexdigest()


def test_write_manifest(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manifest.json"
    write_manifest({"schema_version": 1}, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"schema_version": 1}
