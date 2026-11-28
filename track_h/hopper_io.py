# SPDX-License-Identifier: Apache-2.0
"""Hash-bound asset transport and result sealing for the Hopper study."""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

MANIFEST = "payload_manifest.json"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        f.write("\n")


def hashes(root):
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()}


def make_payload(data, archive):
    data, archive = Path(data).resolve(), Path(archive).resolve()
    if not data.is_dir() or archive.is_relative_to(data):
        raise ValueError("asset source must be a directory separate from the archive")
    paths = list(data.rglob("*"))
    if any(path.is_symlink() or not (path.is_dir() or path.is_file()) for path in paths):
        raise ValueError("asset source must contain only regular files and directories")
    pins = hashes(data)
    if not pins or MANIFEST in pins:
        raise ValueError("asset source is empty or contains a reserved manifest filename")
    manifest = json.dumps({"format": "sparseedit-track-h-assets-v1", "files": pins}, sort_keys=True).encode()
    with tarfile.open(archive, "x:gz") as tar:
        for name in pins:
            tar.add(data / name, arcname=name, recursive=False)
        entry = tarfile.TarInfo(MANIFEST)
        entry.size = len(manifest)
        tar.addfile(entry, io.BytesIO(manifest))
    return pins


def _check_payload(root):
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("format") != "sparseedit-track-h-assets-v1":
        raise ValueError("wrong Hopper asset manifest format")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("asset tree contains a symbolic link")
    actual = hashes(root)
    actual.pop(MANIFEST, None)
    if actual != manifest["files"]:
        raise ValueError("asset inventory differs from frozen hashes")
    return actual


@contextmanager
def verified_assets(archive, scratch):
    if archive.is_dir():
        yield archive, {"manifest_sha256": sha(archive / MANIFEST), "files": _check_payload(archive)}
        return
    archive_sha256 = sha(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    if checksum.is_file() and checksum.read_text().split() != [archive_sha256, archive.name]:
        raise ValueError("Hopper asset archive SHA-256 differs from its sidecar")
    with tempfile.TemporaryDirectory(dir=scratch, prefix=".sparseedit-h-assets-") as temp:
        root = Path(temp)
        with tarfile.open(archive, "r:gz") as tar:
            seen = set()
            for member in tar.getmembers():
                rel = PurePosixPath(member.name)
                if (rel.is_absolute() or ".." in rel.parts or not rel.parts or not member.isfile()
                        or rel in seen):
                    raise ValueError(f"unsafe or duplicate Hopper asset archive member: {member.name}")
                seen.add(rel)
                dest = root.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"asset member cannot be read: {member.name}")
                with source, dest.open("xb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
            pins = _check_payload(root)
        yield root, {
            "archive_sha256": archive_sha256, "sidecar_verified": checksum.is_file(),
            "manifest_sha256": sha(root / MANIFEST), "files": pins,
        }


def seal_result(output, receipt):
    output = Path(output).resolve()
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("result directory contains a symbolic link")
    write(output / "completion.json", {**receipt, "output_sha256": hashes(output)})
    archive = output.with_name(output.name + "_receipts.tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(output, arcname=output.name)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    with checksum.open("x", encoding="utf-8") as f:
        f.write(f"{sha(archive)}  {archive.name}\n")
    return archive, checksum


if __name__ == "__main__":
    import argparse
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("--data", type=Path, required=True)
    argp.add_argument("--output", type=Path, required=True)
    args = argp.parse_args()
    count = len(make_payload(args.data, args.output))
    checksum = args.output.with_suffix(args.output.suffix + ".sha256")
    with checksum.open("x", encoding="utf-8") as f:
        f.write(f"{sha(args.output)}  {args.output.name}\n")
    print(f"{count} verified payload file bindings in {args.output}")
