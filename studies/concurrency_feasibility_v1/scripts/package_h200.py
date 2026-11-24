#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package and verify the calibration-only offline H200 study."""

import argparse
from hashlib import file_digest
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile

STUDY = Path(__file__).resolve().parents[1]
EVIDENCE = STUDY.parents[1]
STUDY_REL = STUDY.relative_to(EVIDENCE)
BUNDLE_REL = Path("backend_reference/current_v4/backend_bundle")
MANIFEST = "sparseedit_phase_a_transfer.json"


def transfer_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != name:
        raise ValueError(f"Invalid transfer member: {name}")
    return path


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def add_tree(files, root, subtree, relative_root):
    for path in sorted(subtree.rglob("*")):
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Transfer input contains a symlink: {path}")
        if path.is_file():
            files[(relative_root / path.relative_to(root)).as_posix()] = path


def selected_files(plan, runbook):
    files = {}
    for folder in ("source", "scripts", "controls", "tests", "artifacts/node3_f0", "artifacts/f0_anchor_v1"):
        add_tree(files, STUDY, STUDY / folder, STUDY_REL)
    for name in ("protocol.json", "README.md", "artifacts/baseline_v1/completion.json"):
        files[(STUDY_REL / name).as_posix()] = STUDY / name
    for name in (
        "artifacts/serving_baseline_v1",
        "artifacts/executor_equivalence_attempt3",
        "artifacts/capacity_v1",
        "artifacts/isolation_v1/f1_node3_graph_attempt1",
        "artifacts/isolation_v1/f1_node3_eager_attempt2",
    ):
        if (STUDY / name / "completion.json").is_file():
            add_tree(files, STUDY, STUDY / name, STUDY_REL)
    for name in ("serving_baseline_report.md", "phase_a_summary.json", "phase_a_report.md",
                 "feasibility_summary.json", "feasibility_report.md"):
        path = STUDY / "results" / name
        if path.is_file():
            files[(STUDY_REL / "results" / name).as_posix()] = path

    bundle_root = EVIDENCE / BUNDLE_REL
    bundle_path = bundle_root / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    files[(BUNDLE_REL / "bundle.json").as_posix()] = bundle_path
    for name, expected in bundle["runtime_hashes"].items():
        path = bundle_root / "runtime" / name
        if digest(path) != expected:
            raise ValueError(f"Frozen runtime hash mismatch: {path}")
        files[(BUNDLE_REL / "runtime" / name).as_posix()] = path
    for case in bundle["cases"]["calibration"]:
        for rel, expected in (
            (Path("inputs") / case["input_filename"], case["prepared_sha256"]),
            (Path("traces") / f'{case["case_tag"]}.json', case["trace_sha256"]),
        ):
            path = bundle_root / rel
            if digest(path) != expected:
                raise ValueError(f"Calibration hash mismatch: {path}")
            files[(BUNDLE_REL / rel).as_posix()] = path
    files["H200_RUNBOOK.md"] = runbook
    source_docs = plan.parent.parent
    files["docs/atc2027/2026-11-24-sparseedit-atc2027-experiment-plan.md"] = plan
    for name in ("sparseedit_system_research_plan.md", "superpowers/specs/2026-11-24-sparseedit-atc2027-system-design.md"):
        files[f"docs/{name}"] = source_docs / name
    return files


def package(output, plan, runbook):
    files = selected_files(plan, runbook)
    members = {name: digest(path) for name, path in files.items()}
    manifest = {
        "schema_version": 1, "scope": "SparseEdit ATC 2027 Phase A calibration only",
        "study_directory": STUDY_REL.as_posix(),
        "bundle_metadata_sha256": members[(BUNDLE_REL / "bundle.json").as_posix()],
        "files": members,
    }
    with tarfile.open(output, "x:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=transfer_member(name).as_posix(), recursive=False)
        data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        header = tarfile.TarInfo(MANIFEST)
        header.size = len(data)
        header.mode = 0o644
        archive.addfile(header, BytesIO(data))
    return verify_archive(output)


def verify_archive(path):
    observed = {}
    manifest = None
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            name = transfer_member(entry.name).as_posix()
            if not entry.isfile() or name in observed:
                raise ValueError(f"Unsupported or duplicate archive entry: {name}")
            stream = archive.extractfile(entry)
            if name == MANIFEST:
                if manifest is not None:
                    raise ValueError("Duplicate transfer manifest")
                manifest = json.load(stream)
            else:
                observed[name] = file_digest(stream, "sha256").hexdigest()
    if manifest is None or manifest["files"] != observed:
        raise ValueError("Transfer content differs from its SHA-256 manifest")
    return manifest


def verify_extracted(root, *, assets_and_runtime=False):
    manifest = json.loads((root / MANIFEST).read_text())
    git_managed = tuple(STUDY_REL / folder for folder in ("scripts", "tests")) if assets_and_runtime else ()
    verified = {}
    for relative, expected in manifest["files"].items():
        member = transfer_member(relative)
        if any(member.is_relative_to(directory) for directory in git_managed):
            continue
        path = root / member
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Transfer file differs from SHA-256 manifest: {relative}")
        verified[relative] = expected
    return {**manifest, "files": verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--plan", required=True, type=Path)
    build.add_argument("--runbook", required=True, type=Path)
    checked = commands.add_parser("verify")
    checked.add_argument("archive", type=Path)
    extracted = commands.add_parser("verify-extracted")
    extracted.add_argument("root", type=Path)
    extracted.add_argument("--assets-and-runtime", action="store_true",
                           help="Verify frozen files, leaving Git-managed scripts and tests to the checked-out revision")
    args = parser.parse_args()
    if args.command == "build":
        manifest = package(args.output, args.plan, args.runbook)
        print(f"PASS {args.output} {digest(args.output)}")
    elif args.command == "verify":
        manifest = verify_archive(args.archive)
        print(f"PASS {args.archive} {digest(args.archive)}")
    else:
        manifest = verify_extracted(args.root.resolve(), assets_and_runtime=args.assets_and_runtime)
    print(f"{len(manifest['files'])} verified transfer files")


if __name__ == "__main__":
    main()
