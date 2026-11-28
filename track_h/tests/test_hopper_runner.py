"""CPU tests of launcher, device admission and artifact transport."""

import io
import tarfile

import pytest

from hopper_io import make_payload, verified_assets
from run_track_h import STAGES, parse_device, selected_stages


def test_hopper_admission_checks_uuid_model_and_foreign_compute():
    uuid = "GPU-b2d4ca11-024c-46d1-a54a-0ff39388bfe4"
    listing = f"GPU 0: NVIDIA H20 (UUID: {uuid})\n"
    info = f"{uuid}, 0, NVIDIA H20, 97871, 95000, 580.0\n"
    assert parse_device(uuid, listing, info, "gpu_uuid, pid\n")["physical_index"] == 0
    with pytest.raises(ValueError, match="compute processes"):
        parse_device(uuid, listing, info, f"gpu_uuid, pid\n{uuid}, 103\n")
    with pytest.raises(ValueError, match="H20/H200"):
        parse_device(uuid, listing.replace("H20", "A40"), info.replace("H20", "A40"), "gpu_uuid, pid\n")


def test_stage_order_restores_dependency_order_and_rejects_duplicates():
    assert selected_stages(["H4", "H1"]) == ("H1", "H4")
    assert selected_stages(list(STAGES)) == STAGES
    with pytest.raises(ValueError, match="duplicate"):
        selected_stages(["H2", "H2"])


def test_payload_fresh_extraction_is_hash_verified(tmp_path):
    data = tmp_path / "payload_data"
    data.mkdir()
    (data / "cases.json").write_text("[]")
    archive = tmp_path / "payload.tar.gz"
    make_payload(data, archive)
    with verified_assets(archive, tmp_path) as (path, identity):
        assert (path / "cases.json").read_text() == "[]"
        assert set(identity["files"]) == {"cases.json"}
    assert not path.exists()
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as stream:
        entry = tarfile.TarInfo("../outside")
        entry.size = 1
        stream.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        with verified_assets(bad, tmp_path):
            pass


def test_one_command_runs_ordered_entries_and_seals_partial_output(tmp_path, monkeypatch):
    import json
    import run_track_h

    uuid = "GPU-b2d4ca11-024c-46d1-a54a-0ff39388bfe4"
    source = tmp_path / "runner"
    source.mkdir()
    for name, partial in (("H1", True), ("H4", False)):
        (source / f"{name.lower()}.py").write_text(
            "from pathlib import Path\nimport json, os, sys\n"
            "out=Path(sys.argv[-1]); out.mkdir()\n"
            f"assert os.environ['CUDA_VISIBLE_DEVICES']=={(uuid if partial else '')!r}\n"
            "conf=json.loads(Path(sys.argv[2]).read_text())\n"
            "assert (Path(conf['assets'])/'cases.json').read_text()=='[]'\n"
            f"(out/'completion.json').write_text(json.dumps({{'phase': {name!r}, "
            f"'status': {('PARTIAL' if partial else 'PASS')!r}}}))\n"
        )
    monkeypatch.setattr(run_track_h, "ROOT", source)
    monkeypatch.setattr(run_track_h, "check_device", lambda gpu_uuid: {"gpu_uuid": gpu_uuid})
    data = tmp_path / "payload_data"
    data.mkdir()
    (data / "cases.json").write_text("[]")
    archive = tmp_path / "payload.tar.gz"
    make_payload(data, archive)
    output = tmp_path / "result"
    returncode = run_track_h.main(["--gpu-uuid", uuid, "--assets", str(archive),
                                   "--output", str(output), "--stages", "H4", "H1"])
    assert returncode == 2
    receipt = json.loads((output / "completion.json").read_text())
    assert receipt["status"] == "PARTIAL"
    assert [k for k in receipt["stages"]] == ["H1", "H4"]
    assert "H1/completion.json" in receipt["output_sha256"]
    with tarfile.open(tmp_path / "result_receipts.tar.gz", "r:gz") as stream:
        assert "result/report.md" in stream.getnames()
