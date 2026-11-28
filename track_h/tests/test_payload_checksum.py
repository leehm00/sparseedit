import pytest

from hopper_io import make_payload, verified_assets


def test_distributed_checksum_must_match_before_payload_extraction(tmp_path):
    source = tmp_path / "files"
    source.mkdir()
    (source / "cases.json").write_text("[]")
    archive = tmp_path / "assets.tar.gz"
    make_payload(source, archive)
    archive.with_suffix(".gz.sha256").write_text("0" * 64 + "  assets.tar.gz\n")
    with pytest.raises(ValueError, match="archive SHA-256"):
        with verified_assets(archive, tmp_path):
            pass
