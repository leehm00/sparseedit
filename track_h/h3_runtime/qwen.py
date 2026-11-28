# SPDX-License-Identifier: Apache-2.0
"""Use exact H1 dense receipts and the resident Qwen native entry point."""

import shutil

from .contracts import artifact, file_sha, read


def prior_dense(prior_results, case, repetition, mode, uuid, model_identity):
    from h1 import source_identity
    from track_h_qwen.contracts import PROTOCOL, digest_json

    if prior_results is None:
        return None
    root = prior_results / "H1"
    completion = root / "completion.json"
    if not completion.is_file():
        return None
    receipt = read(completion)
    if (
        receipt.get("phase") != "H1" or receipt.get("execution_status") != "PASS"
        or receipt.get("run_mode") != mode or receipt.get("gpu_uuid") != uuid
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("source_sha256") != source_identity()
        or receipt.get("model", {}).get("runtime_file_sha256") != model_identity
    ):
        return None
    selected = [row for row in receipt["requests"] if
                row["arm"] == "fresh_dense" and row["case_id"] == case["case_id"]
                and row.get("repetition") == repetition and row["split"] == case["split"]]
    if not selected:
        return None
    if len(selected) != 1:
        raise ValueError("duplicate matching H1 dense cells")
    row = selected[0]
    expected = {
        "status": "PASS", "input_sha256": case["input_sha256"], "source_pixel_sha256": case["source_pixel_sha256"],
        "prompt_sha256": digest_json(case["prompt"]), "model_sha256": model_identity,
        "protocol_sha256": digest_json(PROTOCOL), "budget_percent": None, "timing_role": "measurement",
        "session_instruction_index": None,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        return None
    manifest = artifact(root, row["manifest_path"], row["manifest_sha256"])
    if read(manifest) != {key: value for key, value in row.items() if key not in ("manifest_path", "manifest_sha256")}:
        raise ValueError("matching H1 dense manifest differs from completion row")
    image = artifact(root, row["output_path"], row["output_sha256"])
    trace = artifact(root, row["action_trace_path"], row["action_trace_sha256"])
    return {
        "image": image, "trace": trace, "request_wall_s": row["request_wall_seconds"],
        "native_metadata": {
            "timing_scope": row["timing_scope"], "h1_gpu_uuid": uuid, "h1_request_manifest_sha256": file_sha(manifest),
            "h1_completion_sha256": file_sha(completion),
        },
    }


def copy_prior(receipt, directory):
    shutil.copyfile(receipt["image"], directory / "output.png")
    shutil.copyfile(receipt["trace"], directory / "actions.json")
