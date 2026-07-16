import os

import pytest

import main


@pytest.mark.skipif(
    os.environ.get("GYMREC_HF_SMOKE") != "1",
    reason="set GYMREC_HF_SMOKE=1 to validate the published Level1-1 release",
)
def test_standardized_level1_1_release_smoke():
    source = main.resolve_huggingface_policy_source(
        "hf://tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo",
        device="cpu",
    )

    assert len(source.revision) == 40
    assert source.release_manifest_document["document_type"] == "rlab.release_manifest"
    assert source.env_id == "SuperMarioBros-Nes-v0"
    assert source.state == "Level1-1"
    assert source.collector.endswith(f"@{source.revision}")
