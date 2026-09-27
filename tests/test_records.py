# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from voxelscope.canonical import EvidenceError
from voxelscope.records import OverlapRatio, StudyManifest, WindowConfig


def test_diagnostic_claim_fails_closed_without_nonoverlap_proof() -> None:
    config = WindowConfig(
        (7, 8, 9),
        (6, 5, 4),
        (OverlapRatio(1, 2),) * 3,
        "constant",
        None,
    )
    with pytest.raises(EvidenceError) as caught:
        StudyManifest(
            "voxelscope/v1",
            "forbidden-claim",
            True,
            "diagnostic_accuracy",
            "unresolved",
            (),
            False,
            "volume.json",
            "model.json",
            config,
            (),
            (),
            (),
            None,
        )
    assert caught.value.code == "lineage_gate_failed"
