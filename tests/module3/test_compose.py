import logging

from module3.compose import GeneralDataConfig, compose_dataset


def test_no_general_source_returns_targeted_only(mk_candidate, caplog):
    targeted = [mk_candidate(f"t{i}", 0, 0.5, [0.0] * 1024) for i in range(5)]
    with caplog.at_level(logging.INFO):
        out = compose_dataset(
            targeted,
            general_config=GeneralDataConfig(ratio=0.3, source_path=None),
        )
    assert out["targeted"] == targeted
    assert out["general"] == []
    assert any("general data source" in r.message.lower() for r in caplog.records)


def test_ratio_recorded_in_manifest(mk_candidate):
    targeted = [mk_candidate("t1", 0, 0.5, [0.0] * 1024)]
    out = compose_dataset(
        targeted,
        general_config=GeneralDataConfig(ratio=0.25, source_path=None),
    )
    assert out["manifest"]["general_ratio"] == 0.25
    assert out["manifest"]["targeted_count"] == 1
    assert out["manifest"]["general_count"] == 0
