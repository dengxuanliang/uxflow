from tests.integration.complex_smoke_runner import (
    EXPECTED_NEAR_DUPLICATE,
    EXPECTED_SELECTED,
    run_complex_smoke,
    write_complex_smoke_report,
)


async def test_complex_module_2_3_smoke_report():
    result = await run_complex_smoke()
    write_complex_smoke_report(result)

    assert result.trajectory_count == 20
    assert result.scored_count >= 12
    assert set(result.selected_ids) == EXPECTED_SELECTED
    assert not (set(result.selected_ids) & EXPECTED_NEAR_DUPLICATE)
    assert result.manifest["targeted_count"] == len(EXPECTED_SELECTED)
    assert result.manifest["general_count"] == 0
    assert "smoke_010" in result.decayed_ids
