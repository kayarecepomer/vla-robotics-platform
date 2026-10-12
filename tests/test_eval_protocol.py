import pytest

from vla.eval.protocol import (
    Grid,
    TrialRecord,
    append_record,
    grid_sheet_svg,
    load_records,
    summarize,
    to_markdown,
    trial_schedule,
    wilson_interval,
)


def test_grid_cells_and_centres():
    grid = Grid(rows=4, cols=4, cell_cm=6.0)
    assert grid.cells()[0] == "A1" and grid.cells()[-1] == "D4" and len(grid.cells()) == 16
    assert grid.center_cm("A1") == (3.0, 3.0)
    assert grid.center_cm("D4") == (21.0, 21.0)


def test_schedule_covers_every_slot_once_in_condition_blocks():
    schedule = trial_schedule(Grid(), ["nominal", "dim_light"], repeats=2, seed=1)
    assert len(schedule) == 64 and len({slot.key for slot in schedule}) == 64
    assert [slot.condition for slot in schedule[:32]] == ["nominal"] * 32


def test_wilson_interval_edges():
    low, high = wilson_interval(0, 16)
    assert low == 0.0 and 0.15 < high < 0.25
    low, high = wilson_interval(16, 16)
    assert high == pytest.approx(1.0) and 0.75 < low < 0.85
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_log_round_trip_and_summary(tmp_path):
    log = tmp_path / "trials.jsonl"
    for index, success in enumerate([True, True, False, True]):
        append_record(log, TrialRecord("act", "pick", "nominal", f"A{index + 1}", 0, success, True, 10.0))
    append_record(log, TrialRecord("act", "pick", "dim_light", "A1", 0, False, False, 30.0))
    rows = {(row.label, row.condition): row for row in summarize(load_records(log))}
    assert rows[("act", "nominal")].rate == 0.75
    assert rows[("act", "all")].trials == 5
    assert "| act | nominal | 75% |" in to_markdown(list(rows.values()))


def test_grid_sheet_is_svg_at_real_scale():
    svg = grid_sheet_svg(Grid(cell_cm=6.0))
    assert svg.startswith("<svg") and 'width="270.0mm"' in svg and svg.count("<rect") == 17
