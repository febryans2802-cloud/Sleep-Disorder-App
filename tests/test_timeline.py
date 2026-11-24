import pytest

from rbdcap.timeline import clock_to_offset, format_clock, parse_clock, parse_time_or_offset


def test_parse_and_format():
    assert parse_clock("23:40") == 23 * 3600 + 40 * 60
    assert parse_clock("03:12:30") == 3 * 3600 + 12 * 60 + 30
    assert format_clock(parse_clock("07:05:09")) == "07:05:09"
    with pytest.raises(ValueError):
        parse_clock("25:00")


def test_offsets_cross_midnight():
    start = parse_clock("23:30")
    assert clock_to_offset(parse_clock("23:45"), start) == 15 * 60
    assert clock_to_offset(parse_clock("01:30"), start) == 2 * 3600
    assert parse_time_or_offset("00:30", start) == 3600
    assert parse_time_or_offset("125.5", start) == 125.5
    with pytest.raises(ValueError):
        parse_time_or_offset("01:00", None)
