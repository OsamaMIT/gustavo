import csv
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest

from src.fastf1_adapter import (
    convert_fastf1_records,
    fetch_fastf1_to_csv,
    list_fastf1_events,
    list_fastf1_session_lap_options,
)
from src.telemetry_loader import REQUIRED_COLUMNS


def test_convert_fastf1_records_maps_core_fields_and_motion() -> None:
    rows = convert_fastf1_records(
        [
            {
                "Time": timedelta(seconds=0),
                "Speed": 100,
                "Throttle": 50,
                "Brake": False,
                "RPM": 12000,
                "nGear": 4,
                "DRS": 0,
            },
            {
                "Time": timedelta(seconds=1),
                "Speed": 136,
                "Throttle": 100,
                "Brake": False,
                "RPM": 12300,
                "nGear": 5,
                "DRS": 12,
            },
            {
                "Time": timedelta(seconds=2),
                "Speed": 120,
                "Throttle": 0,
                "Brake": True,
                "RPM": 10000,
                "nGear": 4,
                "DRS": 0,
            },
        ],
        [
            {"Time": timedelta(seconds=0), "X": 0, "Y": 0},
            {"Time": timedelta(seconds=1), "X": 1000, "Y": 0},
            {"Time": timedelta(seconds=2), "X": 2000, "Y": 200},
        ],
        year=2023,
        event="Bahrain",
        session_name="Q",
        driver="VER",
        lap_number=7,
    )

    assert REQUIRED_COLUMNS <= set(rows[0])
    assert rows[0]["throttle"] == pytest.approx(0.5)
    assert rows[2]["brake"] == pytest.approx(1.0)
    assert rows[1]["distance"] > rows[0]["distance"]
    assert rows[1]["longitudinal_accel"] > 0.0
    assert abs(rows[1]["lateral_accel"]) > 0.0
    assert abs(rows[1]["steering_angle"]) > 0.0
    assert rows[2]["segment_id"] == "braking"
    assert rows[0]["front_left_temp"] == ""
    assert rows[0]["setup_id"] == "fastf1_2023_Bahrain_Q_VER_lap_7"


def test_convert_fastf1_records_rejects_empty_car_data() -> None:
    with pytest.raises(ValueError, match="no car data"):
        convert_fastf1_records([])


class FakeFrame:
    def __init__(self, records):
        self.records = records

    def to_dict(self, orient):
        assert orient == "records"
        return self.records

    def add_distance(self):
        return self


class FakeSchedule:
    def to_dict(self, orient):
        assert orient == "records"
        return [
            {"EventName": "Bahrain Grand Prix"},
            {"EventName": "Saudi Arabian Grand Prix"},
        ]


class FakeOptionLaps:
    def to_dict(self, orient):
        assert orient == "records"
        return [
            {"Driver": "VER", "LapNumber": 1},
            {"Driver": "VER", "LapNumber": 2},
            {"Driver": "LEC", "LapNumber": 1},
            {"Driver": "LEC", "LapNumber": 3},
        ]


class FakeLap:
    fail_car = False
    fail_pos = False

    def __getitem__(self, key):
        if key == "LapNumber":
            return 14
        raise KeyError(key)

    def get_car_data(self):
        if self.fail_car:
            raise RuntimeError("The data you are trying to access has not been loaded yet. See Session.load")
        return FakeFrame(
            [
                {
                    "Time": timedelta(seconds=0),
                    "Speed": 296,
                    "Throttle": 100,
                    "Brake": False,
                    "RPM": 10575,
                    "nGear": 8,
                    "Distance": 24.2,
                },
                {
                    "Time": timedelta(seconds=1),
                    "Speed": 305,
                    "Throttle": 100,
                    "Brake": False,
                    "RPM": 10853,
                    "nGear": 8,
                    "Distance": 118.3,
                },
            ]
        )

    def get_pos_data(self):
        if self.fail_pos:
            raise RuntimeError("The data you are trying to access has not been loaded yet. See Session.load")
        return FakeFrame(
            [
                {"Time": timedelta(seconds=0), "X": 0, "Y": 0},
                {"Time": timedelta(seconds=1), "X": 1000, "Y": 0},
            ]
        )


class FakeLaps:
    def __init__(self, lap=None):
        self.selected_driver = None
        self.lap = lap or FakeLap()

    def pick_drivers(self, driver):
        self.selected_driver = driver
        return self

    def pick_fastest(self):
        return self.lap


class FakeSession:
    def __init__(self, lap=None):
        self.laps = FakeLaps(lap)
        self.loaded_kwargs = None

    def load(self, **kwargs):
        self.loaded_kwargs = kwargs


class FakeOptionSession:
    def __init__(self):
        self.laps = FakeOptionLaps()
        self.loaded_kwargs = None

    def load(self, **kwargs):
        self.loaded_kwargs = kwargs


def test_list_fastf1_events_uses_schedule(tmp_path, monkeypatch) -> None:
    cache_calls = []

    class FakeCache:
        @staticmethod
        def enable_cache(cache_dir):
            cache_calls.append(cache_dir)

    monkeypatch.setitem(
        sys.modules,
        "fastf1",
        SimpleNamespace(
            Cache=FakeCache,
            get_event_schedule=lambda year, include_testing: FakeSchedule(),
        ),
    )

    events = list_fastf1_events(2023, cache_dir=tmp_path / "cache")

    assert events == ("Bahrain Grand Prix", "Saudi Arabian Grand Prix")
    assert cache_calls == [str(tmp_path / "cache")]


def test_list_fastf1_session_lap_options_loads_drivers_and_laps(
    tmp_path,
    monkeypatch,
) -> None:
    fake_session = FakeOptionSession()

    class FakeCache:
        @staticmethod
        def enable_cache(cache_dir):
            pass

    def fake_get_session(year, event, session_name):
        assert year == 2023
        assert event == "Bahrain Grand Prix"
        assert session_name == "Q"
        return fake_session

    monkeypatch.setitem(
        sys.modules,
        "fastf1",
        SimpleNamespace(Cache=FakeCache, get_session=fake_get_session),
    )

    options = list_fastf1_session_lap_options(
        2023,
        "Bahrain Grand Prix",
        "Q",
        cache_dir=tmp_path / "cache",
    )

    assert options.drivers == ("VER", "LEC")
    assert options.laps_by_driver == {"VER": (1, 2), "LEC": (1, 3)}
    assert fake_session.loaded_kwargs == {
        "laps": True,
        "telemetry": False,
        "weather": False,
        "messages": False,
    }


def test_fetch_fastf1_to_csv_uses_session_and_writes_csv(tmp_path, monkeypatch) -> None:
    fake_session = FakeSession()
    cache_calls = []

    class FakeCache:
        @staticmethod
        def enable_cache(cache_dir):
            cache_calls.append(cache_dir)

    def fake_get_session(year, event, session_name):
        assert year == 2023
        assert event == "Bahrain"
        assert session_name == "Q"
        return fake_session

    monkeypatch.setitem(
        sys.modules,
        "fastf1",
        SimpleNamespace(Cache=FakeCache, get_session=fake_get_session),
    )

    target = fetch_fastf1_to_csv(
        year=2023,
        event="Bahrain",
        session_name="Q",
        driver="VER",
        output=tmp_path / "fastf1.csv",
        cache_dir=tmp_path / "cache",
    )

    assert target.exists()
    assert cache_calls == [str(tmp_path / "cache")]
    assert fake_session.loaded_kwargs == {
        "laps": True,
        "telemetry": True,
        "weather": False,
        "messages": False,
    }
    assert fake_session.laps.selected_driver == "VER"

    with target.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["speed"] == "296.0"
    assert rows[0]["lap_number"] == "14"
    assert rows[0]["setup_id"] == "fastf1_2023_Bahrain_Q_VER_lap_14"


def test_fetch_fastf1_to_csv_falls_back_when_position_data_not_loaded(
    tmp_path,
    monkeypatch,
) -> None:
    fake_lap = FakeLap()
    fake_lap.fail_pos = True
    fake_session = FakeSession(fake_lap)

    class FakeCache:
        @staticmethod
        def enable_cache(cache_dir):
            pass

    monkeypatch.setitem(
        sys.modules,
        "fastf1",
        SimpleNamespace(
            Cache=FakeCache,
            get_session=lambda year, event, session_name: fake_session,
        ),
    )

    target = fetch_fastf1_to_csv(
        year=2023,
        event="Bahrain",
        session_name="Q",
        driver="VER",
        output=tmp_path / "fastf1.csv",
        cache_dir=tmp_path / "cache",
    )

    with target.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["lateral_accel"] == "0.0"
    assert rows[0]["steering_angle"] == "0.0"


def test_fetch_fastf1_to_csv_explains_missing_car_telemetry(
    tmp_path,
    monkeypatch,
) -> None:
    fake_lap = FakeLap()
    fake_lap.fail_car = True
    fake_session = FakeSession(fake_lap)

    class FakeCache:
        @staticmethod
        def enable_cache(cache_dir):
            pass

    monkeypatch.setitem(
        sys.modules,
        "fastf1",
        SimpleNamespace(
            Cache=FakeCache,
            get_session=lambda year, event, session_name: fake_session,
        ),
    )

    with pytest.raises(RuntimeError, match="did not load car telemetry"):
        fetch_fastf1_to_csv(
            year=2023,
            event="Bahrain",
            session_name="Q",
            driver="VER",
            output=tmp_path / "fastf1.csv",
            cache_dir=tmp_path / "cache",
        )


def test_fetch_fastf1_to_csv_explains_missing_lap_timing(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeSessionWithoutLaps:
        def load(self, **kwargs):
            pass

        @property
        def laps(self):
            raise RuntimeError("The data you are trying to access has not been loaded yet. See Session.load")

    class FakeCache:
        @staticmethod
        def enable_cache(cache_dir):
            pass

    monkeypatch.setitem(
        sys.modules,
        "fastf1",
        SimpleNamespace(
            Cache=FakeCache,
            get_session=lambda year, event, session_name: FakeSessionWithoutLaps(),
        ),
    )

    with pytest.raises(RuntimeError, match="did not load lap timing data"):
        fetch_fastf1_to_csv(
            year=2023,
            event="Bahrain",
            session_name="Q",
            driver="VER",
            output=tmp_path / "fastf1.csv",
            cache_dir=tmp_path / "cache",
        )
