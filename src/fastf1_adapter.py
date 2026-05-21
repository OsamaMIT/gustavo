"""Optional FastF1 API adapter into the internal telemetry CSV schema."""

from __future__ import annotations

import csv
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from .synthetic_data import CSV_COLUMNS


DEFAULT_FASTF1_CACHE_DIR = Path("data/fastf1_cache")
GRAVITY = 9.80665
FASTF1_CONNECTIVITY_URLS = (
    (
        "FastF1 schedule",
        "https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_{year}.json",
    ),
)
FASTF1_PRIMARY_BASE_URL = "https://livetiming.formula1.com"
FASTF1_MIRROR_BASE_URL = "https://livetiming-mirror.fastf1.dev"
FASTF1_SESSION_PROBE_PAGE = "SessionInfo.jsonStream"


@dataclass(frozen=True)
class FastF1SessionLapOptions:
    """Available FastF1 drivers and lap numbers for one loaded session."""

    drivers: tuple[str, ...]
    laps_by_driver: dict[str, tuple[int, ...]]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number(value)
    if isinstance(value, timedelta):
        return value.total_seconds()
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        try:
            return float(total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, datetime):
        return value.timestamp()
    return _number(value)


def _record_time(record: Mapping[str, Any]) -> float | None:
    for key in ("SessionTime", "Time", "Date", "timestamp"):
        if key in record:
            seconds = _to_seconds(record.get(key))
            if seconds is not None:
                return seconds
    return None


def _normalise_percent(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if number > 1.5:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _normalise_brake(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y"}:
            return 1.0
        if lowered in {"false", "f", "no", "n"}:
            return 0.0
    return _normalise_percent(value)


def _get_value(record: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _records_from_dataframe(frame: Any) -> list[Mapping[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, list):
        return frame
    to_dict = getattr(frame, "to_dict", None)
    if callable(to_dict):
        return list(to_dict("records"))
    return list(frame)


def _coordinate_pair(record: Mapping[str, Any]) -> tuple[float, float] | None:
    x = _number(record.get("X"))
    y = _number(record.get("Y"))
    if x is None or y is None:
        return None
    # FastF1 position channels are in 1/10 m. For unit tests or pre-scaled
    # fixtures, this constant scale does not affect curvature.
    return x / 10.0, y / 10.0


def _nearest_positions(
    car_records: Iterable[Mapping[str, Any]],
    position_records: Iterable[Mapping[str, Any]],
) -> list[tuple[float, float] | None]:
    timed_positions = [
        (time_value, position)
        for record in position_records
        if (time_value := _record_time(record)) is not None
        and (position := _coordinate_pair(record)) is not None
    ]
    if not timed_positions:
        return [None for _ in car_records]

    timed_positions.sort(key=lambda item: item[0])
    positions: list[tuple[float, float] | None] = []
    pointer = 0
    for record in car_records:
        time_value = _record_time(record)
        if time_value is None:
            positions.append(None)
            continue
        while (
            pointer + 1 < len(timed_positions)
            and abs(timed_positions[pointer + 1][0] - time_value)
            <= abs(timed_positions[pointer][0] - time_value)
        ):
            pointer += 1
        positions.append(timed_positions[pointer][1])
    return positions


def _curvature(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
    following: tuple[float, float] | None,
) -> float:
    if previous is None or current is None or following is None:
        return 0.0
    ax, ay = previous
    bx, by = current
    cx, cy = following
    ab = math.hypot(bx - ax, by - ay)
    bc = math.hypot(cx - bx, cy - by)
    ca = math.hypot(ax - cx, ay - cy)
    if min(ab, bc, ca) <= 1e-9:
        return 0.0
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    area_twice = abs(cross)
    radius_denominator = ab * bc * ca
    if radius_denominator <= 1e-9:
        return 0.0
    sign = 1.0 if cross >= 0.0 else -1.0
    return sign * (2.0 * area_twice / radius_denominator)


def _estimated_motion(
    speed_kmh: float | None,
    previous_position: tuple[float, float] | None,
    current_position: tuple[float, float] | None,
    following_position: tuple[float, float] | None,
) -> tuple[float, float]:
    if speed_kmh is None:
        return 0.0, 0.0
    speed_ms = speed_kmh / 3.6
    curvature = _curvature(previous_position, current_position, following_position)
    lateral_g = (speed_ms * speed_ms * curvature) / GRAVITY
    lateral_g = max(-6.0, min(6.0, lateral_g))
    # Internal synthetic traces use steering-like degrees. FastF1 has no
    # steering channel, so estimate a diagnostic proxy from lateral demand.
    steering_degrees = max(-30.0, min(30.0, lateral_g * 18.0))
    return lateral_g, steering_degrees


def _longitudinal_accel(
    previous_speed: float | None,
    current_speed: float | None,
    previous_time: float | None,
    current_time: float | None,
) -> float:
    if (
        previous_speed is None
        or current_speed is None
        or previous_time is None
        or current_time is None
    ):
        return 0.0
    delta_seconds = current_time - previous_time
    if delta_seconds <= 1e-9:
        return 0.0
    delta_ms = (current_speed - previous_speed) / 3.6
    return max(-6.0, min(6.0, delta_ms / delta_seconds / GRAVITY))


def _infer_segment(
    speed: float,
    throttle: float,
    brake: float,
    lateral_accel: float,
) -> str:
    lateral = abs(lateral_accel)
    if brake >= 0.20:
        return "braking"
    if throttle >= 0.70 and lateral < 0.35:
        return "straight"
    if throttle >= 0.55:
        return "exit"
    if speed >= 220.0 and lateral >= 0.45:
        return "high_speed"
    if speed <= 115.0 and lateral >= 0.35:
        return "low_speed"
    if lateral >= 0.75 and throttle <= 0.35:
        return "entry"
    if lateral >= 0.35:
        return "mid_corner"
    return "straight"


def convert_fastf1_records(
    car_records: Iterable[Mapping[str, Any]],
    position_records: Iterable[Mapping[str, Any]] | None = None,
    *,
    year: int | None = None,
    event: str | int | None = None,
    session_name: str | None = None,
    driver: str | None = None,
    lap_number: int | None = None,
) -> list[dict[str, Any]]:
    """Convert FastF1 car/position telemetry records into internal rows."""

    car_rows = list(car_records)
    if not car_rows:
        raise ValueError("FastF1 telemetry contained no car data records")

    position_rows = list(position_records or [])
    positions = _nearest_positions(car_rows, position_rows)
    times = [_record_time(record) for record in car_rows]
    valid_times = [time for time in times if time is not None]
    start_time = min(valid_times) if valid_times else 0.0
    setup_id = (
        f"fastf1_{year}_{event}_{session_name}_{driver}_lap_{lap_number}"
        if year is not None and event is not None and session_name and driver
        else "fastf1_import"
    )

    converted: list[dict[str, Any]] = []
    integrated_distance = 0.0
    previous_time: float | None = None
    previous_speed: float | None = None

    for index, record in enumerate(car_rows):
        current_time = times[index]
        speed = _number(_get_value(record, "Speed", "speed", default=0.0)) or 0.0
        throttle = _normalise_percent(_get_value(record, "Throttle", "throttle", default=0.0))
        brake = _normalise_brake(_get_value(record, "Brake", "brake", default=0.0))
        throttle = throttle if throttle is not None else 0.0
        brake = brake if brake is not None else 0.0

        if previous_time is not None and current_time is not None:
            delta_seconds = max(0.0, current_time - previous_time)
            integrated_distance += (speed / 3.6) * delta_seconds

        distance = _number(_get_value(record, "Distance", "distance"))
        if distance is None:
            distance = integrated_distance

        previous_position = positions[index - 1] if index > 0 else None
        current_position = positions[index]
        following_position = positions[index + 1] if index + 1 < len(positions) else None
        lateral_accel, steering_angle = _estimated_motion(
            speed,
            previous_position,
            current_position,
            following_position,
        )
        longitudinal_accel = _longitudinal_accel(
            previous_speed,
            speed,
            previous_time,
            current_time,
        )

        row = {column: "" for column in CSV_COLUMNS}
        row.update(
            {
                "timestamp": (
                    max(0.0, current_time - start_time)
                    if current_time is not None
                    else float(index)
                ),
                "lap_number": int(lap_number or _number(record.get("LapNumber")) or 1),
                "distance": distance,
                "speed": speed,
                "steering_angle": steering_angle,
                "throttle": throttle,
                "brake": brake,
                "lateral_accel": lateral_accel,
                "longitudinal_accel": longitudinal_accel,
                "yaw_rate": "",
                "gear": _get_value(record, "nGear", "Gear", "gear", default=""),
                "rpm": _get_value(record, "RPM", "rpm", default=""),
                "drs": _get_value(record, "DRS", "drs", default=""),
                "setup_id": setup_id,
                "corner_id": "fastf1_unknown",
                "segment_id": _infer_segment(speed, throttle, brake, lateral_accel),
            }
        )
        converted.append(row)
        previous_time = current_time
        previous_speed = speed

    return converted


def _event_value(event: str | int) -> str | int:
    if isinstance(event, int):
        return event
    stripped = str(event).strip()
    return int(stripped) if stripped.isdigit() else stripped


def _select_lap(driver_laps: Any, lap: int | None) -> Any:
    if lap is None:
        selected = driver_laps.pick_fastest()
        if selected is None:
            raise ValueError("FastF1 did not return a fastest lap for the requested driver")
        return selected

    try:
        matching = driver_laps[driver_laps["LapNumber"] == lap]
        if len(matching) == 0:
            raise ValueError
        return matching.iloc[0]
    except Exception as exc:
        raise ValueError(f"Driver has no lap {lap} in the requested session") from exc


def _lap_number(selected_lap: Any, requested_lap: int | None) -> int | None:
    if requested_lap is not None:
        return requested_lap
    try:
        number = selected_lap["LapNumber"]
    except Exception:
        number = getattr(selected_lap, "LapNumber", None)
    numeric = _number(number)
    return int(numeric) if numeric is not None else None


def _driver_laps(session_laps: Any, driver: str) -> Any:
    picker = getattr(session_laps, "pick_drivers", None)
    if callable(picker):
        return picker(driver)
    picker = getattr(session_laps, "pick_driver", None)
    if callable(picker):
        return picker(driver)
    raise RuntimeError("FastF1 session laps object does not support driver selection")


def _import_fastf1() -> Any:
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "FastF1 is not installed. Install it with `pip install fastf1` "
            "or add it to your environment before using FastF1 features."
        ) from exc
    return fastf1


def _enable_fastf1_cache(fastf1: Any, cache_dir: str | Path) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def check_fastf1_data_access(
    year: int = 2023,
    event: str | int | None = None,
    session_name: str | None = None,
    cache_dir: str | Path = DEFAULT_FASTF1_CACHE_DIR,
    timeout: float = 5.0,
) -> list[dict[str, str]]:
    """Check whether the FastF1 package and public data endpoints are reachable."""

    try:
        fastf1 = _import_fastf1()
        version = str(getattr(fastf1, "__version__", "unknown"))
        results = [
            {
                "name": "FastF1 package",
                "url": "pypi:fastf1",
                "status": "ok",
                "detail": version,
            }
        ]
    except RuntimeError as exc:
        return [
            {
                "name": "FastF1 package",
                "url": "pypi:fastf1",
                "status": "failed",
                "detail": str(exc),
            }
        ]

    for name, template in FASTF1_CONNECTIVITY_URLS:
        url = template.format(year=int(year))
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"fastf1-diagnostic/{version}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read(1)
                status = getattr(response, "status", 200)
        except urllib.error.HTTPError as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "failed",
                    "detail": f"HTTP {exc.code}: {exc.reason}",
                }
            )
        except urllib.error.URLError as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "failed",
                    "detail": str(exc.reason),
                }
            )
        except TimeoutError as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "failed",
                    "detail": f"timeout: {exc}",
                }
            )
        else:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "ok",
                    "detail": f"HTTP {status}",
                }
            )

    if event is None or session_name is None:
        results.append(
            {
                "name": "F1 live timing session",
                "url": "pass --event and --session to check a concrete session",
                "status": "skipped",
                "detail": "session-specific check not requested",
            }
        )
        return results

    try:
        _enable_fastf1_cache(fastf1, cache_dir)
        session = fastf1.get_session(int(year), _event_value(event), session_name)
        session_path = session.api_path + FASTF1_SESSION_PROBE_PAGE
    except Exception as exc:
        results.append(
            {
                "name": "FastF1 session path",
                "url": f"{year} {event} {session_name}",
                "status": "failed",
                "detail": str(exc),
            }
        )
        return results

    for name, base_url in (
        ("F1 live timing session", FASTF1_PRIMARY_BASE_URL),
        ("FastF1 live timing mirror session", FASTF1_MIRROR_BASE_URL),
    ):
        url = base_url + session_path
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"fastf1-diagnostic/{version}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read(1)
                status = getattr(response, "status", 200)
        except urllib.error.HTTPError as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "failed",
                    "detail": f"HTTP {exc.code}: {exc.reason}",
                }
            )
        except urllib.error.URLError as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "failed",
                    "detail": str(exc.reason),
                }
            )
        except TimeoutError as exc:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "failed",
                    "detail": f"timeout: {exc}",
                }
            )
        else:
            results.append(
                {
                    "name": name,
                    "url": url,
                    "status": "ok",
                    "detail": f"HTTP {status}",
                }
            )

    return results


def _session_laps(session: Any) -> Any:
    try:
        return session.laps
    except Exception as exc:
        if _looks_like_data_not_loaded(exc):
            raise RuntimeError(
                "FastF1 did not load lap timing data for this session. "
                "FastF1 must download public session data unless it is already "
                "complete in the local cache. Check internet access, try a "
                "different session, or clear data/fastf1_cache and retry."
            ) from exc
        raise


def _looks_like_data_not_loaded(exc: Exception) -> bool:
    return "Session.load" in str(exc) or exc.__class__.__name__ == "DataNotLoadedError"


def _load_session_with_mirror_retry(
    fastf1: Any,
    year: int,
    event: str | int,
    session_name: str,
    *,
    telemetry: bool,
    retry_with_mirror: bool,
) -> Any:
    def load_session() -> Any:
        loaded_session = fastf1.get_session(int(year), _event_value(event), session_name)
        loaded_session.load(
            laps=True,
            telemetry=telemetry,
            weather=False,
            messages=False,
        )
        _session_laps(loaded_session)
        return loaded_session

    try:
        return load_session()
    except RuntimeError as exc:
        if not retry_with_mirror:
            raise
        try:
            import fastf1._api as fastf1_api
        except ImportError:
            raise exc
        original_base_url = fastf1_api.base_url
        mirror_base_url = getattr(fastf1_api, "base_url_mirror", FASTF1_MIRROR_BASE_URL)
        if original_base_url == mirror_base_url:
            raise
        try:
            fastf1_api.base_url = mirror_base_url
            return load_session()
        except RuntimeError as mirror_exc:
            raise RuntimeError(
                f"{exc} Retried through FastF1 live timing mirror, but that "
                f"also failed: {mirror_exc}"
            ) from mirror_exc
        finally:
            fastf1_api.base_url = original_base_url


def list_fastf1_events(
    year: int,
    cache_dir: str | Path = DEFAULT_FASTF1_CACHE_DIR,
) -> tuple[str, ...]:
    """Return event names available from the FastF1 schedule for a season."""

    fastf1 = _import_fastf1()
    _enable_fastf1_cache(fastf1, cache_dir)
    schedule = fastf1.get_event_schedule(int(year), include_testing=False)
    events: list[str] = []
    for record in _records_from_dataframe(schedule):
        event_name = record.get("EventName")
        if event_name is None:
            continue
        event_text = str(event_name)
        if event_text and event_text not in events:
            events.append(event_text)
    if not events:
        raise RuntimeError(f"FastF1 returned no events for {year}")
    return tuple(events)


def list_fastf1_session_lap_options(
    year: int,
    event: str | int,
    session_name: str,
    cache_dir: str | Path = DEFAULT_FASTF1_CACHE_DIR,
    retry_with_mirror: bool = True,
) -> FastF1SessionLapOptions:
    """Return driver and lap dropdown options for a FastF1 session."""

    fastf1 = _import_fastf1()
    _enable_fastf1_cache(fastf1, cache_dir)
    session = _load_session_with_mirror_retry(
        fastf1,
        year,
        event,
        session_name,
        telemetry=False,
        retry_with_mirror=retry_with_mirror,
    )

    laps_by_driver: dict[str, set[int]] = {}
    driver_order: list[str] = []
    for record in _records_from_dataframe(_session_laps(session)):
        driver = _get_value(record, "Driver", "Abbreviation", "DriverNumber")
        lap_number = _number(record.get("LapNumber"))
        if driver is None or lap_number is None:
            continue
        driver_text = str(driver).strip()
        lap_int = int(lap_number)
        if not driver_text or lap_int <= 0:
            continue
        if driver_text not in laps_by_driver:
            driver_order.append(driver_text)
            laps_by_driver[driver_text] = set()
        laps_by_driver[driver_text].add(lap_int)

    if not driver_order:
        raise RuntimeError(f"FastF1 returned no driver lap data for {event} {session_name}")

    return FastF1SessionLapOptions(
        drivers=tuple(driver_order),
        laps_by_driver={
            driver: tuple(sorted(laps))
            for driver, laps in laps_by_driver.items()
        },
    )


def _lap_car_records(selected_lap: Any) -> list[Mapping[str, Any]]:
    try:
        car_data = selected_lap.get_car_data()
    except Exception as exc:
        if _looks_like_data_not_loaded(exc):
            raise RuntimeError(
                "FastF1 did not load car telemetry for this session. "
                "This usually means the requested session has no public telemetry, "
                "the FastF1 download failed, or the cache contains an incomplete "
                "session. Try another session/driver or clear data/fastf1_cache."
            ) from exc
        raise

    add_distance = getattr(car_data, "add_distance", None)
    if callable(add_distance):
        car_data = add_distance()
    records = _records_from_dataframe(car_data)
    if not records:
        raise RuntimeError("FastF1 returned no car telemetry records for the selected lap")
    return records


def _lap_position_records(selected_lap: Any) -> list[Mapping[str, Any]]:
    try:
        return _records_from_dataframe(selected_lap.get_pos_data())
    except Exception as exc:
        if _looks_like_data_not_loaded(exc):
            return []
        raise


def fetch_fastf1_to_csv(
    year: int,
    event: str | int,
    session_name: str,
    driver: str,
    output: str | Path,
    lap: int | None = None,
    cache_dir: str | Path = DEFAULT_FASTF1_CACHE_DIR,
    retry_with_mirror: bool = True,
) -> Path:
    """Fetch one FastF1 driver lap and write it as internal telemetry CSV."""

    fastf1 = _import_fastf1()
    _enable_fastf1_cache(fastf1, cache_dir)
    session = _load_session_with_mirror_retry(
        fastf1,
        year,
        event,
        session_name,
        telemetry=True,
        retry_with_mirror=retry_with_mirror,
    )
    driver_laps = _driver_laps(_session_laps(session), driver)

    selected_lap = _select_lap(driver_laps, lap)
    selected_lap_number = _lap_number(selected_lap, lap)

    rows = convert_fastf1_records(
        _lap_car_records(selected_lap),
        _lap_position_records(selected_lap),
        year=int(year),
        event=event,
        session_name=session_name,
        driver=driver,
        lap_number=selected_lap_number,
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return target
