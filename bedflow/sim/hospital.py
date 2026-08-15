"""
Fake hospital. Generates admission requests, stays, and hourly census.

Two arrival streams, an hourly demand curve, stay lengths fitted from the
MIMIC-IV demo, and discharges clustered late morning.

Run from the repo root:
    python -m bedflow.sim.hospital
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from bedflow.domain.models import Acuity, CareLevel, Source

# --------------------------------------------------------------------------
# fitted from the MIMIC-IV demo. see scripts/fit_los.py
# planned and ED medians are nearly identical, but ED is far more variable.
# that variance is the thing the protection level defends against.
# --------------------------------------------------------------------------
LOS_MU = {Source.PLANNED: 4.9197, Source.ED: 4.9479}
LOS_SIGMA = {Source.PLANNED: 0.5767, Source.ED: 0.8093}

# --------------------------------------------------------------------------
# capacity. sized backwards from Little's Law so occupancy lands near 75%.
#   L = lambda * W
#   ~0.80 requests/hour * ~186 hour mean stay = ~148 beds occupied
#   148 / 200 = 74%
# --------------------------------------------------------------------------
BEDS_PER_UNIT = {
    CareLevel.ICU: 20,
    CareLevel.STEPDOWN: 30,
    CareLevel.TELEMETRY: 50,
    CareLevel.MEDSURG: 100,
}
TOTAL_BEDS = sum(BEDS_PER_UNIT.values())

CARE_LEVEL_MIX = {
    CareLevel.ICU: 0.10,
    CareLevel.STEPDOWN: 0.15,
    CareLevel.TELEMETRY: 0.25,
    CareLevel.MEDSURG: 0.50,
}

# --------------------------------------------------------------------------
# demand. base rate is requests per hour averaged over the day. multipliers
# average close to 1.0, so the base number means what it says and you can
# scale total volume without touching the shape of the curve.
# --------------------------------------------------------------------------
ED_BASE = 0.57
PLANNED_BASE = 0.33

# admission requests, not ED door arrivals. someone walks in at 2pm, gets
# worked up, and the decision to admit lands at 5pm. so this curve sits
# later than a raw arrivals curve would.
ED_HOURLY = [
    0.45, 0.35, 0.30, 0.28, 0.30, 0.35,
    0.50, 0.70, 0.95, 1.15, 1.30, 1.35,
    1.30, 1.30, 1.35, 1.45, 1.60, 1.70,
    1.65, 1.50, 1.30, 1.10, 0.85, 0.60,
]

# planned admissions are not a smooth curve. they land in two morning
# blocks and are zero the rest of the day.
PLANNED_HOURLY = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.5, 3.5, 4.5, 4.0, 3.0, 2.5,
    2.0, 1.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

# the two streams move in opposite directions on weekends. do not apply one
# weekend factor to both, it erases the pattern the model should find.
WEEKEND_FACTOR = {Source.PLANNED: 0.08, Source.ED: 1.03}

# discharges cluster late morning because that is when rounds happen.
# admissions peak in the evening. those peaks do not line up, and that gap
# is the entire reason bed assignment is hard.
DISCHARGE_HOURLY = [
    0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
    0.03, 0.05, 0.07, 0.09, 0.11, 0.11,
    0.10, 0.09, 0.07, 0.05, 0.04, 0.03,
    0.03, 0.02, 0.02, 0.01, 0.01, 0.01,
]

# --------------------------------------------------------------------------
# hidden structure. the model is never told any of this exists. if it picks
# these up from the features it does have, the pipeline works. if it does
# not, that is a real finding rather than a bug.
# --------------------------------------------------------------------------
FLU_START_DAY = 40
FLU_END_DAY = 61
FLU_LOS_MULTIPLIER = 1.15

QUIET_WEEKDAY = 2          # planned admissions dip on Wednesdays
QUIET_WEEKDAY_FACTOR = 0.6

SURGE_DAY_COUNT = 5        # days where ED demand roughly doubles
SURGE_MULTIPLIER = 2.0

# --------------------------------------------------------------------------
WARMUP_DAYS = 14
DEFAULT_DAYS = 90
START = datetime(2025, 1, 6)   # a Monday, so weekday 0 is Monday

OUT_DIR = Path("data")


@dataclass(frozen=True)
class Stay:
    id: str
    source: Source
    care_level: CareLevel
    acuity: Acuity
    arrived_at: datetime
    discharged_at: datetime


def arrival_rate(hour: int, weekday: int, source: Source) -> float:
    """
    Expected requests per hour. Pure lookup, no randomness.

    Kept separate from generation on purpose: this same function drives the
    Locust ramp in M8, just multiplied by a load factor.
    """
    if source is Source.PLANNED:
        rate = PLANNED_BASE * PLANNED_HOURLY[hour]
        if weekday == QUIET_WEEKDAY:
            rate *= QUIET_WEEKDAY_FACTOR
    else:
        rate = ED_BASE * ED_HOURLY[hour]

    if weekday >= 5:
        rate *= WEEKEND_FACTOR[source]

    return rate


def generate_arrivals(
    days: int,
    source: Source,
    surge_days: set[int],
    rng: random.Random,
) -> list[datetime]:
    """
    Nonhomogeneous Poisson process by thinning.

    Generate candidates at the peak rate using exponential gaps, then keep
    each one with probability rate(t) / peak. Provably correct, six lines,
    and it produces natural clumping. Sampling a random hour uniformly
    would not: real arrivals bunch up, and burstiness is exactly what the
    load test needs to be honest.
    """
    horizon = days * 24
    peak = max(
        arrival_rate(h, d, source) for h in range(24) for d in range(7)
    ) * (SURGE_MULTIPLIER if source is Source.ED else 1.0)

    out: list[datetime] = []
    t = 0.0
    while t < horizon:
        t += rng.expovariate(peak)
        if t >= horizon:
            break

        moment = START + timedelta(hours=t)
        day_index = int(t // 24)
        rate = arrival_rate(moment.hour, moment.weekday(), source)

        if source is Source.ED and day_index in surge_days:
            rate *= SURGE_MULTIPLIER

        if rng.random() < rate / peak:
            out.append(moment)

    return out


def sample_stay_hours(source: Source, arrived_at: datetime, rng: random.Random) -> float:
    """Lognormal, because hospital stays are heavily right skewed."""
    hours = rng.lognormvariate(LOS_MU[source], LOS_SIGMA[source])

    day_index = (arrived_at - START).days
    if FLU_START_DAY <= day_index <= FLU_END_DAY:
        hours *= FLU_LOS_MULTIPLIER

    return hours


def snap_to_discharge_hour(ready_at: datetime, rng: random.Random) -> datetime:
    """
    A patient is medically ready at some arbitrary moment, but nobody is
    discharged at 3am. Draw a discharge hour from the rounds curve, then use
    the first time that hour comes around at or after the ready moment.

    Order matters. Drawing from the full curve first and choosing the day
    second keeps the curve's shape intact. Restricting the draw to hours
    still left in the current day would instead pile patients onto whatever
    late hour they happened to become ready at, which is exactly what a
    truncated distribution does to you.

    This is what creates the supply and demand mismatch: beds free up around
    lunchtime, requests peak in the evening.
    """
    hour = rng.choices(range(24), weights=DISCHARGE_HOURLY, k=1)[0]

    day = ready_at.date()
    if hour < ready_at.hour:
        day = (ready_at + timedelta(days=1)).date()

    return datetime(day.year, day.month, day.day, hour)


def pick_care_level(rng: random.Random) -> CareLevel:
    levels = list(CARE_LEVEL_MIX)
    weights = [CARE_LEVEL_MIX[level] for level in levels]
    return rng.choices(levels, weights=weights, k=1)[0]


def pick_acuity(source: Source, rng: random.Random) -> Acuity:
    if source is Source.PLANNED:
        return rng.choices(
            [Acuity.ESI_2, Acuity.ESI_3, Acuity.ESI_4],
            weights=[0.15, 0.60, 0.25],
            k=1,
        )[0]
    return rng.choices(
        [Acuity.ESI_1, Acuity.ESI_2, Acuity.ESI_3, Acuity.ESI_4, Acuity.ESI_5],
        weights=[0.05, 0.25, 0.45, 0.20, 0.05],
        k=1,
    )[0]


def build_stays(days: int, seed: int) -> list[Stay]:
    rng = random.Random(seed)
    surge_days = set(rng.sample(range(days), SURGE_DAY_COUNT))

    stays: list[Stay] = []
    counter = 0

    for source in (Source.ED, Source.PLANNED):
        for arrived_at in generate_arrivals(days, source, surge_days, rng):
            hours = sample_stay_hours(source, arrived_at, rng)
            ready_at = arrived_at + timedelta(hours=hours)
            discharged_at = snap_to_discharge_hour(ready_at, rng)

            counter += 1
            stays.append(
                Stay(
                    id=f"S{counter:06d}",
                    source=source,
                    care_level=pick_care_level(rng),
                    acuity=pick_acuity(source, rng),
                    arrived_at=arrived_at,
                    discharged_at=discharged_at,
                )
            )

    stays.sort(key=lambda s: s.arrived_at)
    return stays


def build_census(stays: list[Stay], days: int) -> list[dict]:
    """
    Occupancy per hour, per care level.

    Uses a delta array rather than counting every stay at every hour: +1 at
    arrival, -1 at discharge, then a running total. Linear instead of
    quadratic, which matters once you push this to a year of history.
    """
    horizon = days * 24
    levels = list(BEDS_PER_UNIT)

    deltas = {level: [0] * (horizon + 1) for level in levels}
    admits_ed = [0] * horizon
    admits_planned = [0] * horizon
    discharges = [0] * horizon

    for stay in stays:
        start = int((stay.arrived_at - START).total_seconds() // 3600)
        end = int((stay.discharged_at - START).total_seconds() // 3600)
        if start < 0 or start >= horizon:
            continue

        deltas[stay.care_level][start] += 1
        if end < horizon:
            deltas[stay.care_level][end] -= 1
            discharges[end] += 1

        if stay.source is Source.ED:
            admits_ed[start] += 1
        else:
            admits_planned[start] += 1

    running = {level: 0 for level in levels}
    rows: list[dict] = []

    for h in range(horizon):
        for level in levels:
            running[level] += deltas[level][h]

        moment = START + timedelta(hours=h)
        row = {
            "timestamp": moment.isoformat(),
            "hour": moment.hour,
            "weekday": moment.weekday(),
            "admissions_ed": admits_ed[h],
            "admissions_planned": admits_planned[h],
            "discharges": discharges[h],
            "occupied_total": sum(running.values()),
        }
        for level in levels:
            row[f"occupied_{level.name.lower()}"] = running[level]
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(days: int = DEFAULT_DAYS, seed: int = 42, out_dir: Path = OUT_DIR) -> None:
    stays = build_stays(days, seed)
    census = build_census(stays, days)

    # the hospital starts empty. the first two weeks show occupancy climbing
    # from zero, which no real hospital ever does. train on that and the
    # model learns a hospital where beds are always plentiful.
    cutoff = START + timedelta(days=WARMUP_DAYS)
    census = [r for r in census if datetime.fromisoformat(r["timestamp"]) >= cutoff]
    kept_stays = [s for s in stays if s.arrived_at >= cutoff]

    write_csv(out_dir / "census.csv", census)
    write_csv(
        out_dir / "stays.csv",
        [
            {
                "id": s.id,
                "source": s.source.value,
                "care_level": s.care_level.name,
                "acuity": s.acuity.value,
                "arrived_at": s.arrived_at.isoformat(),
                "discharged_at": s.discharged_at.isoformat(),
                "los_hours": round(
                    (s.discharged_at - s.arrived_at).total_seconds() / 3600, 2
                ),
            }
            for s in kept_stays
        ],
    )

    summarise(census, kept_stays)


def summarise(census: list[dict], stays: list[Stay]) -> None:
    occ = [r["occupied_total"] for r in census]
    mean_occ = sum(occ) / len(occ)

    print(f"stays          {len(stays)}")
    print(f"census rows    {len(census)}  ({len(census) / 24:.0f} days after warmup)")
    print(f"occupancy      mean {mean_occ:.0f} of {TOTAL_BEDS} beds "
          f"({mean_occ / TOTAL_BEDS:.0%})")
    print(f"               min {min(occ)}  max {max(occ)}")

    ed = sum(1 for s in stays if s.source is Source.ED)
    print(f"stream mix     ed {ed}  planned {len(stays) - ed}")

    # split the streams. a merged histogram hides the ED evening peak behind
    # the planned morning block, and hides bugs in either one.
    ed_h = [0] * 24
    pl_h = [0] * 24
    dc_h = [0] * 24
    for row in census:
        h = row["hour"]
        ed_h[h] += row["admissions_ed"]
        pl_h[h] += row["admissions_planned"]
        dc_h[h] += row["discharges"]

    print()
    print("hour    ed  planned  disch   ed profile          discharge profile")
    ed_peak = max(1, max(ed_h))
    dc_peak = max(1, max(dc_h))
    for h in range(24):
        ed_bar = "#" * round(ed_h[h] / ed_peak * 18)
        dc_bar = "#" * round(dc_h[h] / dc_peak * 18)
        print(
            f"{h:>4} {ed_h[h]:>5} {pl_h[h]:>8} {dc_h[h]:>6}   "
            f"{ed_bar:<20}{dc_bar}"
        )


if __name__ == "__main__":
    run()
