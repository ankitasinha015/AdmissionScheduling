"""
Fit a length-of-stay distribution from the MIMIC-IV demo.

Reads hosp/admissions.csv.gz, computes stay duration, fits a lognormal,
and prints parameters you can paste into the simulator.

Run from the repo root:
    python scripts/fit_los.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MIMIC_DIR = Path("data/mimic")
OUT_DIR = Path("docs/results")

# stays outside this range are almost always data entry errors,
# not real patients. 1 hour to 60 days.
MIN_HOURS = 1.0
MAX_HOURS = 24 * 60

# observation stays are short-stay patients who mostly do not compete for
# the inpatient beds this scheduler assigns. leaving them in creates a
# second hump near 15-30h that no single lognormal can fit.
DROP_OBSERVATION = True


def find_admissions_file(root: Path) -> Path:
    """The demo zip nests things differently depending on how it was unpacked."""
    matches = list(root.rglob("admissions.csv.gz")) + list(root.rglob("admissions.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No admissions file under {root.resolve()}. "
            "Unzip the MIMIC-IV demo into data/mimic/ first."
        )
    return matches[0]


def load_stays(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["admittime", "dischtime"])
    df = df.dropna(subset=["admittime", "dischtime"])

    # dates are shifted per patient for privacy, but the shift is constant
    # within a patient, so the difference is the real duration.
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 3600.0

    before = len(df)
    df = df[(df["los_hours"] >= MIN_HOURS) & (df["los_hours"] <= MAX_HOURS)]
    print(f"kept {len(df)} of {before} admissions after range filter")

    if DROP_OBSERVATION and "admission_type" in df.columns:
        obs = df["admission_type"].str.contains("OBSERVATION", na=False)
        print(f"dropping {int(obs.sum())} observation stays")
        df = df[~obs]
        print(f"{len(df)} inpatient stays remain")

    return df


def fit_lognormal(hours: np.ndarray) -> tuple[float, float]:
    """
    Returns (mu, sigma) for numpy's lognormal, which parameterises by the
    mean and sd of the underlying normal.

    floc=0 pins the location at zero. Left free, scipy will happily shift
    the whole distribution to squeeze out a marginally better fit and give
    you parameters that mean nothing.
    """
    shape, loc, scale = stats.lognorm.fit(hours, floc=0)
    mu = np.log(scale)
    sigma = shape
    return mu, sigma


def report(hours: np.ndarray, mu: float, sigma: float) -> None:
    print()
    print("observed")
    print(f"  n            {len(hours)}")
    print(f"  median       {np.median(hours):.1f} h  ({np.median(hours)/24:.1f} d)")
    print(f"  mean         {hours.mean():.1f} h  ({hours.mean()/24:.1f} d)")
    print(f"  p90          {np.percentile(hours, 90):.1f} h")
    print(f"  p99          {np.percentile(hours, 99):.1f} h")
    print(f"  max          {hours.max():.1f} h")

    print()
    print("fitted lognormal")
    print(f"  mu           {mu:.4f}")
    print(f"  sigma        {sigma:.4f}")
    print(f"  median       {np.exp(mu):.1f} h")
    print(f"  mean         {np.exp(mu + sigma**2 / 2):.1f} h")

    # if the fit is good, a sample from it should look like the real thing.
    sample = np.random.default_rng(0).lognormal(mu, sigma, size=20000)
    print()
    print("sampled from fit")
    print(f"  median       {np.median(sample):.1f} h")
    print(f"  p90          {np.percentile(sample, 90):.1f} h")

    ks = stats.kstest(hours, "lognorm", args=(sigma, 0, np.exp(mu)))
    print()
    print(f"KS statistic   {ks.statistic:.4f}  (lower is better, <0.10 is a decent fit)")
    print(f"p-value        {ks.pvalue:.4f}")


def plot(hours: np.ndarray, mu: float, sigma: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    cutoff = np.percentile(hours, 99)
    visible = hours[hours <= cutoff]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(visible, bins=50, density=True, alpha=0.6, label="MIMIC-IV demo")

    x = np.linspace(MIN_HOURS, cutoff, 500)
    pdf = stats.lognorm.pdf(x, sigma, loc=0, scale=np.exp(mu))
    ax.plot(x, pdf, linewidth=2, label="fitted lognormal")

    ax.set_xlabel("length of stay (hours)")
    ax.set_ylabel("density")
    ax.set_title("Inpatient length of stay, MIMIC-IV demo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nplot written to {out}")


def by_admission_type(df: pd.DataFrame) -> None:
    """MIMIC labels admissions by type. Useful for splitting streams later."""
    if "admission_type" not in df.columns:
        return
    print()
    print("by admission type")
    grouped = df.groupby("admission_type")["los_hours"]
    for name, series in grouped:
        print(f"  {name:<32} n={len(series):>4}  median={np.median(series):>7.1f} h")


def by_stream(df: pd.DataFrame) -> None:
    """Group into the two streams the scheduler actually cares about."""
    if "admission_type" not in df.columns:
        return

    planned_types = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
    planned = df[df["admission_type"].isin(planned_types)]
    ed = df[~df["admission_type"].isin(planned_types)]

    print()
    print("by stream")
    for label, subset in (("planned", planned), ("ed", ed)):
        if len(subset) == 0:
            continue
        hours = subset["los_hours"].to_numpy()
        mu, sigma = fit_lognormal(hours)
        print(
            f"  {label:<8} n={len(hours):>4}  "
            f"median={np.median(hours):>7.1f} h  "
            f"mu={mu:.4f}  sigma={sigma:.4f}"
        )


def main() -> None:
    path = find_admissions_file(MIMIC_DIR)
    print(f"reading {path}")

    df = load_stays(path)
    hours = df["los_hours"].to_numpy()

    mu, sigma = fit_lognormal(hours)
    report(hours, mu, sigma)
    by_admission_type(df)
    by_stream(df)
    plot(hours, mu, sigma, OUT_DIR / "los_fit.png")

    print()
    print("paste into the simulator:")
    print(f"    LOS_MU = {mu:.4f}")
    print(f"    LOS_SIGMA = {sigma:.4f}")


if __name__ == "__main__":
    main()