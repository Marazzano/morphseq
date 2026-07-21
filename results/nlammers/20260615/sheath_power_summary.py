"""Shared analysis code for the sheath-cell power comparison notebook.

Recomputes everything from the cached count tables so the notebook does not
depend on numbers pasted from a previous session:

  * per-embryo sheath counts and library depth, by dataset and timepoint
  * method-of-moments NB dispersion (theta) per dataset/timepoint
  * analytic NB Wald power across effect sizes and per-arm sample sizes
  * minimum detectable effect at a target power

The power model matches the inference model used in the analysis notebooks:
an offset NB GLM contrasting two arms, where the variance of the log rate is
approximately (1/mu + 1/theta) per arm.
"""

import csv
import math
import statistics as s
from collections import defaultdict

DATA_DIR = (
    "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    "results/nlammers/20260615"
)
COMBINED_COUNTS = f"{DATA_DIR}/embryo_cell_counts_long.csv"
V230_COUNTS = f"{DATA_DIR}/atlas_count_cache/embryo_cell_counts_v2.3.0.csv"
GENE7_COUNTS = f"{DATA_DIR}/gene7_28C_ctrl_sheath_by_embryo.csv"
# GENE11 is the most representative OLD-protocol dataset (v3.0.1). 28C throughout;
# controls only (perturbation == "ctrl-inj"). Overlaps the panel at 36 hpf only.
GENE11_COUNTS = f"{DATA_DIR}/gene11_28C_ctrl_sheath_by_embryo.csv"

PANEL_HPF = [24, 30, 36]
ALPHA = 0.05
MIN_EMBRYOS = 5

# Datasets whose theta is an artifact of small-sample underdispersion get
# flagged rather than silently reported as infinitely precise.
THETA_CEILING = 1e6

# Depth is a sequencing-time choice, not a property of the biology, so power is
# compared at a common standardized depth rather than each dataset's native depth.
# The expected sheath count becomes rate * STANDARD_DEPTH; theta (the between-embryo
# CV of the rate) is depth-invariant and is held fixed.
STANDARD_DEPTH = 3000


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load_embryo_counts():
    """Per-embryo (sheath, total) for every dataset in the combined table."""
    totals, sheath, hpf = defaultdict(int), defaultdict(int), {}
    with open(COMBINED_COUNTS) as handle:
        for row in csv.DictReader(handle):
            try:
                count = int(float(row["cell_count"]))
                stage = float(row["hpf"])
            except (TypeError, ValueError):
                continue
            key = (row["dataset"], row["embryo"])
            totals[key] += count
            hpf[key] = stage
            if "sheath" in row["cell_group"].lower():
                sheath[key] += count

    groups = defaultdict(lambda: ([], []))
    for key, total in totals.items():
        if total <= 0 or hpf[key] not in PANEL_HPF:
            continue
        dataset, _ = key
        groups[(dataset, hpf[key])][0].append(sheath.get(key, 0))
        groups[(dataset, hpf[key])][1].append(total)
    return groups


def load_gap16_counts():
    """GAP16 is a single v2.3.0 experiment; embryo IDs carry the prefix."""
    totals, sheath, hpf = defaultdict(int), defaultdict(int), {}
    with open(V230_COUNTS) as handle:
        for row in csv.DictReader(handle):
            try:
                count = int(float(row["cell_count"]))
                stage = float(row["hpf"])
            except (TypeError, ValueError):
                continue
            embryo = row["embryo"]
            totals[embryo] += count
            hpf[embryo] = stage
            if "sheath" in row["cell_group"].lower():
                sheath[embryo] += count

    groups = defaultdict(lambda: ([], []))
    for embryo, total in totals.items():
        if total <= 0 or embryo.split("_")[0] != "GAP16":
            continue
        groups[("GAP16 (v2.3.0)", hpf[embryo])][0].append(sheath.get(embryo, 0))
        groups[("GAP16 (v2.3.0)", hpf[embryo])][1].append(total)
    return groups


def _load_control_extract(path, label):
    """Per-embryo (sheath, total) from a pre-filtered control-only CSV.

    Used for GENE datasets extracted directly from their CDS colData, where the
    control filter (temp / perturbation) was already applied at extraction time.
    """
    groups = defaultdict(lambda: ([], []))
    with open(path) as handle:
        for row in csv.DictReader(handle):
            stage = float(row["hpf"])
            groups[(label, stage)][0].append(int(row["sheath"]))
            groups[(label, stage)][1].append(int(row["total_cells"]))
    return groups


def load_gene7_counts():
    return _load_control_extract(GENE7_COUNTS, "GENE7 28C ctrl")


def load_gene11_counts():
    return _load_control_extract(GENE11_COUNTS, "GENE11 ctrl (old)")


def collect_groups(all_timepoints=False):
    """Merge every source into one {(dataset, hpf): (sheath, total)} mapping."""
    groups = {}
    for source in (load_embryo_counts(), load_gap16_counts(),
                   load_gene7_counts(), load_gene11_counts()):
        for key, value in source.items():
            groups[key] = value

    keep = {}
    for key, (sheath, total) in groups.items():
        if len(sheath) < MIN_EMBRYOS or sum(sheath) == 0:
            continue
        if not all_timepoints and key[1] not in PANEL_HPF:
            continue
        keep[key] = (sheath, total)
    return keep


def nb_theta_mom(sheath, total):
    """Method-of-moments NB dispersion for rate mu_i = r * offset_i.

    Returns (theta, rate). theta is inf when the observed variance falls below
    Poisson, which happens in small groups and should be read as "at least this
    precise", not as a real estimate.
    """
    n = len(sheath)
    rate = sum(sheath) / sum(total)
    mu = [rate * t for t in total]
    excess = sum(((y - m) ** 2 - m) / m**2 for y, m in zip(sheath, mu) if m > 0)
    theta = math.inf if excess <= 0 else n / excess
    return theta, rate


def nb_power(theta, rate, depth, n_per_arm, effect, alpha=ALPHA):
    """Two-sided Wald power for a fractional reduction in an offset NB GLM."""
    mu_ctrl = rate * depth
    if mu_ctrl <= 0:
        return 0.0
    mu_trt = mu_ctrl * (1 - effect)
    var = (1 / mu_ctrl + 1 / theta) / n_per_arm + (1 / mu_trt + 1 / theta) / n_per_arm
    se = math.sqrt(var)
    beta = abs(math.log(1 - effect))
    z = 1.959964 if alpha == 0.05 else -_z_quantile(alpha / 2)
    lam = beta / se
    return _phi(lam - z) + _phi(-lam - z)


def _z_quantile(p):
    """Inverse normal CDF (Acklam), only needed for non-default alpha."""
    a = [-39.696830286653757, 220.94609842452050, -275.92851044696869,
         138.35775186726900, -30.664798066147160, 2.5066282774592392]
    b = [-54.476098798224058, 161.58583685804089, -155.69897985988661,
         66.801311887719720, -13.280681552885721]
    c = [-0.0077848940024376, -0.32239645804113, -2.4007582771618,
         -2.5497325393437, 4.3746641414649, 2.9381639826987]
    d = [0.0077846957090414, 0.32246712907004, 2.4451341681548, 3.7544086619074]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        return -_z_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def mde(theta, rate, depth, n_per_arm, target_power=0.80, hi=0.95):
    """Smallest fractional reduction reaching target power; None if unreachable."""
    if nb_power(theta, rate, depth, n_per_arm, hi) < target_power:
        return None
    lo = 0.005
    for _ in range(60):
        mid = (lo + hi) / 2
        if nb_power(theta, rate, depth, n_per_arm, mid) < target_power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summarize(groups, standard_depth=STANDARD_DEPTH):
    """One record per dataset/timepoint with counts, theta, and depth.

    `standard_depth` is the common depth (cells/embryo) at which power is compared.
    The observed rate and theta are kept as-is; only the expected count is
    re-expressed at the standard depth, via `std_expected_sheath = rate * standard_depth`.
    """
    out = []
    for (dataset, stage), (sheath, total) in groups.items():
        theta, rate = nb_theta_mom(sheath, total)
        depth = s.median(total)
        out.append(
            {
                "dataset": dataset,
                "hpf": stage,
                "n_embryos": len(sheath),
                "median_depth": depth,
                "mean_sheath": s.mean(sheath),
                "median_sheath": s.median(sheath),
                "sd_sheath": s.stdev(sheath) if len(sheath) > 1 else 0.0,
                "per_1000": 1000 * sum(sheath) / sum(total),
                "theta": theta,
                "theta_is_ceiling": theta > THETA_CEILING,
                "rate": rate,
                "standard_depth": standard_depth,
                "std_expected_sheath": rate * standard_depth,
                "sheath_counts": sheath,
                "total_counts": total,
            }
        )
    out.sort(key=lambda r: (r["hpf"], -r["std_expected_sheath"]))
    return out


def std_power(record, n_per_arm, effect):
    """Power at the record's standardized depth (holds abundance and theta fixed)."""
    return nb_power(record["theta"], record["rate"],
                    record["standard_depth"], n_per_arm, effect)


def std_mde(record, n_per_arm, target_power=0.80):
    """Minimum detectable effect at the standardized depth."""
    return mde(record["theta"], record["rate"],
               record["standard_depth"], n_per_arm, target_power)


def variance_split(record, n_per_arm=24, use_standard_depth=True):
    """Poisson vs overdispersion share of the log-rate variance (one arm).

    At the standardized depth by default, so the split reflects the common-depth
    comparison rather than each dataset's native sequencing depth.
    """
    depth = record["standard_depth"] if use_standard_depth else record["median_depth"]
    mu = record["rate"] * depth
    poisson = 1 / mu if mu > 0 else 0.0
    over = 0.0 if record["theta_is_ceiling"] else 1 / record["theta"]
    total = poisson + over
    return poisson, over, (over / total if total > 0 else 0.0)
