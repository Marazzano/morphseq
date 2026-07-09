"""Plot sequenced-only homozygous phenotype predictions from saved CSVs."""

from __future__ import annotations

import pandas as pd

from sci_cilia_qc_config import PLOTS_DIR, PREDICTIONS_DIR, RUN_DIR
import b9d2_homo_ce_hta
import cep290_homo_low_to_high
import make_trajectory_plots_sci


def _read_required(name: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run generate_sequenced_predictions.py first.")
    return pd.read_csv(path)


def main() -> None:
    b9d2_homo_ce_hta.SEQ_OUT = PLOTS_DIR / "phenotype_predictions_sequenced" / "b9d2"
    b9d2_homo_ce_hta.OUT = PLOTS_DIR / "phenotype_predictions_sequenced" / "b9d2" / "reference_qc"
    cep290_homo_low_to_high.SEQ_OUT = PLOTS_DIR / "phenotype_predictions_sequenced" / "cep290"
    cep290_homo_low_to_high.OUT = PLOTS_DIR / "phenotype_predictions_sequenced" / "cep290" / "reference_qc"

    b9 = _read_required("b9d2_homo_ce_hta_embryo_predictions.csv")
    ce = _read_required("cep290_homo_low_to_high_embryo_predictions.csv")
    b9_cv_path = PREDICTIONS_DIR / "b9d2_homo_ce_hta_reference_cv_target_hpf_pm2.csv"
    ce_cv_path = PREDICTIONS_DIR / "cep290_homo_low_to_high_reference_cv_target_hpf_pm2.csv"

    print("Plotting b9d2 homozygous phenotype predictions from saved predictions.")
    b9d2_homo_ce_hta._plot_minibars(b9, homo_only=True)
    b9d2_homo_ce_hta._plot_minibars(b9, homo_only=False)
    b9d2_homo_ce_hta._plot_probability_spectrum(b9)
    if b9_cv_path.exists():
        b9d2_homo_ce_hta._plot_target_confusion(pd.read_csv(b9_cv_path))

    print("\nPlotting cep290 homozygous phenotype predictions from saved predictions.")
    cep290_homo_low_to_high._plot_minibars(ce, homo_only=True)
    cep290_homo_low_to_high._plot_minibars(ce, homo_only=False)
    ce_cv = pd.read_csv(ce_cv_path) if ce_cv_path.exists() else pd.DataFrame()
    cep290_homo_low_to_high._plot_probability_spectrum(ce)
    if not ce_cv.empty:
        cep290_homo_low_to_high._plot_target_confusion(ce_cv)

    print("\nRegenerating sequenced time-series trajectory plots.")
    make_trajectory_plots_sci.main()
    print(f"\nPhenotype plots under: {(PLOTS_DIR / phenotype_predictions_sequenced).relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
