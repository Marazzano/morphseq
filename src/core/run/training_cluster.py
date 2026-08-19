"""Retired compatibility entrypoint for cluster training.

Use the single supported entrypoint and select the former cluster metric config:

    python -m src.core.run.training --config-name base_cluster_metric
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "src.core.run.training_cluster is retired. Use "
        "`python -m src.core.run.training --config-name base_cluster_metric`."
    )


if __name__ == "__main__":
    main()
