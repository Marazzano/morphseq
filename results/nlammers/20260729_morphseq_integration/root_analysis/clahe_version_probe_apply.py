"""Apply this interpreter's scikit-image CLAHE to exact shared raw crops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import skimage
from skimage.exposure import equalize_adapthist


HERE = Path(__file__).resolve().parent


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "frac_ge_250": float(np.mean(values >= 250)),
        "frac_eq_255": float(np.mean(values == 255)),
    }


def main() -> None:
    payload = np.load(HERE / "current_raw_crops_6p5.npz")
    raw = payload["raw"]
    masks = payload["mask"].astype(bool)
    wells = payload["wells"].astype(str)

    per_well = []
    outputs = []
    for well, image, mask in zip(wells, raw, masks):
        out = (equalize_adapthist(image) * 255).astype(np.uint8)
        outputs.append(out)
        row = {"well": str(well)}
        row.update({f"mask_{k}": v for k, v in summarize(out[mask]).items()})
        row.update(
            {
                f"threshold_{k}": v
                for k, v in summarize(out[out > 30]).items()
            }
        )
        per_well.append(row)

    version = skimage.__version__.replace(".", "_")
    np.savez_compressed(
        HERE / f"clahe_outputs_skimage_{version}.npz",
        output=np.stack(outputs),
        wells=wells,
    )
    result = {
        "skimage_version": skimage.__version__,
        "n": len(per_well),
        "aggregate_mask": summarize(
            np.concatenate(
                [out[mask] for out, mask in zip(outputs, masks)]
            )
        ),
        "aggregate_threshold": summarize(
            np.concatenate([out[out > 30] for out in outputs])
        ),
        "per_well": per_well,
    }
    path = HERE / f"clahe_metrics_skimage_{version}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_well"}, indent=2))


if __name__ == "__main__":
    main()
