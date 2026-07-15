"""Make a labeled embryo crop montage from the A02 comparison outputs."""

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("results/mcolon/20260712_legacy_focus_stack_comparison/outputs")
STEM = "20260702_hotchem_30hpf_plate01_A02_BF_t0000"
SOURCES = [
    (
        "Current pipeline",
        Path("pipeline/output/acquisition/20260702_hotchem_30hpf_plate01/materialized_images/")
        / "20260702_hotchem_30hpf_plate01_A02/BF/projection/focus_stack"
        / f"{STEM}.png",
    ),
    ("Legacy Build01", ROOT / f"{STEM}__legacy_build01.png"),
    ("Shared clean", ROOT / f"{STEM}__shared_clean.png"),
    ("Improved unclipped", ROOT / f"{STEM}__improved_unclipped.png"),
]

# Portrait-frame coordinates enclosing the complete embryo and nearby background.
CROP_BOX = (55, 900, 315, 1400)
LABEL_HEIGHT = 32


def main() -> None:
    panels = []
    for label, path in SOURCES:
        crop = Image.open(path).convert("L").crop(CROP_BOX)
        panel = Image.new("L", (crop.width, crop.height + LABEL_HEIGHT), color=255)
        panel.paste(crop, (0, LABEL_HEIGHT))
        ImageDraw.Draw(panel).text((8, 8), label, fill=0)
        panels.append(panel)

    montage = Image.new("L", (panels[0].width * 4, panels[0].height), color=255)
    for index, panel in enumerate(panels):
        montage.paste(panel, (index * panel.width, 0))
    out = ROOT / f"{STEM}__embryo_comparison_montage.png"
    montage.save(out)
    print(out)


if __name__ == "__main__":
    main()
