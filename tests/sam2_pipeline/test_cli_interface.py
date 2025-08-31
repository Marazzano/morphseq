from __future__ import annotations

from src.run_morphseq_pipeline.cli import build_parser


def test_build04_accepts_exp_flag():
    p = build_parser()
    args = p.parse_args(["build04", "--root", "/tmp/fake", "--exp", "20250612_30hpf_ctrl_atf6"])  # should not error
    assert args.cmd == "build04"
    # --exp is accepted (for interface parity) and present in args
    assert getattr(args, "exp", None) == "20250612_30hpf_ctrl_atf6"

