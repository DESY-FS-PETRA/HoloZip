"""Application CLI: retained codec commands plus benchmark and compare."""

from holozip.cli import build_parser as build_codec_parser
from holozip.evaluation.runner import benchmark_command, compare_command


def build_parser():
    parser = build_codec_parser()
    sub = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None) is not None
    )
    for name, func in [("benchmark", benchmark_command), ("compare", compare_command)]:
        p = sub.add_parser(
            name,
            help="Measure selected frames and export visual diagnostics"
            if name == "benchmark"
            else "Evaluate and plot an existing archive without re-encoding",
        )
        p.add_argument("--data-dir", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument(
            "--output-dir",
            required=True,
            help="New or empty output directory; existing files are never overwritten",
        )
        p.add_argument(
            "--cpu-entropy-threads",
            type=int,
            default=8,
            help="CPU threads; 0 keeps the current startup setting",
        )
        p.add_argument("--synthesis-batch-size", type=int, default=0)
        p.add_argument(
            "--profile-row",
            type=int,
            help="Raw intensity profile row (default: middle row)",
        )
        p.add_argument(
            "--no-plots",
            action="store_true",
            help="Skip PNG and profile-CSV diagnostics",
        )
        p.add_argument(
            "--save-decoded",
            action="store_true",
            help="Also save decompressed TIFF files",
        )
        if name == "benchmark":
            p.add_argument("--stats", required=True)
            selection = p.add_mutually_exclusive_group()
            selection.add_argument(
                "--count",
                type=int,
                default=12,
                help="Evenly spaced source frames in sorted filename order (default 12)",
            )
            selection.add_argument(
                "--indices",
                type=int,
                nargs="+",
                help="Explicit zero-based source indices",
            )
            p.add_argument("--entropy-batch-size", type=int, default=32)
        else:
            p.add_argument("--archive", required=True)
        p.set_defaults(func=func)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
