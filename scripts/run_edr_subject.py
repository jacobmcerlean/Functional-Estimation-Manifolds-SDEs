import argparse
from pathlib import Path

from respiratory_pipeline.edr_runner import run_subject


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the respiratory SDE pipeline for one subject."
        )
    )

    parser.add_argument(
        "mat_path",
        type=Path,
        help="Path to the subject MAT file.",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help=(
            "Optional analysis start time in seconds."
        ),
    )

    parser.add_argument(
        "--stop",
        type=float,
        default=None,
        help=(
            "Optional analysis stop time in seconds."
        ),
    )

    parser.add_argument(
        "--simulation-seconds",
        type=float,
        default=7200.0,
        help=(
            "Duration of each simulated trajectory."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/subjects"
        ),
        help=(
            "Root directory for subject outputs."
        ),
    )

    args = parser.parse_args()

    subject_id = (
        args.mat_path.stem
    )

    output_dir = (
        args.output_root
        / subject_id
    )

    run_subject(
        args.mat_path,
        output_dir,
        start_seconds=args.start,
        stop_seconds=args.stop,
        simulation_seconds=(
            args.simulation_seconds
        ),
    )


if __name__ == "__main__":
    main()
