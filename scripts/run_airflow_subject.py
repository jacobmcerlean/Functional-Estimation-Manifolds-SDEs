import argparse

from respiratory_pipeline.airflow_runner import (
    run_airflow_subject,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run direct-airflow Takens SDE "
            "for one subject."
        )
    )

    parser.add_argument(
        "mat_path",
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/airflow"
        ),
    )

    parser.add_argument(
        "--simulation-seconds",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--intrinsic-dim",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run_airflow_subject(
        args.mat_path,
        output_root=(
            args.output_root
        ),
        simulation_seconds=(
            args.simulation_seconds
        ),
        intrinsic_dim=(
            args.intrinsic_dim
        ),
        num_neighbors=(
            args.neighbors
        ),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
