"""Command-line interface entry point for the autex pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autex.config import Config
from autex.logging import setup_logger
from autex.pipeline import ReductionPipeline


def main() -> int:
    """CLI execution entry point."""
    parser = argparse.ArgumentParser(
        prog="autex",
        description="Automated Near-Infrared Reduction and Extraction Pipeline",
    )
    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to folder containing raw observation FITS frames or calibrated slopes",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["slope", "photometry", "spectroscopy", "all"],
        default="all",
        help="Pipeline reduction mode: 'slope', 'photometry', 'spectroscopy', or 'all'",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default_config.yaml",
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Override output directory for reduced data",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        default="autex_run.log",
        help="Path to runtime log file",
    )

    args = parser.parse_args()

    setup_logger(name="autex", log_file=args.log_file)

    if Path(args.config).is_file():
        config = Config.from_yaml(args.config)
    else:
        config = Config({
            "paths": {"output_dir": args.output or "data/reduced"},
            "sutr": {"total_reads": 112},
        })

    if args.output:
        config._config["paths"]["output_dir"] = args.output

    pipeline = ReductionPipeline(config)
    pipeline.run(args.raw_dir, mode=args.mode)

    return 0


if __name__ == "__main__":
    sys.exit(main())
