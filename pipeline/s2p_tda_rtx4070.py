#!/usr/bin/env python3
"""Entry-point shim — all pipeline logic lives in rf_pipeline/.

Run exactly as before:
    python s2p_tda_rtx4070.py --config path/to/config.yaml

The original ~3 500-line implementation has been refactored into the
rf_pipeline/ package (config, io, interpolation, metrics, time_domain,
topology, vector_fit, ml, bayes, plotting, runner).  This file now simply
delegates to rf_pipeline.runner.main so that all existing scripts, cron
jobs, and documentation that reference this filename continue to work.
"""
import sys
from pathlib import Path

# Allow running as a script from the pipeline/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rf_pipeline.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
