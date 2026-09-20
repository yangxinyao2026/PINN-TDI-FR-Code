# -*- coding: utf-8 -*-
"""Merge the completed R1.1 training shards and run the unified evaluation."""

import subprocess
import sys
from pathlib import Path

from Simulator.runners.main_revise1_1_learning_comparison import main


if __name__ == "__main__":
    main(stage="formal", finalize_only=True)
    project_root = Path(__file__).resolve().parents[2]
    print("\nStart an independent process and retest the online query time according to the original single time without preheating..")
    subprocess.run(
        [sys.executable, "-m",
         "Simulator.runners.main_revise1_1_learning_comparison",
         "--stage", "formal", "--retime-only"],
        cwd=project_root,
        check=True,
    )
