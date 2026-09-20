# -*- coding: utf-8 -*-
"""Prepare and validate the 50 shared operating conditions for the R1.1 shards."""

from Simulator.runners.main_revise1_1_learning_comparison import main


if __name__ == "__main__":
    main(stage="formal", prepare_only=True)
