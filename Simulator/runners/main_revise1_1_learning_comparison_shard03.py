# -*- coding: utf-8 -*-
"""Run shard 03 of the formal R1.1 independent-condition experiment."""

from Simulator.runners.main_revise1_1_learning_comparison import main


if __name__ == "__main__":
    main(stage="formal", shard_index=2, num_shards=10, train_only=True)

