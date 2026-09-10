"""ES-original positive control, relay-capable.

Runs the ORIGINAL EvolutionStrategiesTrainer population update (unchanged), but
borrows GRZOTrainer's relay/QoL machinery: latest/ checkpoint + --resume across
4h interactive walls, 4-GPU sharded eval, and answer_acc logging (extracted from
the grader's fmt dict) alongside the shaped pass@1.

The only override is train_step: regenerate the per-iteration seed list exactly
as the original fit() does (GRZOTrainer.fit passes seeds=None), then delegate to
the parent ES population step.
"""

import numpy as np

from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer
from es_at_scale.trainer.grzo_trainer import GRZOTrainer


class ESRelayTrainer(GRZOTrainer):
    def train_step(self, iteration, seeds, input_text, target_text):
        # Original fit()'s deterministic per-iteration seed list.
        loop_rng = np.random.default_rng(seed=(self.global_seed or 42) + iteration)
        seeds = loop_rng.integers(
            0, 2 ** 30, size=self.population_size, dtype=np.int64
        ).tolist()
        return EvolutionStrategiesTrainer.train_step(
            self, iteration, seeds, input_text, target_text
        )
