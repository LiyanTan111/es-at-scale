"""Conciseness task reward (ES-at-Scale paper, Section 4.2 / archive script).

Paper protocol (archive/es_fine-tuning_conciseness.py):
    R = -|len(y) - len(s_k)|   (character lengths)
with 2 fixed training pairs and 8 held-out evaluation pairs, raw prompts
(no chat template), max_new_tokens=100.

Note: the archive script decodes the full sequence (prompt included) before
measuring length; we follow the paper's stated formula instead and measure the
generated response only — the prompt is constant per example, and
completion-only length is the interpretation under which the target lengths
are actually achievable.
"""
from typing import Any, Dict, Tuple


def conciseness_reward_fn(
    model_response: str,
    gt_answer: str,
) -> Tuple[Dict[str, Any], float]:
    """Negative absolute character-length difference vs the target solution.

    Module-level so it stays picklable for the reward-timeout mp pool.
    ``answer_reward`` mirrors the scalar so mean-reward logging works.
    """
    r = -float(abs(len(model_response) - len(str(gt_answer))))
    fmt = {
        "answer_reward": r,
        "response_len": len(model_response),
        "target_len": len(str(gt_answer)),
    }
    return fmt, r
