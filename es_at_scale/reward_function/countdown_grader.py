import re
from typing import Any, Dict, List, Optional, Tuple, Union


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """Score for the <think>...</think><answer>...</answer> format.

    1.0 if the full format matches exactly, otherwise a partial credit
    of 0.1 (think present) + 0.5 (answer present).
    """
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    if re.match(full_format_regex, response, re.DOTALL):
        return 1.0

    reward = 0.0
    if re.search(think_regex, response, re.DOTALL):
        reward += 0.1
    if re.search(answer_regex, response, re.DOTALL):
        reward += 0.5
    return reward


def answer_reward_function(response: str, numbers: List[int], target: int) -> float:
    """1.0 iff the last <answer>...</answer> uses each input number exactly once
    and evaluates to `target`; 0.0 otherwise."""
    all_matches = re.findall(r"<answer>(.*?)<\/answer>", response, re.DOTALL)
    if not all_matches:
        return 0.0

    answer_content = all_matches[-1].strip()
    if not answer_content:
        return 0.0
    if not re.match(r"^[0-9+\-*/() ]+$", answer_content):
        return 0.0

    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def _coerce_number(x: Union[int, float, str]) -> Union[int, float]:
    """Parse a target value into a number, preserving non-integer values.

    Some countdown targets are non-integers (e.g. "27.4", "266.666..."), so we
    must not force `int()`. We keep integral values as `int` for clean logging
    and fall back to `float` otherwise; downstream scoring compares with
    `float(target)` either way.
    """
    if isinstance(x, (int, float)):
        return x
    f = float(str(x).strip())
    return int(f) if f.is_integer() else f


def _unpack_target(gt_answer: Union[Dict[str, Any], Tuple[List[int], Union[int, str]]]
                   ) -> Tuple[List[int], Union[int, float]]:
    if isinstance(gt_answer, dict):
        numbers, target_value = gt_answer["numbers"], gt_answer["target"]
    else:
        numbers, target_value = gt_answer  # (numbers, target_value)
    return list(numbers), _coerce_number(target_value)


def countdown_reward_fn(
    model_response: str,
    gt_answer: Union[Dict[str, Any], Tuple[List[int], Union[int, str]]],
) -> Tuple[Dict[str, Any], float]:
    """Reward function for the Countdown task.

    The trainer calls ``reward_fn(response_text, target)`` where ``target`` is a
    single item produced by your dataset's ``collate_fn``. Pack both the input
    numbers and the desired result value into that item, e.g.::

        def countdown_collate_fn(batch):
            prompts  = [item["context"] for item in batch]
            targets  = [
                {"numbers": item["numbers"], "target": item["target"]}
                for item in batch
            ]
            return prompts, targets

    Total reward = 0.1 * format_reward + answer_reward (matches the original
    countdown_task.py weighting).
    """
    numbers, target_value = _unpack_target(gt_answer)

    # The countdown prompt ends with "<think>", which vLLM does not echo back —
    # prepend it before format scoring so the regex sees a complete envelope.
    format_reward = format_reward_function("<think>" + model_response)
    answer_reward = answer_reward_function(model_response, numbers, target_value)

    fmt = {
        "formatted": format_reward > 0,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
    return fmt, 0.1 * format_reward + answer_reward


def countdown_answer_only_reward_fn(
    model_response: str,
    gt_answer: Union[Dict[str, Any], Tuple[List[int], Union[int, str]]],
) -> Tuple[Dict[str, Any], float]:
    """Binary variant: pure 0/1 answer correctness, no format bonus.

    Keeps the same fmt dict (so answer_acc logging still works) but the scalar
    reward is just ``answer_reward``. Module-level so it stays picklable for the
    reward-timeout multiprocessing pool.
    """
    fmt, _ = countdown_reward_fn(model_response, gt_answer)
    return fmt, float(fmt.get("answer_reward", 0.0))


def _parse_countdown_value(response: str, numbers: List[int]):
    """Extract the last <answer> expression's numeric value, or None if the
    answer is missing/ill-formed/uses the wrong multiset of numbers."""
    all_matches = re.findall(r"<answer>(.*?)<\/answer>", response, re.DOTALL)
    if not all_matches:
        return None
    answer_content = all_matches[-1].strip()
    if not answer_content or not re.match(r"^[0-9+\-*/() ]+$", answer_content):
        return None
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return None
    try:
        return float(eval(answer_content, {"__builtins__": None}, {}))
    except Exception:
        return None


def countdown_margin_reward_fn(
    model_response: str,
    gt_answer: Union[Dict[str, Any], Tuple[List[int], Union[int, str]]],
    tau: float = 0.1,
) -> Tuple[Dict[str, Any], float]:
    """Margin surrogate reward (TCAD smoothed-spec style).

    Instead of the 0/1 indicator 1{|value-target|<eps}, expose the verifier's
    internal continuous margin: relative miss delta = |value-target|/max(1,|target|),
    reward = exp(-delta/tau). Continuous through the pass boundary (exact hit
    -> exp(0)=1), decays with distance; invalid/unparseable answers get 0.
    tau is the calibration knob (TCAD's alpha_k analog): tau=0.1 means a 10%
    relative miss scores ~0.37. The fmt dict keeps the true binary
    ``answer_reward`` so answer_acc eval and the Spearman calibration gate see
    the honest 0/1 signal. Module-level: picklable for the timeout pool.
    """
    import math
    numbers, target_value = _unpack_target(gt_answer)
    val = _parse_countdown_value(model_response, numbers)
    binary = 0.0
    margin_reward = 0.0
    delta_rel = None
    if val is not None:
        delta = abs(val - float(target_value))
        binary = 1.0 if delta < 1e-5 else 0.0
        delta_rel = delta / max(1.0, abs(float(target_value)))
        margin_reward = float(math.exp(-delta_rel / tau)) if tau > 0 else binary
    fmt = {
        "formatted": val is not None,
        "answer_reward": binary,
        "margin_delta_rel": delta_rel,
    }
    return fmt, margin_reward


def countdown_margin_tiebreak_reward_fn(
    model_response: str,
    gt_answer: Union[Dict[str, Any], Tuple[List[int], Union[int, str]]],
    tau: float = 0.2,
    lam: float = 0.2,
) -> Tuple[Dict[str, Any], float]:
    """Margin as TIEBREAKER only: r = binary + lam * margin.

    Pure-margin reward got gamed at scale (B=32 run: margin rose, exactness
    fell, Spearman gate slid 0.49 -> 0.31): 'systematically closer' is an easier
    direction than 'exact'. Here correctness strictly dominates (correct >= 1.0
    beats any near-miss <= lam), and the margin only ORDERS the incorrect mass
    so within-group gradients still exist without exactness ever being
    trade-able for closeness."""
    fmt, margin = countdown_margin_reward_fn(model_response, gt_answer, tau=tau)
    return fmt, float(fmt["answer_reward"]) + lam * margin
