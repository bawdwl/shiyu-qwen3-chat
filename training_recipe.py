"""Pure, testable training-recipe arithmetic; no model or dataset side effects."""
import math

LORA_MODULES = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


def lr_multiplier(step, total_steps, schedule='constant', warmup_ratio=0.0):
    """One-based update index. Constant/zero-warmup preserves legacy behavior."""
    if not 1 <= step <= total_steps or schedule not in ('constant', 'cosine'):
        raise ValueError('Invalid update index or schedule')
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError('warmup_ratio must be in [0, 1)')
    if total_steps == 1:
        return 1.0
    warmup = min(total_steps - 1, math.ceil(total_steps * warmup_ratio))
    if warmup and step <= warmup:
        return step / warmup
    if schedule == 'constant':
        return 1.0
    progress = (step - warmup - 1) / max(1, total_steps - warmup - 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def accumulation_factors(target_lengths, normalization='samples'):
    """Coefficients for per-sample *mean* token loss, including a short final group."""
    if not target_lengths or any(type(n) is not int or n <= 0 for n in target_lengths):
        raise ValueError('Every sample must have positive supervised token count')
    if normalization == 'samples':
        return [1.0 / len(target_lengths)] * len(target_lengths)
    if normalization == 'tokens':
        total = sum(target_lengths)
        return [n / total for n in target_lengths]
    raise ValueError('Unknown loss normalization')
