"""请求级生成配置：校验调用方覆盖项，保持历史默认值不变。"""

import math


def for_request(defaults, max_new_tokens=None, overrides=None):
    settings = dict(defaults)
    if max_new_tokens is not None:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
            raise ValueError('回答长度必须是1–1024之间的整数token')
        settings['max_new_tokens'] = max_new_tokens
    if overrides:
        allowed = {'temperature', 'top_p', 'top_k', 'repetition_penalty'}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError('不支持的生成参数：' + ', '.join(sorted(unknown)))
        for name, value in overrides.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} 必须是有限数值')
            value = float(value)
            if name == 'temperature' and not 0 <= value <= 2:
                raise ValueError('temperature 必须在0到2之间')
            if name == 'top_p' and not 0 < value <= 1:
                raise ValueError('top_p 必须大于0且不超过1')
            if name == 'top_k' and (not value.is_integer() or value < 1 or value > 100):
                raise ValueError('top_k 必须是1到100之间的整数')
            if name == 'repetition_penalty' and not 0.5 <= value <= 2:
                raise ValueError('repetition_penalty 必须在0.5到2之间')
            settings[name] = int(value) if name == 'top_k' else value
        if settings.get('temperature', 0) == 0:
            settings['do_sample'] = False
    return settings


def ui_timeout(max_new_tokens):
    return 240 if max_new_tokens > 512 else 120
