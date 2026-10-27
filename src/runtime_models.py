"""在线服务可选模型；离线实验仍默认固定0.6B。"""
PROFILES={
    'qwen3-0.6b':dict(model_id='Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',local_path=None),
    'qwen3-1.7b':dict(model_id='Qwen/Qwen3-1.7B',revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',local_path='outputs/models/qwen3-1.7b'),
}

def get_profile(name):
    if name not in PROFILES: raise ValueError('未知模型配置')
    return dict(PROFILES[name])
