import copy

from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

flux_config = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.30.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


def get_sampler(sampler: str, kwargs: dict = None, arch: str = "flux"):
    if sampler != "flowmatch":
        raise ValueError(f"PosterGen only supports flowmatch sampler, got {sampler!r}")

    sched_init_args = dict(kwargs or {})
    config = copy.deepcopy(flux_config)
    config.update(sched_init_args)
    return CustomFlowMatchEulerDiscreteScheduler.from_config(config)
