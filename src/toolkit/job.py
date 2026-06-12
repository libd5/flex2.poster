from typing import Union, OrderedDict

from toolkit.config import get_config


def get_job(
        config_path: Union[str, dict, OrderedDict],
        name=None
):
    config = get_config(config_path, name)
    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job = config['job']
    if job == 'extension':
        from jobs import ExtensionJob
        return ExtensionJob(config)

    raise ValueError(f'Unknown job type {job!r}. PosterGen only supports job: extension')
