import os
from typing import List
import importlib
import pkgutil

from toolkit.models.base_model import BaseModel
from toolkit.config_modules import ModelConfig
from toolkit.paths import TOOLKIT_ROOT


def get_all_models() -> List[type]:
    all_model_classes: List[type] = []

    extensions_dir = os.path.join(TOOLKIT_ROOT, "extensions")
    for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
        try:
            module = importlib.import_module(f"extensions.{name}")
            models = getattr(module, "AI_TOOLKIT_MODELS", None)
            if isinstance(models, list):
                all_model_classes.extend(models)
        except ImportError as e:
            print(f"Failed to import extensions.{name}: {e}")

    return all_model_classes


def get_model_class(config: ModelConfig):
    for ModelClass in get_all_models():
        if ModelClass.arch == config.arch:
            return ModelClass
    raise ValueError(f"Unknown model arch: {config.arch!r}. PosterGen only supports flex2.")
