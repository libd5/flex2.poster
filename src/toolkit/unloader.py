import gc
import torch
from toolkit.basic import flush
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def to(self, *args, **kwargs):
        return self


def _detach_and_cpu(te: torch.nn.Module):
    torch.nn.Module.to(te, "cpu")


def unload_text_encoder(model: "BaseModel"):
    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            text_encoder_list = []
            pipe = model.pipeline

            if hasattr(pipe, "text_encoder"):
                _detach_and_cpu(pipe.text_encoder)
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                pipe.text_encoder = te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                real_te = getattr(pipe, f"text_encoder_{i}")
                _detach_and_cpu(real_te)
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                setattr(pipe, f"text_encoder_{i}", te)
                i += 1
            model.text_encoder = text_encoder_list
        else:
            _detach_and_cpu(model.text_encoder)
            model.text_encoder = FakeTextEncoder(
                device=model.device_torch,
                dtype=model.torch_dtype,
            )

    torch.cuda.empty_cache()
    gc.collect()
    flush()
