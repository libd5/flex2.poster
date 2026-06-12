"""Small helpers so poster LoRA code paths avoid importing removed ai-toolkit modules."""


def adapter_is(adapter, class_name: str) -> bool:
    return adapter is not None and type(adapter).__name__ == class_name
