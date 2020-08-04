import importlib
from copy import deepcopy


def config_to_instance(inp):
    x = deepcopy(inp)
    try:
        class_name = x.pop("class_name") if isinstance(x, dict) else x
        split = class_name.split(".")
        module = ".".join(split[:-1])
        name = split[-1]
        instance = getattr(importlib.import_module(module), name)
        return instance(**x) if isinstance(x, dict) else instance
    except KeyError:
        raise KeyError(f"class_name not in keys: {x.keys()}")
    except ValueError as e:
        if isinstance(x, str):
            return x
        raise ValueError(e)
    except AttributeError:
        raise AttributeError(f"class or method {inp} is not existing")
