"""FA3 compatibility and the pinned upstream DiT compile recipe; CPU-testable."""

from functools import wraps
import importlib


def shim_fa3(interface):
    for name in ("flash_attn_func", "flash_attn_varlen_func"):
        original = getattr(interface, name)
        if getattr(original, "_avatar_tuple", False):
            continue

        def wrap(function):
            @wraps(function)
            def compatible(*args, **kwargs):
                result = function(*args, **kwargs)
                return result if isinstance(result, tuple) else (result, None)
            compatible._avatar_tuple = True
            return compatible

        setattr(interface, name, wrap(original))


def select_attention(capability, *, installed=True, importer=importlib.import_module):
    if capability[0] != 9:
        print("AVATAR_FA3 off compute_capability", flush=True)
        return "fa2"
    if not installed:
        print("AVATAR_FA3 off build_unavailable", flush=True)
        return "fa2"
    try:
        shim_fa3(importer("flash_attn_interface"))
    except Exception as error:
        print("AVATAR_FA3 off " + type(error).__name__, flush=True)
        return "fa2"
    return "fa3"


def attention_overrides(attention):
    return {"enable_flashattn3": attention == "fa3", "enable_flashattn2": attention != "fa3"}


def compile_dit(torch, dit, enabled=True):
    if not enabled:
        return False
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    for index, block in enumerate(dit.blocks):
        dit.blocks[index] = torch.compile(
            block, fullgraph=False, dynamic=False,
            options={"triton.cudagraphs": False, "epilogue_fusion": False, "emulate_precision_casts": True})
    return True
