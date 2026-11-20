"""Multi-person preparation derived from the pinned LongCat multi-audio demo."""

import math


def pixel_boxes(boxes, width, height):
    """Upstream slices are [y0, x0, y1, x1], in input-image pixels."""
    if boxes is None:
        half = width // 2
        return [[int(height * .1), int(half * .1), int(height * .9), int(half * .9)],
                [int(height * .1), int(half * .1 + half), int(height * .9), int(half * .9 + half)]]
    return [[math.floor(boxes[p][1] * height), math.floor(boxes[p][0] * width),
             math.ceil(boxes[p][3] * height), math.ceil(boxes[p][2] * width)]
            for p in ("person1", "person2")]


def target_masks(torch, boxes, width, height, device):
    masks = []
    for y0, x0, y1, x1 in pixel_boxes(boxes, width, height):
        mask = torch.zeros([height, width])
        mask[y0:y1, x0:x1] = 1
        masks.append(mask)
    # The contract has no 'others' boxes: exactly two audio embeddings and three
    # spatial masks, as upstream with use_background_silent_audio=False.
    background = (masks[0] + masks[1] == 0).to(masks[0].dtype)
    return torch.stack([*masks, background], dim=0).to(device)


SILENCE_PEAK = 1e-4  # -80 dBFS: under any recorded voice; the studio's listener track is exact zeros


def silent_speech(samples):
    """A track with no sound at all: the listener of a one-speaker dialog. The vocal separator refuses an all-zero file
    (audio-separator «is empty or not valid», live 2026-11-20), and upstream pads a shorter track with zeros anyway."""
    import numpy as np
    samples = np.asarray(samples)
    return samples.size > 0 and bool(np.all(np.abs(samples) < SILENCE_PEAK))


def pad_speeches(speeches, sample_rate, duration):
    import numpy as np
    length = max(math.ceil(duration * sample_rate), *(len(s) for s in speeches))
    return [np.pad(s, (0, length - len(s))) for s in speeches]


REF_MAP_CHUNK = 4096


def chunked_ref_attn_map(noise_q, ref_k, ref_target_masks, attn_bias=None, q_chunk=REF_MAP_CHUNK):
    """Upstream calculate_x_ref_attn_map (evidence/upstream-torch_utils.py) over query chunks.

    Upstream builds the whole query x reference-frame attention at once and copies it per mask: at 720p on 2xH100 that
    is several 5 GB tensors at a time, and the first denoising step of a dialog ran out of GPU memory (live 2026-11-20).
    Each query's softmax row is independent, so chunks give the same map; the per-mask masked sums are one product with
    the masks divided by their sizes. Inputs (B, L, heads, dim); result (masks * B, L) in the query dtype, as upstream."""
    import torch
    if attn_bias is not None:
        raise ValueError("attn_bias is not supported")  # upstream never passes one
    ref_k = ref_k.to(noise_q.dtype).to(noise_q.device)
    scale = 1.0 / noise_q.shape[-1] ** 0.5
    queries = noise_q.transpose(1, 2)
    keys = ref_k.transpose(1, 2).transpose(-2, -1)
    masks = ref_target_masks.to(device=queries.device, dtype=torch.float32)
    weights = (masks / masks.sum(-1, keepdim=True)).t()
    batch, _, length, _ = queries.shape
    maps = torch.empty(batch, length, masks.shape[0], device=queries.device, dtype=torch.float32)
    for start in range(0, length, q_chunk):
        scores = (queries[:, :, start:start + q_chunk] * scale) @ keys
        maps[:, start:start + q_chunk] = (scores.float().softmax(-1) @ weights).mean(1)
        del scores
    return maps.permute(2, 0, 1).reshape(-1, length).to(noise_q.dtype)


def install_chunked_ref_map(torch):
    """Upstream get_attn_map_with_target looks the function up in its module at call time; compile does not trace it."""
    from longcat_video.audio_process import torch_utils
    torch_utils.calculate_x_ref_attn_map = torch.compiler.disable(chunked_ref_attn_map)
    print("AVATAR_REFMAP chunked", REF_MAP_CHUNK, flush=True)
