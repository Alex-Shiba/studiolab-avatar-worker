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
