# Avatar worker wire contract, revision 1 (the seam between `avatar-worker/` and the studio)

Owner decision 2026-11-17: LongCat-Video-Avatar 1.6 on the studio's own Vast machines, 2×H100 SXM (fallback 1×H100 SXM),
first as a model of the video request window; the voice comes only from a ready sound (a project voice), no text-to-speech.

## Transport (unchanged from comfy-worker revision 3)
Before every call the studio asks `POST https://run.vast.ai/route/` for the endpoint (cost 100 for start, 1 for state/file) and
sends `{"auth_data": <route answer>, "payload": <body>}` over HTTPS with the pinned Vast root CA, exactly as for `studiolab-comfy`.
Endpoint names: `studiolab-avatar` (2×H100 SXM) and `studiolab-avatar-b` (1×H100 SXM, the fallback), a pool like the video pool.

## POST /clip/start
Exactly these keys, nothing else (unknown or repeated keys, NaN, numeric strings, booleans for integers → 400):
- `request`: 32 lowercase hex.
- `mode`: `"avatar"` (the only mode of this worker).
- `prompt`: 1–2000 characters: the scene in the person's words (Russian or English; the text encoder is multilingual UMT5).
- `size`: `"480p"` | `"720p"`.
- `seed`: integer 0–2147483647.
- `image`: base64 (no data: prefix) single JPEG or PNG, ≤ 12 MiB decoded, ≤ 16 777 216 pixels, ≤ 8192 a side. The studio has
  already cut it to the order's ratio (16:9, 9:16 or 1:1); the worker EXIF-transposes it and passes it on (no further crop).
- `audio`: base64 of one sound file: WAV, MP3, M4A/AAC or OGG/Opus, ≤ 24 MiB decoded, one audio stream, duration 2.0–60.0 s
  (the worker measures it with ffprobe; outside → 400 `audio must last 2 to 60 seconds`).

Answers as comfy-worker: 200 `{"id","state":"queued"}`; the same request with the same fields (bytes of image and audio included in
the fingerprint) → the same id and its current state; the same request with other fields → 409 `conflict`; 400
`{"error":"invalid","detail":"<short English>"}`; 429 `busy` when one running and two queued places are taken or the backend is
unavailable or retiring.

## The take
- 25 fps. Segments: the first has 93 frames (3.72 s), each continuation adds 80 frames (3.2 s), with 13 conditioning frames.
  `segments = 1 + ceil(max(0, duration − 3.72) / 3.2)`; the video is cut to the audio's duration (rounded to whole frames).
- Size: the model's own bucket for the image's ratio at `480p` or `720p` (observed: 16:9 → 832×480 at 480p; a 4:5 portrait →
  544×704 at 480p and 864×1056 at 720p). The worker reports the real width/height in state.
- Sound: the ORIGINAL voice file, re-encoded to AAC, muxed over the whole clip (the model's vocal separator only feeds the
  audio embedding; the separated stem is never the output).
- MP4 H.264 (yuv420p, CRF 16 or better) + AAC.

## POST /clip/state and /clip/file
Same as comfy-worker revision 3: body `{"id"}` only. State: `state` queued|running|done|failed, `progress` 0..1 (segments done /
segments), `stage` (e.g. `downloading_models`, `loading`, `segment 3/7`, `muxing`), `fps` 25, `width`, `height`, `frames`,
`segments`, `gpus` (how many GPUs the engine used), after done `exec_s`, on failure a short `error`, and `machine` (exam numbers +
`boot_s`) like the video worker. File: the whole MP4, `video/mp4`, exact Content-Length; 409 `not_ready` before done; TTL 3600 s
after done; 404 `unknown` for an unknown id.

## Measured on the trial (2026-11-17, upstream demo script, bf16, 8 distilled steps, LoRA mix DMD 4 steps + OPSD)
Seconds of denoising per segment: H100 SXM ×1: 480p 54, 720p 199; ×2 (context parallel 2): 480p 27, 720p 98; ×4: 720p 50;
H100 PCIe: 480p 84, 720p 290. Plus per job: model load + text/audio encoding + VAE ≈ 60–120 s when the engine is started per job.
Peak VRAM bf16: 480p 64 GB, 720p ≈ 80 GB on one GPU, ≈ 70 GB per GPU on two.
