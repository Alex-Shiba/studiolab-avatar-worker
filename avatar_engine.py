"""Resident bf16 Avatar 1.6 inference, derived from Meituan's MIT demo.

Source: LongCat-Video@58eb336edc5c8c63389838727e40b8c18434e54e.
Derived from run_demo_avatar_single_audio_to_video.py and
run_demo_avatar_multi_audio_to_video.py at that commit. GPU imports are lazy.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from acceleration import attention_overrides, compile_dit, select_attention
from dialog import pad_speeches, silent_speech, target_masks
import gpu_watch
from resident_protocol import atomic_json, wait_job

from media import encode_frames, mix_audio, write_progress

MODEL_TYPE = "avatar-v1.6"
MIX_STEPS = 4
MIX_ALPHA = 1.0
INFERENCE_STEPS = 8
# Upstream negative prompt plus text words: a 25 s take whose scene said "recites a poem" grew fake captions in its last segments
# (live run 2026-11-17); the same take with a neutral scene had none.
NEGATIVE_PROMPT = 'Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, text, captions, letters, watermark, logo'


def torch_gc():
    import torch
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_runtime(directory, context_parallel_size):
    directory = Path(directory).resolve()
    sys.path.insert(0, os.environ["AVATAR_SOURCE"])
    import torch
    import torch.distributed as dist
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    control = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=7))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    status = Path(os.environ["AVATAR_MODEL_ROOT"]).parent / "fa3-status.json"
    installed = json.loads(status.read_text(encoding="utf-8")).get("available") is True if status.exists() else False
    # Patch before any LongCat attention module binds the imported functions.
    attention = select_attention(torch.cuda.get_device_capability(local_rank), installed=installed)
    choices = [None] * num_processes
    dist.all_gather_object(choices, attention, group=control)
    attention = "fa3" if all(choice == "fa3" for choice in choices) else "fa2"

    from transformers import AutoTokenizer, UMT5EncoderModel
    
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
    from longcat_video.context_parallel import context_parallel_util
    
    # -------- avatar related --------
    from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
    from audio_separator.separator import Separator
    

    checkpoint_dir = str(Path(os.environ["AVATAR_MODEL_ROOT"]) / "LongCat-Video-Avatar-1.6")
    model_type = "avatar-v1.5" if MODEL_TYPE == "avatar-v1.6" else MODEL_TYPE
    # initialize context parallel
    context_parallel_util.init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    # DeviceMesh creates subgroups with PyTorch defaults; bound their NCCL work to the same hour.
    for group in (context_parallel_util.get_cp_group(), context_parallel_util.get_dp_group()):
        dist.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=3600), group=group)
    if global_rank == 0:
        gpu_watch.topology()
        try:
            gpu_watch.peer_copy(torch)
        except Exception as error:  # a diagnostic never fails the job
            print("AVATAR_GPU_PEER failed " + type(error).__name__, flush=True)
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="base_model", cp_split_hw=cp_split_hw,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        **attention_overrides(attention)).to(local_rank)
    dit.load_lora(os.path.join(checkpoint_dir, "loraA", "lora.safetensors"), "dmd",
                  multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
    dit.load_lora(os.path.join(checkpoint_dir, "loraB", "lora.safetensors"), "opsd",
                  multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
    dit.enable_lora_mix("dmd", "opsd", MIX_ALPHA, MIX_STEPS)
    compiled = compile_dit(torch, dit, os.environ.get("AVATAR_COMPILE", "1") != "0")
    vocal_separator = None
    audio_checkpoint = os.path.join(checkpoint_dir, "whisper-large-v3")
    audio_encoder = get_audio_encoder(audio_checkpoint, model_type).to(local_rank)
    audio_feature_extractor = get_audio_feature_extractor(audio_checkpoint, model_type)
    if global_rank == 0:
        audio_output = directory / "audio_temp"
        audio_output.mkdir(exist_ok=True)
        vocal_separator = Separator(output_dir=audio_output / "vocals", output_single_stem="vocals",
                                    model_file_dir=os.path.join(checkpoint_dir, "vocal_separator"))
        # The separator is a small ONNX model: on CPU it takes seconds. onnxruntime-gpu 1.20.2's CUDA provider fails against the
        # cuDNN 9.1 that torch 2.6 brings ("Failed to initialize CUDNN Frontend", live run 2026-11-17), so it is never used here.
        vocal_separator.onnx_execution_provider = ["CPUExecutionProvider"]
        vocal_separator.load_model("Kim_Vocal_2.onnx")
    # initialize pipeline
    pipe = LongCatVideoAvatarPipeline(
        tokenizer = tokenizer,
        text_encoder = text_encoder,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
        audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor,
        model_type=model_type
    )
    pipe.to(local_rank)


    dist.barrier(group=control)
    return SimpleNamespace(pipe=pipe, separator=vocal_separator, local_rank=local_rank,
                           global_rank=global_rank, cp_rank=cp_rank, control=control,
                           state={"attention": attention, "compiled": compiled})


def generate(directory, job, runtime):
    import PIL.Image
    import numpy as np
    import torch
    import torch.distributed as dist
    from diffusers.utils import load_image
    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.audio_process.vocal_separator import separate_vocals

    directory = Path(directory).resolve()
    os.chdir(directory)
    for name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[name] = str(directory)
    tempfile.tempdir = str(directory)
    pipe, vocal_separator = runtime.pipe, runtime.separator
    local_rank, global_rank, cp_rank = runtime.local_rank, runtime.global_rank, runtime.cp_rank
    raw_speech_path = str(directory / "audio.input")
    prompt, resolution, num_segments = job["prompt"], job["size"], job["segments"]
    model_type = "avatar-v1.5" if MODEL_TYPE == "avatar-v1.6" else MODEL_TYPE
    use_loudness_norm = MODEL_TYPE != "avatar-v1.6"
    use_distill, num_inference_steps = True, INFERENCE_STEPS
    text_guidance_scale = audio_guidance_scale = 1.0
    save_fps, audio_stride, num_frames, num_cond_frames = 25, 1, 93, 13
    negative_prompt = NEGATIVE_PROMPT
    dialog = job.get("mode", "avatar") == "dialog"
    image = load_image(str(directory / "image.png"))
    masks = target_masks(torch, job.get("boxes"), *image.size, local_rank) if dialog else None

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(job["seed"] + global_rank)
    full_embeddings = []
    track_names = ("audio1.input", "audio2.input") if dialog else ("audio.input",)
    if cp_rank == 0:
        generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
        audio_process_duration = max(generate_duration, job["duration"]) + 1.0
        speeches = []
        for name in track_names:
            if dialog:
                import librosa
                raw, _ = librosa.load(str(directory / name), sr=16000, mono=True)
                if silent_speech(raw):
                    # The silent listener: zeros, exactly what the padding below adds to a shorter track.
                    speeches.append(np.zeros(len(raw), dtype=np.float32))
                    continue
            speech, sr = separate_vocals(vocal_separator, str(directory / name), duration=audio_process_duration)
            if not speech.size or sr != 16000:
                raise ValueError("invalid separated audio")
            speeches.append(speech)
        # Parallel audio: retain the longer track; silence-pad the shorter one.
        for speech in pad_speeches(speeches, 16000, audio_process_duration):
            embedding = pipe.get_audio_embedding(
                speech, fps=save_fps * audio_stride, device=local_rank, sample_rate=16000,
                model_type=model_type, use_loudness_norm=use_loudness_norm)
            if not torch.isfinite(embedding).all():
                raise ValueError("invalid audio embedding")
            full_embeddings.append(embedding)
        if dialog:
            raw_speech_path = str(mix_audio(directory / "audio1.input", directory / "audio2.input",
                                            directory, job["duration"]))

    if context_parallel_util.get_cp_size() > 1:
        shape = (torch.tensor(list(full_embeddings[0].size()), dtype=torch.int64, device=local_rank)
                 if cp_rank == 0 else torch.zeros(3, dtype=torch.int64, device=local_rank))
        context_parallel_util.cp_broadcast(shape)
        if cp_rank != 0:
            full_embeddings = [torch.zeros(*shape.tolist(), dtype=torch.float32, device=local_rank)
                               for _ in track_names]
        for embedding in full_embeddings:
            context_parallel_util.cp_broadcast(embedding)
    if any(embedding.shape != full_embeddings[0].shape for embedding in full_embeddings):
        raise ValueError("inconsistent audio embedding shape")

    # prepare audio embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_embeddings[0].shape[0]-1)
    audio_emb = torch.cat([embedding[center_indices][None, ...].to(local_rank) for embedding in full_embeddings])


    output_tuple = pipe.generate_ai2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        output_type='both',
        generator=generator,
        audio_emb=audio_emb,
        ref_target_masks=masks,
        use_distill=use_distill,
    )
    output, latent = output_tuple
    output = output[0]
    video_array = (output * 255).astype(np.uint8)
    video = [PIL.Image.fromarray(img) for img in video_array]
    del output, output_tuple
    if global_rank == 0:
        write_progress(directory, 1, num_segments, width=video[0].width, height=video[0].height)
    torch_gc()
    if context_parallel_util.get_cp_size() > 1:
        torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    # =========================================
    #         long video generation (480P)
    # =========================================
    # load parsed long video args
    ref_img_index = 10
    mask_frame_range = 0

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    total_output_frames = len(video_array) + max(0, num_segments - 1) * (num_frames - num_cond_frames)
    all_generated_frames = np.empty((total_output_frames, height, width, 3), dtype=np.uint8)
    generated_frame_count = len(video_array)
    all_generated_frames[:generated_frame_count] = video_array

    for segment_idx in range(1, num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx+1}/{num_segments}...")

        # prepare audio embedding for the next clip
        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx   = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_embeddings[0].shape[0]-1)
        audio_emb = torch.cat([embedding[center_indices][None, ...].to(local_rank) for embedding in full_embeddings])

        output_tuple = pipe.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=False,
            offload_kv_cache=False,
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_emb,
            ref_target_masks=masks,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            use_distill=use_distill,
        )
        output, latent = output_tuple

        output = output[0]
        new_video_array = (output * 255).astype(np.uint8)
        new_video = [PIL.Image.fromarray(img) for img in new_video_array]
        del output, output_tuple

        appended_frames = new_video_array[num_cond_frames:]
        next_frame_count = generated_frame_count + len(appended_frames)
        all_generated_frames[generated_frame_count:next_frame_count] = appended_frames
        generated_frame_count = next_frame_count

        current_video = new_video


        if global_rank == 0:
            write_progress(directory, segment_idx + 1, num_segments, width=width, height=height)
        torch_gc()
    if global_rank == 0:
        write_progress(directory, num_segments, num_segments, stage="muxing", width=width, height=height)
        encode_frames((frame.tobytes() for frame in all_generated_frames[:generated_frame_count]),
                      raw_speech_path, directory, job["frames"], width, height)
    # Completion is published only after every rank has finished this job.
    dist.barrier(group=runtime.control)


def run(inbox, context_parallel_size):
    inbox = inbox.resolve()
    runtime = load_runtime(inbox, context_parallel_size)
    import torch.distributed as dist
    if runtime.global_rank == 0:
        atomic_json(inbox / "ready.json", runtime.state)
    try:
        while True:
            message = [wait_job(inbox) if runtime.global_rank == 0 else None]
            # Never park an NCCL collective across the idle period.
            dist.broadcast_object_list(message, src=0, group=runtime.control)
            request = message[0]
            directory = Path(request["directory"])
            sampler = gpu_watch.Sampler().start() if runtime.global_rank == 0 else None
            try:
                generate(directory, request["job"], runtime)
            finally:
                if sampler is not None:
                    sampler.stop()
            os.chdir(inbox)
            tempfile.tempdir = str(inbox)
            if runtime.global_rank == 0:
                atomic_json(directory / "completion.json", {"token": request["token"], **runtime.state})
                (inbox / "active.json").unlink(missing_ok=True)
            del request, message
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--context_parallel_size", type=int, choices=(1, 2, 4, 8), required=True)
    args = parser.parse_args()
    try:
        run(args.inbox, args.context_parallel_size)
    except Exception as error:
        print("AVATAR_ENGINE_FAILED " + type(error).__name__, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
