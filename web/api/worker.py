"""
web.api.worker — Background task runner for the clipping pipeline.

Wraps ``clipping.runner.run_pipeline()`` in an asyncio task with
progress reporting via the job store.
"""

from __future__ import annotations

import asyncio
import glob
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .config_adapter import build_config_from_payload
from .models import ClipDetail, JobStatus
from . import store

# Semaphore to control max concurrent jobs
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)

# Store settings overrides (API keys etc.) in memory
_settings_env: dict[str, str] = {}


def set_settings_env(env: dict[str, str]) -> None:
    """Update runtime settings environment."""
    global _settings_env
    _settings_env.update(env)


def get_settings_env() -> dict[str, str]:
    """Get current settings environment."""
    return dict(_settings_env)


def _run_pipeline_sync(job_id: str, payload: dict) -> None:
    """
    Run the clipping pipeline synchronously (called from thread pool).

    This function updates the job store at each pipeline step so the
    frontend can poll or receive SSE progress updates.
    """
    try:
        # Build config from API payload
        cfg = build_config_from_payload(
            payload, job_id, env_overrides=_settings_env
        )

        # Validate API key (Gemini provider only)
        if cfg.ai_provider == "gemini" and not cfg.api_key_gemini:
            store.set_error(
                job_id,
                "GOOGLE_API_KEY tidak ditemukan. Set via Settings atau .env file.",
            )
            return

        # --- Step 1: Download ---
        store.set_status(job_id, JobStatus.DOWNLOADING)
        store.update_progress(
            job_id,
            step="download",
            step_number=1,
            total_steps=7,
            message="Mengunduh video...",
            percent=5.0,
        )

        from clipping import engine

        source_platform = getattr(cfg, "source_platform", "youtube")

        # If upload file, skip download
        if payload.get("upload_filename"):
            upload_path = os.path.join(os.getcwd(), "uploads", payload["upload_filename"])
            if not os.path.exists(upload_path):
                store.set_error(job_id, f"File upload tidak ditemukan: {payload['upload_filename']}")
                return
            cfg.original_video_path = upload_path
            store.update_progress(
                job_id,
                step="download",
                step_number=1,
                total_steps=7,
                message="Menggunakan file upload.",
                percent=14.0,
            )
        else:
            if not cfg.video_url:
                if os.path.exists(cfg.original_video_path):
                    store.update_progress(
                        job_id,
                        step="download",
                        step_number=1,
                        total_steps=7,
                        message="Bypass download: menggunakan video lama.",
                        percent=14.0,
                    )
                else:
                    store.set_error(job_id, "Video asli tidak ditemukan di Job ID tersebut. File mungkin sudah terhapus.")
                    return
            else:
                engine.download_video(
                    cfg.video_url,
                    cfg.original_video_path,
                    getattr(cfg, "use_dlp_subs", False),
                    getattr(cfg, "download_source_height", "max"),
                    source_platform=source_platform,
                )
                store.update_progress(
                    job_id,
                    step="download",
                    step_number=1,
                    total_steps=7,
                    message="Video berhasil diunduh.",
                    percent=14.0,
                )

        # --- Step 2: Transcribe ---
        store.set_status(job_id, JobStatus.TRANSCRIBING)
        store.update_progress(
            job_id,
            step="transcribe",
            step_number=2,
            total_steps=7,
            message="Memulai transkripsi...",
            percent=15.0,
        )

        full_transcript = ""
        segment_data = []

        # Try YouTube JSON3 subs first
        json3_files = glob.glob(cfg.original_video_path.replace(".mp4", ".*.json3"))
        file_json3 = json3_files[0] if json3_files else None

        if source_platform == "youtube" and getattr(cfg, "use_dlp_subs", False) and file_json3 and os.path.exists(file_json3):
            full_transcript, segment_data = engine.parse_youtube_json3_subs(
                file_json3, max_words_per_subtitle=cfg.max_words_per_subtitle
            )

        if not full_transcript or not segment_data:
            full_transcript, segment_data = engine.transcribe_video(
                cfg.original_video_path,
                max_words_per_subtitle=cfg.max_words_per_subtitle,
                model_size=cfg.whisper_model,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
            )

        store.update_progress(
            job_id,
            step="transcribe",
            step_number=2,
            total_steps=7,
            message="Transkripsi selesai.",
            percent=35.0,
        )

        # --- Step 3: AI Analysis ---
        store.set_status(job_id, JobStatus.ANALYZING)
        store.update_progress(
            job_id,
            step="analyze",
            step_number=3,
            total_steps=7,
            message="Menganalisis dengan AI...",
            percent=36.0,
        )

        import json

        gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")

        if getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
            with open(gemini_output_path, "r", encoding="utf-8") as f:
                analysis_result = json.load(f)
        else:
            analysis_result = engine.analyze_with_ai(full_transcript, cfg)
            with open(gemini_output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, indent=4, ensure_ascii=False)

        store.update_progress(
            job_id,
            step="analyze",
            step_number=3,
            total_steps=7,
            message=f"AI menemukan {len(analysis_result)} klip viral.",
            percent=50.0,
        )

        # --- Step 4: Metadata ---
        from clipping import metadata

        analysis_result = metadata.normalize_and_validate(analysis_result)
        metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
        metadata.save_metadata_preview(analysis_result, path=metadata_path)

        store.update_progress(
            job_id,
            step="metadata",
            step_number=4,
            total_steps=7,
            message="Metadata dinormalisasi.",
            percent=55.0,
        )

        # --- Step 5: Diarization (optional) ---
        diarization_data = None
        from clipping import studio, diarization as diarization_mod

        if (
            (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
            or getattr(cfg, "use_camera_switch", False)
        ) and studio._is_vertical_ratio(cfg.selected_ratio):
            try:
                store.update_progress(
                    job_id,
                    step="diarization",
                    step_number=5,
                    total_steps=7,
                    message="Menjalankan speaker diarization...",
                    percent=56.0,
                )
                audio_path = cfg.original_video_path.replace(".mp4", "_audio.wav")
                diarization_mod.extract_audio(cfg.original_video_path, audio_path)
                num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
                min_spk = None
                max_spk = None

                if str(num_speakers_arg).lower() == "auto":
                    max_faces = studio.estimate_speaker_count_from_video(cfg.original_video_path, cfg)
                    num_speakers_arg = "auto"
                    min_spk = max(1, max_faces)
                    max_spk = min_spk + 2

                diarization_data = diarization_mod.run_diarization(
                    audio_path,
                    hf_token=cfg.hf_token,
                    num_speakers=num_speakers_arg,
                    min_speakers=min_spk,
                    max_speakers=max_spk,
                )
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                store.update_progress(
                    job_id,
                    step="diarization",
                    step_number=5,
                    total_steps=7,
                    message=f"Diarization gagal: {e}. Fallback ke mode biasa.",
                    percent=58.0,
                )
                diarization_data = None

        # --- Step 6: Render Preparation ---
        store.set_status(job_id, JobStatus.RENDERING)
        store.update_progress(
            job_id,
            step="render",
            step_number=6,
            total_steps=7,
            message="Menyiapkan rendering...",
            percent=60.0,
        )

        os.environ["OSC_VIDEO_SCALE_ALGO"] = str(getattr(cfg, "video_scale_algo", "lanczos"))

        import cv2

        cap_e = cv2.VideoCapture(cfg.original_video_path)
        src_h_e = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_e.release()

        target_w_e, target_h_e = studio._get_render_dims(cfg, cfg.selected_ratio, source_h=src_h_e)
        video_encoder = studio.detect_video_encoder(cfg, target_h=target_h_e)

        glitch_video_path = None
        if cfg.use_hook_glitch:
            cap_g = cv2.VideoCapture(cfg.original_video_path)
            source_h_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_g.release()
            glitch_video_path = studio.prepare_glitch_video(
                cfg.selected_ratio, cfg, video_encoder, source_h=source_h_g
            )

        # --- Step 7: Render Each Clip ---
        from clipping import hook_manager

        render_manifest: list[dict] = []
        total_clips = len(analysis_result)

        custom_hook_path = None
        if getattr(cfg, "hook_source", None):
            custom_hook_path = hook_manager.download_custom_hook(cfg)

        for idx, clip in enumerate(sorted(analysis_result, key=lambda x: x["rank"])):
            clip_num = idx + 1
            store.update_progress(
                job_id,
                step="render",
                step_number=6,
                total_steps=7,
                message=f"Merender klip {clip_num}/{total_clips}...",
                percent=60.0 + (35.0 * clip_num / total_clips),
            )

            if custom_hook_path:
                clip["custom_hook_info"] = {"file_path": custom_hook_path}

            render_result = studio.process_clip(
                clip["rank"],
                clip,
                cfg.selected_ratio,
                glitch_video_path,
                segment_data,
                cfg,
                video_encoder,
                diarization_data=diarization_data,
            )
            if render_result:
                render_manifest.append(render_result)

        # --- Save manifest ---
        manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(render_manifest, f, ensure_ascii=False, indent=2)

        # --- Build clip details for the job store ---
        clips: list[ClipDetail] = []
        for entry in render_manifest:
            filename = os.path.basename(entry.get("output_file") or entry.get("video_path") or "")
            clips.append(
                ClipDetail(
                    rank=entry.get("rank", 0),
                    viral_score=entry.get("viral_score"),
                    title=entry.get("title_indonesia", ""),
                    title_en=entry.get("title_inggris", ""),
                    filename=filename,
                    duration=entry.get("duration"),
                    start_time=entry.get("start_time"),
                    end_time=entry.get("end_time"),
                    download_url=f"/api/outputs/{job_id}/{filename}",
                    metadata=entry,
                )
            )

        store.set_clips(job_id, clips)
        store.update_progress(
            job_id,
            step="done",
            step_number=7,
            total_steps=7,
            message=f"Selesai! {len(clips)} klip berhasil dirender.",
            percent=100.0,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        error_msg = f"{type(exc).__name__}: {exc}"
        store.set_error(job_id, error_msg)
        store.update_progress(
            job_id,
            step="error",
            step_number=0,
            total_steps=7,
            message=f"Pipeline gagal: {error_msg}",
            percent=0.0,
        )
        print(f"[Worker] Job {job_id} failed:\n{tb}", file=sys.stderr)


async def submit_job(job_id: str, payload: dict) -> None:
    """
    Submit a job to the background worker queue.

    Uses a semaphore to limit concurrency and runs the pipeline
    in a thread pool to avoid blocking the async event loop.
    """
    async def _run():
        async with _semaphore:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _executor, _run_pipeline_sync, job_id, payload
            )

    asyncio.create_task(_run())
