"""
clipping.runner — Pipeline Orchestrator

Maps to Cell 4 (Execute) of the notebook.
Orchestrates the full clip generation pipeline.
"""

import json
import os
import time

from . import diarization as diarization_mod
from . import engine, metadata, studio, hook_manager
from .clip_config import load_external_clip_config, resolve_clip_cfg


def _vprint(cfg, *args, **kwargs):
    """Print only when verbose mode is enabled."""
    if getattr(cfg, "verbose", False):
        print(*args, **kwargs)


def run_pipeline(cfg) -> list[dict]:
    """
    Run the full clipping pipeline:
      1. Download YouTube video
      2. Transcribe with Whisper
      3. Analyse with Gemini AI
      4. Normalize metadata
      5. Prepare glitch transition
      6. Render each clip
      7. Save render_manifest.json

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object from ``config.build_config()``.

    Returns
    -------
    list[dict]
        Render manifest (one dict per clip).
    """

    # ── Load external per-clip overrides (if provided) ──
    external_clip_cfg = load_external_clip_config(
        getattr(cfg, "clip_config_path", None)
    )
    _vprint(
        cfg,
        f"   📋 External clip config: {cfg.clip_config_path}"
        if external_clip_cfg
        else "   📋 No external clip config provided",
    )

    # Step 1 — Download
    source_platform = getattr(cfg, "source_platform", "youtube")
    _vprint(
        cfg,
        f"\n[1/7] 📥 Downloading video from {source_platform}...",
        f"   Source URL : {cfg.url_youtube}",
        f"   Output     : {cfg.file_video_asli}",
        f"   Max height : {getattr(cfg, 'download_source_height', 'max')}",
    )
    t0 = time.time()
    engine.download_video(
        cfg.url_youtube,
        cfg.file_video_asli,
        getattr(cfg, "use_dlp_subs", False),
        getattr(cfg, "download_source_height", "max"),
        source_platform=source_platform,
        cookies_path=getattr(cfg, "cookies_path", None),
        cookies_from_browser=getattr(cfg, "cookies_from_browser", None),
    )
    _vprint(cfg, f"   ⏱ Download took {time.time() - t0:.1f}s")

    # Optional Colab cleanup between stages
    if getattr(cfg, "colab_cleanup", False):
        from .colab_utils import free_gpu
        free_gpu()

    # Step 2 — Transcribe
    transkrip_lengkap = ""
    data_segmen = []
    _vprint(
        cfg,
        "\n[2/7] 🎙️ Transcribing audio...",
        f"   Preferred sources: yt-transcript-api → yt-dlp json3 → Whisper ({cfg.whisper_model})",
    )
    t1 = time.time()

    # Priority 1: YouTube Transcript API (fastest, no local model)
    if (
        getattr(cfg, "use_yt_transcript_api", False)
        and source_platform == "youtube"
    ):
        transkrip_lengkap, data_segmen = engine.fetch_youtube_transcript_api(
            cfg.url_youtube,
            max_words_per_subtitle=cfg.max_kata_per_subtitle,
        )
        if transkrip_lengkap and data_segmen:
            print("✅ Berhasil mengambil subtitle via YouTube Transcript API, melewati Whisper.")

    # Priority 2: yt-dlp JSON3 subtitles (fast, no Whisper)
    if not transkrip_lengkap or not data_segmen:
        import glob

        # Mencari file json3 apapun (karena bahasanya bisa .id.json3 atau .en.json3)
        json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
        file_json3 = json3_files[0] if json3_files else None
        _vprint(cfg, f"   🔍 Looking for downloaded json3 subtitles: {json3_files}")

        if source_platform == "youtube":
            if (
                getattr(cfg, "use_dlp_subs", False)
                and file_json3
                and os.path.exists(file_json3)
            ):
                transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
                    file_json3, max_words_per_subtitle=cfg.max_kata_per_subtitle
                )
                if transkrip_lengkap and data_segmen:
                    print(
                        f"✅ Berhasil memparsing subtitle dari YouTube ({os.path.basename(file_json3)}), melewati proses Whisper."
                    )

    # Priority 3: Whisper (slowest but most accurate)
    if not transkrip_lengkap or not data_segmen:
        _vprint(
            cfg,
            f"   🔄 Falling back to Whisper: model={cfg.whisper_model}, device={cfg.whisper_device}, compute={cfg.whisper_compute_type}",
        )
        transkrip_lengkap, data_segmen = engine.transcribe_video(
            cfg.file_video_asli,
            max_words_per_subtitle=cfg.max_kata_per_subtitle,
            model_size=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
        )

    _vprint(
        cfg,
        f"   ⏱ Transcription took {time.time() - t1:.1f}s",
        f"   📝 Transcript length: {len(transkrip_lengkap)} chars",
        f"   🧩 Segments: {len(data_segmen)}",
    )

    # Free VRAM after Whisper (large model) before loading Gemini/Pyannote
    if getattr(cfg, "colab_cleanup", False):
        from .colab_utils import free_gpu
        free_gpu()

    # Step 3 — Gemini AI analysis
    gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")
    _vprint(
        cfg,
        f"\n[3/7] 🤖 AI analysis with {cfg.ai_provider}...",
        f"   Model: {cfg.gemini_model}" if cfg.ai_provider == "gemini" else "",
    )
    t2 = time.time()
    
    if getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
        print(f"\n🔄 [3/3] Memuat data AI ({cfg.ai_provider}) dari file lokal: {gemini_output_path}")
        with open(gemini_output_path, "r", encoding="utf-8") as f:
            hasil_json = json.load(f)
        _vprint(cfg, f"   ⏱ Loaded saved AI response in {time.time() - t2:.1f}s")
    else:
        hasil_json = engine.analyze_with_ai(transkrip_lengkap, cfg)
        _vprint(cfg, f"   ⏱ AI analysis took {time.time() - t2:.1f}s")
        
        # Save raw gemini json for future loading/reproduction
        with open(gemini_output_path, "w", encoding="utf-8") as f:
            json.dump(hasil_json, f, indent=4, ensure_ascii=False)
        print(f"💾 Raw AI response tersimpan di: {gemini_output_path}")

    # Step 4 — Metadata normalisation
    hasil_json = metadata.normalize_and_validate(hasil_json)
    _vprint(
        cfg,
        "\n[4/7] 📊 Normalizing metadata...",
        f"   Clips returned: {len(hasil_json)}",
    )
    metadata.print_preview(hasil_json)

    metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
    metadata.save_metadata_preview(hasil_json, path=metadata_path)

    # Step 5 — Diarization (split-screen / camera-switch)
    diarization_data = None
    needs_diarization = (
        (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
        or getattr(cfg, "use_camera_switch", False)
    ) and studio._is_vertical_ratio(cfg.pilihan_rasio)
    _vprint(
        cfg,
        f"\n[5/7] 🎙️ Diarization (split-screen/camera-switch)...",
        f"   Enabled: {needs_diarization}",
        f"   Speakers: {getattr(cfg, 'diarization_num_speakers', 2)}",
    )
    if needs_diarization:
        try:
            mode_label = (
                "Split-Screen"
                if getattr(cfg, "use_split_screen", False)
                else "Camera-Switch"
            )
            print(f"\n🎙️ [{mode_label}] Menjalankan speaker diarization...")
            audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
            diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
            num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
            min_spk = None
            max_spk = None

            if str(num_speakers_arg).lower() == "auto":
                max_faces = studio.estimate_speaker_count_from_video(
                    cfg.file_video_asli, cfg
                )
                num_speakers_arg = "auto"
                min_spk = max(1, max_faces)
                max_spk = min_spk + 2
                print(f"   ℹ️ Instruksi Pyannote: {min_spk} hingga {max_spk} speaker.")

            t3 = time.time()
            diarization_data = diarization_mod.run_diarization(
                audio_path,
                hf_token=cfg.hf_token,
                num_speakers=num_speakers_arg,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )
            _vprint(cfg, f"   ⏱ Diarization took {time.time() - t3:.1f}s")
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
            # Free VRAM after Pyannote before heavy render
            if getattr(cfg, "colab_cleanup", False):
                from .colab_utils import free_gpu
                free_gpu()
        except Exception as e:
            print(f"⚠️ Diarization gagal: {e}")
            print("   Fallback ke mode render biasa (tanpa split-screen).")
            diarization_data = None

    # Step 6 — Render each clip (encoder & glitch computed per-clip)
    import cv2
    cap_src = cv2.VideoCapture(cfg.file_video_asli)
    src_h_global = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_src.release()
    _vprint(
        cfg,
        f"\n[6/7] 🎬 Rendering clips...",
        f"   Source height: {src_h_global}px",
        f"   Total clips : {len(hasil_json)}",
    )
    t4 = time.time()

    render_manifest: list[dict] = []

    custom_hook_path = None
    if getattr(cfg, "hook_source", None):
        print("\n🎣 Mengunduh sumber klip Hook kustom...")
        custom_hook_path = hook_manager.download_custom_hook(cfg)
        _vprint(cfg, f"   Custom hook path: {custom_hook_path}")

    for klip in sorted(hasil_json, key=lambda x: x["rank"]):
        rank = klip["rank"]
        title = klip.get("title", "")
        
        if custom_hook_path:
            klip["custom_hook_info"] = {"file_path": custom_hook_path}

        # Resolve per-clip configuration (global + external JSON + clip metadata)
        clip_cfg = resolve_clip_cfg(klip, cfg, external=external_clip_cfg)
        clip_ratio = getattr(clip_cfg, "pilihan_rasio", cfg.pilihan_rasio)

        # Per-clip encoder detection (handles quality overrides like video_cq, video_preset)
        os.environ["OSC_VIDEO_SCALE_ALGO"] = str(
            getattr(clip_cfg, "video_scale_algo", "lanczos")
        )
        target_w, target_h = studio._get_render_dims(clip_cfg, clip_ratio, source_h=src_h_global)
        video_encoder = studio.detect_video_encoder(clip_cfg, target_h=target_h)

        _vprint(
            cfg,
            f"\n   🎞 Clip #{rank}: {title[:60]}",
            f"      Ratio  : {clip_ratio} ({target_w}x{target_h})",
            f"      Encoder: {video_encoder}",
            f"      B-Roll : {'ON' if getattr(clip_cfg, 'use_broll', False) else 'OFF'}",
            f"      Hook   : {'ON' if getattr(clip_cfg, 'use_hook_glitch', False) else 'OFF'}",
            f"      Subs   : {'OFF' if getattr(clip_cfg, 'no_subs', False) else 'ON'}",
            f"      Split  : {'ON' if getattr(clip_cfg, 'use_split_screen', False) else 'OFF'}",
        )

        # Per-clip glitch transition (only if hook enabled for this clip)
        file_glitch_ts = None
        if getattr(clip_cfg, "use_hook_glitch", False):
            print("⚙️ Menyiapkan Video Glitch Transisi...")
            t_glitch = time.time()
            file_glitch_ts = studio.siapkan_glitch_video(
                clip_ratio, clip_cfg, video_encoder, source_h=src_h_global
            )
            _vprint(cfg, f"      ⏱ Glitch prep took {time.time() - t_glitch:.1f}s")

        t_clip = time.time()
        hasil_render = studio.proses_klip(
            rank,
            klip,
            clip_ratio,
            file_glitch_ts,
            data_segmen,
            clip_cfg,
            video_encoder,
            diarization_data=diarization_data,
        )
        _vprint(
            cfg,
            f"      ⏱ Render took {time.time() - t_clip:.1f}s",
            f"      Output: {hasil_render.get('output_path') if hasil_render else 'FAILED'}",
        )
        if hasil_render:
            render_manifest.append(hasil_render)

    _vprint(cfg, f"\n   ⏱ Total render loop took {time.time() - t4:.1f}s")

    # Step 7 — Save manifest
    manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(render_manifest, f, ensure_ascii=False, indent=2)

    _vprint(
        cfg,
        "\n[7/7] 💾 Saving manifest...",
        f"   Path: {manifest_path}",
        f"   Items: {len(render_manifest)}",
    )
    print(
        f"\n💾 Render manifest disimpan ke {manifest_path} ({len(render_manifest)} item)"
    )
    return render_manifest
