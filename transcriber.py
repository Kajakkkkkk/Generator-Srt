import os
import logging
import subprocess
import tempfile
import srt
from datetime import timedelta
import time
from typing import List, Dict, Optional, Tuple, Callable
import shutil

from faster_whisper import WhisperModel

# -------------------------------
# Logi
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------------
# Cache modeli
# -------------------------------
_MODEL_CACHE: dict[str, WhisperModel] = {}

# -------------------------------
# FFmpeg
# -------------------------------
def get_ffmpeg_path() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    local = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return "ffmpeg"

# -------------------------------
# Audio → WAV 16k/mono
# -------------------------------
def extract_audio(video_path: str, channels: int = 1, sample_rate: int = 16000) -> Optional[str]:
    t0 = time.time()
    ffmpeg = get_ffmpeg_path()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio_path = tmp.name

    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-ac", str(channels), "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        audio_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        logging.info("✅ Audio extracted: %s (%.2fs)", audio_path, time.time() - t0)
        return audio_path
    except subprocess.CalledProcessError as e:
        logging.error("FFmpeg failed: %s", e)
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
        return None

# -------------------------------
# (Opcja) pre-split WAV na kawałki
# -------------------------------
def split_audio_wav(audio_path: str, chunk_seconds: int) -> List[str]:
    """
    Dzieli WAV bez rekompresji na równe kawałki co chunk_seconds.
    Zwraca listę ścieżek do plików tymczasowych.
    """
    if chunk_seconds <= 0:
        return [audio_path]

    ffmpeg = get_ffmpeg_path()
    out_dir = tempfile.mkdtemp(prefix="chunks_")
    pattern = os.path.join(out_dir, "part_%03d.wav")

    cmd = [
        ffmpeg, "-y", "-i", audio_path,
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", "-loglevel", "error",
        pattern
    ]
    subprocess.run(cmd, check=True)
    parts = sorted(
        [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("part_") and f.endswith(".wav")]
    )
    return parts or [audio_path]

# -------------------------------
# Prefetch (opcjonalnie)
# -------------------------------
MODEL_REPOS = {
    "large-v3": "Systran/faster-whisper-large-v3",
}
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

def _local_hf_dir(repo_id: str) -> str:
    safe = repo_id.replace("/", "--")
    return os.path.join(os.path.dirname(__file__), "models", safe)

def local_model_dir(model_size: str) -> str:
    repo = MODEL_REPOS.get(model_size, model_size)
    return _local_hf_dir(repo)

def prefetch_model(model_size: str) -> str:
    repo = MODEL_REPOS.get(model_size, model_size)
    target_dir = local_model_dir(model_size)
    os.makedirs(target_dir, exist_ok=True)
    if snapshot_download and not any(os.scandir(target_dir)):
        logging.info("⬇️  Prefetch Whisper: %s → %s", repo, target_dir)
        snapshot_download(repo_id=repo, local_dir=target_dir)
        logging.info("✅ Prefetch Whisper done")
    else:
        logging.info("ℹ️ Model already present: %s", target_dir)
    return target_dir

# -------------------------------
# Ładowanie modelu
# -------------------------------
def load_model(
    model_size: str = "large-v3",
    compute_type: str = "float16",
    device: str = "cuda",
    *,
    num_workers: int = 6
) -> WhisperModel:
    key = f"{model_size}:{compute_type}:{device}:{num_workers}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_path = local_model_dir(model_size)
    src = model_path if os.path.isdir(model_path) and any(os.scandir(model_path)) else MODEL_REPOS.get(model_size, model_size)

    model = WhisperModel(
        src,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers,
    )
    _MODEL_CACHE[key] = model
    logging.info("📦 Loaded faster-whisper: %s (device=%s, type=%s, workers=%d)", src, device, compute_type, num_workers)
    return model

# -------------------------------
# SRT helpers
# -------------------------------
def segments_to_srt(segments: List[Dict]) -> str:
    subs = []
    for i, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            text = "…"

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            end = start + 0.001  # minimalny odstęp

        subs.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=text
            )
        )
    return srt.compose(subs)

def save_srt(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# -------------------------------
# Łączenie krótkich segmentów
# -------------------------------
def merge_segments(
    segments: List[Dict],
    *,
    max_chars: int = 140,
    max_gap: float = 0.8
) -> List[Dict]:
    """Łączy sąsiednie segmenty, aby były dłuższe i czytelniejsze."""
    if not segments:
        return []
    out: List[Dict] = []
    cur = {"start": segments[0]["start"], "end": segments[0]["end"], "text": segments[0]["text"]}
    for s in segments[1:]:
        gap = s["start"] - cur["end"]
        candidate = (cur["text"] + " " + s["text"]).strip()
        if gap <= max_gap and len(candidate) <= max_chars:
            cur["end"] = s["end"]
            cur["text"] = candidate
        else:
            out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
    out.append(cur)
    return out

# -------------------------------
# Miękki/twardy limit znaków 80/90
# -------------------------------
def _wrap_text_soft(text: str, soft_max: int, hard_max: int) -> list[str]:
    """
    Zawijanie <= soft_max z możliwością rozlania do hard_max.
    Jeśli słowo > hard_max – tniemy je na sztywno.
    """
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    chunks: list[str] = []
    cur = ""

    for w in words:
        if len(w) > hard_max:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(w), hard_max):
                chunks.append(w[i:i+hard_max])
            continue

        if not cur:
            cur = w
            continue

        candidate_len = len(cur) + 1 + len(w)
        if candidate_len <= soft_max:
            cur = f"{cur} {w}"
        elif candidate_len <= hard_max:
            cur = f"{cur} {w}"
        else:
            chunks.append(cur)
            cur = w

    if cur:
        chunks.append(cur)

    return chunks

def split_segment_soft(seg: dict, soft_max: int = 80, hard_max: int = 90, min_dur: float = 0.7) -> list[dict]:
    """
    Rozbija segment na wpisy SRT z limitem (soft/hard).
    Czas dzielony proporcjonalnie do znaków; każdy wpis ma >= min_dur.
    """
    text = (seg.get("text") or "").strip()
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    total_dur = max(0.001, end - start)

    if len(text) <= hard_max:
        return [seg | {"start": start, "end": end, "text": text}]

    parts_text = _wrap_text_soft(text, soft_max, hard_max)
    total_chars = sum(len(p) for p in parts_text) or 1
    per_char = total_dur / total_chars

    out: list[dict] = []
    t = start
    n = len(parts_text)

    for i, chunk in enumerate(parts_text):
        alloc = max(min_dur, len(chunk) * per_char)
        remain = (n - 1) - i
        latest_end = end - max(0, remain) * min_dur
        t_next = min(t + alloc, latest_end)
        if i == n - 1:
            t_next = end

        out.append({
            "start": t,
            "end": max(t_next, t + 0.001),
            "text": chunk
        })
        t = t_next

    return out

def enforce_soft_limits(segments: list[dict], soft_max: int = 80, hard_max: int = 90, min_dur: float = 0.7) -> list[dict]:
    """Zastosuj miękki/twardy limit do wszystkich segmentów."""
    out: list[dict] = []
    for s in segments:
        out.extend(split_segment_soft(s, soft_max=soft_max, hard_max=hard_max, min_dur=min_dur))
    return out

# -------------------------------
# Transkrypcja jednego pliku audio
# -------------------------------
ProgressCB = Optional[Callable[[int, Optional[str]], None]]  # (percent, eta_text)

def _iterate_segments(
    model: WhisperModel,
    audio_path: str,
    *,
    language: Optional[str],
    vad_filter: bool,
    min_silence_ms: int,
    beam_size: int,
    cancel_flag: List[bool],
    progress_cb: ProgressCB,
    base_progress: int,
    max_progress: int,
) -> Tuple[List[Dict], float]:
    t0 = time.time()
    segs, info = model.transcribe(
        audio_path,
        task="transcribe",
        language=language,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": int(min_silence_ms)},
        beam_size=beam_size,

        # --- ANTY-PĘTLA / stabilizacja ---
        temperature=[0.0, 0.2, 0.4, 0.6],
        patience=1,
        condition_on_previous_text=False,
        suppress_blank=True,
        suppress_tokens=None,
        no_speech_threshold=0.65,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        # ---
        word_timestamps=False,
    )

    out: List[Dict] = []
    duration = getattr(info, "duration", None) or 0.0
    if duration:
        logging.info("Processing chunk duration %s", time.strftime("%H:%M:%S", time.gmtime(duration)))
    removed = getattr(info, "vad_silence_duration", 0.0)
    if removed:
        logging.info("VAD removed %s of audio", time.strftime("%M:%S", time.gmtime(removed)))

    last_p = -1
    for s in segs:
        if cancel_flag and cancel_flag[0]:
            logging.warning("⏹️ Canceled by user during transcription.")
            break
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text})

        if progress_cb and duration > 0:
            frac = min(1.0, s.end / duration)
            p = base_progress + int((max_progress - base_progress) * frac)
            if p != last_p:
                elapsed = time.time() - t0
                eta_txt = None
                if frac > 0:
                    total_est = elapsed / max(frac, 1e-9)
                    rem = total_est - elapsed
                    rem = max(0, int(rem))
                    m, ssec = divmod(rem, 60)
                    h, m = divmod(m, 60)
                    eta_txt = f"ETA ~{h}h {m}m {ssec}s" if h else (f"ETA ~{m}m {ssec}s" if m else f"ETA ~{ssec}s")
                progress_cb(p, eta_txt)
                last_p = p

    if progress_cb:
        progress_cb(max_progress, None)

    elapsed = time.time() - t0
    if duration:
        rtf = elapsed / duration
        logging.info("Chunk done in %.1fs (RTF=%.3f)", elapsed, rtf)
    else:
        logging.info("Chunk done in %.1fs", elapsed)

    return out, float(duration)

# -------------------------------
# Główny proces
# -------------------------------
def process_video(
    video_path: str,
    output_base: str,
    model_size: str = "large-v3",
    *,
    compute_type: str = "float16",
    device: str = "cuda",
    num_workers: int = 6,
    vad_filter: bool = True,
    min_silence_ms: int = 600,
    beam_size: int = 5,
    progress_cb: ProgressCB = None,
    cancel_flag: Optional[List[bool]] = None,
    prefetch: bool = True,
    # scalanie
    merge_max_chars: int = 140,
    merge_max_gap: float = 0.8,
    # pre-split
    enable_presplit: bool = False,
    presplit_minutes: int = 0,
    # NOWE: miękki/twardy limit
    soft_max_chars: int = 80,
    hard_max_chars: int = 90,
) -> Tuple[bool, List[str]]:
    """
    Zwraca (success, [ścieżki_plików]).
    - Tylko transkrypcja PL do SRT.
    - Stabilne parametry ograniczające pętle.
    - Opcjonalny pre-split długich plików po czasie.
    - Miękki/twardy limit znaków per wpis (domyślnie 80/90).
    """
    t_all = time.time()
    if cancel_flag is None:
        cancel_flag = [False]

    created: List[str] = []

    # Prefetch
    if prefetch:
        try:
            prefetch_model(model_size)
        except Exception as e:
            logging.warning("Prefetch Whisper failed: %s", e)

    # Audio
    audio_path = extract_audio(video_path)
    if not audio_path:
        return False, created

    try:
        # Model
        model = load_model(
            model_size,
            compute_type=compute_type,
            device=device,
            num_workers=num_workers,
        )

        if progress_cb:
            progress_cb(1, "Starting transcription…")

        # split, jeśli włączony
        chunks = [audio_path]
        if enable_presplit and presplit_minutes and presplit_minutes > 0:
            try:
                chunks = split_audio_wav(audio_path, presplit_minutes * 60)
                logging.info("Pre-split created %d chunks", len(chunks))
            except Exception as e:
                logging.warning("Pre-split failed (%s). Falling back to single pass.", e)
                chunks = [audio_path]

        # przelot po chunkach
        all_segments: List[Dict] = []
        base_prog = 0
        per_chunk_span = 60  # ile % paska zajmuje cała transkrypcja (0–60)
        span_per_chunk = max(1, per_chunk_span // max(1, len(chunks)))
        time_offset = 0.0

        for idx, ch in enumerate(chunks, 1):
            logging.info("=== Chunk %d/%d ===", idx, len(chunks))
            ch_segments, ch_dur = _iterate_segments(
                model, ch,
                language="pl",
                vad_filter=vad_filter, min_silence_ms=min_silence_ms,
                beam_size=beam_size,
                cancel_flag=cancel_flag,
                progress_cb=progress_cb,
                base_progress=base_prog,
                max_progress=min(60, base_prog + span_per_chunk),
            )
            if cancel_flag[0]:
                return False, created

            # dodaj offset czasowy chunku
            for s in ch_segments:
                all_segments.append({
                    "start": s["start"] + time_offset,
                    "end": s["end"] + time_offset,
                    "text": s["text"]
                })
            time_offset += ch_dur
            base_prog += span_per_chunk

        logging.info("Collected %d segments (pre-merge)", len(all_segments))

        # 1) scal lekko
        all_segments = merge_segments(all_segments, max_chars=merge_max_chars, max_gap=merge_max_gap)
        logging.info("Segments after merge: %d", len(all_segments))

        # 2) miękki/twardy limit 80/90 (lub z GUI)
        all_segments = enforce_soft_limits(all_segments, soft_max=soft_max_chars, hard_max=hard_max_chars, min_dur=0.7)
        logging.info("Segments after soft<=%d hard<=%d: %d", soft_max_chars, hard_max_chars, len(all_segments))

        # zapis SRT (PL)
        if progress_cb:
            progress_cb(90, "Writing subtitles…")
        pl_srt = segments_to_srt(all_segments)
        pl_path = f"{output_base}.pl.srt"
        save_srt(pl_path, pl_srt)
        created.append(pl_path)
        logging.info("Saved: %s", pl_path)

        if progress_cb:
            progress_cb(100, "Done.")
        logging.info("✅ Total pipeline time: %.1fs", time.time() - t_all)
        return True, created

    finally:
        # czyścimy tylko oryginalny audio_path; chanki są w temp dirze i znikną same
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
