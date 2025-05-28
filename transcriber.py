import os
import subprocess
import logging
import srt
from datetime import timedelta
import whisper
import torch
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio(video_path, audio_path="temporary_audio.wav", channels=1, sample_rate=16000):

    if os.path.exists(audio_path):
        os.remove(audio_path)
        logging.info("Old audio file removed.")
    start_time = time.time()
    ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
    command = f"\"{ffmpeg_path}\" -i \"{video_path}\" -ac {channels} -ar {sample_rate} -vn \"{audio_path}\""
    try:
        subprocess.run(command, shell=True, check=True)
        logging.info("✅ Audio extracted successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio: {e}")
        return None
    elapsed_time = time.time() - start_time
    logging.info(f"Audio extraction took {elapsed_time:.2f} seconds.")
    return audio_path

def transcribe_with_whisper(audio_path, model_size="base", update_progress=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size).to(device)
    logging.info(f"📦 Whisper model: {model_size} | Device: {device}")

    result = model.transcribe(audio_path, language="pl")

    if update_progress:
        update_progress(100)

    return result["segments"]


def generate_srt_from_segments(segments):
    start_time = time.time()
    subtitles = []
    for i, seg in enumerate(segments):
        start = timedelta(seconds=seg["start"])
        end = timedelta(seconds=seg["end"])
        text = seg["text"].strip()

        subtitle = srt.Subtitle(index=i + 1, start=start, end=end, content=text)
        subtitles.append(subtitle)

    srt_content = srt.compose(subtitles)
    elapsed_time = time.time() - start_time
    logging.info(f"Subtitle generation took {elapsed_time:.2f} seconds.")
    return srt_content

def transcribe_video_to_srt(video_path, output_path="output.srt", model_size="base", update_progress=None):
    start_time = time.time()
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return False

    segments = transcribe_with_whisper(audio_path, model_size=model_size, update_progress=update_progress)

    srt_content = generate_srt_from_segments(segments)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    logging.info(f"✅ Subtitle file saved as: {output_path}")
    total_elapsed_time = time.time() - start_time
    logging.info(f"Total processing time: {total_elapsed_time:.2f} seconds.")
    return True
