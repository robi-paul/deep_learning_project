# audio_pipeline.py
# Usage:
#   python audio_pipeline.py input.wav output_folder \
#         --whisper_model_dir /path/to/whisper-small-cache \
#         --nllb_dir /path/to/nllb-200-distilled-600M

import os
import sys
import argparse
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pyttsx3

# Optional Jupyter display
try:
    from IPython.display import Audio, display
except ImportError:
    Audio = None
    display = None

# Optional CLI playback
try:
    import simpleaudio as sa
except ImportError:
    sa = None

import subprocess

def play_audio_cli(wav_path: str):
    """
    Play audio in CLI: prefer simpleaudio; else open with OS default.
    """
    # Try simpleaudio first
    if sa:
        try:
            wave_obj = sa.WaveObject.from_wave_file(wav_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            return
        except Exception as e:
            print(f"simpleaudio error: {e}")
    # Fallback: OS default player
    print(f"Opening audio with default player: {wav_path}")
    if sys.platform.startswith('win'):
        os.startfile(wav_path)
    elif sys.platform.startswith('darwin'):
        subprocess.call(['open', wav_path])
    else:
        subprocess.call(['xdg-open', wav_path])


def transcribe_mandarin(wav_path: str, whisper_model_dir: str = None, model_size: str = "small") -> str:
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != whisper.audio.SAMPLE_RATE:
        audio = resample_poly(audio, whisper.audio.SAMPLE_RATE, sr)
    if whisper_model_dir and os.path.isdir(whisper_model_dir):
        model = whisper.load_model(model_size, download_root=whisper_model_dir)
    else:
        model = whisper.load_model(model_size)
    result = model.transcribe(audio, language="zh", temperature=0.0, best_of=1, beam_size=1)
    return result["text"].strip()


def translate_mandarin_to_english(text: str, model_dir: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_dir, src_lang="zho_Hans")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    eng_id = tok.convert_tokens_to_ids("eng_Latn")
    model.config.forced_bos_token_id = eng_id
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True)
    seq_len = inputs["input_ids"].size(1)
    outputs = model.generate(**inputs, max_length=seq_len+500, min_length=seq_len,
                             num_beams=6, no_repeat_ngram_size=5, early_stopping=True)
    return tok.decode(outputs[0], skip_special_tokens=True)


def tts_save(text: str, output_wav: str, voice_index: int = 0, rate: int = 150):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice_index].id)
    engine.setProperty('rate', rate)
    engine.save_to_file(text, output_wav)
    engine.runAndWait()


def process(input_wav: str, output_dir: str, whisper_model_dir: str, nllb_dir: str):
    wav = input_wav.lstrip('r').replace('\\', '/')
    base = os.path.splitext(os.path.basename(wav))[0]
    print(f"Input WAV: {wav}")
    print("Playing input audio...")
    if display and Audio:
        display(Audio(filename=wav, autoplay=True))
    else:
        play_audio_cli(wav)
    mandarin_text = transcribe_mandarin(wav, whisper_model_dir)
    print("Input text (Mandarin):", mandarin_text)
    english_text = translate_mandarin_to_english(mandarin_text, nllb_dir)
    print("Output text (English):", english_text)
    os.makedirs(output_dir, exist_ok=True)
    output_wav = os.path.join(output_dir, f"{base}_en.wav")
    tts_save(english_text, output_wav)
    print("Saved English speech to:", output_wav)
    print("Playing output audio...")
    if display and Audio:
        display(Audio(filename=output_wav, autoplay=True))
    else:
        play_audio_cli(output_wav)
    return mandarin_text, english_text, output_wav


def main():
    p = argparse.ArgumentParser(description="ASR+Translate+TTS Pipeline")
    p.add_argument('input_wav', help='Path to input WAV file')
    p.add_argument('output_dir', help='Directory to save English WAV')
    p.add_argument('--whisper_model_dir', default=None, help='Local whisper-small cache path')
    p.add_argument('--nllb_dir', required=True, help='Local nllb-200-distilled-600M folder path')
    args = p.parse_args()
    process(args.input_wav, args.output_dir, args.whisper_model_dir, args.nllb_dir)

if __name__ == '__main__':
    main()
