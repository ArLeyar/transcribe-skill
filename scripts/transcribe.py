#!/usr/bin/env python3
"""Audio transcription: OpenAI API or local MLX models. Auto-chunks long files."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Local default: antony66/whisper-large-v3-russian converted to MLX (WER 6.39 on Common Voice,
# vs ~9.8 for base large-v3). Falls back to turbo if the converted model dir is missing.
_RU_DIR = Path.home() / ".cache/whisper-models/whisper-large-v3-russian-mlx"
RUSSIAN_MLX_MODEL = str(_RU_DIR) if _RU_DIR.exists() else "mlx-community/whisper-large-v3-turbo"
# General multilingual MLX model — used for non-Russian local transcription.
TURBO_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"


def load_env(path=None):
    """Load ~/.env into os.environ so HF_TOKEN/OPENAI_API_KEY are available regardless
    of whether the calling shell sourced it.

    Parsed in Python (NOT shell-sourced) so a stray line in .env can't execute. Only the
    fixed ~/.env path is read, existing env vars are never overridden, values aren't printed.
    """
    path = path or (Path.home() / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_LANG_ALIASES = {
    "ru": "ru", "rus": "ru", "russian": "ru", "русский": "ru", "рус": "ru", "ру": "ru",
    "en": "en", "eng": "en", "english": "en", "английский": "en", "англ": "en", "ен": "en",
}


def normalize_lang(lang):
    """Map ru/russian/русский/en/english/английский... to ISO code. Pass through others."""
    if not lang:
        return "ru"
    return _LANG_ALIASES.get(lang.strip().lower(), lang.strip().lower())


def local_model_for(lang):
    """Russian fine-tune for ru, general turbo for everything else."""
    return RUSSIAN_MLX_MODEL if lang == "ru" else TURBO_MLX_MODEL


def clean_hallucinations(text):
    """Collapse whisper repetition loops: glued in-word repeats ('енитьенитьенить...')
    and a word/phrase repeated 3+ times in a row ('просила просила просила ...').
    Two legit repeats are kept; only runaway loops are collapsed."""
    # In-word glued loops: a 2-12 char unit repeated 3+ times inside one token
    text = re.sub(r"(\S{2,12}?)\1{2,}", r"\1", text)

    # Consecutive repeated phrases (1-6 words), 3+ occurrences -> keep one
    words = text.split(" ")
    out = []
    i = 0
    n = len(words)
    while i < n:
        collapsed = False
        for size in range(1, 7):
            if i + 2 * size > n:
                continue
            phrase = words[i:i + size]
            reps = 1
            while words[i + reps * size:i + (reps + 1) * size] == phrase:
                reps += 1
            if reps >= 3:
                out.extend(phrase)
                i += reps * size
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def get_duration(path):
    """Get audio duration in seconds via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


# Video containers (and uncompressed audio) get ffmpeg-extracted to 16kHz mono mp3
# before transcription — mlx_whisper chokes on raw 300MB+ webm/mp4, the audio track is tiny.
_FFMPEG_FORCE_EXTS = (".wav", ".caf", ".aiff", ".flac",
                      ".webm", ".mp4", ".mov", ".mkv", ".m4v", ".avi", ".flv", ".wmv")


# Conservative denoise: high/low-pass to cut rumble + hiss, gentle FFT denoise.
# nf=-25 is mild on purpose — aggressive settings eat quiet roadside speech.
_DENOISE_FILTER = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25"


def _ffmpeg_codec_args(lossless):
    """Audio output codec args. Lossless 16k mono WAV for local ASR (best quality),
    32k mono MP3 for OpenAI upload (size/bandwidth limit)."""
    if lossless:
        return ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
    return ["-ar", "16000", "-ac", "1", "-b:a", "32k"]


def _af(denoise):
    """ffmpeg audio-filter args for optional denoise (empty when off)."""
    return ["-af", _DENOISE_FILTER] if denoise else []


def _extract_wav(path, denoise=False):
    """Decode `path` to a temp 16k mono WAV (for pyannote). Caller owns the returned file."""
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", path, "-vn", *_af(denoise), "-ar", "16000", "-ac", "1", wav, "-y"],
        capture_output=True, check=True,
    )
    return wav


def split_audio(path, chunk_sec=300, lossless=False, denoise=False):
    """Split audio into chunks. Returns (chunks, tmpdir_or_None) where each chunk is
    a (path, start_sec, end_sec) tuple with its absolute offset in the source — callers
    add start_sec to local segment timestamps to map onto a full-file timeline (diarization).

    lossless=True -> 16k mono WAV (local engines, no MP3 quality loss).
    lossless=False -> 32k mono MP3 (OpenAI upload).
    denoise=True -> conservative high/low-pass + afftdn (roadside/noisy recordings).
    """
    ext_out = "wav" if lossless else "mp3"
    codec = _ffmpeg_codec_args(lossless)
    filt = _af(denoise)
    duration = get_duration(path)
    if duration <= chunk_sec:
        # Short file — only re-encode if format needs it (or denoise/lossless requested)
        ext = Path(path).suffix.lower()
        needs_reencode = (ext in _FFMPEG_FORCE_EXTS or os.path.getsize(path) > 20_000_000
                          or denoise or lossless)
        if needs_reencode:
            tmpdir = tempfile.mkdtemp()
            out = os.path.join(tmpdir, f"audio.{ext_out}")
            # -vn: drop video stream so ffmpeg doesn't decode a huge video track for tiny audio.
            subprocess.run(
                ["ffmpeg", "-i", path, "-vn", *filt, *codec, out, "-y"],
                capture_output=True, check=True,
            )
            return [(out, 0.0, duration)], tmpdir
        return [(path, 0.0, duration)], None

    tmpdir = tempfile.mkdtemp()
    pattern = os.path.join(tmpdir, f"chunk_%03d.{ext_out}")
    subprocess.run(
        ["ffmpeg", "-i", path, "-vn", *filt, "-f", "segment", "-segment_time", str(chunk_sec),
         *codec, pattern, "-y"],
        capture_output=True, check=True,
    )
    paths = sorted(Path(tmpdir).glob(f"chunk_*.{ext_out}"))
    print(f"Split into {len(paths)} chunks ({duration:.0f}s total)", file=sys.stderr)
    # Real offsets from actual chunk durations (segment muxer boundaries drift from the
    # nominal grid; cumulative real durations keep diarization timestamps aligned).
    chunks = []
    t0 = 0.0
    for c in paths:
        d = get_duration(str(c))
        chunks.append((str(c), t0, t0 + d))
        t0 += d
    return chunks, tmpdir


def group_speech_into_chunks(speech, max_chunk=300, merge_gap=0.5):
    """Pure: turn silero-vad speech timestamps into chunk (start, end) ranges.

    Adjacent speech segments closer than merge_gap are merged; segments are then packed
    into chunks no longer than max_chunk, cutting ONLY on silence (segment boundaries).
    A single speech run longer than max_chunk stays whole (no mid-speech cut).
    """
    merged = []
    for s in speech:
        if merged and s["start"] - merged[-1]["end"] < merge_gap:
            merged[-1]["end"] = s["end"]
        else:
            merged.append({"start": s["start"], "end": s["end"]})

    chunks = []
    cur_start = cur_end = None
    for seg in merged:
        if cur_start is None:
            cur_start, cur_end = seg["start"], seg["end"]
        elif seg["end"] - cur_start <= max_chunk:
            cur_end = seg["end"]
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = seg["start"], seg["end"]
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def _vad_speech_timestamps(path):
    """silero-vad speech timestamps [{start,end}] in seconds. Raises if unavailable.

    Decodes audio with ffmpeg (16k mono f32 PCM) instead of silero's read_audio, which
    depends on a torchaudio backend (torchcodec/sox) that is often missing/incompatible.
    """
    import numpy as np
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    r = subprocess.run(
        ["ffmpeg", "-i", path, "-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    audio = np.frombuffer(r.stdout, dtype=np.float32).copy()
    wav = torch.from_numpy(audio)
    model = load_silero_vad()
    return get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)


def vad_split(path, chunk_sec=300, denoise=False):
    """VAD-based chunking. Same (chunks, tmpdir) contract as split_audio but cuts on
    silence and trims leading/trailing silence (less whisper hallucination). Lossless WAV.
    Raises on any VAD failure so the caller can fall back to fixed-time split_audio."""
    speech = _vad_speech_timestamps(path)
    total = get_duration(path)
    spoken = sum(s["end"] - s["start"] for s in speech)
    print(f"VAD: speech {spoken:.0f}s / total {total:.0f}s", file=sys.stderr)
    if not speech:
        raise RuntimeError("VAD found no speech")
    ranges = group_speech_into_chunks(speech, chunk_sec)
    tmpdir = tempfile.mkdtemp()
    codec = _ffmpeg_codec_args(True)
    filt = _af(denoise)
    chunks = []
    try:
        for i, (s, e) in enumerate(ranges):
            out = os.path.join(tmpdir, f"chunk_{i:03d}.wav")
            subprocess.run(
                ["ffmpeg", "-i", path, "-vn", "-ss", str(s), "-to", str(e), *filt, *codec, out, "-y"],
                capture_output=True, check=True,
            )
            chunks.append((out, s, e))
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    print(f"VAD: {len(chunks)} speech chunks", file=sys.stderr)
    return chunks, tmpdir


def chunk_audio(path, vad=True, denoise=False, chunk_sec=300):
    """Lossless chunking for local engines: try VAD, fall back to fixed-time on any failure
    (silero-vad/torch missing, model download failed, no speech detected)."""
    if vad:
        try:
            return vad_split(path, chunk_sec=chunk_sec, denoise=denoise)
        except Exception as ex:  # noqa: BLE001 — any VAD failure must not break transcription
            print(f"VAD unavailable ({type(ex).__name__}: {ex}); using fixed-time chunks",
                  file=sys.stderr)
    return split_audio(path, chunk_sec=chunk_sec, lossless=True, denoise=denoise)


def _file_start_time(path):
    """Recording start as epoch seconds. Cascade: creation_time tag -> filesystem
    birthtime -> mtime. Voice Memo m4a carry creation_time but not every file does."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format_tags=creation_time",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    val = r.stdout.strip()
    if val:
        from datetime import datetime
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    st = os.stat(path)
    return getattr(st, "st_birthtime", None) or st.st_mtime


def group_by_time(items, gap=300):
    """Group recordings made back-to-back. items: list of {path,start,end} in epoch
    seconds. Consecutive files separated by less than `gap` seconds of silence land in
    the same group. Returns list of [path, ...] groups, ordered by time."""
    groups = []
    cur = []
    prev_end = None
    for it in sorted(items, key=lambda x: x["start"]):
        if cur and it["start"] - prev_end < gap:
            cur.append(it["path"])
        else:
            if cur:
                groups.append(cur)
            cur = [it["path"]]
        prev_end = it["end"]
    if cur:
        groups.append(cur)
    return groups


def concat_audio(files):
    """Normalize each input to 16k mono WAV, then concat via ffmpeg concat demuxer.
    Returns (merged_wav_path, tmpdir). Normalizing first avoids codec-mismatch glitches;
    paths are single-quote-escaped for the demuxer list file."""
    tmpdir = tempfile.mkdtemp()
    try:
        for i, f in enumerate(files):
            out = os.path.join(tmpdir, f"part_{i:03d}.wav")
            subprocess.run(
                ["ffmpeg", "-i", f, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out, "-y"],
                capture_output=True, check=True,
            )
        listfile = os.path.join(tmpdir, "list.txt")
        with open(listfile, "w") as fh:
            for i in range(len(files)):
                safe = os.path.join(tmpdir, f"part_{i:03d}.wav").replace("'", "'\\''")
                fh.write(f"file '{safe}'\n")
        merged = os.path.join(tmpdir, "merged.wav")
        subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", merged, "-y"],
            capture_output=True, check=True,
        )
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    return merged, tmpdir


def record(duration, output_path, sample_rate=16000):
    """Record audio from microphone."""
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    if duration and duration > 0:
        print(f"Recording {duration}s...", file=sys.stderr)
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    else:
        print("Recording... (Ctrl+C to stop)", file=sys.stderr)
        chunks = []
        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                while True:
                    data, _ = stream.read(int(sample_rate * 0.5))
                    chunks.append(data.copy())
        except KeyboardInterrupt:
            pass
        audio = np.concatenate(chunks) if chunks else np.array([])

    sf.write(output_path, audio, sample_rate)
    print(f"Recorded {len(audio)/sample_rate:.1f}s -> {output_path}", file=sys.stderr)


def transcribe_openai(path, lang, model, prompt=None, denoise=False):
    """Transcribe via OpenAI API. Auto-splits long files, parallel upload."""
    import concurrent.futures
    from openai import OpenAI

    client = OpenAI()
    chunks, tmpdir = split_audio(path, denoise=denoise)

    def _transcribe(chunk):
        chunk_path = chunk[0]
        kwargs = {"model": model, "language": lang}
        if prompt:
            kwargs["prompt"] = prompt
        with open(chunk_path, "rb") as f:
            r = client.audio.transcriptions.create(file=f, **kwargs)
        return r.text

    try:
        if len(chunks) == 1:
            text = _transcribe(chunks[0])
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 5)) as ex:
                results = list(ex.map(_transcribe, chunks))
            text = "\n\n".join(results)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return text


def transcribe_local(path, lang, model_repo, prompt=None, denoise=False, keep_temp=False,
                     vad=True, only_lang=None, chunk_sec=300):
    """Transcribe locally via mlx-whisper. Returns RAW text (caller cleans).

    Chunks via VAD (cut on silence, less hallucination) with fixed-time fallback, so
    video containers/large files get ffmpeg-extracted to clean 16kHz mono WAV and long
    files are transcribed chunk-by-chunk (progress + bounded memory).

    only_lang set -> per-chunk language auto-detect (whisper picks the language from each
    chunk instead of being forced to `lang`), keeping ONLY chunks whose detected language
    == only_lang. For mixed-language recordings ("give me just the Spanish"). Whisper
    detects one language per chunk, so a chunk that mixes two languages is kept/dropped
    whole by its dominant language — caller uses small chunk_sec (~30s) to keep that rare.
    """
    import mlx_whisper

    detect = only_lang is not None
    # condition_on_previous_text=False: stops repetition loops from leaking into
    # the next 30s window (main whisper hallucination vector on long-form audio)
    kwargs = {"path_or_hf_repo": model_repo,
              "language": None if detect else lang,  # None -> whisper auto-detects per chunk
              "condition_on_previous_text": False}
    if prompt:
        kwargs["initial_prompt"] = prompt

    chunks, tmpdir = chunk_audio(path, vad=vad, denoise=denoise, chunk_sec=chunk_sec)
    parts = []
    kept = dropped = 0
    try:
        for i, (chunk_path, _start, _end) in enumerate(chunks):
            if len(chunks) > 1:
                print(f"Transcribing chunk {i + 1}/{len(chunks)}...", file=sys.stderr)
            r = mlx_whisper.transcribe(chunk_path, **kwargs)
            if detect and r.get("language") != only_lang:
                dropped += 1
                continue
            kept += 1
            parts.append(r["text"])
    finally:
        if tmpdir and keep_temp:
            print(f"[keep-temp] chunks in {tmpdir}", file=sys.stderr)
        elif tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    if detect:
        print(f"Language filter '{only_lang}': kept {kept} chunks, dropped {dropped}",
              file=sys.stderr)
    return "\n".join(parts)


def _diarize_pipeline_kwargs(speakers):
    """num_speakers for pyannote when the caller knows the count (e.g. --speakers 2).
    Pinning the count sharply stabilizes diarization on 2-person conversations."""
    return {"num_speakers": speakers} if speakers else {}


def transcribe_diarize(path, lang, speakers=None, denoise=False):
    """OpenAI transcription + pyannote speaker diarization (fast hybrid). HF_TOKEN from env."""
    import concurrent.futures

    import torch
    from openai import OpenAI
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN required for diarization.", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for diarization.", file=sys.stderr)
        sys.exit(1)

    wav_path = _extract_wav(path, denoise)  # for pyannote

    # Run OpenAI transcription + pyannote diarization in PARALLEL
    print("Running transcription + diarization in parallel...", file=sys.stderr)

    def _openai_transcribe():
        client = OpenAI()
        chunks, tmpdir = split_audio(path, denoise=denoise)
        def _t(chunk):
            with open(chunk[0], "rb") as f:
                r = client.audio.transcriptions.create(
                    model="whisper-1", file=f, language=lang,
                    response_format="verbose_json", timestamp_granularities=["segment"],
                )
            return chunk[1], r  # (start_offset, response)
        try:
            if len(chunks) == 1:
                results = [_t(chunks[0])]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                    results = list(ex.map(_t, chunks))
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
        # Merge segments using each chunk's absolute start offset
        all_segments = []
        for offset, r in results:
            for seg in r.segments:
                all_segments.append({
                    "start": seg.start + offset,
                    "end": seg.end + offset,
                    "text": seg.text,
                })
        return all_segments

    def _pyannote_diarize():
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token,
        )
        if torch.backends.mps.is_available():
            pipeline = pipeline.to(torch.device("mps"))
        return pipeline(wav_path, **_diarize_pipeline_kwargs(speakers))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_transcript = ex.submit(_openai_transcribe)
            f_diarize = ex.submit(_pyannote_diarize)
            segments = f_transcript.result()
            diarization = f_diarize.result()
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    # Handle DiarizeOutput (pyannote >= 3.3) or Annotation
    annotation = getattr(diarization, "speaker_diarization", diarization)
    return _diarize_assign_and_format(segments, annotation)


def transcribe_diarize_local(path, lang, model_repo, prompt=None,
                             speakers=None, denoise=False, keep_temp=False, vad=True):
    """Fully local: mlx-whisper (Russian model) transcription + pyannote diarization.

    No audio leaves the machine. Transcription runs chunk-by-chunk (bounded memory +
    progress on long files). Each whisper segment is assigned to the pyannote speaker it
    overlaps most (nearest turn if it sits in a gap, so never a bare '?'), then tiny
    speaker islands are smoothed out.

    word_timestamps stays OFF: mlx-whisper's DTW word alignment duplicates text on
    long/repetitive speech, which corrupted transcripts worse than segment-level mapping
    improved them.
    """
    import mlx_whisper
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN required for diarization (pyannote download).", file=sys.stderr)
        sys.exit(1)

    # Full wav for pyannote (it streams internally, memory is fine on the whole file)
    wav_tmp = _extract_wav(path, denoise)

    kwargs = {"path_or_hf_repo": model_repo, "language": lang, "word_timestamps": False,
              "condition_on_previous_text": False}
    if prompt:
        kwargs["initial_prompt"] = prompt
    try:
        # Chunk transcription so long files don't blow memory / stall with no progress
        chunks, tmpdir = chunk_audio(path, vad=vad, denoise=denoise)
        tokens = []
        try:
            for i, (cpath, cstart, _cend) in enumerate(chunks):
                if len(chunks) > 1:
                    print(f"Transcribing chunk {i + 1}/{len(chunks)}...", file=sys.stderr)
                else:
                    print("Transcribing locally (mlx-whisper)...", file=sys.stderr)
                result = mlx_whisper.transcribe(cpath, **kwargs)
                tokens.extend(_segments_to_tokens(result.get("segments", []), cstart))
        finally:
            if tmpdir and keep_temp:
                print(f"[keep-temp] chunks in {tmpdir}", file=sys.stderr)
            elif tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

        print("Diarizing locally (pyannote 3.1)...", file=sys.stderr)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        if torch.backends.mps.is_available():
            pipeline = pipeline.to(torch.device("mps"))
        diarization = pipeline(wav_tmp, **_diarize_pipeline_kwargs(speakers))
    finally:
        if os.path.exists(wav_tmp):
            os.unlink(wav_tmp)

    annotation = getattr(diarization, "speaker_diarization", diarization)
    return _diarize_assign_and_format(tokens, annotation)


def _segments_to_tokens(segments, offset=0.0):
    """Shift each whisper segment to the full-file timeline: {start,end,text} + offset."""
    return [{"start": s["start"] + offset, "end": s["end"] + offset, "text": s.get("text", "")}
            for s in segments]


def _assign_speaker(start, end, turns):
    """Speaker whose diarization turn overlaps [start,end] most. If nothing overlaps
    (token sits in a gap), the nearest turn by time — never a bare '?'."""
    best_spk, best_ov = None, 0.0
    for ts, te, spk in turns:
        ov = min(end, te) - max(start, ts)
        if ov > best_ov:
            best_ov, best_spk = ov, spk
    if best_spk is not None:
        return best_spk
    nearest, ndist = "?", None
    for ts, te, spk in turns:
        dist = max(ts - end, start - te, 0.0)
        if ndist is None or dist < ndist:
            ndist, nearest = dist, spk
    return nearest


def _group_tokens(tokens):
    """Join contiguous same-speaker tokens into [(speaker, text), ...] blocks."""
    groups = []
    for tok in tokens:
        text = tok.get("text", "").strip()
        if not text:
            continue
        spk = tok.get("speaker", "?")
        if not groups or groups[-1][0] != spk:
            groups.append((spk, [text]))
        else:
            groups[-1][1].append(text)
    return [(spk, " ".join(words)) for spk, words in groups]


def _render_diarized(groups, clean):
    """[(speaker, text)] -> `**SPEAKER:** text` blocks. clean=True collapses repetition loops."""
    fmt = clean_hallucinations if clean else (lambda x: x)
    return "\n\n".join(f"**{spk}:** {fmt(text)}" for spk, text in groups).strip()


def _smooth_speakers(tokens, min_run=0.8):
    """Absorb tiny speaker islands (shorter than min_run seconds) into the preceding run.

    Word-level assignment ping-pongs when pyannote turn edges don't line up with word
    edges — single fillers ('а', 'ага', '-то') land on the wrong speaker. Collapsing
    sub-second islands removes that noise while real short turns survive."""
    if not tokens:
        return tokens
    runs = []
    for tok in tokens:
        if runs and runs[-1]["spk"] == tok["speaker"]:
            runs[-1]["toks"].append(tok)
        else:
            runs.append({"spk": tok["speaker"], "toks": [tok]})
    for i, run in enumerate(runs):
        span = run["toks"][-1]["end"] - run["toks"][0]["start"]
        if span < min_run and len(runs) > 1:
            new_spk = runs[i - 1]["spk"] if i > 0 else runs[i + 1]["spk"]
            for tok in run["toks"]:
                tok["speaker"] = new_spk
    return tokens


def _diarize_assign_and_format(tokens, annotation):
    """Assign tokens to speakers (max overlap), smooth ping-pong. Returns (raw, cleaned)
    diarized transcripts so the cleaned output stays readable while raw is kept for audit."""
    turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]
    for tok in tokens:
        tok["speaker"] = _assign_speaker(tok["start"], tok["end"], turns)
    _smooth_speakers(tokens)
    groups = _group_tokens(tokens)
    return _render_diarized(groups, clean=False), _render_diarized(groups, clean=True)


_NO_TOKEN_WARNING = "WARNING: diarization unavailable (no HF_TOKEN), returning plain transcription"


def transcribe_one(audio_path, args, lang):
    """Transcribe a single resolved audio file. Returns (raw_text, cleaned_text).

    Handles the no-HF_TOKEN fallback for diarize engines: warn loudly (stderr + a banner
    line prepended to the output) and fall back to plain local transcription, unless
    --no-fallback is set (then exit)."""
    engine = args.engine
    if engine in ("diarize", "diarize-cloud") and not os.environ.get("HF_TOKEN"):
        if args.no_fallback:
            print("Error: HF_TOKEN required for diarization (--no-fallback set).", file=sys.stderr)
            sys.exit(1)
        print(_NO_TOKEN_WARNING, file=sys.stderr)
        # diarize-cloud runs without mlx-whisper installed -> fall back to its cloud
        # transcriber, not local MLX (which may not be importable in a cloud-only env).
        if engine == "diarize-cloud":
            if not os.environ.get("OPENAI_API_KEY"):
                print("Error: no HF_TOKEN and no OPENAI_API_KEY for diarize-cloud fallback.",
                      file=sys.stderr)
                sys.exit(1)
            model = args.model or "gpt-4o-mini-transcribe"
            raw = transcribe_openai(audio_path, lang, model, args.prompt, denoise=args.denoise)
        else:
            model = args.model or local_model_for(lang)
            raw = transcribe_local(audio_path, lang, model, args.prompt,
                                   denoise=args.denoise, keep_temp=args.keep_temp,
                                   vad=not args.no_vad)
        return raw, _NO_TOKEN_WARNING + "\n\n" + clean_hallucinations(raw)

    if engine == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set. Use -e local for offline.", file=sys.stderr)
            sys.exit(1)
        model = args.model or "gpt-4o-mini-transcribe"
        raw = transcribe_openai(audio_path, lang, model, args.prompt, denoise=args.denoise)
        return raw, clean_hallucinations(raw)
    if engine == "diarize":
        model = args.model or local_model_for(lang)
        return transcribe_diarize_local(audio_path, lang, model, args.prompt,
                                        speakers=args.speakers, denoise=args.denoise,
                                        keep_temp=args.keep_temp, vad=not args.no_vad)
    if engine == "diarize-cloud":
        return transcribe_diarize(audio_path, lang, speakers=args.speakers, denoise=args.denoise)
    # local (default)
    only_lang = normalize_lang(args.only) if args.only else None
    # Auto-detect needs the general multilingual model; the Russian fine-tune skews detection.
    model = args.model or (TURBO_MLX_MODEL if only_lang else local_model_for(lang))
    raw = transcribe_local(audio_path, lang, model, args.prompt,
                           denoise=args.denoise, keep_temp=args.keep_temp,
                           vad=not args.no_vad, only_lang=only_lang,
                           chunk_sec=30 if only_lang else 300)
    return raw, clean_hallucinations(raw)


def _emit(raw, cleaned, out_path, args):
    """Print to stdout, optionally save (cleaned + sibling .raw), copy to clipboard."""
    shown = raw if args.raw else cleaned
    print(shown)
    if out_path:
        Path(out_path).write_text(shown, encoding="utf-8")
        print(f"Saved to {out_path}", file=sys.stderr)
        # Save raw alongside cleaned (audit) only when they actually differ
        if not args.raw and raw != cleaned:
            raw_path = Path(out_path).with_suffix(".raw" + Path(out_path).suffix)
            raw_path.write_text(raw, encoding="utf-8")
            print(f"Saved raw to {raw_path}", file=sys.stderr)
    if args.clipboard:
        subprocess.run(["pbcopy"], input=shown.encode(), check=True)
        print("(copied to clipboard)", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Transcribe audio")
    p.add_argument("file", nargs="*", help="Audio file(s) to transcribe")
    p.add_argument("-r", "--record", type=float, nargs="?", const=0, metavar="SEC",
                   help="Record from mic (seconds, 0=until Ctrl+C)")
    p.add_argument("-e", "--engine", default="local",
                   choices=["openai", "local", "diarize", "diarize-cloud"],
                   help="Engine: local (mlx Russian, default), openai (cloud), "
                   "diarize (fully local: mlx Russian + pyannote), "
                   "diarize-cloud (OpenAI transcribe + local pyannote)")
    p.add_argument("-l", "--lang", default="ru",
                   help="Language: ru/русский (default) | en/английский | any ISO code")
    p.add_argument("--only", metavar="LANG",
                   help="Mixed-language audio: auto-detect language per chunk and keep ONLY "
                   "chunks in LANG (e.g. es). Forces the multilingual turbo model + small "
                   "chunks. Local engine only.")
    p.add_argument("-m", "--model", help="Model override")
    p.add_argument("-p", "--prompt", help="Initial prompt: bias toward domain vocab/names "
                   "(e.g. IT terms). Leave empty for general speech (psychology, casual).")
    p.add_argument("-s", "--speakers", type=int,
                   help="Pin number of speakers for diarization (e.g. 2). Stabilizes 2-person talks.")
    p.add_argument("--denoise", action="store_true",
                   help="Conservative denoise (high/low-pass + afftdn) for noisy/roadside audio")
    p.add_argument("--raw", action="store_true",
                   help="Output raw transcript (skip hallucination cleanup)")
    p.add_argument("--no-fallback", action="store_true",
                   help="For diarize engines: fail instead of falling back to plain transcription")
    p.add_argument("--keep-temp", action="store_true",
                   help="Keep temp audio chunks and print their path (debug)")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable VAD chunking, use fixed-time 5-min chunks (local engines)")
    p.add_argument("--merge", action="store_true",
                   help="Merge multiple files recorded back-to-back into one session each")
    p.add_argument("--merge-gap", type=float, default=300,
                   help="Max silence (sec) between recordings to count as one session (default 300)")
    p.add_argument("-o", "--output", help="Save transcript to file (single file input)")
    p.add_argument("--clipboard", action="store_true", help="Copy result to clipboard")
    args = p.parse_args()

    load_env()
    lang = normalize_lang(args.lang)

    if args.only and args.engine != "local":
        p.error("--only works only with -e local (auto-detect path)")

    if args.record is not None and args.file:
        p.error("Use either files or --record, not both")

    # Determine audio sources
    rec_tmp = None
    if args.record is not None:
        rec_tmp = tempfile.mktemp(suffix=".wav")
        record(args.record, rec_tmp)
        files = [rec_tmp]
    else:
        files = args.file

    if not files:
        p.error("Provide audio file(s) or use --record to record from mic")
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        p.error("File(s) not found: " + ", ".join(missing))

    if args.merge and len(files) > 1:
        items = []
        for f in files:
            start = _file_start_time(f)
            items.append({"path": f, "start": start, "end": start + get_duration(f)})
        groups = group_by_time(items, args.merge_gap)
        print(f"Merging {len(files)} files into {len(groups)} session(s)", file=sys.stderr)
        for g in groups:
            if len(g) > 1:
                print(f"\n=== session: {len(g)} fragments from {Path(g[0]).name} ===",
                      file=sys.stderr)
                merged, mtmp = concat_audio(g)
                out_path = str(Path(g[0]).with_suffix(".merged.md"))
                try:
                    raw, cleaned = transcribe_one(merged, args, lang)
                finally:
                    shutil.rmtree(mtmp, ignore_errors=True)
            else:
                out_path = str(Path(g[0]).with_suffix(".md"))
                raw, cleaned = transcribe_one(g[0], args, lang)
            _emit(raw, cleaned, out_path, args)
    else:
        # Multiple files without --merge: process each independently (no implicit concat).
        for f in files:
            if len(files) > 1:
                print(f"\n=== {f} ===", file=sys.stderr)
                out_path = str(Path(f).with_suffix(".md"))
            else:
                out_path = args.output
            raw, cleaned = transcribe_one(f, args, lang)
            _emit(raw, cleaned, out_path, args)

    if rec_tmp and os.path.exists(rec_tmp):
        os.unlink(rec_tmp)


if __name__ == "__main__":
    main()
