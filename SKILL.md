---
name: transcribe
description: Transcribe audio from file or microphone. Russian + IT slang optimized. Auto-chunks long files. Local MLX whisper (default, offline, free), OpenAI gpt-4o-mini-transcribe, or local speaker diarization. Use when user asks to transcribe, convert speech to text, or record and transcribe audio. ALSO auto-activate (no "transcribe" word needed) whenever the user sends, pastes, or links a path/URL to an audio file — .m4a (most common), .mp3, .wav, .caf, .ogg, .flac — or a video file (.mp4, .mov, .webm, .mkv); a bare audio path/link means "transcribe this".
allowed-tools: Read, Bash, Glob
user_invocable: true
version: 1.1.0
---

# Audio Transcription

Transcribe audio files or record from microphone. Optimized for Russian with IT anglicisms.
Default is **fully local** (offline, free) using Whisper on Apple Silicon.
Auto-splits long files into chunks so long recordings just work.

> **ALWAYS run transcription in the background / a parallel agent — never block the main thread.**
> Model load + chunking + diarization take minutes (and silero-vad/pyannote download on first
> run). Launch every `transcribe.py` invocation in the background (e.g. Bash with
> `run_in_background: true`, or a parallel sub-agent), tell the user what was launched, keep
> working, and read the output file when it completes. Do NOT wait on it inline.

**Video files work directly** (`.webm`, `.mp4`, `.mov`, `.mkv`, ...) — the audio track is
ffmpeg-extracted automatically. Feed the recording as-is, no manual audio extraction needed.

`__SKILL_DIR__` below is this skill's own directory (set by the installer).

## Quick Reference

Dependencies are declared inside the script (PEP 723), so **`uv run` installs everything
automatically on first run** — no `--with` flags needed. Always launch it in the
background / a parallel agent (see note above).

### Transcribe a file (local, default — offline, free)

```bash
uv run __SKILL_DIR__/scripts/transcribe.py <file>
```

VAD chunking (cut on silence, less hallucination) is on by default; falls back to fixed-time
5-min chunks only if VAD genuinely fails at runtime.

### Transcribe + save to markdown

```bash
uv run __SKILL_DIR__/scripts/transcribe.py <file> -o transcript.md
```

### English (or other language)

```bash
# uses general turbo model automatically for non-Russian
uv run __SKILL_DIR__/scripts/transcribe.py <file> -l en
```

### Mixed-language audio — keep only one language

```bash
# per-chunk language auto-detect; keeps ONLY chunks in the target language.
# For recordings that mix languages ("give me just the Spanish"). Forces the
# multilingual turbo model + small (~30s) chunks. Local engine only.
uv run __SKILL_DIR__/scripts/transcribe.py <file> --only es
```

Whisper detects one language per chunk, so a chunk mixing two languages is kept or
dropped whole by its dominant language (small chunks keep that rare). A chunk of the
*other* language misdetected as the target gets force-decoded to gibberish — eyeball
the output and delete those. Prints `Language filter 'es': kept N, dropped M`.

### Speaker diarization — who said what (fully local, best quality)

```bash
uv run __SKILL_DIR__/scripts/transcribe.py -e diarize -s 2 <file>
```

`HF_TOKEN` is read from `~/.env` automatically (the script loads it; a token already
exported in the shell also works and takes precedence).
Pass `-s N` to pin the speaker count (e.g. `-s 2` for a two-person talk) — this sharply
stabilizes diarization. Transcription runs on the local whisper model, diarization on
pyannote 3.1 (MPS). Nothing leaves the machine. Each whisper segment is assigned to the
speaker it overlaps most (nearest turn on gaps, so no bare `?`), then tiny speaker islands
are smoothed out.

If `HF_TOKEN` is missing, diarize falls back to a plain transcription with a `WARNING:`
banner at the top of the output (use `--no-fallback` to hard-fail instead).

### OpenAI cloud (audio leaves machine, costs money)

```bash
uv run __SKILL_DIR__/scripts/transcribe.py -e openai <file>
```

### Jargon-heavy audio — bias toward domain vocabulary

```bash
# initial prompt nudges the model toward your terms/names. Use ONLY for jargon-heavy audio.
# Do NOT use for general/personal talk — it would force wrong vocab.
uv run __SKILL_DIR__/scripts/transcribe.py <file> \
  -p "Kubernetes, gRPC, Rust, Go, Postgres"
```

### Record from mic + transcribe

```bash
uv run __SKILL_DIR__/scripts/transcribe.py -r 30   # 30 seconds
uv run __SKILL_DIR__/scripts/transcribe.py -r      # until Ctrl+C
```

### Noisy / roadside audio

```bash
# conservative high/low-pass + afftdn denoise. Only when there's real background noise.
uv run __SKILL_DIR__/scripts/transcribe.py --denoise <file>
```

### Merge fragments recorded back-to-back

```bash
# group files recorded close together (gap < --merge-gap, default 300s) into one session.
# Time source: creation_time tag -> filesystem birthtime -> mtime.
uv run __SKILL_DIR__/scripts/transcribe.py --merge -e diarize -s 2 frag1.m4a frag2.m4a frag3.m4a
```

Each session is saved as `<first-fragment>.merged.md`. Multiple files WITHOUT `--merge` are
transcribed independently (each to `<file>.md`), never concatenated implicitly.

## All Options

```
transcribe.py [file ...] [-r SEC] [-e ENGINE] [-l LANG] [--only LANG] [-m MODEL] [-p PROMPT]
              [-s N] [--denoise] [--raw] [--no-vad] [--no-fallback] [--keep-temp]
              [--merge] [--merge-gap SEC] [-o FILE] [--clipboard]

  file            Audio file(s) (.wav, .mp3, .m4a, .caf, .ogg, .flac) or video (.mp4, .mov, ...)
  -r, --record    Record from mic (SEC=seconds, omit=until Ctrl+C)
  -e, --engine    local (default) | openai | diarize | diarize-cloud
  -l, --lang      ru/русский (default) | en/английский | any ISO code
  --only          Mixed audio: auto-detect per chunk, keep ONLY this language (e.g. es). Local only.
  -m, --model     Model override
  -p, --prompt    Initial prompt — bias toward domain vocab/names. Jargon audio only.
  -s, --speakers  Pin speaker count for diarization (e.g. 2). Stabilizes 2-person talks.
  --denoise       Conservative denoise (high/low-pass + afftdn) for noisy/roadside audio
  --raw           Output raw transcript (skip hallucination cleanup)
  --no-vad        Disable VAD chunking, use fixed-time 5-min chunks
  --no-fallback   Diarize: fail instead of falling back to plain transcription on missing token
  --keep-temp     Keep temp audio chunks and print their path (debug)
  --merge         Merge multiple files recorded back-to-back into one session each
  --merge-gap     Max silence (sec) between recordings to count as one session (default 300)
  -o, --output    Save transcript to file (single input). Also writes <name>.raw.<ext> if raw differs.
  --clipboard     Copy result to clipboard (pbcopy)
```

## Engines

| Engine | Transcription | Diarization | Local? | Cost | Notes |
|--------|---------------|-------------|--------|------|-------|
| `local` (default) | mlx whisper | — | ✅ fully | Free | offline, no key needed |
| `openai` | gpt-4o-mini-transcribe | — | ❌ cloud | $0.003/min | fast, auto-chunk, parallel |
| `diarize` | mlx whisper | pyannote 3.1 (local) | ✅ fully | Free | best local who-said-what |
| `diarize-cloud` | OpenAI whisper-1 | pyannote 3.1 (local) | ❌ hybrid | $ | legacy; faster on long files |

### Language → model

- `ru` (default) → `~/.cache/whisper-models/whisper-large-v3-russian-mlx` **if that directory
  exists** (antony66 fine-tune, WER 6.39 vs ~9.8 base). It is optional and absent by default.
- Otherwise, and for any other language → `mlx-community/whisper-large-v3-turbo` (general
  multilingual, downloaded from HuggingFace on first run).
- Override either with `-m`.

The optional Russian model is built locally — see "Optional: better Russian model" in README.md.

### Model overrides

```bash
-m gpt-4o-transcribe                          # OpenAI: more accurate (2x cost)
-m mlx-community/whisper-large-v3-mlx          # local: full v3 (slower than turbo)
-m mlx-community/whisper-large-v3-turbo        # local: general, fast
```

## Output formatting (skill's job, not the script's)

The script outputs plain text, or `**SPEAKER_00:** ...` lines for diarized audio. When **saving
a transcript to a file**, the skill (you) should wrap it:

1. **Add a header** with: title (infer topic from content), date, source filename, engine used,
   and participants if known. Example:
   ```markdown
   # Transcript — <topic>

   > Date: YYYY-MM-DD | Source: <filename> | Engine: diarize (local)
   > Participants: <names>

   ...transcript...
   ```
2. **Rename speakers** from generic `SPEAKER_00/01` to real names when:
   - the user told you who's talking ("this is a call between Anna and a recruiter"), or
   - it's inferable from the conversation content.
   Keep `SPEAKER_XX` only if genuinely unknown.

Do NOT bake name-mapping into the script — names differ every time; that's a reasoning task.

## Anti-hallucination (built into local engines)

Whisper loops on silence/breathing/long pauses ("просила просила просила…",
glued "енитьенитьенить", same phrase across many segments). Mitigations baked in:

1. `condition_on_previous_text=False` — loop can't leak into the next 30s window.
2. VAD chunking trims leading/trailing silence and cuts on silence, so whisper sees
   less dead air to loop on (local engines, when `silero-vad` is available).
3. `clean_hallucinations()` — collapses in-word glued repeats and word/phrase repeated
   3+ times in a row (2 legit repeats kept). Applied to raw text; `-o` also saves the raw
   alongside the cleaned as `.raw.md` for audit.

`word_timestamps` is deliberately OFF for diarization: mlx-whisper's DTW word alignment
duplicates text on long/repetitive speech, which hurt transcripts more than word-level
speaker mapping helped. Speakers are mapped at the segment level (max overlap + nearest turn).

Residual artifacts are still possible on long silent stretches. If a transcript loops
heavily anyway, re-run with `-m mlx-community/whisper-large-v3-turbo`.

## Auto-chunking

Long files are chunked automatically:
- **local / diarize**: VAD chunks (cut on silence) via `silero-vad`, lossless 16k mono WAV.
  Falls back to fixed-time 5-min chunks only if VAD fails at runtime (deps are bundled via
  PEP 723, so a missing-module fallback shouldn't happen).
- **openai**: fixed 5-min chunks (32k mono mp3 for upload limit), transcribed in parallel
  (up to 5 concurrent), merged into one output.

Diarization chunks long files too (transcription per chunk with absolute offsets; pyannote
runs once on the full wav). Best quality local = lossless + VAD (the default); openai stays
mp3 for speed/upload size. Large uncompressed files are re-encoded before processing.

## Requirements

- **Python deps**: bundled in the script via PEP 723 — `uv run` installs them on first run
  (mlx-whisper, silero-vad, torch, numpy, soundfile, sounddevice, pyannote.audio, openai).
- **ffmpeg**: `brew install ffmpeg` (required for all engines; not a pip package)
- **Local engines**: Apple Silicon Mac (M1+), 16GB RAM is enough. Models are downloaded
  from HuggingFace on first run and cached in `~/.cache/huggingface/`.
- **OpenAI engine**: `OPENAI_API_KEY` in `~/.env` (or exported in the shell)
- **Diarize**: `HF_TOKEN` in `~/.env` (or exported in the shell), accept pyannote terms on huggingface.co
