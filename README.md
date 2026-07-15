# transcribe — audio to text, locally

*[Русская версия](README.ru.md)*

A skill for **Codex** and **Claude Code** that turns audio and video recordings into text
right on your Mac: offline, free, and nothing ever leaves your machine. Tuned for Russian
speech with English tech jargon mixed in, but works with any language Whisper supports.

What it does:

- transcribes any file: `.m4a`, `.mp3`, `.wav`, `.caf`, `.ogg`, `.flac`, and video too (`.mp4`, `.mov`, `.webm`, `.mkv`)
- records from the microphone and transcribes on the spot
- separates speakers in a conversation (who said what)
- handles hour-long recordings: they are split on silence automatically
- cleans up Whisper's classic hallucination loops on silent stretches

## Requirements

- Apple Silicon Mac (M1 or newer), 16GB RAM is plenty
- Codex or Claude Code
- Homebrew — if you don't have it, grab it at [brew.sh](https://brew.sh)

The installer pulls in the rest (ffmpeg, uv, the model) by itself.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ArLeyar/transcribe-skill/main/install.sh | bash
```

The installer detects Codex and/or Claude Code and drops the skill where each of them looks
for it. Restart Codex afterwards.

## Usage

Drag an audio file into the chat window (that pastes its path) and say:

```
transcribe /Users/you/Downloads/recording.m4a
```

From there just use plain language — the skill picks the right mode:

- `transcribe recording.m4a and save it to a file`
- `there are two people talking, split it by speaker` — diarization
- `this one is in English` — another language
- `record 30 seconds from the mic and transcribe it`
- `the recording is noisy, it's from the street` — turns on denoising

The first run downloads the model (~1.6GB). One time only; after that it works offline.
Transcription runs roughly 5-10x faster than the recording's length.

## Optional extras

### Better Russian accuracy

The installer offers to build this for you — say `y` when it asks. This section is only for
doing it later.

By default the skill uses the general `whisper-large-v3-turbo`. For Russian there is a much
better fine-tune, [antony66/whisper-large-v3-russian](https://huggingface.co/antony66/whisper-large-v3-russian)
(WER 6.39 vs ~9.8 for the base model). No MLX build of it is published, so it gets converted
locally — a ~3GB download, no token needed, one time:

```bash
bash install.sh --ru-model
```

Or by hand, if you don't have the repo checked out:

```bash
uv run --with mlx-whisper --with torch --with numpy --with tqdm --with huggingface_hub \
  python3 ~/.codex/skills/transcribe/scripts/convert.py \
  --torch-name-or-path antony66/whisper-large-v3-russian \
  --mlx-path ~/.cache/whisper-models/whisper-large-v3-russian-mlx
mv ~/.cache/whisper-models/whisper-large-v3-russian-mlx/model.safetensors \
   ~/.cache/whisper-models/whisper-large-v3-russian-mlx/weights.safetensors
```

The script picks that directory up automatically for Russian once it exists. Measured on real
noisy speech, the fine-tune keeps punctuation and profanity that `turbo` drops entirely.

### Speaker diarization

Needs a free HuggingFace token:

1. Sign up at [huggingface.co](https://huggingface.co)
2. Accept the terms on [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Create a token in [settings](https://huggingface.co/settings/tokens)
4. Put it in `~/.env` as: `HF_TOKEN=hf_your_token`

Without a token diarization simply stays off and plain transcription keeps working.

### OpenAI cloud engine

Faster on long files, but the audio leaves your machine and it costs money ($0.003/min).
Put `OPENAI_API_KEY=sk-...` in `~/.env` and ask for "transcribe with openai".

## Updating

Run the install command again — it overwrites the skill with the current version.

## Troubleshooting

- `command not found: uv` — close and reopen your terminal, then run the installer again
- `ffmpeg not found` — `brew install ffmpeg`
- Codex doesn't see the skill — restart the app; check that `~/.codex/skills/transcribe` exists
- Text loops on repeated words — ask to "re-run with the turbo model"

## Notes

Local engines are Apple Silicon only — they run on [MLX](https://github.com/ml-explore/mlx),
Apple's array framework. The installer refuses to pretend otherwise on other hardware.

## License

MIT.
