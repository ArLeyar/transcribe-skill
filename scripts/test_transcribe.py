"""Unit tests for transcribe.py pure functions + ffmpeg arg construction.

Run: uv run --with pytest pytest skills/transcribe/scripts/test_transcribe.py
mlx/pyannote/openai paths are integration-tested by running on real audio, not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import transcribe as t  # noqa: E402


# --- load_env -------------------------------------------------------------

def test_load_env_parses_and_strips(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('export FOO=bar\n# comment\n\nBAZ="quoted"\nQUX=\'single\'\n')
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    monkeypatch.delenv("QUX", raising=False)
    t.load_env(env)
    assert t.os.environ["FOO"] == "bar"
    assert t.os.environ["BAZ"] == "quoted"
    assert t.os.environ["QUX"] == "single"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=fromfile\n")
    monkeypatch.setenv("FOO", "fromenv")
    t.load_env(env)
    assert t.os.environ["FOO"] == "fromenv"


def test_load_env_ignores_comments_blanks_and_no_eq(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# just a comment\n\nNOEQUALS\nGOOD=1\n")
    monkeypatch.delenv("GOOD", raising=False)
    t.load_env(env)
    assert t.os.environ["GOOD"] == "1"
    assert "NOEQUALS" not in t.os.environ


def test_load_env_missing_file_is_noop(tmp_path):
    t.load_env(tmp_path / "does-not-exist.env")  # must not raise


# --- normalize_lang / local_model_for ------------------------------------

def test_normalize_lang_aliases():
    assert t.normalize_lang("русский") == "ru"
    assert t.normalize_lang("English") == "en"
    assert t.normalize_lang(None) == "ru"
    assert t.normalize_lang("fr") == "fr"  # passthrough


def test_local_model_for():
    assert t.local_model_for("ru") == t.RUSSIAN_MLX_MODEL
    assert t.local_model_for("en") == t.TURBO_MLX_MODEL


# --- clean_hallucinations -------------------------------------------------

def test_clean_collapses_runaway_phrase():
    out = t.clean_hallucinations("просила просила просила просила да")
    assert out == "просила да"


def test_clean_keeps_two_legit_repeats():
    out = t.clean_hallucinations("да да хорошо")
    assert out == "да да хорошо"


def test_clean_collapses_glued_inword():
    assert t.clean_hallucinations("енитьенитьенить") == "енить"


# --- ffmpeg arg construction ---------------------------------------------

def test_codec_args_lossless_vs_mp3():
    assert "pcm_s16le" in t._ffmpeg_codec_args(True)
    assert "-c:a" in t._ffmpeg_codec_args(True)
    assert "32k" in t._ffmpeg_codec_args(False)


def test_diarize_pipeline_kwargs():
    assert t._diarize_pipeline_kwargs(2) == {"num_speakers": 2}
    assert t._diarize_pipeline_kwargs(None) == {}


def _capture_ffmpeg(monkeypatch, calls):
    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        class R:
            stdout = "1.0"
        return R()
    monkeypatch.setattr(t.subprocess, "run", fake_run)
    monkeypatch.setattr(t, "get_duration", lambda p: 1.0)
    monkeypatch.setattr(t.tempfile, "mkdtemp", lambda: "/tmp/fake")


def test_split_audio_lossless_uses_wav(tmp_path, monkeypatch):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    calls = []
    _capture_ffmpeg(monkeypatch, calls)
    chunks, _ = t.split_audio(str(f), lossless=True)
    cmd = calls[-1]
    assert "pcm_s16le" in cmd
    # chunks are (path, start, end) tuples
    path, start, end = chunks[0]
    assert path.endswith(".wav")
    assert start == 0.0


def test_split_audio_denoise_adds_filter(tmp_path, monkeypatch):
    f = tmp_path / "a.m4a"
    f.write_bytes(b"x")
    calls = []
    _capture_ffmpeg(monkeypatch, calls)
    t.split_audio(str(f), lossless=True, denoise=True)
    cmd = calls[-1]
    assert "-af" in cmd
    assert t._DENOISE_FILTER in cmd


def test_split_audio_long_file_offsets(tmp_path, monkeypatch):
    """Each chunk carries its absolute (start, end) offset from real chunk durations."""
    src = tmp_path / "long.wav"
    src.write_bytes(b"x")
    outdir = tmp_path / "out"
    outdir.mkdir()
    for i in range(3):
        (outdir / f"chunk_{i:03d}.wav").write_bytes(b"x")

    def fake_dur(p):
        return 700.0 if str(p).endswith("long.wav") else 250.0  # source vs each chunk

    monkeypatch.setattr(t, "get_duration", fake_dur)
    monkeypatch.setattr(t.tempfile, "mkdtemp", lambda: str(outdir))
    monkeypatch.setattr(t.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "1"})())
    chunks, _ = t.split_audio(str(src), chunk_sec=300, lossless=True)
    assert [c[1] for c in chunks] == [0.0, 250.0, 500.0]   # cumulative real durations
    assert chunks[-1][2] == 750.0


# --- VAD chunk grouping ---------------------------------------------------

def test_vad_merges_close_segments():
    speech = [{"start": 0.0, "end": 2.0}, {"start": 2.3, "end": 4.0}]  # gap 0.3 < 0.5
    chunks = t.group_speech_into_chunks(speech, max_chunk=300)
    assert chunks == [(0.0, 4.0)]


def test_vad_cuts_on_silence_at_max_chunk():
    speech = [{"start": 0.0, "end": 200.0}, {"start": 260.0, "end": 400.0}]  # gap 60 > 0.5
    chunks = t.group_speech_into_chunks(speech, max_chunk=300)
    # second segment would push past 300s -> new chunk starts at the silence boundary
    assert chunks == [(0.0, 200.0), (260.0, 400.0)]


def test_chunk_audio_falls_back_when_vad_raises(tmp_path, monkeypatch):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    monkeypatch.setattr(t, "vad_split", lambda *a, **k: (_ for _ in ()).throw(ImportError("no silero")))
    monkeypatch.setattr(t, "split_audio", lambda *a, **k: ([("fixed", 0.0, 1.0)], None))
    chunks, _ = t.chunk_audio(str(f), vad=True)
    assert chunks == [("fixed", 0.0, 1.0)]


# --- word-level diarization mapping --------------------------------------

TURNS = [(0.0, 5.0, "A"), (5.0, 12.0, "B")]


def test_assign_speaker_by_max_overlap():
    assert t._assign_speaker(1.0, 4.0, TURNS) == "A"
    assert t._assign_speaker(6.0, 11.0, TURNS) == "B"
    # straddles boundary, more time in B
    assert t._assign_speaker(4.0, 9.0, TURNS) == "B"


def test_assign_speaker_nearest_when_no_overlap():
    # token entirely in a gap after all turns -> nearest is B
    assert t._assign_speaker(20.0, 21.0, TURNS) == "B"
    # no turns at all -> '?'
    assert t._assign_speaker(0.0, 1.0, []) == "?"


def test_segments_to_tokens_offsets_to_timeline():
    segs = [{"start": 1, "end": 3, "text": "seg"}]
    toks = t._segments_to_tokens(segs, offset=10.0)
    assert toks == [{"start": 11.0, "end": 13.0, "text": "seg"}]


def test_smooth_absorbs_tiny_island():
    # B island lasts 0.2s between two A runs -> absorbed into A
    toks = [
        {"start": 0.0, "end": 2.0, "text": "раз", "speaker": "A"},
        {"start": 2.0, "end": 2.2, "text": "а", "speaker": "B"},
        {"start": 2.2, "end": 5.0, "text": "два", "speaker": "A"},
    ]
    t._smooth_speakers(toks, min_run=0.8)
    assert [x["speaker"] for x in toks] == ["A", "A", "A"]


def test_smooth_keeps_real_short_turn():
    toks = [
        {"start": 0.0, "end": 2.0, "text": "раз", "speaker": "A"},
        {"start": 2.0, "end": 3.2, "text": "да точно", "speaker": "B"},  # 1.2s, real
    ]
    t._smooth_speakers(toks, min_run=0.8)
    assert [x["speaker"] for x in toks] == ["A", "B"]


def test_group_tokens_aggregates_contiguous():
    toks = [{"text": "привет", "speaker": "A"}, {"text": "как", "speaker": "A"},
            {"text": "дела", "speaker": "B"}]
    assert t._group_tokens(toks) == [("A", "привет как"), ("B", "дела")]


def test_render_diarized_raw_vs_clean():
    groups = [("A", "да да да да")]  # 4x -> clean collapses, raw keeps
    assert t._render_diarized(groups, clean=False) == "**A:** да да да да"
    assert t._render_diarized(groups, clean=True) == "**A:** да"


# --- merge by recording time ---------------------------------------------

def test_group_by_time_merges_back_to_back():
    items = [
        {"path": "a", "start": 0, "end": 100},
        {"path": "b", "start": 150, "end": 200},   # 50s gap < 300 -> same group
        {"path": "c", "start": 1000, "end": 1100},  # 800s gap -> new group
    ]
    assert t.group_by_time(items, gap=300) == [["a", "b"], ["c"]]


def test_group_by_time_sorts_unordered_input():
    items = [
        {"path": "late", "start": 1000, "end": 1100},
        {"path": "early", "start": 0, "end": 100},
    ]
    assert t.group_by_time(items, gap=300) == [["early"], ["late"]]


def test_group_by_time_all_separate():
    items = [{"path": f"f{i}", "start": i * 1000, "end": i * 1000 + 10} for i in range(3)]
    assert t.group_by_time(items, gap=300) == [["f0"], ["f1"], ["f2"]]


def test_file_start_time_falls_back_to_mtime(tmp_path, monkeypatch):
    f = tmp_path / "x.m4a"
    f.write_bytes(b"x")
    # no creation_time tag returned -> cascade to filesystem time
    monkeypatch.setattr(t.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": ""})())
    ts = t._file_start_time(str(f))
    assert isinstance(ts, float) and ts > 0
