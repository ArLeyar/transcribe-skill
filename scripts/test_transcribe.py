"""Unit tests for transcribe.py pure functions + ffmpeg arg construction.

Run: uv run --with pytest --with httpx pytest scripts/test_transcribe.py
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


def test_local_model_for(monkeypatch):
    monkeypatch.setattr(t, "LOW_MEM", False)
    assert t.local_model_for("ru") == t.RUSSIAN_MLX_MODEL
    assert t.local_model_for("en") == t.TURBO_MLX_MODEL
    assert t.general_model() == t.TURBO_MLX_MODEL


def test_local_model_for_low_mem(monkeypatch):
    monkeypatch.setattr(t, "LOW_MEM", True)
    assert t.local_model_for("ru") == t.LOW_MEM_MLX_MODEL
    assert t.local_model_for("en") == t.LOW_MEM_MLX_MODEL
    assert t.general_model() == t.LOW_MEM_MLX_MODEL


def test_total_ram_gb_is_sane():
    assert 1 < t.total_ram_gb() < 4096


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


# --- resolve_engine: the cloud engine stays off unless asked for ---------

def test_resolve_engine_defaults_to_local():
    assert t.resolve_engine(None, env={}) == "local"


def test_resolve_engine_key_alone_does_not_enable_cloud():
    """Having a key for some other tool is not consent to upload recordings."""
    assert t.resolve_engine(None, env={"ELEVENLABS_API_KEY": "k"}) == "local"


def test_resolve_engine_explicit_and_pinned():
    assert t.resolve_engine("eleven", env={}) == "eleven"
    assert t.resolve_engine(None, env={"TRANSCRIBE_ENGINE": "eleven"}) == "eleven"


def test_resolve_engine_ignores_unknown_pin():
    assert t.resolve_engine(None, env={"TRANSCRIBE_ENGINE": "scribe"}) == "local"


# --- Scribe response -> transcript ---------------------------------------

def _words(*triples):
    return [{"text": txt, "type": typ, "speaker_id": sid} for txt, typ, sid in triples]


def test_eleven_two_speakers_render_as_blocks():
    payload = {"text": "ignored", "words": _words(
        ("Привет", "word", "speaker_1"), (",", "word", "speaker_1"),
        (" ", "spacing", "speaker_1"), ("как", "word", "speaker_1"),
        (" ", "spacing", "speaker_1"), ("дела", "word", "speaker_1"),
        ("Нормально", "word", "speaker_2"),
    )}
    raw, cleaned = t._eleven_words_to_transcript(payload)
    assert raw == "**SPEAKER_00:** Привет, как дела\n\n**SPEAKER_01:** Нормально"
    assert cleaned == raw


def test_eleven_single_speaker_returns_plain_text():
    payload = {"text": "  Один голос  ", "words": _words(("Один", "word", "speaker_1"))}
    assert t._eleven_words_to_transcript(payload)[0] == "Один голос"


def test_eleven_diarize_off_returns_plain_text():
    payload = {"text": "Два голоса", "words": _words(
        ("Два", "word", "speaker_1"), ("голоса", "word", "speaker_2"))}
    assert t._eleven_words_to_transcript(payload, diarize=False)[0] == "Два голоса"


def test_eleven_payload_without_words_key():
    assert t._eleven_words_to_transcript({"text": "только текст"})[0] == "только текст"


def test_eleven_drops_audio_events():
    payload = {"text": "", "words": _words(
        ("Да", "word", "speaker_1"), ("(laughter)", "audio_event", "speaker_1"),
        ("Нет", "word", "speaker_2"))}
    raw, _ = t._eleven_words_to_transcript(payload)
    assert "laughter" not in raw and raw.startswith("**SPEAKER_00:** Да")


def test_eleven_unlabelled_word_joins_the_current_speaker():
    """A gap in the API's labelling is a gap, not a third person in the room."""
    payload = {"text": "", "words": _words(
        ("Раз", "word", "speaker_1"), (" ", "spacing", "speaker_1"),
        ("два", "word", None), ("три", "word", "speaker_2"))}
    raw, _ = t._eleven_words_to_transcript(payload)
    assert raw == "**SPEAKER_00:** Раз два\n\n**SPEAKER_01:** три"


def test_eleven_leading_spacing_does_not_steal_speaker_00():
    payload = {"text": "", "words": _words(
        (" ", "spacing", "speaker_9"), ("Первый", "word", "speaker_1"),
        ("Второй", "word", "speaker_2"))}
    raw, _ = t._eleven_words_to_transcript(payload)
    assert raw.startswith("**SPEAKER_00:** Первый")


def test_eleven_unlabelled_words_before_any_label_go_to_the_first_speaker():
    payload = {"text": "", "words": _words(
        ("Начало", "word", None), (" ", "spacing", None),
        ("речи", "word", "speaker_1"), ("ответ", "word", "speaker_2"))}
    raw, _ = t._eleven_words_to_transcript(payload)
    assert raw == "**SPEAKER_00:** Начало речи\n\n**SPEAKER_01:** ответ"


# --- the key never rides out in an error message -------------------------

def test_redact_removes_the_key():
    assert t._redact("Illegal header value b'sk_secret'", "sk_secret") == \
        "Illegal header value b'***'"


def test_eleven_http_error_text_carries_no_key(monkeypatch):
    """httpx quotes a rejected header value back verbatim, and that text is prepended to
    the transcript — so it reaches stdout, the saved file and the clipboard."""
    import httpx
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_supersecret")
    monkeypatch.setattr(t, "_eleven_prepare", lambda *a, **k: ("f.m4a", None))
    monkeypatch.setattr("builtins.open", lambda *a, **k: __import__("io").BytesIO(b"x"))

    def raise_with_key(*a, **k):
        raise httpx.LocalProtocolError("Illegal header value b'sk_supersecret'")
    monkeypatch.setattr(httpx, "post", raise_with_key)
    import pytest
    with pytest.raises(t.ElevenUnavailable) as exc:
        t.transcribe_eleven("f.m4a", "ru")
    assert "sk_supersecret" not in str(exc.value)


def test_eleven_rejects_a_key_with_control_characters(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_bad\x01key")
    import pytest
    with pytest.raises(t.ElevenUnavailable) as exc:
        t.transcribe_eleven("f.m4a", "ru")
    assert "printable ASCII" in str(exc.value) and "sk_bad" not in str(exc.value)


def test_eleven_rejects_a_non_ascii_key(monkeypatch):
    """Printable but non-ASCII raises UnicodeEncodeError inside httpx — not an HTTPError,
    so it would escape the fallback entirely and print the offending character."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_é")
    import pytest
    with pytest.raises(t.ElevenUnavailable) as exc:
        t.transcribe_eleven("f.m4a", "ru")
    assert "printable ASCII" in str(exc.value)


# --- the request we actually send ----------------------------------------

class _FakeResponse:
    def __init__(self, payload=None, bad_json=False):
        self._payload, self._bad = payload, bad_json

    def raise_for_status(self):
        return None

    def json(self):
        if self._bad:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


def _capture_post(monkeypatch, response):
    """Stand in for httpx.post; returns the dict the request kwargs land in."""
    import httpx
    seen = {}
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test")
    monkeypatch.setattr(t, "_eleven_prepare", lambda *a, **k: ("f.m4a", None))
    monkeypatch.setattr("builtins.open", lambda *a, **k: __import__("io").BytesIO(b"x"))
    monkeypatch.setattr(httpx, "post",
                        lambda url, **kw: (seen.update(url=url, **kw), response)[1])
    return seen


def test_eleven_request_shape(monkeypatch):
    seen = _capture_post(monkeypatch, _FakeResponse({"text": "ок", "words": []}))
    t.transcribe_eleven("f.m4a", "ru", speakers=3)
    assert seen["url"] == t.ELEVEN_URL
    assert seen["timeout"] == 1800
    name, handle = seen["files"]["file"]
    assert name == "f.m4a"
    assert handle.closed, "the upload handle must be closed by the with-block"
    data = seen["data"]
    # without this the single-speaker passthrough keeps "(laughter)" and the diarized
    # path strips it — same audio, different output depending on speaker count
    assert data["tag_audio_events"] == "false"
    assert data["model_id"] == "scribe_v2"
    assert data["diarize"] == "true" and data["num_speakers"] == "3"
    assert data["language_code"] == "rus"
    assert seen["headers"]["xi-api-key"] == "sk_test"


def test_eleven_num_speakers_dropped_when_diarize_off(monkeypatch):
    seen = _capture_post(monkeypatch, _FakeResponse({"text": "ок", "words": []}))
    t.transcribe_eleven("f.m4a", "ru", speakers=3, diarize=False)
    assert "num_speakers" not in seen["data"] and seen["data"]["diarize"] == "false"


def test_eleven_non_json_body_is_an_outage_not_a_crash(monkeypatch):
    _capture_post(monkeypatch, _FakeResponse(bad_json=True))
    import pytest
    with pytest.raises(t.ElevenUnavailable) as exc:
        t.transcribe_eleven("f.m4a", "ru")
    assert "not JSON" in str(exc.value)


def test_eleven_prepare_cleans_up_when_ffmpeg_fails(tmp_path, monkeypatch):
    f = tmp_path / "call.mp4"
    f.write_bytes(b"x")
    made = []
    real_mkdtemp = t.tempfile.mkdtemp
    monkeypatch.setattr(t.tempfile, "mkdtemp",
                        lambda *a, **k: made.append(real_mkdtemp()) or made[-1])

    def ffmpeg_dies(*a, **k):
        raise t.subprocess.CalledProcessError(1, "ffmpeg")
    monkeypatch.setattr(t.subprocess, "run", ffmpeg_dies)
    import pytest
    with pytest.raises(t.subprocess.CalledProcessError):
        t._eleven_prepare(str(f))
    assert made and not t.os.path.exists(made[0])


# --- fallback when the cloud is unavailable ------------------------------

def _args(**over):
    base = dict(engine="eleven", model=None, prompt=None, speakers=None, denoise=False,
                keep_temp=False, no_vad=False, no_fallback=False, no_diarize=False, only=None)
    base.update(over)
    return t.argparse.Namespace(**base)


def test_eleven_failure_falls_back_with_banner(monkeypatch):
    monkeypatch.setattr(t, "transcribe_eleven",
                        lambda *a, **k: (_ for _ in ()).throw(t.ElevenUnavailable("no key")))
    monkeypatch.setattr(t, "transcribe_local", lambda *a, **k: "локальный текст")
    raw, cleaned = t.transcribe_one("x.m4a", _args(), "ru")
    assert raw == "локальный текст"
    assert cleaned.startswith("WARNING: ElevenLabs unavailable")


def test_eleven_failure_with_no_fallback_exits(monkeypatch):
    monkeypatch.setattr(t, "transcribe_eleven",
                        lambda *a, **k: (_ for _ in ()).throw(t.ElevenUnavailable("nope")))
    import pytest
    with pytest.raises(SystemExit) as exc:
        t.transcribe_one("x.m4a", _args(no_fallback=True), "ru")
    assert exc.value.code == 1


def test_eleven_both_engines_failing_exits_instead_of_raising(monkeypatch):
    monkeypatch.setattr(t, "transcribe_eleven",
                        lambda *a, **k: (_ for _ in ()).throw(t.ElevenUnavailable("no net")))

    def local_dead(*a, **k):
        raise OSError("model cache missing")
    monkeypatch.setattr(t, "transcribe_local", local_dead)
    import pytest
    with pytest.raises(SystemExit) as exc:
        t.transcribe_one("x.m4a", _args(), "ru")
    assert exc.value.code == 1


def test_eleven_parsing_bug_is_not_disguised_as_an_outage(monkeypatch):
    """A defect in our own response handling must surface, not fall back silently."""
    def bug(*a, **k):
        raise TypeError("bad call signature")
    monkeypatch.setattr(t, "transcribe_eleven", bug)
    monkeypatch.setattr(t, "transcribe_local", lambda *a, **k: "should not be reached")
    import pytest
    with pytest.raises(TypeError):
        t.transcribe_one("x.m4a", _args(), "ru")


def test_eleven_prompt_is_reported_as_ignored(monkeypatch, capsys):
    monkeypatch.setattr(t, "transcribe_eleven", lambda *a, **k: ("r", "c"))
    t.transcribe_one("a.m4a", _args(prompt="Kubernetes, gRPC"), "ru")
    assert "-p (initial prompt) has no effect" in capsys.readouterr().err


def test_eleven_flags_are_forwarded(monkeypatch):
    seen = {}
    monkeypatch.setattr(t, "transcribe_eleven",
                        lambda *a, **k: (seen.update(k) or ("r", "c")))
    t.transcribe_one("a.m4a", _args(speakers=2, denoise=True, no_diarize=True), "ru")
    assert seen["speakers"] == 2 and seen["denoise"] is True and seen["diarize"] is False


# --- _eleven_prepare: upload the audio track, not the video container ----

def test_eleven_prepare_passes_audio_through(tmp_path):
    f = tmp_path / "voice.m4a"
    f.write_bytes(b"x")
    assert t._eleven_prepare(str(f)) == (str(f), None)


def test_eleven_prepare_extracts_video(tmp_path, monkeypatch):
    f = tmp_path / "call.mp4"
    f.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(t.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    out, tmpdir = t._eleven_prepare(str(f))
    assert out.endswith("audio.mp3") and tmpdir and "-vn" in calls[0]
    t.shutil.rmtree(tmpdir, ignore_errors=True)


def test_eleven_prepare_denoise_forces_pass_on_plain_audio(tmp_path, monkeypatch):
    f = tmp_path / "voice.m4a"
    f.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(t.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    out, tmpdir = t._eleven_prepare(str(f), denoise=True)
    assert out != str(f) and "-af" in calls[0]
    t.shutil.rmtree(tmpdir, ignore_errors=True)
