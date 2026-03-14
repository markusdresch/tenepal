"""Tests for CLI backend flag parsing and fallback behavior."""

import pytest

from tenepal import cli
from tenepal.phoneme import backend as backend_module


class TestBackendArgParsing:
    def test_default_backend(self):
        parser = cli.build_parser()
        args = parser.parse_args([])
        assert args.backend == "allosaurus"

    def test_explicit_allosaurus(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--backend", "allosaurus"])
        assert args.backend == "allosaurus"

    def test_explicit_omnilingual(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--backend", "omnilingual"])
        assert args.backend == "omnilingual"

    def test_invalid_backend_rejected(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "invalid"])

    def test_build_parser_dual_backend_choice(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--backend", "dual", "test.wav"])
        assert args.backend == "dual"

    def test_omnilingual_model_flag_default(self):
        parser = cli.build_parser()
        args = parser.parse_args(["test.wav"])
        assert args.omnilingual_model == "300M"

    def test_omnilingual_model_flag_7b(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--omnilingual-model", "7B", "test.wav"])
        assert args.omnilingual_model == "7B"

    def test_omnilingual_model_invalid(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--omnilingual-model", "1B", "test.wav"])

    def test_setup_omnilingual_subcommand_parsed(self):
        parser = cli.build_parser()
        args = parser.parse_args(["setup-omnilingual"])
        assert args.command == "setup-omnilingual"


class TestBackendFallback:
    def test_unavailable_backend_falls_back(self, monkeypatch, capsys):
        def fake_get_backend(name):
            raise ValueError("not available")

        monkeypatch.setattr(backend_module, "get_backend", fake_get_backend)
        backend = cli.validate_backend("omnilingual")
        assert backend == "allosaurus"
        captured = capsys.readouterr()
        assert "Falling back to allosaurus" in captured.err

    def test_available_backend_no_fallback(self, monkeypatch, capsys):
        class FakeBackend:
            pass

        def fake_get_backend(name):
            return FakeBackend()

        monkeypatch.setattr(backend_module, "get_backend", fake_get_backend)
        backend = cli.validate_backend("allosaurus")
        assert backend == "allosaurus"
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_validate_backend_dual_available(self, monkeypatch):
        class FakeBackend:
            pass

        def fake_get_backend(name):
            return FakeBackend()

        monkeypatch.setattr(backend_module, "get_backend", fake_get_backend)
        monkeypatch.setattr(
            "tenepal.phoneme.omnilingual_backend.OmnilingualBackend.is_available",
            classmethod(lambda cls: True),
        )
        assert cli.validate_backend("dual") == "dual"

    def test_validate_backend_dual_unavailable_fallback(self, monkeypatch, capsys):
        class FakeBackend:
            pass

        def fake_get_backend(name):
            return FakeBackend()

        monkeypatch.setattr(backend_module, "get_backend", fake_get_backend)
        monkeypatch.setattr(
            "tenepal.phoneme.omnilingual_backend.OmnilingualBackend.is_available",
            classmethod(lambda cls: False),
        )
        assert cli.validate_backend("dual") == "allosaurus"
        captured = capsys.readouterr()
        assert "Dual mode will use Allosaurus only" in captured.err

    def test_backend_kwargs_dual(self):
        assert cli._backend_kwargs("dual", "300M") == {"model_size": "300M"}

    def test_backend_kwargs_dual_7b(self):
        assert cli._backend_kwargs("dual", "7B") == {"model_size": "7B"}


class TestWhisperModelArgParsing:
    def test_build_parser_has_whisper_model_arg(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--whisper-model", "base", "test.wav"])
        assert args.whisper_model == "base"

    def test_whisper_model_default_is_none(self):
        parser = cli.build_parser()
        args = parser.parse_args(["test.wav"])
        assert args.whisper_model is None

    def test_whisper_model_choices(self):
        parser = cli.build_parser()
        # Valid choices
        for model in ["tiny", "base", "small", "medium", "large"]:
            args = parser.parse_args(["--whisper-model", model, "test.wav"])
            assert args.whisper_model == model

    def test_whisper_model_invalid_rejected(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--whisper-model", "invalid", "test.wav"])

    def test_setup_whisper_in_commands(self):
        parser = cli.build_parser()
        args = parser.parse_args(["setup-whisper"])
        assert args.command == "setup-whisper"
