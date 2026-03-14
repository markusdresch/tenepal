"""Tests for profile loader module."""

import json
from pathlib import Path

import pytest

from tenepal.language import (
    load_profile,
    load_profiles_from_directory,
    load_default_profiles,
    default_registry,
)


class TestProfileLoading:
    """Test loading individual profiles from JSON."""

    def test_load_nah_profile(self):
        """Load Nahuatl profile and verify structure."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        nah_path = profiles_dir / "nah.json"

        profile = load_profile(nah_path)

        assert profile.code == "nah"
        assert profile.name == "Nahuatl"
        assert profile.family == "Uto-Aztecan"
        assert isinstance(profile.marker_phonemes, set)
        assert isinstance(profile.absent_phonemes, set)
        assert isinstance(profile.marker_sequences, set)
        assert profile.priority == 10

    def test_load_spa_profile(self):
        """Load Spanish profile and verify structure."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        spa_path = profiles_dir / "spa.json"

        profile = load_profile(spa_path)

        assert profile.code == "spa"
        assert profile.name == "Spanish"
        assert profile.family == "Indo-European"
        assert isinstance(profile.marker_phonemes, set)
        assert profile.priority == 1

    def test_load_eng_profile(self):
        """Load English profile and verify structure."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        eng_path = profiles_dir / "eng.json"

        profile = load_profile(eng_path)

        assert profile.code == "eng"
        assert profile.name == "English"
        assert profile.family == "Indo-European"
        assert isinstance(profile.marker_phonemes, set)
        assert profile.priority == 2

    def test_load_deu_profile(self):
        """Load German profile and verify structure."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        deu_path = profiles_dir / "deu.json"

        profile = load_profile(deu_path)

        assert profile.code == "deu"
        assert profile.name == "German"
        assert profile.family == "Indo-European"
        assert isinstance(profile.marker_phonemes, set)
        assert profile.priority == 3

    def test_load_all_profiles(self):
        """Load all profiles and verify expected languages.

        NOTE: Phase 32 added MAY (Yucatec Maya) as a 7th language.
        This test now verifies all 7 languages are loaded.
        """
        profiles = load_default_profiles()

        codes = {p.code for p in profiles}
        assert codes == {"nah", "may", "spa", "eng", "deu", "fra", "ita"}

        # Verify sorted order (alphabetical)
        assert [p.code for p in profiles] == ["deu", "eng", "fra", "ita", "may", "nah", "spa"]


class TestGhostMarkersRemoved:
    """Test that ghost markers have been removed from profiles."""

    def test_ghost_markers_removed_eng(self):
        """Verify ENG profile keeps explicit English cues and excludes outdated ghosts."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        eng_path = profiles_dir / "eng.json"

        profile = load_profile(eng_path)

        # Intentional ENG cues for NAH/ENG separation.
        required_markers = {"θ", "ð", "ɹ", "ʌ", "dʒ"}
        for marker in required_markers:
            assert marker in profile.marker_phonemes, f"ENG marker {marker} missing from profile"

        # Legacy marker that remains excluded.
        assert "ŋ" not in profile.marker_phonemes, "ENG ghost marker ŋ should stay excluded"

        # Core vowel markers should remain present.
        real_markers = {"æ", "ʊ", "ɪ"}
        for real in real_markers:
            assert real in profile.marker_phonemes, f"Real marker {real} missing from ENG profile"

    def test_ghost_markers_removed_deu(self):
        """Verify German ghost markers (ç, ø, ʏ) are NOT in profile."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        deu_path = profiles_dir / "deu.json"

        profile = load_profile(deu_path)

        # Ghost markers that should NOT be present
        ghost_markers = {"ç", "ø", "ʏ"}
        for ghost in ghost_markers:
            assert ghost not in profile.marker_phonemes, f"Ghost marker {ghost} found in DEU profile"

        # Real markers that SHOULD be present (appeared in Phase 19 validation)
        real_markers = {"x", "ʁ"}
        for real in real_markers:
            assert real in profile.marker_phonemes, f"Real marker {real} missing from DEU profile"

    def test_ghost_markers_removed_spa(self):
        """Verify Spanish ghost markers (β, r, ɣ) are NOT in profile."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        spa_path = profiles_dir / "spa.json"

        profile = load_profile(spa_path)

        # Ghost markers that should NOT be present
        ghost_markers = {"β", "r", "ɣ"}
        for ghost in ghost_markers:
            assert ghost not in profile.marker_phonemes, f"Ghost marker {ghost} found in SPA profile"

        # Real markers that SHOULD be present (appeared in Phase 19 validation)
        # b is kept per user decision (voiced plosive), d/ɡ/ɲ/ɾ appeared in validation
        real_markers = {"b", "d", "ɡ", "ɲ", "ɾ"}
        for real in real_markers:
            assert real in profile.marker_phonemes, f"Real marker {real} missing from SPA profile"

    def test_nah_tl_kept(self):
        """Verify Nahuatl tɬ is kept as safety net exception."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        nah_path = profiles_dir / "nah.json"

        profile = load_profile(nah_path)

        # tɬ should be present (safety net exception per user decision)
        assert "tɬ" in profile.marker_phonemes, "NAH tɬ safety net marker missing"


class TestRegistryIntegration:
    """Test that default_registry loads from JSON configs."""

    def test_default_registry_uses_config(self):
        """Verify default_registry() loads profiles from JSON.

        NOTE: Phase 32 added MAY (Yucatec Maya) as a 7th language.
        """
        registry = default_registry()

        codes = registry.codes()
        assert set(codes) == {"nah", "may", "spa", "eng", "deu", "fra", "ita"}

        # Verify profiles are properly loaded
        nah = registry.get("nah")
        assert nah is not None
        assert nah.name == "Nahuatl"
        assert nah.priority == 10

        spa = registry.get("spa")
        assert spa is not None
        assert spa.name == "Spanish"
        assert spa.priority == 1

        eng = registry.get("eng")
        assert eng is not None
        assert eng.name == "English"
        assert eng.priority == 2

        deu = registry.get("deu")
        assert deu is not None
        assert deu.name == "German"
        assert deu.priority == 3

    def test_registry_markers_match_current_eng_policy(self):
        """Verify registry reflects current ENG marker policy."""
        registry = default_registry()

        # ENG: explicit cues enabled for NAH/ENG separation.
        eng = registry.get("eng")
        for marker in ("θ", "ð", "ɹ", "ʌ", "dʒ"):
            assert marker in eng.marker_phonemes
        assert "ŋ" not in eng.marker_phonemes

        # DEU ghost markers remain excluded.
        deu = registry.get("deu")
        assert "ç" not in deu.marker_phonemes
        assert "ø" not in deu.marker_phonemes
        assert "ʏ" not in deu.marker_phonemes

        # SPA ghost markers remain excluded.
        spa = registry.get("spa")
        assert "β" not in spa.marker_phonemes
        assert "r" not in spa.marker_phonemes
        assert "ɣ" not in spa.marker_phonemes


class TestTypeConversion:
    """Test that types are correctly converted from JSON."""

    def test_profile_roundtrip_types(self):
        """Verify sets are sets, sequences are tuple sets."""
        profiles = load_default_profiles()

        for profile in profiles:
            # marker_phonemes should be a set
            assert isinstance(profile.marker_phonemes, set)
            assert all(isinstance(p, str) for p in profile.marker_phonemes)

            # absent_phonemes should be a set
            assert isinstance(profile.absent_phonemes, set)
            assert all(isinstance(p, str) for p in profile.absent_phonemes)

            # marker_sequences should be a set of tuples
            assert isinstance(profile.marker_sequences, set)
            for seq in profile.marker_sequences:
                assert isinstance(seq, tuple)
                assert len(seq) == 2
                assert all(isinstance(p, str) for p in seq)

            # marker_weights should be a dict
            assert isinstance(profile.marker_weights, dict)

            # sequence_weights should be a dict with tuple keys
            assert isinstance(profile.sequence_weights, dict)
            for key in profile.sequence_weights.keys():
                assert isinstance(key, tuple)

            # marker_trigrams should be a set of 3-tuples
            assert isinstance(profile.marker_trigrams, set)
            for tri in profile.marker_trigrams:
                assert isinstance(tri, tuple)
                assert len(tri) == 3
                assert all(isinstance(p, str) for p in tri)

            # trigram_weights should be a dict with tuple keys
            assert isinstance(profile.trigram_weights, dict)
            for key in profile.trigram_weights.keys():
                assert isinstance(key, tuple)
                assert len(key) == 3

            # negative_markers should be a dict
            assert isinstance(profile.negative_markers, dict)

    def test_sequence_loading(self):
        """Verify marker_sequences loaded as set of tuples."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        nah_path = profiles_dir / "nah.json"

        profile = load_profile(nah_path)

        # NAH has 3 marker sequences
        assert len(profile.marker_sequences) == 3
        assert ("k", "w") in profile.marker_sequences
        assert ("t", "l") in profile.marker_sequences
        assert ("t", "ɬ") in profile.marker_sequences

    def test_weights_loading(self):
        """Verify marker_weights dict loads correctly."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        nah_path = profiles_dir / "nah.json"

        profile = load_profile(nah_path)

        # Current profiles have empty weights (default 1.0)
        assert isinstance(profile.marker_weights, dict)
        assert isinstance(profile.sequence_weights, dict)

        # Verify threshold loads
        assert profile.threshold == 0.0


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_nonexistent_file(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_profile(Path("/nonexistent/path.json"))

    def test_nonexistent_directory(self):
        """Loading from nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_profiles_from_directory(Path("/nonexistent/directory"))


class TestTrigramLoading:
    """Test loading trigram fields from JSON profiles."""

    def test_deu_trigrams_loaded(self):
        """DEU profile should have trigrams loaded from JSON."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        deu_path = profiles_dir / "deu.json"
        profile = load_profile(deu_path)

        assert len(profile.marker_trigrams) == 5
        assert ("ʃ", "t", "ʁ") in profile.marker_trigrams
        assert ("ʃ", "p", "ʁ") in profile.marker_trigrams
        assert ("p", "f", "l") in profile.marker_trigrams
        assert ("t", "s", "v") in profile.marker_trigrams
        assert ("x", "t", "ʁ") in profile.marker_trigrams

    def test_deu_trigram_weights_loaded(self):
        """DEU trigram weights should be loaded with correct values."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        deu_path = profiles_dir / "deu.json"
        profile = load_profile(deu_path)

        assert profile.trigram_weights[("ʃ", "t", "ʁ")] == 1.5
        assert profile.trigram_weights[("p", "f", "l")] == 1.2

    def test_profiles_without_trigrams_default_empty(self):
        """Profiles without trigram fields should have empty defaults."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        spa_path = profiles_dir / "spa.json"
        profile = load_profile(spa_path)

        assert profile.marker_trigrams == set()
        assert profile.trigram_weights == {}
        assert profile.negative_markers == {}

    def test_deu_bigrams_count(self):
        """DEU profile should have 17 bigrams after adding new ones."""
        profiles_dir = Path(__file__).parent.parent / "src" / "tenepal" / "language" / "profiles"
        deu_path = profiles_dir / "deu.json"
        profile = load_profile(deu_path)

        assert len(profile.marker_sequences) == 17
