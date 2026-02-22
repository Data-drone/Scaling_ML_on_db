"""Tests for src/config.py — size presets and parameter validation."""

import pytest
from src.config import PRESETS, get_preset, validate_params, SizePreset


class TestSizePreset:
    def test_all_presets_exist(self):
        expected = {"tiny", "small", "medium", "medium_large", "large", "xlarge"}
        assert set(PRESETS.keys()) == expected

    def test_preset_total_features(self):
        medium = get_preset("medium")
        assert medium.total_features == medium.num_features + medium.cat_features

    def test_preset_table_name(self):
        small = get_preset("small")
        assert small.table_name == "imbalanced_1m"

    def test_preset_full_table_name(self):
        medium = get_preset("medium")
        assert medium.full_table_name() == "brian_gen_ai.xgb_scaling.imbalanced_10m"

    def test_preset_full_table_name_custom(self):
        tiny = get_preset("tiny")
        assert tiny.full_table_name("my_catalog", "my_schema") == "my_catalog.my_schema.imbalanced_10k"

    def test_preset_immutable(self):
        preset = get_preset("small")
        with pytest.raises(AttributeError):
            preset.name = "changed"

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_row_counts_increase(self):
        sizes = ["tiny", "small", "medium", "medium_large", "large", "xlarge"]
        rows = [PRESETS[s].rows for s in sizes]
        assert rows == sorted(rows), "Presets should have increasing row counts"


class TestValidateParams:
    def test_valid_params(self):
        result = validate_params("small", "D16sv5", 4, 14)
        assert result["data_size"] == "small"
        assert result["preset"].rows == 1_000_000

    def test_invalid_data_size(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            validate_params("nonexistent")

    def test_large_data_single_node_warning(self):
        with pytest.warns(UserWarning, match="100,000,000 rows"):
            validate_params("large", num_workers=1)

    def test_large_data_distributed_no_warning(self):
        # Should not warn with enough workers
        result = validate_params("large", num_workers=8)
        assert result["num_workers"] == 8
