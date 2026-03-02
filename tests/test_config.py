"""Tests for configuration module."""

from src.data.config import (
    DATA_DIR,
    DURATION_MAP,
    LAND_REGISTRY_COLUMNS,
    LAND_REGISTRY_COMPLETE_CSV,
    LAND_REGISTRY_YEARLY_CSV,
    OLD_NEW_MAP,
    PPD_CATEGORY_MAP,
    PROCESSED_DIR,
    PROJECT_DIR,
    PROPERTY_TYPE_MAP,
    RAW_DIR,
)


def test_project_dir_is_absolute():
    assert PROJECT_DIR.is_absolute()


def test_data_dirs_under_project():
    assert str(DATA_DIR).startswith(str(PROJECT_DIR))
    assert str(RAW_DIR).startswith(str(DATA_DIR))
    assert str(PROCESSED_DIR).startswith(str(DATA_DIR))


def test_land_registry_columns_count():
    assert len(LAND_REGISTRY_COLUMNS) == 16


def test_land_registry_urls():
    assert "pp-complete.csv" in LAND_REGISTRY_COMPLETE_CSV
    assert "{year}" in LAND_REGISTRY_YEARLY_CSV


def test_property_type_map_values():
    assert set(PROPERTY_TYPE_MAP.keys()) == {"D", "S", "T", "F", "O"}


def test_old_new_map():
    assert set(OLD_NEW_MAP.keys()) == {"Y", "N"}


def test_duration_map():
    assert set(DURATION_MAP.keys()) == {"F", "L", "U"}


def test_ppd_category_map():
    assert set(PPD_CATEGORY_MAP.keys()) == {"A", "B"}
