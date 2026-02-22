"""
Dataset size presets and parameter validation for XGBoost scaling experiments.

Presets define the dataset dimensions used across all tracks (single-node, Ray, GPU).
Each preset maps to a Unity Catalog table: brian_gen_ai.xgb_scaling.imbalanced_{suffix}.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SizePreset:
    """Dataset size preset with generation parameters."""
    name: str
    rows: int
    num_features: int       # Total numerical features
    cat_features: int       # Total categorical features
    table_suffix: str       # Unity Catalog table name suffix
    imbalance_ratio: float = 0.05  # Minority class fraction

    @property
    def total_features(self) -> int:
        return self.num_features + self.cat_features

    @property
    def table_name(self) -> str:
        return f"imbalanced_{self.table_suffix}"

    def full_table_name(self, catalog: str = "brian_gen_ai", schema: str = "xgb_scaling") -> str:
        return f"{catalog}.{schema}.{self.table_name}"


# Standard presets used across all experiments
PRESETS = {
    "tiny":         SizePreset("tiny",         10_000,      15,   5, "10k"),
    "small":        SizePreset("small",      1_000_000,     80,  20, "1m"),
    "medium":       SizePreset("medium",    10_000_000,    200,  50, "10m"),
    "medium_large": SizePreset("medium_large", 30_000_000, 200,  50, "30m"),
    "large":        SizePreset("large",    100_000_000,    400, 100, "100m"),
    "xlarge":       SizePreset("xlarge",   500_000_000,    400, 100, "500m"),
}


def get_preset(name: str) -> SizePreset:
    """Get a size preset by name, raising ValueError if not found."""
    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")
    return PRESETS[name]


def validate_params(
    data_size: str,
    node_type: Optional[str] = None,
    num_workers: Optional[int] = None,
    cpus_per_worker: Optional[int] = None,
) -> dict:
    """
    Validate experiment parameters and return a normalised dict.

    Raises ValueError for invalid combinations.
    """
    preset = get_preset(data_size)

    params = {
        "preset": preset,
        "data_size": data_size,
        "node_type": node_type,
        "num_workers": num_workers,
        "cpus_per_worker": cpus_per_worker,
    }

    # Warn about data sizes that likely need distributed training
    if preset.rows >= 100_000_000 and (num_workers is None or num_workers <= 1):
        import warnings
        warnings.warn(
            f"Dataset '{data_size}' has {preset.rows:,} rows. "
            "Single-node training may OOM or be very slow. "
            "Consider using Ray distributed training with num_workers >= 4."
        )

    return params
