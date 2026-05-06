"""
compat.py — Module aliasing shim.

Adds all subdirectories to sys.path and creates aliases so bare-name
imports like `from betting_projections import X` work throughout the
codebase regardless of which file triggers the import first.

Safe to import multiple times — guarded by sys.modules check.
"""
import sys
import importlib
from pathlib import Path

# Find repo root (the directory containing this file)
ROOT = Path(__file__).resolve().parent

# Add every source directory to sys.path so bare imports work
_paths = [
    ROOT,
    ROOT / "config",
    ROOT / "core" / "pipelines",
    ROOT / "core" / "models",
    ROOT / "core" / "features",
    ROOT / "data" / "api",
    ROOT / "app" / "tabs",
    ROOT / "scripts",
]
for _p in _paths:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def _alias(dotted_path: str, bare_name: str) -> None:
    """Register dotted_path module under bare_name in sys.modules."""
    if bare_name in sys.modules:
        return
    try:
        mod = importlib.import_module(dotted_path)
        sys.modules[bare_name] = mod
    except ImportError:
        pass  # optional — skip silently


# Config
_alias("config.settings",             "config")

# Core models
_alias("core.models.sport_model",     "sport_model")
_alias("core.models.nhl_model",       "model_trainer")  # NHLModel / backward-compat wrapper

# Core features
_alias("core.features.nhl_features",  "feature_engineering")

# Core pipelines
_alias("core.pipelines.nhl_pipeline", "data_pipeline")
_alias("core.pipelines.mlb_pipeline", "mlb_pipeline")
_alias("core.pipelines.nba_pipeline", "nba_pipeline")
_alias("core.pipelines.nhl_betting",  "betting_projections")

# Data APIs
_alias("data.api.nhl_api",            "nhl_api")
_alias("data.api.nst_scraper",        "nst_scraper")
_alias("data.api.mlb_api",            "mlb_api")
_alias("data.api.nba_client",         "nba_client")

# App tabs
_alias("app.tabs.mlb_tab",            "mlb_tab")
_alias("app.tabs.nba_tab",            "nba_tab")
_alias("app.tabs.nhl_tab",            "nhl_tab")
