"""
compat.py — adds all subdirectory packages to sys.path so that
internal modules (e.g. `from config import ...`) continue working
without rewriting every import in every file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "core" / "pipelines"))
sys.path.insert(0, str(ROOT / "core" / "models"))
sys.path.insert(0, str(ROOT / "core" / "features"))
sys.path.insert(0, str(ROOT / "data" / "api"))
sys.path.insert(0, str(ROOT / "app" / "tabs"))
sys.path.insert(0, str(ROOT / "scripts"))

# Aliases so `import config` finds config/settings.py etc.
import importlib, types

def _alias(real_module: str, alias: str):
    mod = importlib.import_module(real_module)
    sys.modules[alias] = mod

_alias("config.settings",          "config")
_alias("core.models.sport_model",  "sport_model")
_alias("core.models.nhl_model",    "model_trainer")
_alias("core.features.nhl_features","feature_engineering")
_alias("core.pipelines.nhl_pipeline","data_pipeline")
_alias("core.pipelines.mlb_pipeline","mlb_pipeline")
_alias("core.pipelines.nba_pipeline","nba_pipeline")
_alias("core.pipelines.nhl_betting","betting_projections")
_alias("data.api.nhl_api",         "nhl_api")
_alias("data.api.nst_scraper",     "nst_scraper")
_alias("data.api.mlb_api",         "mlb_api")
_alias("data.api.nba_client",      "nba_client")
_alias("app.tabs.mlb_tab",         "mlb_tab")
_alias("app.tabs.nba_tab",         "nba_tab")
