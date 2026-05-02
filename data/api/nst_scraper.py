"""
nst_scraper.py — Scrapes advanced stats from naturalstatrick.com

NST provides the richest public source of advanced NHL stats:
  - Individual Corsi (iCF), Fenwick (iFF)
  - Individual Expected Goals (ixG)
  - High-danger chances (iHDCF, iHDGF)
  - On-ice CF%, FF%, xGF%, etc.
  - Power-play TOI breakdown

All methods return pandas DataFrames and cache responses to disk.
"""

import os
import time
import json
import hashlib
from io import StringIO
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import (
    NST_BASE, NST_PLAYER_URL, REQUEST_HEADERS, REQUEST_DELAY,
    CACHE_DIR, CURRENT_SEASON, NST_TEAM_MAP
)

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_params(params: dict) -> str:
    key = json.dumps(params, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"nst_{name}.csv")


def _save_df_cache(name: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(name), index=False)


def _load_df_cache(name: str, max_age_minutes: int = 120) -> Optional[pd.DataFrame]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    age = (time.time() - os.path.getmtime(path)) / 60
    if age > max_age_minutes:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _fix_team_code(code: str) -> str:
    return NST_TEAM_MAP.get(code, code)


# ── Core scraper ──────────────────────────────────────────────────────────────

def _fetch_nst_table(params: dict) -> pd.DataFrame:
    """
    Request the NST playerteams.php page with given params and parse
    the first HTML table found into a DataFrame.
    """
    cache_id = _hash_params(params)
    cached = _load_df_cache(cache_id)
    if cached is not None:
        return cached

    time.sleep(REQUEST_DELAY)
    try:
        session = requests.Session()
        # Mimic a real browser more closely
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://naturalstatrick.com/",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
        })
        # First visit the home page to get cookies
        session.get("https://naturalstatrick.com/", timeout=15)
        time.sleep(1.0)

        resp = session.get(
            NST_PLAYER_URL,
            params=params,
            timeout=25,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[NST] Request failed: {e}")
        return pd.DataFrame()

    # Debug: print first 500 chars to see what we got
    snippet = resp.text[:500].replace('\n', ' ').strip()
    print(f"[NST] Response status: {resp.status_code}, length: {len(resp.text)}")
    print(f"[NST] Page snippet: {snippet[:300]}")

    soup = BeautifulSoup(resp.text, "lxml")

    # Check for bot-block / Cloudflare pages
    title = soup.find("title")
    page_title = title.text.strip() if title else ""
    print(f"[NST] Page title: {page_title}")

    if any(x in page_title.lower() for x in ["blocked", "cloudflare", "attention", "access denied", "just a moment"]):
        print("[NST] Blocked by bot protection — skipping NST stats.")
        return pd.DataFrame()

    # Try multiple possible table selectors
    table = (
        soup.find("table", {"id": "players"})
        or soup.find("table", {"id": "tbl"})
        or soup.find("table", {"class": "tablesorter"})
        or soup.find("div", {"id": "playerreport"})
        or soup.find("table")
    )

    if table and table.name != "table":
        table = table.find("table")

    if table is None:
        all_tables = soup.find_all("table")
        print(f"[NST] Tables found on page: {len(all_tables)}")
        all_divs = [d.get("id","") for d in soup.find_all("div") if d.get("id")]
        print(f"[NST] Div IDs on page: {all_divs[:20]}")
        print("[NST] No usable table found.")
        return pd.DataFrame()

    try:
        df = pd.read_html(StringIO(str(table)))[0]
    except Exception as e:
        print(f"[NST] Could not parse table: {e}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if "Player" in df.columns:
        df = df[df["Player"] != "Player"].reset_index(drop=True)

    for col in df.columns:
        if col not in ("Player", "Team", "Position", "Pos", "Name"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Team" in df.columns:
        df["Team"] = df["Team"].apply(_fix_team_code)

    if "Player" not in df.columns and "Name" in df.columns:
        df = df.rename(columns={"Name": "Player"})

    print(f"[NST] Successfully parsed table with {len(df)} rows, columns: {list(df.columns[:8])}")
    _save_df_cache(cache_id, df)
    return df


# ── Public accessors ──────────────────────────────────────────────────────────

def _base_params(season: str = CURRENT_SEASON, situation: str = "all",
                 stdoi: str = "std", rate: str = "n") -> dict:
    return {
        "season":    season,
        "stype":     2,
        "sit":       situation,
        "score":     "all",
        "stdoi":     stdoi,
        "rate":      rate,
        "team":      "ALL",
        "pos":       "S",
        "loc":       "B",
        "toi":       0,
        "gpfilt":    "none",
        "fd":        "",
        "td":        "",
        "tgp":       410,
        "lines":     "single",
        "draftteam": "ALL",
    }


def get_standard_stats(season: str = CURRENT_SEASON,
                        situation: str = "all") -> pd.DataFrame:
    """
    Fetch standard individual stats for all skaters.
    Includes: GP, TOI, Goals, Assists, Points, SOG, SH%, ixG, iCF, iFF, iSCF, iHDCF
    """
    params = _base_params(season=season, situation=situation, stdoi="std")
    df = _fetch_nst_table(params)
    df = _rename_std_cols(df)
    return df


def get_rate_stats(season: str = CURRENT_SEASON,
                   situation: str = "all") -> pd.DataFrame:
    """
    Fetch per-60 rate stats for all skaters.
    """
    params = _base_params(season=season, situation=situation,
                          stdoi="std", rate="y")
    df = _fetch_nst_table(params)
    df = _rename_rate_cols(df)
    return df


def get_on_ice_stats(season: str = CURRENT_SEASON,
                      situation: str = "all") -> pd.DataFrame:
    """
    Fetch on-ice relative stats: CF%, FF%, GF%, xGF%, HDCF%, etc.
    """
    params = _base_params(season=season, situation=situation, stdoi="oi")
    df = _fetch_nst_table(params)
    df = _rename_oi_cols(df)
    return df


def get_pp_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Fetch power-play individual stats.
    """
    params = _base_params(season=season, situation="pp", stdoi="std")
    df = _fetch_nst_table(params)
    df = _rename_std_cols(df, prefix="pp_")
    return df


def get_all_situations_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Merge standard, on-ice, and PP stats into one comprehensive DataFrame.
    This is the main method used for feature engineering.
    """
    std   = get_standard_stats(season, "all")
    oi    = get_on_ice_stats(season, "all")
    pp    = get_pp_stats(season)
    ev    = get_standard_stats(season, "5v5")   # 5-on-5 only

    frames = [std]

    if not oi.empty and "player_name" in oi.columns:
        oi_merge = oi[["player_name", "team"] + [c for c in oi.columns
                       if c not in std.columns and c not in ("player_name","team")]]
        frames.append(oi_merge.set_index(["player_name", "team"]))

    base = frames[0].set_index(["player_name", "team"])

    if not oi.empty and "player_name" in oi.columns:
        oi_cols = [c for c in oi.columns if c not in base.columns
                   and c not in ("player_name", "team")]
        base = base.join(oi[["player_name","team"] + oi_cols].set_index(
            ["player_name","team"]), how="left")

    # PP TOI
    if not pp.empty and "player_name" in pp.columns:
        pp_cols = ["pp_toi", "pp_goals", "pp_ixg"]
        pp_cols = [c for c in pp_cols if c in pp.columns]
        if pp_cols:
            pp_sub = pp[["player_name","team"] + pp_cols].set_index(["player_name","team"])
            base = base.join(pp_sub, how="left")

    # 5v5 goals + shots for even-strength metrics
    if not ev.empty and "player_name" in ev.columns:
        ev_cols = ["ev_goals", "ev_shots", "ev_ixg"]
        if "goals" in ev.columns:
            ev = ev.rename(columns={"goals": "ev_goals", "shots_on_goal": "ev_shots",
                                    "ixg": "ev_ixg"})
            ev_sub = ev[["player_name","team"] + [c for c in ["ev_goals","ev_shots","ev_ixg"]
                          if c in ev.columns]].set_index(["player_name","team"])
            base = base.join(ev_sub, how="left")

    return base.reset_index()


# ── Column renaming helpers ───────────────────────────────────────────────────

def _rename_std_cols(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    rename = {
        "Player":              "player_name",
        "Team":                "team",
        "Position":            "position",
        "Pos":                 "position",
        "GP":                  f"{prefix}gp",
        "TOI":                 f"{prefix}toi",
        "Goals":               f"{prefix}goals",
        "Total Assists":       f"{prefix}total_assists",
        "First Assists":       f"{prefix}first_assists",
        "Second Assists":      f"{prefix}second_assists",
        "Total Points":        f"{prefix}total_points",
        "IPP":                 f"{prefix}ipp",
        "SOG":                 f"{prefix}shots_on_goal",
        "SH%":                 f"{prefix}shooting_pct",
        "Shots":               f"{prefix}shots_on_goal",
        "ixG":                 f"{prefix}ixg",
        "iCF":                 f"{prefix}icf",
        "iFF":                 f"{prefix}iff",
        "iSCF":                f"{prefix}iscf",
        "iHDCF":               f"{prefix}ihdcf",
        "iHDGF":               f"{prefix}ihdgf",
        "Rush Attempts":       f"{prefix}rush_attempts",
        "Rebounds Created":    f"{prefix}rebounds_created",
        "PIM":                 f"{prefix}pim",
        "+/-":                 f"{prefix}plus_minus",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _rename_rate_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Player":   "player_name",
        "Team":     "team",
        "Position": "position",
        "GP":       "gp",
        "TOI":      "toi",
        "Goals":    "goals_per60",
        "ixG":      "ixg_per60",
        "iCF":      "icf_per60",
        "SOG":      "shots_per60",
        "iHDCF":    "ihdcf_per60",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _rename_oi_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Player":     "player_name",
        "Team":       "team",
        "Position":   "position",
        "GP":         "gp",
        "TOI":        "toi",
        "CF%":        "cf_pct",
        "CF% Rel":    "cf_pct_rel",
        "FF%":        "ff_pct",
        "FF% Rel":    "ff_pct_rel",
        "SF%":        "sf_pct",
        "GF%":        "gf_pct",
        "xGF%":       "xgf_pct",
        "SCF%":       "scf_pct",
        "HDCF%":      "hdcf_pct",
        "HDGF%":      "hdgf_pct",
        "PDO":        "pdo",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
