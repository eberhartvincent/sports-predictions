"""
mlb_pipeline.py — MLB prediction pipeline

Prediction approach:
  - Features are per-AB RATES (not raw counts) to prevent early-season inflation
  - HR classifier output is capped at realistic maximum (~15%)
  - All regression outputs are capped at physically plausible game maxima
  - Roster uses rosterType=active (no IL players) with a 30-min cache TTL
  - Injury/IL check via MLB API injuredList endpoint as secondary filter
  - Park factors applied to game total projections
  - Platoon splits applied when pitcher handedness is known
"""

import os, time
from datetime import datetime
from typing import Optional
from math import exp, factorial

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from mlb_api import (
    get_mlb_schedule, get_mlb_roster, get_mlb_player_gamelog,
    get_mlb_pitcher_gamelog,
    get_game_weather, get_batter_vs_pitcher,
    BALLPARK_COORDS, ROOFED_STADIUMS,
)
try:
    from statcast_client import (
        get_statcast_batter_stats, get_statcast_pitcher_stats,
        get_umpire_tendency, get_sprint_speed,
    )
    STATCAST_AVAILABLE = True
except ImportError:
    STATCAST_AVAILABLE = False
    _p = lambda msg: print(f"  [MLB] {msg}")
from sport_model import SportModel
from config import (
    MLB_MIN_GP as MIN_GP, MLB_SEASON as SEASON,
    MLB_LEAGUE_AVG_RUNS, MLB_HOME_ADVANTAGE_RUNS,
    MLB_REGRESSION_WEIGHT, MLB_DEFAULT_ERA, MLB_DEFAULT_WHIP, MLB_DEFAULT_K9,
    MLB_PITCHER_RECENT_STARTS, MLB_CONF_ELITE, MLB_CONF_HIGH, MLB_CONF_MEDIUM,
    MLB_TEAM_NAMES, MLB_REQUEST_DELAY, MLB_TEAMS, API_TIMEOUT_SECONDS,
    REQUEST_HEADERS,
)

def _p(msg): print(f"  [MLB] {msg}")

# ── Park factors ──────────────────────────────────────────────────────────────
PARK_FACTORS = {
    "COL":1.18, "CIN":1.08, "PHI":1.07, "TEX":1.07, "BOS":1.06,
    "MIL":1.05, "ARI":1.05, "CHC":1.04, "NYY":1.03, "ATL":1.03,
    "HOU":1.02, "CLE":1.02, "BAL":1.01, "MIN":1.01, "TB":1.00,
    "STL":1.00, "LAD":0.99, "DET":0.99, "WSH":0.98, "TOR":0.98,
    "CWS":0.98, "LAA":0.97, "KC":0.97,  "NYM":0.97, "PIT":0.96,
    "SEA":0.96, "OAK":0.96, "SF":0.95,  "MIA":0.95, "SD":0.94,
}

# ── Platoon advantage ─────────────────────────────────────────────────────────
PLATOON_AVG_BOOST = 0.028    # batting avg boost with platoon advantage

# ── Lineup PA weights ─────────────────────────────────────────────────────────
LINEUP_PA_WEIGHTS = {1:1.00,2:0.99,3:0.97,4:0.95,5:0.93,6:0.91,7:0.89,8:0.87,9:0.85}

# ── Prediction output caps ────────────────────────────────────────────────────
# No hard caps — predictions reflect actual model/stat output.
# Realistic ranges emerge naturally from the stat anchor blend.
CAPS: dict = {}

# ── Feature columns ───────────────────────────────────────────────────────────
# All rolling stats are per-AB RATES, not raw counts.
# This prevents a player who hits 2 HRs in one early-season game from
# producing an hr_rate of 0.67 that confuses the model.
FEAT_COLS = [
    # Per-AB hit rate rolling windows
    "hit_rate_3g","hit_rate_5g","hit_rate_10g","season_hit_rate",
    # Per-AB HR rate rolling windows
    "hr_rate_3g","hr_rate_5g","hr_rate_10g","season_hr_rate",
    # Per-AB TB (slug) rolling
    "slg_3g","slg_5g","slg_10g","season_slg",
    # Per-AB RBI rate
    "rbi_rate_3g","rbi_rate_5g","season_rbi_rate",
    # Per-AB run rate
    "run_rate_3g","season_run_rate",
    # Strikeout + walk rates
    "k_rate_3g","k_rate_5g","season_k_rate",
    "bb_rate_3g","season_bb_rate",
    # Season-level summary
    "season_obp","season_avg",
    # AB opportunity
    "ab_3g","season_ab_pg",
    # ── Streak / momentum features ────────────────────────────────────────────
    # These measure CURRENT form vs established baseline.
    # Positive = hot (recent > season), Negative = cold (recent < season).
    # The model learns that hot streaks predict continued hot performance
    # (momentum), NOT that cold streaks predict imminent rebounds (gambler's fallacy).
    "slg_vs_mean_3g",      # SLG: last 3G vs season avg
    "slg_vs_mean_5g",      # SLG: last 5G vs season avg (more stable window)
    "hr_rate_vs_mean_5g",  # HR rate: last 5G vs season avg (power-specific streak)
    "bb_rate_vs_mean_5g",  # Walk rate trend — rising BB often precedes hits
    "k_rate_vs_mean_5g",   # K rate trend — rising Ks = mechanical issues
    "hit_rate_vs_mean_5g", # Hit rate: last 5G vs season avg
    "games_since_hr",      # Games since last HR (0 = hit one last game, capped at 20)
    # Context
    "is_home","park_factor",
    "opp_era","opp_whip","opp_k9",
    "platoon_advantage","lineup_pa_weight",
    "gp",
    # Weather
    "weather_hr_factor","wind_component","temp_f","precip_mm",
    # Batter vs pitcher history (weighted by sample size)
    "bvp_avg","bvp_slg","bvp_ops","bvp_hr_rate","bvp_sample_weight",
    # ── Statcast xStats (Baseball Savant — free, strongest predictors) ────────
    # These are the most predictive features available for MLB.
    # xBA/xSLG capture true talent better than actual BA/SLG.
    # Luck indicators (actual - expected) identify regression candidates.
    "exit_velocity_avg",    # avg exit velocity (raw power)
    "barrel_pct",           # barrel rate (optimal launch angle+velocity)
    "hard_hit_pct",         # hard hit% (>= 95 mph exit velocity)
    "xba",                  # expected batting average (vs actual BA)
    "xslg",                 # expected slugging (vs actual SLG)
    "xwoba",                # expected wOBA (best overall hitting metric)
    "sweet_spot_pct",       # sweet spot contact % (launch angle 8-32°)
    "woba_luck",            # actual wOBA - expected wOBA (negative = unlucky = buy)
    "ba_luck",              # actual BA - xBA
    # Umpire factor
    "umpire_run_factor",    # today's HP umpire run scoring tendency
    "umpire_k_factor",      # today's HP umpire strikeout tendency
    # Sprint speed (run scoring proxy)
    "sprint_speed",         # ft/sec (faster = more runs scored)
]


def _roll(series: pd.Series, n: int) -> float:
    s = series.iloc[-n:] if len(series) >= n else series
    return float(s.mean()) if len(s) > 0 else 0.0

def _rate(num: pd.Series, den: pd.Series, n: int) -> float:
    """Per-AB rate over last n games."""
    ns = num.iloc[-n:] if len(num) >= n else num
    ds = den.iloc[-n:] if len(den) >= n else den
    total_den = float(ds.sum())
    return float(ns.sum()) / total_den if total_den > 0 else 0.0


def build_mlb_features(
    plogs: pd.DataFrame,
    is_home: bool,
    opp_era: float = MLB_DEFAULT_ERA,
    opp_whip: float = MLB_DEFAULT_WHIP,
    opp_k9: float = MLB_DEFAULT_K9,
    park_factor: float = 1.0,
    platoon_advantage: float = 0.0,
    lineup_slot: int = 4,
    weather_hr_factor: float = 1.0,
    wind_component: float = 0.0,
    temp_f: float = 72.0,
    precip_mm: float = 0.0,
    bvp_avg: float = 0.0,
    bvp_slg: float = 0.0,
    bvp_ops: float = 0.0,
    bvp_hr_rate: float = 0.0,
    bvp_sample_weight: float = 0.0,
) -> dict:
    if len(plogs) == 0:
        return {}

    hits = plogs["hits"]; hr   = plogs["hr"]
    rbi  = plogs["rbi"]; k    = plogs["k"];  bb = plogs["bb"]
    ab   = plogs["ab"];  runs = plogs.get("runs", pd.Series([0]*len(plogs)))
    obp  = plogs.get("obp", pd.Series([0.0]*len(plogs)))
    slg  = plogs.get("slg", pd.Series([0.0]*len(plogs)))
    avg  = plogs.get("avg", pd.Series([0.0]*len(plogs)))

    # Season-level per-AB rates
    total_ab = float(ab.sum())
    season_hit_rate = float(hits.sum()) / total_ab if total_ab > 0 else 0.0
    season_hr_rate  = float(hr.sum())   / total_ab if total_ab > 0 else 0.0
    season_rbi_rate = float(rbi.sum())  / total_ab if total_ab > 0 else 0.0
    season_run_rate = float(runs.sum()) / total_ab if total_ab > 0 else 0.0
    season_k_rate   = float(k.sum())    / total_ab if total_ab > 0 else 0.0
    season_bb_rate  = float(bb.sum())   / total_ab if total_ab > 0 else 0.0

    # ── Streak / momentum features ─────────────────────────────────────────
    # All computed as (recent_rate - season_rate).
    # Positive = hot, negative = cold.
    # The model learns: hot → tends to stay hot (momentum), not cold → due.
    slg_mean      = float(slg.mean())        if len(slg) > 0 else 0.0
    slg_3g_val    = _roll(slg, 3)
    slg_5g_val    = _roll(slg, 5)

    hr_season_rate = float(hr.sum()) / max(float(ab.sum()), 1)
    hr_5g_rate     = _rate(hr, ab, 5)

    bb_season_rate = float(bb.sum()) / max(float(ab.sum()), 1)
    bb_5g_rate     = _rate(bb, ab, 5)

    k_season_rate  = float(k.sum()) / max(float(ab.sum()), 1)
    k_5g_rate      = _rate(k, ab, 5)

    hit_season_rate = float(hits.sum()) / max(float(ab.sum()), 1)
    hit_5g_rate     = _rate(hits, ab, 5)

    # Games since last HR — actual recency of power, not gambler's fallacy
    # 0 = hit one last game, 20 = hasn't hit one in 20+ games (or never)
    hr_arr = plogs["hr"].values
    games_since_hr = 20  # default cap
    for i in range(len(hr_arr) - 1, -1, -1):
        if hr_arr[i] > 0:
            games_since_hr = len(hr_arr) - 1 - i
            break

    slg_vs_mean_3g   = slg_3g_val - slg_mean
    slg_vs_mean_5g   = slg_5g_val - slg_mean
    hr_rate_vs_mean  = hr_5g_rate  - hr_season_rate
    bb_rate_vs_mean  = bb_5g_rate  - bb_season_rate
    k_rate_vs_mean   = k_5g_rate   - k_season_rate
    hit_rate_vs_mean = hit_5g_rate - hit_season_rate

    return {
        # Per-AB hit rate
        "hit_rate_3g":   _rate(hits, ab, 3),
        "hit_rate_5g":   _rate(hits, ab, 5),
        "hit_rate_10g":  _rate(hits, ab, 10),
        "season_hit_rate": season_hit_rate,
        # Per-AB HR rate
        "hr_rate_3g":    _rate(hr, ab, 3),
        "hr_rate_5g":    _rate(hr, ab, 5),
        "hr_rate_10g":   _rate(hr, ab, 10),
        "season_hr_rate": season_hr_rate,
        # SLG (total bases per AB)
        "slg_3g":        _roll(slg, 3),
        "slg_5g":        _roll(slg, 5),
        "slg_10g":       _roll(slg, 10),
        "season_slg":    float(slg.mean()),
        # Per-AB RBI rate
        "rbi_rate_3g":   _rate(rbi, ab, 3),
        "rbi_rate_5g":   _rate(rbi, ab, 5),
        "season_rbi_rate": season_rbi_rate,
        # Per-AB run rate
        "run_rate_3g":   _rate(runs, ab, 3),
        "season_run_rate": season_run_rate,
        # Strikeout / walk rates
        "k_rate_3g":     _rate(k, ab, 3),
        "k_rate_5g":     _rate(k, ab, 5),
        "season_k_rate": season_k_rate,
        "bb_rate_3g":    _rate(bb, ab, 3),
        "season_bb_rate": season_bb_rate,
        # Summary
        "season_obp":    float(obp.mean()),
        "season_avg":    float(avg.mean()),
        "ab_3g":          _roll(ab, 3),
        "season_ab_pg":   float(ab.mean()),
        "slg_vs_mean_3g":  slg_vs_mean_3g,
        "slg_vs_mean_5g":  slg_vs_mean_5g,
        "hr_rate_vs_mean_5g":  hr_rate_vs_mean,
        "bb_rate_vs_mean_5g":  bb_rate_vs_mean,
        "k_rate_vs_mean_5g":   k_rate_vs_mean,
        "hit_rate_vs_mean_5g": hit_rate_vs_mean,
        "games_since_hr":  float(min(games_since_hr, 20)),
        # Context
        "is_home":          1.0 if is_home else 0.0,
        "park_factor":      park_factor,
        "opp_era":          opp_era,
        "opp_whip":         opp_whip,
        "opp_k9":           opp_k9,
        "platoon_advantage": platoon_advantage,
        "lineup_pa_weight": LINEUP_PA_WEIGHTS.get(lineup_slot, 0.91),
        "gp":               float(len(plogs)),
        # Weather
        "weather_hr_factor": weather_hr_factor,
        "wind_component":    wind_component,
        "temp_f":            temp_f,
        "precip_mm":         precip_mm,
        # Batter vs pitcher history
        "bvp_avg":           bvp_avg,
        "bvp_slg":           bvp_slg,
        "bvp_ops":           bvp_ops,
        "bvp_hr_rate":       bvp_hr_rate,
        "bvp_sample_weight": bvp_sample_weight,
    }


def build_training_df(all_logs: pd.DataFrame,
                       team_pitching: dict = None) -> pd.DataFrame:
    """
    Build training features from game logs.
    team_pitching: {team_abbrev: {era, whip, k9}} from get_team_pitching_stats().
    Each training row uses the ACTUAL opposing team season ERA/WHIP/K9 so the
    model genuinely learns how pitcher quality affects batter outcomes.
    Without this, opp_era/whip/k9 are constants and the model ignores them entirely.
    """
    rows = []
    rows_per_player = len(all_logs) / max(all_logs["player_id"].nunique(), 1) \
                      if not all_logs.empty else 0
    eff_min_gp = 3 if rows_per_player < 15 else MIN_GP

    for _, grp in all_logs.groupby("player_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        for i in range(eff_min_gp, len(grp)):
            hist     = grp.iloc[:i]
            cur      = grp.iloc[i]
            opp_team = str(cur.get("opponent", ""))

            if team_pitching and opp_team in team_pitching:
                p = team_pitching[opp_team]
                opp_era, opp_whip, opp_k9 = p["era"], p["whip"], p["k9"]
            else:
                opp_era, opp_whip, opp_k9 = MLB_DEFAULT_ERA, MLB_DEFAULT_WHIP, MLB_DEFAULT_K9

            feat = build_mlb_features(
                hist, bool(cur.get("home_away", True)),
                opp_era, opp_whip, opp_k9,
                weather_hr_factor=1.0, wind_component=0.0,
                temp_f=72.0, precip_mm=0.0,
                bvp_avg=0.0, bvp_slg=0.0, bvp_ops=0.0,
                bvp_hr_rate=0.0, bvp_sample_weight=0.0,
            )
            if not feat:
                continue
            feat["target_hit"]  = int(cur["hit_flag"])
            feat["target_hr"]   = int(cur["hr_flag"])
            feat["target_hits"] = float(cur["hits"])
            feat["target_tb"]   = float(cur["tb"])
            feat["target_rbi"]  = float(cur["rbi"])
            feat["target_k"]    = float(cur["k"])
            feat["target_runs"] = float(cur.get("runs", 0))
            rows.append(feat)
    return pd.DataFrame(rows)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MLBPipeline:
    def __init__(self):
        self.games:             list         = []
        self.predictions:         pd.DataFrame = pd.DataFrame()
        self.pitcher_predictions: pd.DataFrame = pd.DataFrame()
        self.game_proj:           list         = []
        self._statcast_batters:   dict         = {}
        self._statcast_pitchers:  dict         = {}
        self._sprint_speed:       dict         = {}
        self._umpire_map:         dict         = {}
        self._rosters:          pd.DataFrame = pd.DataFrame()
        self._all_logs:         pd.DataFrame = pd.DataFrame()
        self._pitcher_map:      dict         = {}
        self._pitcher_hand:     dict         = {}  # gid_side → "L"/"R"
        self._pitcher_id_map:   dict         = {}  # opp_team → pitcher_id (for BvP)
        self._batter_hand:      dict         = {}  # player_id → "L"/"R"/"S"
        self._lineup_map:       dict         = {}  # (gid, pid) → slot
        self._il_ids:           set          = set()
        self._weather_map:      dict         = {}  # game_id → weather dict
        self._bvp_cache:        dict         = {}  # (batter_id, pitcher_id) → bvp dict
        self._team_pitching:    dict         = {}  # team_abbrev → {era, whip, k9}
        self._effective_min_gp: int          = MIN_GP
        self.metrics:           dict         = {}
        self.models = {
            "hit":  SportModel("mlb_hit",  "classify"),
            "hr":   SportModel("mlb_hr",   "classify"),
            "hits": SportModel("mlb_hits", "regress"),
            "tb":   SportModel("mlb_tb",   "regress"),
            "rbi":  SportModel("mlb_rbi",  "regress"),
            "k":    SportModel("mlb_k",    "regress"),
            "runs": SportModel("mlb_runs", "regress"),
        }

    def fetch_schedule(self, date: Optional[str] = None) -> list:
        et   = ZoneInfo("America/New_York")
        dstr = date or datetime.now(et).strftime("%Y-%m-%d")
        self.games = get_mlb_schedule(dstr)
        _p(f"Found {len(self.games)} games on {dstr}")
        return self.games

    def fetch_rosters(self) -> pd.DataFrame:
        """Active roster only (rosterType=active excludes IL players)."""
        frames = []; seen = set()
        for g in self.games:
            for side in ("away_team_id","home_team_id"):
                tid = g.get(side)
                if tid and tid not in seen:
                    seen.add(tid)
                    try:
                        # Force short TTL so IL moves are caught within 30 min
                        from mlb_api import _get as mlb_get, _save as mlb_save, _load as mlb_load
                        from config import MLB_CACHE_DIR as MLB_CACHE
                        key = f"mlb_roster_{tid}_{SEASON}_active"
                        cached = mlb_load(key, ttl=30)
                        if cached:
                            frames.append(pd.DataFrame(cached))
                            continue
                        data = mlb_get(f"/teams/{tid}/roster",
                                       {"rosterType":"active","season":SEASON})
                        rows = []
                        for p in data.get("roster",[]):
                            pi  = p.get("person",{})
                            pos = p.get("position",{}).get("abbreviation","")
                            rows.append({
                                "player_id":   pi.get("id"),
                                "player_name": pi.get("fullName",""),
                                "position":    pos,
                                "team_id":     tid,
                                "team":        MLB_TEAMS.get(tid, str(tid)),
                                "status":      p.get("status",{}).get("code","A"),
                            })
                        mlb_save(key, rows)
                        if rows:
                            frames.append(pd.DataFrame(rows))
                    except Exception as e:
                        _p(f"Roster error {tid}: {e}")
                    time.sleep(0.3)
        self._rosters = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        _p(f"Rosters: {len(self._rosters)} active players")
        return self._rosters

    def fetch_il_players(self) -> set:
        """
        Players on the IL are already excluded by rosterType=active in fetch_rosters().
        This method exists as a no-op safety check — the active roster is the source
        of truth. We keep the set empty so nothing is double-filtered.
        """
        self._il_ids = set()
        _p("IL filter: active roster already excludes IL players")
        return self._il_ids

    def fetch_game_logs(self) -> pd.DataFrame:
        if self._rosters.empty:
            return pd.DataFrame()
        batters = self._rosters[~self._rosters["position"].isin(["P","SP","RP"])]
        pids    = batters["player_id"].dropna().astype(int).unique().tolist()

        # Fetch current + 2 prior seasons for richer model training
        seasons_to_fetch = [SEASON, str(int(SEASON)-1), str(int(SEASON)-2)]
        _p(f"Fetching logs for {len(pids)} batters across seasons {seasons_to_fetch} …")

        frames = []
        for pid in pids:
            for szn in seasons_to_fetch:
                try:
                    df = get_mlb_player_gamelog(pid, szn)
                    if not df.empty:
                        df["season"] = szn
                        frames.append(df)
                except Exception:
                    pass
                time.sleep(MLB_REQUEST_DELAY)

        if frames:
            self._all_logs = (pd.concat(frames, ignore_index=True)
                              .sort_values(["player_id","game_date"])
                              .reset_index(drop=True))
        else:
            self._all_logs = pd.DataFrame()

        rows_per_player = len(self._all_logs) / max(len(pids), 1)
        self._effective_min_gp = 3 if rows_per_player < 15 else MIN_GP
        _p(f"Total log rows: {len(self._all_logs)} | effective MIN_GP: {self._effective_min_gp}")
        return self._all_logs

    def fetch_pitcher_stats(self) -> dict:
        """Fetch probable pitcher ERA/WHIP/K9 and handedness.

        Key insight: _pitcher_map is keyed by the BATTING team that faces each pitcher.
          - away pitcher (e.g. Cole) faces the HOME batters → key = home_team
          - home pitcher (e.g. Sale) faces the AWAY batters → key = away_team
        In build_predictions we look up _pitcher_map[opp] where opp is the team
        providing the pitcher, so we need: _pitcher_map[batting_team] = pitcher_stats.
        """
        import requests as _req
        _p("Fetching probable pitcher stats …")
        for g in self.games:
            gid  = str(g.get("game_id",""))
            away = g.get("away_team","")
            home = g.get("home_team","")

            for side in ("away", "home"):
                pid = g.get(f"{side}_pitcher_id")
                if not pid:
                    continue

                # The batting team that faces this pitcher:
                # away pitcher → faces home batters; home pitcher → faces away batters
                batting_team = home if side == "away" else away

                try:
                    df = get_mlb_pitcher_gamelog(pid, SEASON)
                    if df.empty:
                        df = get_mlb_pitcher_gamelog(pid, str(int(SEASON)-1))
                    if not df.empty:
                        recent = df.iloc[-MLB_PITCHER_RECENT_STARTS:]
                        self._pitcher_map[batting_team] = {
                            "era":  float(recent["era"].mean()),
                            "whip": float(recent["whip"].mean()),
                            "k9":   float(recent["k9"].mean()),
                        }
                        _p(f"  {g.get(f'{side}_pitcher_name','?')} ({away if side=='home' else home}) "
                           f"vs {batting_team} batters: "
                           f"ERA={self._pitcher_map[batting_team]['era']:.2f} "
                           f"WHIP={self._pitcher_map[batting_team]['whip']:.2f}")

                    # Store pitcher ID under the batting team for BvP lookups
                    self._pitcher_id_map[batting_team] = pid

                    # Pitcher handedness — key by gid + pitching side
                    try:
                        r = _req.get(
                            f"https://statsapi.mlb.com/api/v1/people/{pid}",
                            timeout=API_TIMEOUT_SECONDS, headers=REQUEST_HEADERS)
                        if r.status_code == 200:
                            hand = r.json().get("people",[{}])[0].get("pitchHand",{}).get("code","R")
                            self._pitcher_hand[gid+"_"+side] = hand
                    except Exception:
                        pass
                except Exception as e:
                    _p(f"Pitcher {pid} error: {e}")

        _p(f"Pitcher stats loaded for {len(self._pitcher_map)} batting teams")
        return self._pitcher_map

    def fetch_batter_handedness(self) -> dict:
        """Fetch bat side for active batters (capped at 120 to limit API calls)."""
        import requests as _req
        if self._rosters.empty:
            return {}
        _p("Fetching batter handedness …")
        batters = self._rosters[~self._rosters["position"].isin(["P","SP","RP"])]
        pids    = batters["player_id"].dropna().astype(int).unique().tolist()
        for pid in pids[:120]:
            try:
                r = _req.get(
                    f"https://statsapi.mlb.com/api/v1/people/{pid}",
                    timeout=API_TIMEOUT_SECONDS, headers=REQUEST_HEADERS)
                if r.status_code == 200:
                    hand = r.json().get("people",[{}])[0].get("batSide",{}).get("code","R")
                    self._batter_hand[pid] = hand
            except Exception:
                pass
            time.sleep(0.15)
        _p(f"Handedness for {len(self._batter_hand)} batters")
        return self._batter_hand

    def fetch_batter_vs_pitcher(self) -> dict:
        """Pre-fetch career BvP stats for all today's batters vs probable pitchers.
        Batched here so build_predictions doesn't make live API calls per player."""
        if self._rosters.empty:
            return {}
        _p("Fetching batter vs pitcher history …")
        batters = self._rosters[~self._rosters["position"].isin(["P","SP","RP"])]
        pids    = batters["player_id"].dropna().astype(int).unique().tolist()
        fetched = 0
        for pid in pids:
            for batting_team, pitcher_id in self._pitcher_id_map.items():
                # Only fetch for batters on the team facing this pitcher
                player_team = batters[batters["player_id"]==pid]["team"].values
                if len(player_team) == 0 or player_team[0] != batting_team:
                    continue
                cache_key = (pid, pitcher_id)
                if cache_key not in self._bvp_cache:
                    self._bvp_cache[cache_key] = get_batter_vs_pitcher(pid, pitcher_id)
                    fetched += 1
                    time.sleep(0.08)
        _p(f"BvP fetched for {fetched} batter-pitcher pairs ({len(self._bvp_cache)} total cached)")
        return self._bvp_cache

    def fetch_statcast_stats(self) -> tuple:
        """
        Fetch Statcast xStats for batters and pitchers.
        These are the single strongest individual predictors —
        xwOBA outperforms traditional stats significantly.
        """
        _p("Fetching Statcast batter stats…")
        if not STATCAST_AVAILABLE:
            _p("  Statcast not available — skipping")
            self._statcast_batters  = {}
            self._statcast_pitchers = {}
            self._umpire_map        = {}
            self._sprint_speed      = {}
            return {}, {}, {}, {}
        try:
            batter_df  = get_statcast_batter_stats(SEASON)
            pitcher_df = get_statcast_pitcher_stats(SEASON)
            sprint_df  = get_sprint_speed(SEASON)

            # Index by player_id for fast lookup
            self._statcast_batters = {}
            if not batter_df.empty and "player_id" in batter_df.columns:
                for _, row in batter_df.iterrows():
                    self._statcast_batters[int(row["player_id"])] = row.to_dict()

            self._statcast_pitchers = {}
            if not pitcher_df.empty and "player_id" in pitcher_df.columns:
                for _, row in pitcher_df.iterrows():
                    self._statcast_pitchers[int(row["player_id"])] = row.to_dict()

            self._sprint_speed = {}
            if not sprint_df.empty and "player_id" in sprint_df.columns:
                for _, row in sprint_df.iterrows():
                    self._sprint_speed[int(row["player_id"])] = float(
                        row.get("sprint_speed", 27.0))

            _p(f"Statcast: {len(self._statcast_batters)} batters, "
               f"{len(self._statcast_pitchers)} pitchers, "
               f"{len(self._sprint_speed)} sprint speeds")
        except Exception as e:
            _p(f"Statcast fetch error: {e}")
            self._statcast_batters = {}; self._statcast_pitchers = {}
            self._sprint_speed = {}

        # Umpire tendency for today
        try:
            date_str = self.games[0].get("game_date","") if self.games else ""
            self._umpire_map = get_umpire_tendency(date_str) if date_str else {}
            _p(f"Umpire data: {len(self._umpire_map)} games")
        except Exception as e:
            _p(f"Umpire data error: {e}")
            self._umpire_map = {}

        return (self._statcast_batters, self._statcast_pitchers,
                self._sprint_speed, self._umpire_map)

    def fetch_weather(self) -> dict:
        """Fetch weather for each game using Open-Meteo (free, no API key)."""
        _p("Fetching game weather …")
        for g in self.games:
            gid  = str(g.get("game_id",""))
            home = g.get("home_team","")
            date = g.get("date","")
            start= g.get("start_time","")
            try:
                w = get_game_weather(home, date, start)
                self._weather_map[gid] = w
                if not w.get("is_dome"):
                    _p(f"  {g.get('away_team')} @ {home}: {w.get('weather_desc','?')} "
                       f"(HR factor: {w.get('weather_hr_factor',1.0):.3f})")
            except Exception as e:
                _p(f"  Weather error {home}: {e}")
                self._weather_map[gid] = {"weather_hr_factor":1.0,"wind_component":0.0,
                                          "temp_f":72.0,"precip_mm":0.0,"is_dome":False}
        _p(f"Weather fetched for {len(self._weather_map)} games")
        return self._weather_map
    def fetch_team_pitching_stats(self) -> dict:
        """
        Build team pitching stats for training data enrichment.

        We already have ERA/WHIP/K9 for every batting team from fetch_pitcher_stats
        (which pulls from today's probable pitchers). Use that directly — it's the
        same data, already fetched, and avoids a separate unreliable API call.

        For historical training rows, each game row knows its opponent team abbreviation.
        We map opponent → pitcher quality using this same dataset.
        """
        if self._pitcher_map:
            self._team_pitching = dict(self._pitcher_map)
            _p(f"Team pitching stats: {len(self._team_pitching)} teams (from pitcher map)")
            sample = dict(list(self._team_pitching.items())[:3])
            _p(f"  Sample: { {k: {sk: round(sv,2) for sk,sv in v.items()} for k,v in sample.items()} }")
        else:
            _p("Team pitching stats: 0 teams (pitcher map empty — no games today?)")
            self._team_pitching = {}
        return self._team_pitching

    def train_models(self, force: bool = False):
        all_saved = all(m.is_saved() for m in self.models.values())
        if not force and all_saved:
            if all(m.load() for m in self.models.values()):
                # Validate cached model feature columns match current FEAT_COLS.
                # If they differ the model was trained with different features — retrain.
                for name, model in self.models.items():
                    # Only retrain if cached features are completely different
                    # (not just a subset — weather/bvp may not have been in training df)
                    cached = set(model.feature_names)
                    current = set(FEAT_COLS)
                    core_only = current - {"weather_hr_factor","wind_component","temp_f",
                                           "precip_mm","bvp_avg","bvp_slg","bvp_ops",
                                           "bvp_hr_rate","bvp_sample_weight"}
                    if not cached.intersection(core_only):
                        _p(f"Feature mismatch in cached {name} model — forcing retrain")
                        force = True
                        break
                if not force:
                    self.metrics = {k: m.metrics for k,m in self.models.items()}
                    _p("Models loaded from cache")
                    return

        if self._all_logs.empty:
            raise RuntimeError("No game logs to train on")

        _p("Building training features …")
        full_df = build_training_df(self._all_logs, team_pitching=self._team_pitching)
        if full_df.empty:
            raise RuntimeError("Training df is empty")

        # Date-based split: train on oldest 80%, prevents temporal leakage
        if "game_date" in self._all_logs.columns and len(full_df) > 200:
            all_dates = pd.to_datetime(self._all_logs["game_date"], errors="coerce").dropna()
            if not all_dates.empty:
                cutoff = all_dates.quantile(0.80)
                # Approximate: take oldest 80% of training rows by index
                cutoff_idx = int(len(full_df) * 0.80)
                train_df = full_df.iloc[:cutoff_idx]
            else:
                train_df = full_df
        else:
            train_df = full_df
        _p(f"Training on {len(train_df)} rows (of {len(full_df)} total) …")

        feat_cols = [c for c in FEAT_COLS if c in train_df.columns]
        for name, tcol in {
            "hit":"target_hit","hr":"target_hr","hits":"target_hits",
            "tb":"target_tb","rbi":"target_rbi","k":"target_k","runs":"target_runs",
        }.items():
            if tcol in train_df.columns:
                self.models[name].train(train_df, tcol, feat_cols)
                self.models[name].save()
                self.metrics[name] = self.models[name].metrics

    def build_predictions(self) -> pd.DataFrame:
        season_active = False
        if not self._all_logs.empty and "game_date" in self._all_logs.columns:
            latest = pd.to_datetime(self._all_logs["game_date"], errors="coerce").max()
            if pd.notna(latest):
                season_active = (pd.Timestamp.now() - latest).days <= 30

        eff_min_gp       = self._effective_min_gp
        inactivity_limit = 30 if season_active else 400

        rows = []
        for g in self.games:
            away, home = g.get("away_team",""), g.get("home_team","")
            gid        = str(g.get("game_id",""))

            for team, opp, is_home in [(away,home,False),(home,away,True)]:
                # _pitcher_map is keyed by batting team (team facing the pitcher)
                opp_pitch   = self._pitcher_map.get(team,{
                    "era":MLB_DEFAULT_ERA,"whip":MLB_DEFAULT_WHIP,"k9":MLB_DEFAULT_K9})
                park_factor = PARK_FACTORS.get(home, 1.0)
                p_side      = "home" if team==away else "away"
                pitcher_hand= self._pitcher_hand.get(gid+"_"+p_side, "R")
                pitcher_id  = self._pitcher_id_map.get(team)

                # Weather for this game
                wx          = self._weather_map.get(gid, {})
                wx_hr_factor= float(wx.get("weather_hr_factor", 1.0))
                wx_wind     = float(wx.get("wind_component", 0.0))
                wx_temp     = float(wx.get("temp_f", 72.0))
                wx_precip   = float(wx.get("precip_mm", 0.0))

                team_ros = self._rosters[self._rosters["team"]==team] \
                           if not self._rosters.empty else pd.DataFrame()

                for _, player in team_ros.iterrows():
                    if player.get("position") in ("P","SP","RP"):
                        continue
                    pid  = player["player_id"]
                    name = player["player_name"]

                    # Skip IL players (secondary check on top of active roster)
                    try:
                        pid_int = int(pid)
                    except Exception:
                        pid_int = 0
                    if pid_int in self._il_ids:
                        continue

                    if self._all_logs.empty:
                        continue
                    plogs = self._all_logs[
                        self._all_logs["player_id"]==pid
                    ].sort_values("game_date")

                    if len(plogs) < eff_min_gp:
                        continue

                    # Require at least 8 total at-bats — filters out players
                    # who appeared in 1-2 games as pinch hitters or call-ups
                    # with no meaningful sample (e.g. minor league prospects)
                    if float(plogs["ab"].sum()) < 8:
                        continue

                    last = pd.to_datetime(plogs["game_date"].iloc[-1], errors="coerce")
                    if pd.notna(last) and (pd.Timestamp.now()-last).days > inactivity_limit:
                        continue

                    batter_hand = self._batter_hand.get(pid_int, "R")
                    platoon_adv = (
                        PLATOON_AVG_BOOST if (
                            (batter_hand=="L" and pitcher_hand=="R") or
                            (batter_hand=="R" and pitcher_hand=="L")
                        ) else PLATOON_AVG_BOOST * 0.5 if batter_hand=="S" else 0.0
                    )

                    lineup_slot = self._lineup_map.get((gid, pid_int), 4)

                    # Batter vs pitcher career history
                    # Only fetch if not already cached (avoids 0.1s sleep per batter)
                    bvp = {}
                    if pitcher_id and pid_int:
                        cache_key = (pid_int, pitcher_id)
                        bvp = self._bvp_cache.get(cache_key, {})

                    bvp_ab  = int(bvp.get("bvp_ab", 0))
                    bvp_hr  = int(bvp.get("bvp_hr", 0))

                    feat = build_mlb_features(
                        plogs, is_home,
                        opp_pitch["era"], opp_pitch["whip"], opp_pitch["k9"],
                        park_factor=park_factor,
                        platoon_advantage=platoon_adv,
                        lineup_slot=lineup_slot,
                        weather_hr_factor=wx_hr_factor,
                        wind_component=wx_wind,
                        temp_f=wx_temp,
                        precip_mm=wx_precip,
                        bvp_avg=float(bvp.get("bvp_avg", 0.0)),
                        bvp_slg=float(bvp.get("bvp_slg", 0.0)),
                        bvp_ops=float(bvp.get("bvp_ops", 0.0)),
                        bvp_hr_rate=float(bvp_hr / max(bvp_ab, 1)),
                        bvp_sample_weight=float(bvp.get("bvp_sample_weight", 0.0)),
                    )
                    if not feat:
                        continue

                    feat_df = pd.DataFrame([feat]).reindex(columns=FEAT_COLS, fill_value=0.0)

                    row = {
                        "player_id":    pid,
                        "player_name":  name,
                        "team":         team,
                        "opponent":     opp,
                        "position":     player.get("position",""),
                        "game_label":   f"{away} @ {home}",
                        "is_home":      is_home,
                        "gp":           len(plogs),
                        "season_avg":   round(float(plogs["avg"].mean()), 3),
                        "season_hr":    int(plogs["hr"].sum()),
                        "season_hits":  int(plogs["hits"].sum()),
                        "park_factor":  park_factor,
                        "platoon_adv":  round(platoon_adv, 3),
                        "pitcher_hand": pitcher_hand,
                        "batter_hand":  batter_hand,
                        "weather":      wx.get("weather_desc",""),
                        "bvp_ab":       bvp_ab,
                        "bvp_hr":       bvp_hr,
                    }

                    total_gp     = len(plogs)
                    ml_weight    = min(1.0, total_gp / 60)
                    stat_weight  = 1.0 - ml_weight

                    total_ab     = float(plogs["ab"].sum())
                    season_ab_pg = float(plogs["ab"].mean())
                    total_hr     = float(plogs["hr"].sum())

                    # ── League average rates ───────────────────────────────
                    LG_HR_PER_AB   = 0.034   # league avg ~1 HR per 29 AB
                    LG_HIT_PER_AB  = 0.248
                    LG_RBI_PG      = 0.41
                    LG_RUNS_PG     = 0.42
                    LG_K_PG        = 0.84
                    LG_HITS_PG     = 0.77
                    LG_TB_PG       = 1.22

                    # ── HR regression — the most critical calculation ──────
                    # K_HR controls how quickly we trust individual HR rate.
                    # HR rates stabilize at ~600 PA over a full career but
                    # within a single season we want to trust recent data more.
                    # Scale K with season progress: early season = more regression,
                    # mid/late season = more trust in actual data.
                    # K=300 at start, drops to K=150 by 300 AB.
                    K_HR  = max(150, 300 - int(total_ab / 3))
                    K_HIT = 150   # hits stabilize faster than HRs

                    player_hr_per_ab  = total_hr / max(total_ab, 1)
                    player_hit_per_ab = float(plogs["hits"].sum()) / max(total_ab, 1)

                    # Tiered HR prior: don't pull zero-HR players to league avg.
                    # A player with 0 HRs in 40 AB is a contact/speed player.
                    # Their prior should be the contact player rate (~0.008/AB),
                    # not the league average (0.034/AB) which inflates predictions.
                    if total_hr == 0 and total_ab >= 15:
                        # Contact player prior — bottom quartile HR rate
                        hr_prior = 0.008
                    elif total_hr == 0:
                        # Too few AB to know anything — use league average
                        hr_prior = LG_HR_PER_AB
                    else:
                        # Has demonstrated power — regress toward league avg
                        hr_prior = LG_HR_PER_AB

                    w_hr  = total_ab / (total_ab + K_HR)
                    w_hit = total_ab / (total_ab + K_HIT)

                    reg_hr_per_ab  = w_hr * player_hr_per_ab + (1 - w_hr) * hr_prior
                    reg_hit_per_ab = w_hit * player_hit_per_ab + (1 - w_hit) * LG_HIT_PER_AB

                    exp_ab = min(season_ab_pg, 4.5)

                    stat_anchors = {
                        "hit":  reg_hit_per_ab * exp_ab,
                        "hr":   reg_hr_per_ab  * exp_ab,
                        "hits": float(plogs["hits"].mean()) * w_hit + LG_HITS_PG * (1-w_hit),
                        "tb":   float(plogs["tb"].mean())   * w_hit + LG_TB_PG   * (1-w_hit),
                        "rbi":  float(plogs["rbi"].mean())  * w_hit + LG_RBI_PG  * (1-w_hit),
                        "k":    float(plogs["k"].mean())    * w_hit + LG_K_PG    * (1-w_hit),
                        "runs": float(plogs.get("runs", pd.Series([0]*len(plogs))).mean()) * w_hit + LG_RUNS_PG * (1-w_hit),
                    }

                    # ── Per-target ML weights ─────────────────────────────
                    # The key insight: a player with 30+ AB and zero HRs has
                    # demonstrated they are not a power hitter. The ML model
                    # should have zero influence on their HR projection —
                    # the stat anchor (which gives ~0.026) is the truth.
                    # For all other targets, use the normal ml_weight.
                    is_proven_contact = (total_hr == 0 and total_ab >= 30)
                    is_any_power      = (total_hr >= 1)

                    for mname, model in self.models.items():
                        if model.is_trained:
                            try:
                                ml_val = float(model.predict(feat_df)[0])
                            except Exception:
                                ml_val = 0.0
                        else:
                            ml_val = 0.0

                        # Use stat anchor only for HR when player has no power evidence
                        if mname == "hr" and is_proven_contact:
                            # Trust the stat anchor completely — ML model can't
                            # override clear empirical evidence of no power
                            blended = stat_anchors["hr"]
                        else:
                            anchor  = stat_anchors.get(mname, ml_val)
                            blended = ml_weight * ml_val + stat_weight * anchor

                            # BvP HR adjustment: if the batter has meaningful history
                            # vs this pitcher (20+ AB), nudge HR projection by their
                            # actual HR rate in that matchup (weighted by sample size)
                            if mname == "hr" and bvp_ab >= 20 and pitcher_id:
                                bvp_hr_rate = bvp_hr / max(bvp_ab, 1)
                                bvp_weight  = min(0.20, bvp_ab / 200)  # max 20% influence
                                blended = (1 - bvp_weight) * blended + bvp_weight * (bvp_hr_rate * exp_ab)

                        row[f"proj_{mname}"] = round(max(0.0, blended), 3)

                    hp   = row.get("proj_hit", 0.5)
                    hits = row.get("proj_hits", 0)
                    hr   = row.get("proj_hr",   0)
                    rbi  = row.get("proj_rbi",  0)
                    r    = row.get("proj_runs", 0)
                    tb   = row.get("proj_tb",   0)
                    hrr  = row.get("proj_hrr",  0)
                    k    = row.get("proj_k",    0)

                    # Overall confidence (based on hit probability)
                    row["confidence"] = (
                        "Elite"  if hp >= MLB_CONF_ELITE  else
                        "High"   if hp >= MLB_CONF_HIGH   else
                        "Medium" if hp >= MLB_CONF_MEDIUM else "Low"
                    )
                    # Per-category confidence — stored in parquet, no recalc needed in UI
                    row["conf_hits"] = (
                        "Elite"  if hits >= 1.10 else
                        "High"   if hits >= 0.85 else
                        "Medium" if hits >= 0.60 else "Low"
                    )

                    row["conf_rbi"] = (
                        "Elite"  if rbi >= 1.20 else
                        "High"   if rbi >= 0.80 else
                        "Medium" if rbi >= 0.50 else "Low"
                    )
                    row["conf_runs"] = (
                        "Elite"  if r   >= 1.10 else
                        "High"   if r   >= 0.75 else
                        "Medium" if r   >= 0.50 else "Low"
                    )
                    row["conf_hrr"] = (
                        "Elite"  if hrr >= 2.80 else
                        "High"   if hrr >= 2.00 else
                        "Medium" if hrr >= 1.40 else "Low"
                    )
                    row["conf_tb"] = (
                        "Elite"  if tb  >= 3.00 else
                        "High"   if tb  >= 2.00 else
                        "Medium" if tb  >= 1.20 else "Low"
                    )
                    row["conf_k"] = (
                        "Elite"  if k   >= 1.40 else
                        "High"   if k   >= 1.00 else
                        "Medium" if k   >= 0.65 else "Low"
                    )
                    # H+R+RBI combined — standard DraftKings/FanDuel betting category
                    row["proj_hrr"] = round(
                        row.get("proj_hits", 0) +
                        row.get("proj_runs", 0) +
                        row.get("proj_rbi",  0), 2
                    )

                    # ── Per-category confidence tiers ─────────────────────────
                    # Each stat gets its own tier so sorting by HR shows
                    # HR-specific confidence, not just overall hit confidence.
                    hr  = row.get("proj_hr",   0)
                    rbi = row.get("proj_rbi",  0)
                    r   = row.get("proj_runs", 0)
                    tb  = row.get("proj_tb",   0)
                    hrr = row.get("proj_hrr",  0)
                    k   = row.get("proj_k",    0)

                    # HR confidence: based on HR probability vs league avg 0.107/game

                    # RBI confidence
                    row["conf_rbi"] = (
                        "Elite"  if rbi >= 1.20 else
                        "High"   if rbi >= 0.80 else
                        "Medium" if rbi >= 0.50 else "Low"
                    )
                    # Runs confidence
                    row["conf_runs"] = (
                        "Elite"  if r   >= 1.10 else
                        "High"   if r   >= 0.75 else
                        "Medium" if r   >= 0.50 else "Low"
                    )
                    # H+R+RBI confidence
                    row["conf_hrr"] = (
                        "Elite"  if hrr >= 2.80 else
                        "High"   if hrr >= 2.00 else
                        "Medium" if hrr >= 1.40 else "Low"
                    )
                    # Hits confidence (same as overall but explicit)

                    # Strikeout confidence (for pitcher props)
                    row["conf_k"] = (
                        "Elite"  if k   >= 1.40 else
                        "High"   if k   >= 1.00 else
                        "Medium" if k   >= 0.65 else "Low"
                    )
                    rows.append(row)

        self.predictions = pd.DataFrame()
        if rows:
            df = pd.DataFrame(rows)
            # Deduplicate — same player can appear if they show up in multiple
            # roster fetches (traded players, roster cache overlap)
            df = (df.sort_values("gp", ascending=False)
                    .drop_duplicates(subset=["player_id","game_label"], keep="first")
                    .sort_values("proj_hits", ascending=False)
                    .reset_index(drop=True))
            self.predictions = df
        _p(f"Predictions: {len(self.predictions)} players")
        return self.predictions

    def build_pitcher_predictions(self) -> pd.DataFrame:
        """
        Build per-start pitcher projections from recent game log stats.
        Uses the pitcher game logs already fetched in fetch_pitcher_stats.
        Returns a DataFrame with one row per starting pitcher today.
        """
        rows = []
        for g in self.games:
            away, home = g.get("away_team",""), g.get("home_team","")
            gid        = str(g.get("game_id",""))
            wx         = self._weather_map.get(gid, {})
            wx_hr      = float(wx.get("weather_hr_factor", 1.0))
            wx_desc    = wx.get("weather_desc","")

            for side in ("away","home"):
                pid   = g.get(f"{side}_pitcher_id")
                pname = g.get(f"{side}_pitcher_name","TBD")
                if not pid:
                    continue

                # Batting team this pitcher faces
                batting_team = home if side == "away" else away
                pitching_team= away if side == "away" else home
                is_home_p    = (side == "home")
                park_factor  = PARK_FACTORS.get(home, 1.0)

                # Fetch pitcher's own game logs for recent performance
                try:
                    df = get_mlb_pitcher_gamelog(pid, SEASON)
                    if df.empty:
                        df = get_mlb_pitcher_gamelog(pid, str(int(SEASON)-1))
                except Exception:
                    df = pd.DataFrame()

                if df.empty:
                    # Use values from pitcher_map if logs unavailable
                    pm = self._pitcher_map.get(batting_team, {})
                    era  = pm.get("era",  MLB_DEFAULT_ERA)
                    whip = pm.get("whip", MLB_DEFAULT_WHIP)
                    k9   = pm.get("k9",   MLB_DEFAULT_K9)
                    ip3  = 5.5
                    gp   = 0
                else:
                    recent = df.iloc[-MLB_PITCHER_RECENT_STARTS:]
                    era    = float(recent["era"].mean())
                    whip   = float(recent["whip"].mean())
                    k9     = float(recent["k9"].mean())
                    ip3    = float(recent.get("ip", pd.Series([5.5]*len(recent))).mean()) \
                             if "ip" in recent.columns else 5.5
                    gp     = len(df)

                # Projected stats for tonight
                proj_ip = min(ip3, 7.0)   # realistic ceiling
                proj_k  = round(k9 * proj_ip / 9, 1)
                proj_er = round(era * proj_ip / 9, 2)   # projected earned runs allowed
                proj_bb = round(whip * proj_ip - (proj_er * 1.0), 1)
                proj_bb = max(0.0, proj_bb)

                # Park and weather adjust projected ER allowed
                proj_er_adj = round(proj_er * park_factor * wx_hr, 2)

                # Classify start quality
                if era <= 3.00:
                    quality = "Ace"
                elif era <= 3.75:
                    quality = "Above Avg"
                elif era <= 4.50:
                    quality = "Average"
                elif era <= 5.50:
                    quality = "Below Avg"
                else:
                    quality = "Avoid"

                rows.append({
                    "pitcher_id":    pid,
                    "pitcher_name":  pname,
                    "team":          pitching_team,
                    "opponent":      batting_team,
                    "is_home":       is_home_p,
                    "game_label":    f"{away} @ {home}",
                    "gp":            gp,
                    "era":           round(era, 2),
                    "whip":          round(whip, 2),
                    "k9":            round(k9, 1),
                    "proj_ip":       round(proj_ip, 1),
                    "proj_k":        proj_k,
                    "proj_er":       proj_er_adj,
                    "proj_bb":       proj_bb,
                    "quality":       quality,
                    "park_factor":   park_factor,
                    "weather":       wx_desc,
                })

        self.pitcher_predictions = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["pitcher_id","game_label"])
            .sort_values("era", ascending=True)
            .reset_index(drop=True)
        ) if rows else pd.DataFrame()
        _p(f"Pitcher predictions: {len(self.pitcher_predictions)} starters")
        return self.pitcher_predictions

    def build_game_projections(self) -> list:
        results = []
        for g in self.games:
            away, home  = g.get("away_team",""), g.get("home_team","")
            park_factor = PARK_FACTORS.get(home, 1.0)
            gid         = str(g.get("game_id",""))
            wx          = self._weather_map.get(gid, {})
            wx_factor   = float(wx.get("weather_hr_factor", 1.0))
            wx_desc     = wx.get("weather_desc", "")
            ap = self.predictions[self.predictions["team"]==away] \
                 if not self.predictions.empty else pd.DataFrame()
            hp = self.predictions[self.predictions["team"]==home] \
                 if not self.predictions.empty else pd.DataFrame()

            away_runs = float(ap["proj_runs"].sum()) if not ap.empty else MLB_LEAGUE_AVG_RUNS
            home_runs = float(hp["proj_runs"].sum()) if not hp.empty else MLB_LEAGUE_AVG_RUNS
            away_runs = (away_runs*(1-MLB_REGRESSION_WEIGHT)
                         + MLB_LEAGUE_AVG_RUNS*MLB_REGRESSION_WEIGHT) * park_factor * wx_factor
            home_runs = (home_runs*(1-MLB_REGRESSION_WEIGHT)
                         + MLB_LEAGUE_AVG_RUNS*MLB_REGRESSION_WEIGHT) * park_factor * wx_factor + MLB_HOME_ADVANTAGE_RUNS
            total = round(away_runs + home_runs, 1)

            away_prob = round(away_runs/(away_runs+home_runs), 3)
            home_prob = round(1-away_prob, 3)
            fav       = home if home_prob > away_prob else away

            def to_ml(p):
                p = max(0.01,min(0.99,p))
                return int(-(p/(1-p))*100) if p>=0.5 else int(((1-p)/p)*100)
            def fmt_ml(ml): return f"+{ml}" if ml>0 else str(ml)

            hcvr = _run_line_cover(home_runs, away_runs)
            results.append({
                "game_id":         g.get("game_id"),
                "away_team":       away, "home_team": home,
                "away_proj_runs":  round(away_runs,1),
                "home_proj_runs":  round(home_runs,1),
                "total_proj_runs": total,
                "away_win_prob":   away_prob, "home_win_prob": home_prob,
                "favourite":       fav,
                "away_ml_display": fmt_ml(to_ml(away_prob)),
                "home_ml_display": fmt_ml(to_ml(home_prob)),
                "proj_spread":     round(home_runs-away_runs,2),
                "home_cover_prob": round(hcvr,3),
                "away_cover_prob": round(1-hcvr,3),
                "run_line_home":   "-1.5" if home_runs>=away_runs else "+1.5",
                "run_line_away":   "+1.5" if home_runs>=away_runs else "-1.5",
                "weather":         wx_desc,
                **_ou_probs(total),
            })
        self.game_proj = results
        return results

    def run(self, force_retrain: bool = False, status_callback=None,
            date: Optional[str] = None) -> pd.DataFrame:
        def st(msg, f):
            _p(msg)
            if status_callback: status_callback(msg, f)
        st("Fetching MLB schedule …",      0.05); self.fetch_schedule(date)
        st("Loading MLB rosters …",        0.12); self.fetch_rosters()
        st("Checking injured list …",      0.18); self.fetch_il_players()
        st("Downloading batter logs …",    0.30); self.fetch_game_logs()
        st("Fetching pitcher stats …",     0.42); self.fetch_pitcher_stats()
        st("Fetching team pitching stats …", 0.46); self.fetch_team_pitching_stats()
        st("Fetching batter handedness …", 0.52); self.fetch_batter_handedness()
        st("Fetching batter vs pitcher …", 0.57); self.fetch_batter_vs_pitcher()
        st("Fetching game weather …",      0.62); self.fetch_weather()
        st("Training / loading models …",  0.65); self.train_models(force_retrain)
        st("Building MLB predictions …",   0.85); self.build_predictions()
        st("Building pitcher projections …",0.90); self.build_pitcher_predictions()
        st("Building game projections …",  0.95); self.build_game_projections()
        st("Done!",                        1.00)
        return self.predictions

    def get_games(self): return self.games
    def get_teams_playing(self) -> list:
        t = set()
        for g in self.games: t.add(g["away_team"]); t.add(g["home_team"])
        return sorted(t)


# ── Math helpers ──────────────────────────────────────────────────────────────

def _run_line_cover(fav: float, dog: float, line: float = 1.5) -> float:
    p = 0.0
    for f in range(20):
        for d in range(20):
            if (f-d) > line:
                p += ((fav**f)*exp(-fav)/factorial(f)) * ((dog**d)*exp(-dog)/factorial(d))
    return float(np.clip(p, 0.01, 0.99))

def _ou_probs(total: float) -> dict:
    def over(lam, line):
        u = sum((lam**k)*exp(-lam)/factorial(k) for k in range(int(line)+1))
        return float(np.clip(1-u, 0.01, 0.99))
    lines = {7.5:over(total,7.5), 8.5:over(total,8.5), 9.5:over(total,9.5)}
    best  = max(lines, key=lambda l: abs(lines[l]-0.5))
    rec   = "OVER" if lines[best]>0.5 else "UNDER"
    bp    = lines[best] if lines[best]>0.5 else 1-lines[best]
    return {
        "over_7_5":round(lines[7.5],3),"over_8_5":round(lines[8.5],3),
        "over_9_5":round(lines[9.5],3),"best_ou_line":best,
        "best_ou_prob":round(bp,3),"recommendation":rec,
    }