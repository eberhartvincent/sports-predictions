"""
scripts/backtest.py — Backtesting engine comparing past predictions to actual results.

Fetches actual game results from official APIs and matches against
saved prediction snapshots in data/cache/predictions/history/.

Run from Codespaces terminal:
    python scripts/backtest.py --days 7
"""

import argparse, json, os, sys, requests, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
for _p in [_ROOT, _ROOT/"config", _ROOT/"core"/"pipelines", _ROOT/"data"/"api"]:
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

ET       = ZoneInfo("America/New_York")
PRED_DIR = _ROOT / "data" / "cache" / "predictions"
HIST_DIR = PRED_DIR / "history"
OUT_FILE = PRED_DIR / "backtest_results.json"

# Use full browser headers — helps bypass 403 on some endpoints
HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nhl.com/",
}


def log(msg): print(f"[{datetime.now(ET).strftime('%H:%M:%S')}] {msg}", flush=True)


def _hist_files(sport: str, days: int) -> list:
    """Return history parquet files for past dates only (not today)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if not HIST_DIR.exists():
        return []
    files = sorted(
        f for f in HIST_DIR.glob(f"{sport}_*.parquet")
        if f.stem.replace(f"{sport}_", "") < today
    )
    return files[-days:]


# ── Calibration metrics ───────────────────────────────────────────────────────

def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0; n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        total += abs(float(probs[mask].mean()) - float(labels[mask].mean())) * mask.sum()
    return total / max(n, 1)


def brier(probs, labels):
    return float(np.mean((probs - labels.astype(float))**2))


def auc(probs, labels):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs))
    except Exception:
        return None


def roi(labels, odds=150):
    dec = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
    wins = int(labels.sum()); total = len(labels)
    return (wins * (dec - 1) - (total - wins)) / total if total > 0 else 0.0


def tier_breakdown(df):
    out = {}
    for tier in ["Elite", "High", "Medium", "Low"]:
        sub = df[df["conf"] == tier]
        if len(sub) < 5: continue
        probs  = sub["pred_prob"].values.astype(float)
        labels = sub["scored"].values.astype(int)
        out[tier] = {
            "n":            int(len(sub)),
            "accuracy":     round(float(labels.mean()), 4),
            "avg_pred_prob":round(float(probs.mean()),  4),
            "ece":          round(ece(probs, labels),   4),
            "brier":        round(brier(probs, labels), 4),
            "auc":          round(auc(probs, labels) or 0, 4),
        }
    return out


# ── NHL Backtest ──────────────────────────────────────────────────────────────

def _nhl_fetch_results(date_str: str) -> dict:
    """
    Fetch NHL game results via ESPN (no cloud IP blocking).
    Returns {player_id: {goals, shots, assists}} dict.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(_ROOT / "data" / "api"))
        from nhl_api import get_game_results
        return get_game_results(date_str)
    except Exception as e:
        log(f"  NHL ESPN fetch error: {e}")
        return {}


def backtest_nhl(days: int) -> dict:
    log("NHL backtest…")
    files = _hist_files("nhl", days)
    if not files:
        return {"sport": "nhl",
                "message": "No history snapshots found. These are saved daily by the workflow.",
                "rows": []}

    log(f"  Found {len(files)} history files")
    rows = []
    api_failures = 0

    for pred_file in files:
        date_str = pred_file.stem.replace("nhl_", "")
        try:
            preds   = pd.read_parquet(pred_file)
            required_cols = {"player_id", "goal_probability", "confidence"}
            if not required_cols.issubset(preds.columns):
                log(f"  {date_str}: skipping — parquet missing prediction columns")
                continue
            results = _nhl_fetch_results(date_str)

            if not results:
                api_failures += 1
                log(f"  {date_str}: no API results (possible 403)")
                continue

            log(f"  {date_str}: {len(results)} player results from API")
            for pid, stats in results.items():
                match = preds[preds["player_id"] == pid]
                if match.empty: continue
                r = match.iloc[0]
                rows.append({
                    "date":       date_str,
                    "player_id":  pid,
                    "name":       r.get("player_name", ""),
                    "team":       r.get("team", ""),
                    "conf":       str(r.get("confidence", "Low")),
                    "pred_prob":  float(r.get("goal_probability", 0)),
                    "scored":     int(stats["goals"] > 0),
                    "goals":      stats["goals"],
                    "actual_sog": stats["shots"],
                    "pred_sog":   float(r.get("projected_sog", 0)),
                })
        except Exception as e:
            log(f"  NHL {date_str} error: {e}")

    msg = ""
    if api_failures > 0 and not rows:
        msg = (f"NHL API returned no data for {api_failures}/{len(files)} dates. "
               "This usually means the API is blocking cloud IPs. "
               "Try running the backtest from your local machine or Codespaces directly.")
    elif api_failures > 0:
        msg = f"Note: {api_failures} date(s) had no API data — results may be incomplete."

    log(f"  NHL: {len(rows)} rows matched")
    return {"sport": "nhl", "rows": rows, "message": msg,
            "files_found": len(files), "api_failures": api_failures}


def aggregate_nhl(rows: list, meta: dict = None) -> dict:
    meta = meta or {}
    if not rows:
        return {
            "files_found":  meta.get("files_found", 0),
            "api_failures": meta.get("api_failures", 0),
            "message":      meta.get("message", ""),
        }
    df  = pd.DataFrame(rows)
    out = {
        "sport":    "nhl",
        "n_games":  int(df["date"].nunique()),
        "n_rows":   int(len(df)),
        "files_found":  meta.get("files_found", 0),
        "api_failures": meta.get("api_failures", 0),
        "message":      meta.get("message", ""),
    }
    probs  = df["pred_prob"].values.astype(float)
    labels = df["scored"].values.astype(int)
    out["overall_accuracy"] = round(float(labels.mean()), 4)
    out["overall_brier"]    = round(brier(probs, labels), 4)
    out["overall_ece"]      = round(ece(probs, labels),   4)
    out["overall_auc"]      = round(auc(probs, labels) or 0, 4)
    out["tiers"]            = tier_breakdown(df)

    if "actual_sog" in df.columns and "pred_sog" in df.columns:
        out["sog_mae"] = round(float((df["pred_sog"] - df["actual_sog"]).abs().mean()), 3)

    elite = df[df["conf"] == "Elite"]
    if len(elite) >= 5:
        el = elite["scored"].values
        out["elite_n"]           = int(len(elite))
        out["elite_accuracy"]    = round(float(el.mean()), 4)
        out["elite_roi_plus150"] = round(roi(el, 150), 4)
        out["elite_roi_plus130"] = round(roi(el, 130), 4)
        out["elite_roi_plus110"] = round(roi(el, 110), 4)

    cal_bins = []
    for lo in np.arange(0, 1, 0.1):
        mask = (probs >= lo) & (probs < lo + 0.1)
        if mask.sum() >= 5:
            cal_bins.append({
                "bin_lo":     round(float(lo), 1),
                "bin_hi":     round(float(lo + 0.1), 1),
                "n":          int(mask.sum()),
                "avg_pred":   round(float(probs[mask].mean()), 3),
                "avg_actual": round(float(labels[mask].mean()), 3),
            })
    out["calibration_curve"] = cal_bins
    return out


# ── MLB Backtest ──────────────────────────────────────────────────────────────

def backtest_mlb(days: int) -> dict:
    log("MLB backtest…")
    files = _hist_files("mlb", days)
    if not files:
        return {"sport": "mlb",
                "message": "No history snapshots found.",
                "rows": []}

    log(f"  Found {len(files)} history files")
    rows = []; api_failures = 0
    MLB_API = "https://statsapi.mlb.com/api/v1"

    for pred_file in files:
        date_str = pred_file.stem.replace("mlb_", "")
        try:
            preds = pd.read_parquet(pred_file)
            required_cols = {"player_id", "proj_hits", "proj_hr", "confidence"}
            if not required_cols.issubset(preds.columns):
                log(f"  {date_str}: skipping — parquet missing prediction columns")
                continue

            # Step 1: get game IDs for this date
            r = requests.get(f"{MLB_API}/schedule",
                params={"sportId": 1, "date": date_str},
                headers=HEADERS, timeout=20)
            if r.status_code != 200:
                api_failures += 1
                log(f"  {date_str}: MLB API {r.status_code}")
                continue

            game_ids = []
            for d in r.json().get("dates", []):
                for game in d.get("games", []):
                    gid = game.get("gamePk")
                    if gid: game_ids.append(gid)

            # Step 2: fetch boxscore per game (same as pipeline does)
            matched = 0
            for gid in game_ids:
                bs = requests.get(f"{MLB_API}/game/{gid}/boxscore",
                    headers=HEADERS, timeout=15)
                if bs.status_code != 200: continue
                bs_data = bs.json()
                for side in ("away", "home"):
                    team_data = bs_data.get("teams", {}).get(side, {})
                    for pid_str, pdata in team_data.get("players", {}).items():
                        pid  = pdata.get("person", {}).get("id")
                        stat = pdata.get("stats", {}).get("batting", {})
                        ab   = int(stat.get("atBats", 0))
                        if ab == 0 or pid is None: continue
                        match = preds[preds["player_id"] == int(pid)]
                        if match.empty: continue
                        pr = match.iloc[0]
                        rows.append({
                            "date":        date_str,
                            "player_id":   int(pid),
                            "name":        pr.get("player_name", ""),
                            "team":        pr.get("team", ""),
                            "conf":        str(pr.get("confidence", "Low")),
                            "pred_h":      float(pr.get("proj_hits",  0)),
                            "pred_hr":     float(pr.get("proj_hr",    0)),
                            "pred_rbi":    float(pr.get("proj_rbi",   0)),
                            "pred_runs":   float(pr.get("proj_runs",  0)),
                            "pred_tb":     float(pr.get("proj_tb",    0)),
                            "pred_k":      float(pr.get("proj_k",     0)),
                            "pred_hrr":    float(pr.get("proj_hrr",   0)),
                            "actual_h":    int(stat.get("hits",       0)),
                            "actual_hr":   int(stat.get("homeRuns",   0)),
                            "actual_rbi":  int(stat.get("rbi",        0)),
                            "actual_runs": int(stat.get("runs",       0)),
                            "actual_tb":   int(stat.get("totalBases", 0)),
                            "actual_k":    int(stat.get("strikeOuts", 0)),
                            "actual_hrr":  int(stat.get("hits", 0)) +
                                           int(stat.get("rbi",  0)) +
                                           int(stat.get("runs", 0)),
                            "hr_scored":   int(int(stat.get("homeRuns", 0)) > 0),
                        })
                        matched += 1
                time.sleep(0.1)
            log(f"  {date_str}: {matched} players matched from {len(game_ids)} games")
        except Exception as e:
            log(f"  MLB {date_str} error: {e}")

    msg = ""
    if api_failures > 0 and not rows:
        msg = f"MLB API returned no data for {api_failures}/{len(files)} dates."
    elif api_failures > 0:
        msg = f"Note: {api_failures} date(s) had no API data."

    log(f"  MLB: {len(rows)} rows matched")
    return {"sport": "mlb", "rows": rows, "message": msg,
            "files_found": len(files), "api_failures": api_failures}


def aggregate_mlb(rows: list, meta: dict = None) -> dict:
    meta = meta or {}
    if not rows:
        return {
            "files_found":  meta.get("files_found", 0),
            "api_failures": meta.get("api_failures", 0),
            "message":      meta.get("message", ""),
        }
    df  = pd.DataFrame(rows)
    out = {
        "sport":   "mlb",
        "n_games": int(df["date"].nunique()),
        "n_rows":  int(len(df)),
        "files_found":  meta.get("files_found", 0),
        "api_failures": meta.get("api_failures", 0),
        "message":      meta.get("message", ""),
    }
    for stat, pc, ac in [
        ("h",   "pred_h",   "actual_h"),
        ("hr",  "pred_hr",  "actual_hr"),
        ("rbi", "pred_rbi", "actual_rbi"),
        ("runs","pred_runs","actual_runs"),
        ("tb",  "pred_tb",  "actual_tb"),
        ("k",   "pred_k",   "actual_k"),
        ("hrr", "pred_hrr", "actual_hrr"),
    ]:
        if pc in df and ac in df:
            out[f"mae_{stat}"]  = round(float((df[pc] - df[ac]).abs().mean()), 3)
            out[f"rmse_{stat}"] = round(float(np.sqrt(((df[pc]-df[ac])**2).mean())), 3)
            out[f"bias_{stat}"] = round(float((df[pc] - df[ac]).mean()), 3)

    if "hr_scored" in df:
        hp = df["pred_hr"].values.astype(float)
        hl = df["hr_scored"].values.astype(int)
        out["hr_brier"]        = round(brier(hp, hl), 4)
        out["hr_ece"]          = round(ece(hp, hl),   4)
        out["hr_auc"]          = round(auc(hp, hl) or 0, 4)
        out["hr_direction_acc"]= round(float(((hp >= 0.08) == (hl == 1)).mean()), 4)

    tiers = {}
    for tier in ["Elite", "High", "Medium", "Low"]:
        sub = df[df["conf"] == tier]
        if len(sub) < 5: continue
        t = {"n": int(len(sub))}
        for stat, pc, ac in [("h","pred_h","actual_h"),
                              ("hr","pred_hr","actual_hr"),
                              ("hrr","pred_hrr","actual_hrr")]:
            if pc in sub and ac in sub:
                t[f"mae_{stat}"]        = round(float((sub[pc]-sub[ac]).abs().mean()),3)
                t[f"avg_pred_{stat}"]   = round(float(sub[pc].mean()),3)
                t[f"avg_actual_{stat}"] = round(float(sub[ac].mean()),3)
                t[f"bias_{stat}"]       = round(float((sub[pc]-sub[ac]).mean()),3)
        tiers[tier] = t
    out["tiers"] = tiers

    cal = []
    for lo in [0, 0.25, 0.50, 0.75]:
        mask = (df["pred_h"] >= lo) & (df["pred_h"] < lo + 0.25)
        if mask.sum() >= 5:
            cal.append({
                "bin":          f"{lo:.2f}-{lo+0.25:.2f}",
                "n":            int(mask.sum()),
                "avg_pred_h":   round(float(df[mask]["pred_h"].mean()),  3),
                "avg_actual_h": round(float(df[mask]["actual_h"].mean()), 3),
            })
    out["hit_calibration_curve"] = cal
    return out


# ── NBA Backtest ──────────────────────────────────────────────────────────────

def backtest_nba(days: int) -> dict:
    log("NBA backtest…")
    files = _hist_files("nba", days)
    if not files:
        return {"sport": "nba",
                "message": "No history snapshots found.",
                "rows": []}

    log(f"  Found {len(files)} history files")
    rows = []; api_failures = 0
    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    for pred_file in files:
        date_str = pred_file.stem.replace("nba_", "")
        date_fmt = date_str.replace("-", "")
        try:
            preds = pd.read_parquet(pred_file)
            required_cols = {"player_id", "proj_pts", "confidence"}
            if not required_cols.issubset(preds.columns):
                log(f"  {date_str}: skipping — parquet missing prediction columns")
                continue

            # Step 1: get game IDs from scoreboard
            url = f"{ESPN_BASE}/scoreboard?dates={date_fmt}"
            r   = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                api_failures += 1
                log(f"  {date_str}: ESPN API {r.status_code}")
                continue

            game_ids = [e.get("id") for e in r.json().get("events", []) if e.get("id")]

            # Step 2: fetch box score per game via summary endpoint
            matched = 0
            for gid in game_ids:
                bs = requests.get(f"{ESPN_BASE}/summary",
                    params={"event": gid}, headers=HEADERS, timeout=15)
                if bs.status_code != 200: continue
                bs_data = bs.json()

                # ESPN summary has boxscore.players array
                for team_data in bs_data.get("boxscore", {}).get("players", []):
                    for stat_group in team_data.get("statistics", []):
                        for athlete in stat_group.get("athletes", []):
                            pid   = athlete.get("athlete", {}).get("id")
                            stats = athlete.get("stats", [])
                            if not pid or not stats: continue
                            # ESPN stats order: MIN,FG,3PT,FT,OREB,DREB,REB,AST,STL,BLK,TO,PF,+/-,PTS
                            try:
                                mins  = float(str(stats[0]).replace(":",".")
                                              .split(".")[0]) if stats else 0
                                pts   = int(float(stats[13])) if len(stats) > 13 else 0
                                reb   = int(float(stats[6]))  if len(stats) > 6  else 0
                                ast_  = int(float(stats[7]))  if len(stats) > 7  else 0
                                stl   = int(float(stats[8]))  if len(stats) > 8  else 0
                                blk   = int(float(stats[9]))  if len(stats) > 9  else 0
                                fg3m  = int(float(str(stats[2]).split("-")[0]))                                         if len(stats) > 2 and "-" in str(stats[2]) else 0
                            except (ValueError, IndexError):
                                continue
                            if mins < 5: continue
                            match = preds[preds["player_id"] == int(pid)]
                            if match.empty: continue
                            pr = match.iloc[0]
                            rows.append({
                                "date":          date_str,
                                "player_id":     int(pid),
                                "name":          pr.get("player_name", ""),
                                "team":          pr.get("team", ""),
                                "conf":          str(pr.get("confidence", "Low")),
                                "pred_pts":      float(pr.get("proj_pts",    0)),
                                "pred_reb":      float(pr.get("proj_reb",    0)),
                                "pred_ast":      float(pr.get("proj_ast",    0)),
                                "pred_fg3m":     float(pr.get("proj_fg3m",   0)),
                                "pred_stocks":   float(pr.get("proj_stocks", 0)),
                                "pred_dd":       float(pr.get("proj_dd",     0)),
                                "actual_pts":    pts,
                                "actual_reb":    reb,
                                "actual_ast":    ast_,
                                "actual_fg3m":   fg3m,
                                "actual_stocks": stl + blk,
                                "actual_dd":     int(
                                    (pts >= 10 and reb >= 10) or
                                    (pts >= 10 and ast_ >= 10) or
                                    (reb >= 10 and ast_ >= 10)
                                ),
                            })
                            matched += 1
            log(f"  {date_str}: {matched} players matched")
        except Exception as e:
            log(f"  NBA {date_str} error: {e}")

    msg = ""
    if api_failures > 0 and not rows:
        msg = f"NBA API returned no data for {api_failures}/{len(files)} dates."
    elif api_failures > 0:
        msg = f"Note: {api_failures} date(s) had no API data."

    log(f"  NBA: {len(rows)} rows matched")
    return {"sport": "nba", "rows": rows, "message": msg,
            "files_found": len(files), "api_failures": api_failures}


def aggregate_nba(rows: list, meta: dict = None) -> dict:
    meta = meta or {}
    if not rows:
        return {
            "files_found":  meta.get("files_found", 0),
            "api_failures": meta.get("api_failures", 0),
            "message":      meta.get("message", ""),
        }
    df  = pd.DataFrame(rows)
    out = {
        "sport":   "nba",
        "n_games": int(df["date"].nunique()),
        "n_rows":  int(len(df)),
        "files_found":  meta.get("files_found", 0),
        "api_failures": meta.get("api_failures", 0),
        "message":      meta.get("message", ""),
    }
    for stat, pc, ac in [
        ("pts",    "pred_pts",    "actual_pts"),
        ("reb",    "pred_reb",    "actual_reb"),
        ("ast",    "pred_ast",    "actual_ast"),
        ("fg3m",   "pred_fg3m",   "actual_fg3m"),
        ("stocks", "pred_stocks", "actual_stocks"),
    ]:
        if pc in df and ac in df:
            out[f"mae_{stat}"]  = round(float((df[pc]-df[ac]).abs().mean()), 3)
            out[f"bias_{stat}"] = round(float((df[pc]-df[ac]).mean()),       3)

    if "pred_dd" in df and "actual_dd" in df:
        dp = df["pred_dd"].values.astype(float)
        dl = df["actual_dd"].values.astype(int)
        out["dd_brier"]         = round(brier(dp, dl), 4)
        out["dd_ece"]           = round(ece(dp, dl),   4)
        out["dd_direction_acc"] = round(float(((dp >= 0.35) == (dl == 1)).mean()), 4)

    tiers = {}
    for tier in ["Elite", "High", "Medium", "Low"]:
        sub = df[df["conf"] == tier]
        if len(sub) < 5: continue
        t = {"n": int(len(sub))}
        for stat, pc, ac in [("pts","pred_pts","actual_pts"),
                              ("reb","pred_reb","actual_reb"),
                              ("ast","pred_ast","actual_ast")]:
            if pc in sub and ac in sub:
                t[f"mae_{stat}"]        = round(float((sub[pc]-sub[ac]).abs().mean()),3)
                t[f"avg_pred_{stat}"]   = round(float(sub[pc].mean()),3)
                t[f"avg_actual_{stat}"] = round(float(sub[ac].mean()),3)
        tiers[tier] = t
    out["tiers"] = tiers
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(days: int = 30) -> dict:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Backtesting last {days} days…")

    results = {}
    for sport, bt_fn, agg_fn in [
        ("nhl", backtest_nhl, aggregate_nhl),
        ("mlb", backtest_mlb, aggregate_mlb),
        ("nba", backtest_nba, aggregate_nba),
    ]:
        data = bt_fn(days)
        agg  = agg_fn(data.get("rows", []))
        results[sport] = {
            "aggregate":    agg,
            "updated_at":   datetime.now(ET).isoformat(),
            "days":         days,
            "n_rows":       len(data.get("rows", [])),
            "files_found":  data.get("files_found", 0),
            "api_failures": data.get("api_failures", 0),
            "message":      data.get("message", ""),
        }
        log(f"  {sport.upper()}: {len(data.get('rows',[]))} rows")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"Saved → {OUT_FILE}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    run_backtest(args.days)
