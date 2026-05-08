"""
scripts/send_predictions_email.py
Sends a daily HTML email digest of ELITE predictions only, all categories.

Env vars (set as GitHub secrets):
    EMAIL_SENDER      — Gmail address
    EMAIL_PASSWORD    — Gmail app password (16 chars)
    EMAIL_RECIPIENTS  — comma-separated recipient list
"""

import json, os, sys, smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

ET           = ZoneInfo("America/New_York")
PRED_DIR     = _ROOT / "data" / "cache" / "predictions"
TODAY        = datetime.now(ET).strftime("%Y-%m-%d")
TODAY_LONG   = datetime.now(ET).strftime("%A, %B %d, %Y")
TODAY_SHORT  = datetime.now(ET).strftime("%b %d")
TOP_N        = 10   # overall list
TOP_CAT      = 10   # per-category lists


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(sport: str) -> dict:
    out = {"predictions": pd.DataFrame(), "pitcher_predictions": pd.DataFrame(),
           "game_projections": [], "games": [], "meta": {}}
    try:
        for key, fname, loader in [
            ("predictions",        f"{sport}_predictions.parquet",        pd.read_parquet),
            ("pitcher_predictions",f"{sport}_pitcher_predictions.parquet", pd.read_parquet),
        ]:
            f = PRED_DIR / fname
            if f.exists():
                out[key] = loader(f)
        for key, fname in [
            ("game_projections", f"{sport}_game_projections.json"),
            ("games",            f"{sport}_games.json"),
            ("meta",             f"{sport}_meta.json"),
        ]:
            f = PRED_DIR / fname
            if f.exists():
                out[key] = json.loads(f.read_text())
    except Exception as e:
        print(f"  Warning loading {sport}: {e}")
    return out


def stat(val, fmt=".2f", color="#212529", bold=False) -> str:
    s = f'<span style="color:{color};{"font-weight:700;" if bold else ""}'
    s += f'font-size:13px;">{val:{fmt}}</span>'
    return s


def badge(text, bg) -> str:
    return (f'<span style="background:{bg};color:#fff;padding:2px 8px;'
            f'border-radius:10px;font-size:11px;font-weight:700;">{text}</span>')


def section(emoji, title, subtitle="") -> str:
    sub = f'<div style="color:#6c757d;font-size:12px;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    return (f'<div style="margin:28px 0 14px;padding:12px 16px;'
            f'background:linear-gradient(135deg,#1a237e11,#1565c011);'
            f'border-left:4px solid #1565c0;border-radius:0 8px 8px 0;">'
            f'<h2 style="margin:0;font-size:16px;color:#1a237e;font-weight:700;">'
            f'{emoji}&nbsp; {title}</h2>{sub}</div>')


def table_header(*cols) -> str:
    cells = "".join(
        f'<th align="left" style="padding:8px 8px;color:#495057;'
        f'font-size:11px;text-transform:uppercase;letter-spacing:.05em;'
        f'border-bottom:2px solid #1565c0;white-space:nowrap;">{c}</th>'
        for c in cols
    )
    return f'<tr style="background:#f8f9fa;">{cells}</tr>'


def row_start(i) -> str:
    bg = "#ffffff" if i % 2 else "#f8f9fa"
    return f'<tr style="background:{bg};border-bottom:1px solid #e9ecef;">'


def td(content, bold=False, color="#212529", align="left") -> str:
    return (f'<td style="padding:8px;color:{color};font-size:13px;'
            f'{"font-weight:700;" if bold else ""}text-align:{align};">'
            f'{content}</td>')


def no_data(msg) -> str:
    return (f'<p style="color:#6c757d;font-size:13px;padding:14px 16px;'
            f'background:#f8f9fa;border-radius:6px;margin:8px 0;">'
            f'ℹ️ {msg}</p>')


def proj_cell(val, fmt=".2f", color="#1565c0") -> str:
    return td(f'<span style="font-weight:700;color:{color};">{val:{fmt}}</span>')


# ── NHL ───────────────────────────────────────────────────────────────────────

def nhl_section(data: dict) -> str:
    html = section("🏒", "NHL — Today's Top Picks", f"Top {TOP_CAT} per category")
    df = data["predictions"]
    if df.empty:
        return html + no_data("No NHL data available.")

    def _cat_table(title, sort_col, val_col, val_label, val_fmt=".3f", color="#c0392b"):
        if sort_col not in df.columns: return ""
        top = df.sort_values(sort_col, ascending=False).head(TOP_CAT)
        h  = f'<div style="margin-bottom:16px;">'
        h += f'<h3 style="font-size:13px;color:#495057;margin:0 0 6px;">{title}</h3>'
        h += '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        h += table_header("#","Player","Team","Opp",val_label,"Conf","Szn G/A")
        for i,(_, r) in enumerate(top.iterrows(),1):
            val  = float(r.get(sort_col, 0))
            conf = str(r.get("conf_goals" if "goal" in sort_col else "conf_sog", r.get("confidence","Low")))
            cc   = "#c0392b" if conf=="Elite" else "#e67e22" if conf=="High" else "#2980b9"
            sg   = int(r.get("season_goals",0)); sa = int(r.get("season_assists",0))
            h += row_start(i)
            h += td(f'<span style="color:#6c757d;font-weight:700;">#{i}</span>')
            h += td(f'<strong>{r.get("player_name","")}</strong>')
            h += td(r.get("team",""), color="#1565c0", bold=True)
            h += td(r.get("opponent",""), color="#6c757d")
            h += td(f'<span style="color:{color};font-weight:800;">{val:{val_fmt}}</span>')
            h += td(f'<span style="color:{cc};font-weight:700;">{conf}</span>')
            h += td(f'{sg}G / {sa}A', color="#6c757d")
            h += '</tr>'
        h += '</table></div>'
        return h

    html += _cat_table(f"🥅 Top {TOP_CAT} — Goal Scorers",    "goal_probability",  "goal_probability",  "Goal Prob", ".3f", "#c0392b")
    html += _cat_table(f"🏒 Top {TOP_CAT} — Shots on Goal",   "projected_sog",     "projected_sog",     "Proj SOG",  ".1f", "#2980b9")
    html += _cat_table(f"🍎 Top {TOP_CAT} — Assists",         "projected_assists",  "projected_assists", "Proj Ast",  ".2f", "#16a085")
    html += _cat_table(f"⭐ Top {TOP_CAT} — Points",          "projected_points",   "projected_points",  "Proj Pts",  ".2f", "#8e44ad")

    if data.get("game_projections") and len(data["game_projections"]) > 0:
        html += _nhl_game_proj(data["game_projections"])
    return html


def _nhl_game_proj(projs: list) -> str:
    html = ('<div style="margin-top:18px;">'
            '<h3 style="font-size:13px;color:#495057;margin:0 0 8px;">'
            '🎰 NHL Game Projections</h3>'
            '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">')
    html += table_header("Matchup","Away Win","Home Win","Total","Best Bet")
    for i, p in enumerate(projs):
        away = p.get("away_team",""); home = p.get("home_team","")
        rec  = p.get("recommendation",""); line = p.get("best_ou_line","")
        prob = float(p.get("best_ou_prob",0))
        rc   = "#c0392b" if rec=="OVER" else "#2980b9"
        html += row_start(i)
        html += td(f'<strong>{away} @ {home}</strong>')
        html += td(f'{p.get("away_ml_display","")} '
                   f'<span style="color:#6c757d;">({float(p.get("away_win_prob",0)):.0%})</span>')
        html += td(f'{p.get("home_ml_display","")} '
                   f'<span style="color:#6c757d;">({float(p.get("home_win_prob",0)):.0%})</span>')
        html += td(f'{float(p.get("proj_total",0)):.1f}')
        html += td(f'<span style="color:{rc};font-weight:700;">'
                   f'{rec} {line} ({prob:.0%})</span>')
        html += '</tr>'
    html += '</table></div>'
    return html


def _mlb_pitcher_table(preds: "pd.DataFrame") -> str:
    if preds.empty: return ""

    def _ptable(title, rows):
        h  = f'<div style="margin-bottom:16px;">'
        h += f'<h3 style="font-size:13px;color:#495057;margin:0 0 6px;">{title}</h3>'
        h += '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        h += table_header("Pitcher","Team","Opp","Hand","Value","ERA","Proj IP")
        for i,(_, r) in enumerate(rows.iterrows()):
            name = r.get("pitcher_name", r.get("player_name",""))
            h += row_start(i)
            h += td(f'<strong>{name}</strong>')
            h += td(r.get("team",""), color="#1565c0", bold=True)
            h += td(r.get("opponent",""), color="#6c757d")
            h += td(r.get("pitcher_hand",""), color="#6c757d")
            yield_col = r.get("proj_k",0)
            h += proj_cell(float(yield_col), color="#8e44ad")
            h += td(f'{float(r.get("era",0)):.2f}', color="#6c757d")
            h += proj_cell(float(r.get("proj_ip",0)), ".1f", color="#16a085")
            h += '</tr>'
        h += '</table></div>'
        return h

    html  = '<div style="margin-top:18px;">'
    # Top K pitchers (sorted by proj strikeouts)
    top_k   = preds.sort_values("proj_k",  ascending=False).head(TOP_CAT)
    # Best ERA matchups (sorted by ERA ascending = easiest to hit against)
    top_era = preds.sort_values("era",     ascending=True ).head(TOP_CAT)

    html += _ptable(f"🔥 Top {TOP_CAT} Pitchers — Strikeouts", top_k)
    html += _ptable(f"💥 Top {TOP_CAT} Pitchers — Best ERA (hitter matchup)", top_era)
    html += '</div>'
    return html


def _mlb_game_proj(projs: list) -> str:
    if not projs: return ""
    html = ('<div style="margin-top:18px;">'
            '<h3 style="font-size:13px;color:#495057;margin:0 0 8px;">'
            '🎰 MLB Game Projections</h3>'
            '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">')
    html += table_header("Matchup","Favourite","Total","Best Bet","Weather")
    for i, p in enumerate(projs):
        away = p.get("away_team",""); home = p.get("home_team","")
        rec  = p.get("recommendation",""); line = p.get("best_ou_line","")
        prob = float(p.get("best_ou_prob",0))
        fav  = p.get("favourite","")
        rc   = "#c0392b" if rec=="OVER" else "#2980b9"
        weather = p.get("weather","")
        html += row_start(i)
        html += td(f'<strong>{away} @ {home}</strong>')
        html += td(f'<span style="font-weight:700;">{fav}</span>', color="#27ae60")
        html += td(f'{float(p.get("total_proj_runs",0)):.1f} runs')
        html += td(f'<span style="color:{rc};font-weight:700;">{rec} {line} ({prob:.0%})</span>')
        html += td(weather[:30] if weather else "—", color="#6c757d")
        html += '</tr>'
    html += '</table></div>'
    return html


def _nba_game_proj(projs: list) -> str:
    if not projs: return ""
    html = ('<div style="margin-top:18px;">'
            '<h3 style="font-size:13px;color:#495057;margin:0 0 8px;">'
            '🎰 NBA Game Projections</h3>'
            '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">')
    html += table_header("Matchup","Favourite","Total","Spread")
    for i, p in enumerate(projs):
        away = p.get("away_team",""); home = p.get("home_team","")
        fav  = p.get("favourite","")
        spread = p.get("proj_spread","")
        html += row_start(i)
        html += td(f'<strong>{away} @ {home}</strong>')
        html += td(f'<span style="font-weight:700;">{fav}</span>', color="#27ae60")
        html += td(f'{float(p.get("total_proj_pts",0)):.0f} pts')
        html += td(str(spread), color="#6c757d")
        html += '</tr>'
    html += '</table></div>'
    return html


# ── MLB ───────────────────────────────────────────────────────────────────────

def mlb_section(data: dict) -> str:
    html = section("⚾", "MLB — Today's Top Picks", f"Top {TOP_CAT} per category")
    df = data["predictions"]
    if df.empty:
        return html + no_data("No MLB data available.")

    def _cat(title, sort_col, conf_col, val_label, val_fmt, color):
        if sort_col not in df.columns: return ""
        top = df.sort_values(sort_col, ascending=False).head(TOP_CAT)
        h  = f'<div style="margin-bottom:16px;">'
        h += f'<h3 style="font-size:13px;color:#495057;margin:0 0 6px;">{title}</h3>'
        h += '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        h += table_header("#","Player","Team","Opp",val_label,"Conf","BvP")
        for i,(_, r) in enumerate(top.iterrows(),1):
            val  = float(r.get(sort_col, 0))
            conf = str(r.get(conf_col, r.get("confidence","Low")))
            cc   = "#c0392b" if conf=="Elite" else "#e67e22" if conf=="High" else "#2980b9"
            bvp  = int(r.get("bvp_ab",0)); bvp_hr = int(r.get("bvp_hr",0))
            bvp_s= f"{bvp}AB {bvp_hr}HR" if bvp>=5 else "—"
            h += row_start(i)
            h += td(f'<span style="color:#6c757d;font-weight:700;">#{i}</span>')
            h += td(f'<strong>{r.get("player_name","")}</strong>')
            h += td(r.get("team",""), color="#1565c0", bold=True)
            h += td(r.get("opponent",""), color="#6c757d")
            h += td(f'<span style="color:{color};font-weight:800;">{val:{val_fmt}}</span>')
            h += td(f'<span style="color:{cc};font-weight:700;">{conf}</span>')
            h += td(bvp_s, color="#6c757d")
            h += '</tr>'
        h += '</table></div>'
        return h

    html += _cat(f"🎯 Top {TOP_CAT} — H+R+RBI",    "proj_hrr",  "conf_hrr",  "Proj HRR",  ".2f", "#f59e0b")
    html += _cat(f"🏆 Top {TOP_CAT} — Hits",        "proj_hits", "conf_hits", "Proj H",    ".3f", "#27ae60")
    html += _cat(f"💣 Top {TOP_CAT} — Home Runs",   "proj_hr",   "conf_hr",   "HR Prob",   ".3f", "#c0392b")
    html += _cat(f"🏃 Top {TOP_CAT} — RBI",         "proj_rbi",  "conf_rbi",  "Proj RBI",  ".2f", "#e67e22")
    html += _cat(f"⚡ Top {TOP_CAT} — Runs Scored", "proj_runs", "conf_runs", "Proj R",    ".2f", "#16a085")
    html += _cat(f"💥 Top {TOP_CAT} — Total Bases",  "proj_tb",   "conf_hrr",  "Proj TB",   ".2f", "#8e44ad")

    preds = data.get("pitcher_predictions", pd.DataFrame())
    if not preds.empty:
        html += _mlb_pitcher_table(preds)
    if data.get("game_projections") and len(data["game_projections"]) > 0:
        html += _mlb_game_proj(data["game_projections"])
    return html


# ── NBA ───────────────────────────────────────────────────────────────────────

def nba_section(data: dict) -> str:
    html = section("🏀", "NBA — Today's Top Picks", f"Top {TOP_CAT} per category")
    df = data["predictions"]
    if df.empty:
        return html + no_data("No NBA data available.")

    def _cat(title, sort_col, conf_col, val_label, val_fmt, color):
        if sort_col not in df.columns: return ""
        top = df.sort_values(sort_col, ascending=False).head(TOP_CAT)
        h  = f'<div style="margin-bottom:16px;">'
        h += f'<h3 style="font-size:13px;color:#495057;margin:0 0 6px;">{title}</h3>'
        h += '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        h += table_header("#","Player","Team","Opp",val_label,"Conf","Szn Avg")
        for i,(_, r) in enumerate(top.iterrows(),1):
            val  = float(r.get(sort_col, 0))
            conf = str(r.get(conf_col, r.get("confidence","Low")))
            cc   = "#c0392b" if conf=="Elite" else "#e67e22" if conf=="High" else "#2980b9"
            spts = float(r.get("season_pts",0))
            sreb = float(r.get("season_reb",0))
            sast = float(r.get("season_ast",0))
            h += row_start(i)
            h += td(f'<span style="color:#6c757d;font-weight:700;">#{i}</span>')
            h += td(f'<strong>{r.get("player_name","")}</strong>')
            h += td(r.get("team",""), color="#1565c0", bold=True)
            h += td(r.get("opponent",""), color="#6c757d")
            h += td(f'<span style="color:{color};font-weight:800;">{val:{val_fmt}}</span>')
            h += td(f'<span style="color:{cc};font-weight:700;">{conf}</span>')
            h += td(f'{spts:.0f}P/{sreb:.0f}R/{sast:.0f}A', color="#6c757d")
            h += '</tr>'
        h += '</table></div>'
        return h

    html += _cat(f"🏀 Top {TOP_CAT} — Points",       "proj_pts",    "conf_pts",    "Proj Pts", ".1f", "#1565c0")
    html += _cat(f"💪 Top {TOP_CAT} — Rebounds",     "proj_reb",    "conf_reb",    "Proj Reb", ".1f", "#16a085")
    html += _cat(f"🎯 Top {TOP_CAT} — Assists",      "proj_ast",    "conf_ast",    "Proj Ast", ".1f", "#e67e22")
    html += _cat(f"🔥 Top {TOP_CAT} — 3-Pointers",  "proj_fg3m",   "conf_fg3m",   "Proj 3PM", ".1f", "#8e44ad")
    html += _cat(f"🛡 Top {TOP_CAT} — Stl+Blk",     "proj_stocks", "conf_stocks", "Proj Stk", ".1f", "#c0392b")
    html += _cat(f"⭐ Top {TOP_CAT} — Double-Double","proj_dd",     "conf_dd",     "DD Prob",  ".0%", "#f59e0b")

    if data.get("game_projections") and len(data["game_projections"]) > 0:
        html += _nba_game_proj(data["game_projections"])
    return html

def _nba_game_proj(projs: list) -> str:
    html = ('<div style="margin-top:18px;">'
            '<h3 style="font-size:13px;color:#495057;margin:0 0 8px;">'
            '🎰 NBA Game Projections</h3>'
            '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">')
    html += table_header("Matchup","Away Win","Home Win","Total Pts","Spread")
    for i, p in enumerate(projs):
        away = p.get("away_team",""); home = p.get("home_team","")
        total = float(p.get("total_proj_pts", 0))
        html += row_start(i)
        html += td(f'<strong>{away} @ {home}</strong>')
        html += td(f'{p.get("away_ml_display","")} '
                   f'<span style="color:#6c757d;">({float(p.get("away_win_prob",0)):.0%})</span>')
        html += td(f'{p.get("home_ml_display","")} '
                   f'<span style="color:#6c757d;">({float(p.get("home_win_prob",0)):.0%})</span>')
        html += td(f'<span style="font-weight:700;">{total:.1f}</span>')
        html += td(f'{p.get("puck_line_away","")}&nbsp;'
                   f'<span style="color:#6c757d;font-size:11px;">'
                   f'({float(p.get("away_cover_prob",0)):.0%})</span>')
        html += '</tr>'
    html += '</table></div>'
    return html


# ── Build full email ──────────────────────────────────────────────────────────

def _apply_prob_ceiling(df):
    """
    Vectorised physics-based probability ceiling.
    Also writes conf_goals and conf_sog so the email shows category-specific confidence.
    """
    if df is None or df.empty or "goal_probability" not in df.columns:
        return df
    import numpy as np

    K = 150; LEAGUE_SH = 0.104; BEST_CASE = 0.130

    shots = df.get("season_shots_pg",  df.get("rolling_5g_shots",
            pd.Series([0]*len(df), index=df.index))).fillna(0).values.astype(float)
    gp    = df.get("gp_season", df.get("gp",
            pd.Series([1]*len(df), index=df.index))).fillna(1).clip(lower=1).values.astype(float)
    goals = df.get("season_goals",
            pd.Series([0]*len(df), index=df.index)).fillna(0).values.astype(float)

    total  = shots * gp
    raw_sh = np.where(total > 0, goals / total, 0.0)
    w      = total / (total + K)
    reg_sh = w * raw_sh + (1 - w) * LEAGUE_SH
    mult   = np.minimum(1.5, 1.0 + 0.5 * np.minimum(total / 100.0, 1.0))
    ceil   = shots * np.minimum(reg_sh * mult, BEST_CASE)
    ceil   = np.where((goals == 0) & (gp < 15), np.minimum(ceil, 0.08), ceil)
    ceil   = np.clip(ceil, 0.02, 0.65)

    raw   = df["goal_probability"].values.astype(float)
    final = np.round(0.90 * np.minimum(raw, ceil) + 0.10 * np.minimum(raw, 0.65), 4)

    df = df.copy()
    df["goal_probability"] = final
    conf = pd.cut(final,
        bins  = [-np.inf, 0.14, 0.22, 0.32, np.inf],
        labels= ["Low", "Medium", "High", "Elite"]).astype(str)
    df["confidence"] = conf
    df["conf_goals"] = conf
    if "projected_sog" in df.columns:
        df["conf_sog"] = pd.cut(
            df["projected_sog"].fillna(0).values.astype(float),
            bins  = [-np.inf, 2.0, 3.0, 4.0, np.inf],
            labels= ["Low", "Medium", "High", "Elite"]).astype(str)
    return df.sort_values("goal_probability", ascending=False).reset_index(drop=True)


def build_html() -> str:
    nhl = load("nhl")
    mlb = load("mlb")
    nba = load("nba")

    # Apply physics-based ceiling to NHL predictions before emailing
    # Prevents 0-goal/low-shot players like Tyson Hinds appearing with 0.60 prob
    nhl["predictions"] = _apply_prob_ceiling(nhl["predictions"])

    n_picks = sum(min(TOP_N, len(d["predictions"])) for d in [nhl, mlb, nba] if not d["predictions"].empty)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sports Predictor — {TODAY_LONG}</title>
</head>
<body style="margin:0;padding:0;background:#f0f2f5;
  font-family:'Segoe UI',Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f2f5;">
<tr><td align="center" style="padding:20px 10px;">
<table width="680" cellpadding="0" cellspacing="0"
  style="max-width:680px;width:100%;background:#fff;
  border-radius:12px;overflow:hidden;
  box-shadow:0 4px 20px rgba(0,0,0,.10);">

  <!-- Header -->
  <tr><td style="background:linear-gradient(135deg,#1a237e 0%,#1565c0 60%,#0288d1 100%);
    padding:32px;text-align:center;">
    <div style="font-size:40px;margin-bottom:8px;">🏆</div>
    <h1 style="color:#fff;margin:0;font-size:24px;font-weight:800;
      letter-spacing:-.5px;">Sports Predictor</h1>
    <p style="color:#90caf9;margin:6px 0 0;font-size:15px;">{TODAY_LONG}</p>
    <div style="margin-top:14px;display:inline-block;background:rgba(255,255,255,.15);
      border-radius:20px;padding:6px 16px;color:#fff;font-size:13px;font-weight:600;">
      {n_picks} Top picks across NHL · MLB · NBA
    </div>
  </td></tr>

  <!-- Body -->
  <tr><td style="padding:8px 32px 32px;">
"""

    html += nhl_section(nhl)
    html += mlb_section(mlb)
    html += nba_section(nba)

    html += f"""
    <!-- Footer -->
    <div style="margin-top:32px;padding:20px;background:#f8f9fa;border-radius:8px;
      text-align:center;color:#adb5bd;font-size:11px;line-height:1.6;">
      Sports Predictor · Generated {TODAY_LONG}<br>
      Predictions are probabilistic estimates for entertainment only.<br>
      Not financial or gambling advice.
    </div>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""
    return html


def send_email(html: str, sender: str, password: str, recipients: list):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🏆 Sports Predictor — {TODAY_SHORT} Top Picks"
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))
    print(f"Sending to: {', '.join(recipients)}")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())
    print("✅ Email sent")


if __name__ == "__main__":
    sender    = os.environ.get("EMAIL_SENDER","")
    password  = os.environ.get("EMAIL_PASSWORD","")
    rcpts_raw = os.environ.get("EMAIL_RECIPIENTS","")

    if not all([sender, password, rcpts_raw]):
        print("❌ Missing EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECIPIENTS")
        sys.exit(1)

    recipients = [r.strip() for r in rcpts_raw.split(",") if r.strip()]

    print(f"Building email for {TODAY_LONG}…")
    html = build_html()

    out = PRED_DIR / "daily_email.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"HTML preview saved → {out}")

    send_email(html, sender, password, recipients)