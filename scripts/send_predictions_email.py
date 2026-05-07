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
    html = section("🏒", "NHL — Elite Goal Scorer Picks",
                   "Players projected goal probability ≥ 0.32")
    df = data["predictions"]
    if df.empty:
        return html + no_data("No NHL data available.")

    elite = df[df["confidence"] == "Elite"].sort_values(
        "goal_probability", ascending=False)

    if elite.empty:
        return html + no_data("No Elite-tier NHL picks today.")

    html += f'<p style="color:#6c757d;font-size:12px;margin:0 0 8px;">{len(elite)} elite picks today</p>'
    html += '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
    html += table_header("#","Player","Team","Opp","Matchup",
                         "Goal Prob","Proj Ast","Proj Pts","Proj SOG","Szn G/A")

    for i, (_, r) in enumerate(elite.iterrows(), 1):
        prob = float(r.get("goal_probability",0))
        ast_ = float(r.get("projected_assists",0))
        pts  = float(r.get("projected_points",0))
        sog  = float(r.get("projected_sog",0))
        sg   = int(r.get("season_goals",0))
        sa   = int(r.get("season_assists",0))
        pc   = "#c0392b" if prob>=0.40 else "#e67e22" if prob>=0.32 else "#2980b9"

        html += row_start(i)
        html += td(f'<span style="color:#6c757d;font-weight:700;">#{i}</span>')
        html += td(f'<strong>{r.get("player_name","")}</strong>')
        html += td(r.get("team",""),   color="#1565c0", bold=True)
        html += td(r.get("opponent",""), color="#6c757d")
        html += td(r.get("game_label",""), color="#6c757d")
        html += td(f'<span style="color:{pc};font-weight:800;font-size:14px;">{prob:.3f}</span>')
        html += proj_cell(ast_,  color="#16a085")
        html += proj_cell(pts,   color="#8e44ad")
        html += proj_cell(sog, ".1f", color="#2980b9")
        html += td(f'{sg}G / {sa}A', color="#6c757d")
        html += '</tr>'

    html += '</table>'

    # Game projections
    if data["game_projections"]:
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


# ── MLB ───────────────────────────────────────────────────────────────────────

def mlb_section(data: dict) -> str:
    html = section("⚾", "MLB — Elite Batter Picks",
                   "Players with projected hit probability ≥ 0.80")
    df = data["predictions"]
    if df.empty:
        return html + no_data("No MLB data available.")

    elite = df[df["confidence"] == "Elite"].sort_values(
        "proj_hrr", ascending=False) if "proj_hrr" in df.columns else pd.DataFrame()

    if elite.empty:
        return html + no_data("No Elite-tier MLB picks today.")

    html += f'<p style="color:#6c757d;font-size:12px;margin:0 0 8px;">{len(elite)} elite picks today</p>'
    html += '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
    html += table_header("#","Player","Team","Opp","H+R+RBI","Proj H",
                         "Proj HR","Proj RBI","Proj R","Proj TB","Proj K","vs P")

    for i, (_, r) in enumerate(elite.iterrows(), 1):
        hrr  = float(r.get("proj_hrr",  0))
        hits = float(r.get("proj_hits", 0))
        hr   = float(r.get("proj_hr",   0))
        rbi  = float(r.get("proj_rbi",  0))
        runs = float(r.get("proj_runs", 0))
        tb   = float(r.get("proj_tb",   0))
        k    = float(r.get("proj_k",    0))
        bvp  = int(r.get("bvp_ab", 0))
        bvp_hr = int(r.get("bvp_hr", 0))
        bvp_str = f"{bvp}AB {bvp_hr}HR" if bvp >= 5 else "—"
        hrr_col = "#f59e0b" if hrr >= 2.5 else "#c0392b" if hrr >= 2.0 else "#212529"

        html += row_start(i)
        html += td(f'<span style="color:#6c757d;font-weight:700;">#{i}</span>')
        html += td(f'<strong>{r.get("player_name","")}</strong>')
        html += td(r.get("team",""),     color="#1565c0", bold=True)
        html += td(r.get("opponent",""), color="#6c757d")
        html += td(f'<span style="font-size:16px;font-weight:800;color:{hrr_col};">{hrr:.2f}</span>')
        html += proj_cell(hits,  color="#27ae60")
        html += proj_cell(hr, ".3f", color="#c0392b")
        html += proj_cell(rbi,   color="#e67e22")
        html += proj_cell(runs,  color="#16a085")
        html += proj_cell(tb,    color="#8e44ad")
        html += proj_cell(k,     color="#6c757d")
        html += td(bvp_str, color="#6c757d")
        html += '</tr>'

    html += '</table>'

    # Pitcher projections
    preds = data.get("pitcher_predictions", pd.DataFrame())
    if not preds.empty:
        html += _mlb_pitcher_table(preds)

    # Game projections
    if data["game_projections"]:
        html += _mlb_game_proj(data["game_projections"])

    return html


def _mlb_pitcher_table(df: pd.DataFrame) -> str:
    html = ('<div style="margin-top:18px;">'
            '<h3 style="font-size:13px;color:#495057;margin:0 0 8px;">'
            '⚾ Starting Pitchers</h3>'
            '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">')
    html += table_header("Pitcher","Team","Opp","ERA","WHIP","K/9",
                         "Proj IP","Proj K","Proj ER","Quality")
    for i, (_, r) in enumerate(df.iterrows(), 1):
        era  = float(r.get("era", 4.50))
        ec   = "#27ae60" if era<=3.50 else "#e67e22" if era<=4.50 else "#c0392b"
        qual = str(r.get("quality","Average"))
        qc   = {"Ace":"#c0392b","Above Avg":"#e67e22","Average":"#2980b9",
                "Below Avg":"#6c757d","Avoid":"#7f8c8d"}.get(qual,"#6c757d")
        html += row_start(i)
        html += td(f'<strong>{r.get("pitcher_name","")}</strong> '
                   f'{"🏠" if r.get("is_home") else "✈️"}')
        html += td(r.get("team",""),     color="#1565c0", bold=True)
        html += td(f'vs {r.get("opponent","")}', color="#6c757d")
        html += td(f'<span style="color:{ec};font-weight:700;">{era:.2f}</span>')
        html += td(f'{float(r.get("whip",1.30)):.2f}', color="#6c757d")
        html += td(f'{float(r.get("k9",8.5)):.1f}',   color="#6c757d")
        html += proj_cell(float(r.get("proj_ip",5.5)), ".1f", "#2980b9")
        html += proj_cell(float(r.get("proj_k", 5.0)), ".1f", "#27ae60")
        html += td(f'<span style="color:{ec};font-weight:700;">'
                   f'{float(r.get("proj_er",2.5)):.2f}</span>')
        html += td(badge(qual, qc))
        html += '</tr>'
    html += '</table></div>'
    return html


def _mlb_game_proj(projs: list) -> str:
    html = ('<div style="margin-top:18px;">'
            '<h3 style="font-size:13px;color:#495057;margin:0 0 8px;">'
            '🎰 MLB Game Projections</h3>'
            '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">')
    html += table_header("Matchup","Away Win","Home Win","Total Proj R","Best Bet")
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
        html += td(f'{float(p.get("total_proj_runs",0)):.1f} R')
        html += td(f'<span style="color:{rc};font-weight:700;">'
                   f'{rec} {line} ({prob:.0%})</span>')
        html += '</tr>'
    html += '</table></div>'
    return html


# ── NBA ───────────────────────────────────────────────────────────────────────

def nba_section(data: dict) -> str:
    html = section("🏀", "NBA — Elite Player Picks",
                   "Players projected 30+ points")
    df = data["predictions"]
    if df.empty:
        return html + no_data("No NBA data available.")

    elite = df[df["confidence"] == "Elite"].sort_values(
        "proj_pts", ascending=False) if "proj_pts" in df.columns else pd.DataFrame()

    if elite.empty:
        return html + no_data("No Elite-tier NBA picks today.")

    html += f'<p style="color:#6c757d;font-size:12px;margin:0 0 8px;">{len(elite)} elite picks today</p>'
    html += '<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
    html += table_header("#","Player","Team","Opp","Proj Pts","Proj Reb",
                         "Proj Ast","Proj 3PM","Proj Stk","DD Prob","Szn Avg")

    for i, (_, r) in enumerate(elite.iterrows(), 1):
        pts    = float(r.get("proj_pts",    0))
        reb    = float(r.get("proj_reb",    0))
        ast_   = float(r.get("proj_ast",    0))
        fg3    = float(r.get("proj_fg3m",   0))
        stk    = float(r.get("proj_stocks", 0))
        dd     = float(r.get("proj_dd",     0))
        spts   = float(r.get("season_pts",  0))
        sreb   = float(r.get("season_reb",  0))
        sast   = float(r.get("season_ast",  0))

        html += row_start(i)
        html += td(f'<span style="color:#6c757d;font-weight:700;">#{i}</span>')
        html += td(f'<strong>{r.get("player_name","")}</strong>')
        html += td(r.get("team",""),     color="#1565c0", bold=True)
        html += td(r.get("opponent",""), color="#6c757d")
        html += td(f'<span style="font-size:16px;font-weight:800;color:#1565c0;">{pts:.1f}</span>')
        html += proj_cell(reb,  color="#16a085")
        html += proj_cell(ast_, color="#e67e22")
        html += proj_cell(fg3, ".1f", color="#8e44ad")
        html += proj_cell(stk, ".1f", color="#c0392b")
        html += td(f'<span style="color:{"#27ae60" if dd>=0.4 else "#6c757d"};'
                   f'font-weight:700;">{dd:.0%}</span>')
        html += td(f'{spts:.0f}P / {sreb:.0f}R / {sast:.0f}A', color="#6c757d")
        html += '</tr>'

    html += '</table>'

    # Game projections
    if data["game_projections"]:
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

def build_html() -> str:
    nhl = load("nhl")
    mlb = load("mlb")
    nba = load("nba")

    n_elite = 0
    for d in [nhl, mlb, nba]:
        if not d["predictions"].empty and "confidence" in d["predictions"].columns:
            n_elite += int((d["predictions"]["confidence"]=="Elite").sum())

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
      {n_elite} Elite picks across NHL · MLB · NBA
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
    msg["Subject"] = f"🏆 Sports Predictor — {TODAY_SHORT} Elite Picks"
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
