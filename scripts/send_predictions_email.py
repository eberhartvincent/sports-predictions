"""
scripts/send_predictions_email.py
Reads saved predictions and emails a formatted HTML digest.
Called by GitHub Actions after warm_cache.py runs.

Usage:
    python scripts/send_predictions_email.py
    
Environment variables (set as GitHub secrets):
    EMAIL_SENDER      — Gmail address to send from
    EMAIL_PASSWORD    — Gmail app password (16 chars, no spaces)
    EMAIL_RECIPIENTS  — comma-separated recipient addresses
"""

import json
import os
import sys
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

ET       = ZoneInfo("America/New_York")
PRED_DIR = _ROOT / "data" / "cache" / "predictions"
TODAY    = datetime.now(ET).strftime("%Y-%m-%d")
TODAY_DISPLAY = datetime.now(ET).strftime("%A, %B %d, %Y")


def load(sport: str) -> dict:
    result = {"predictions": pd.DataFrame(), "game_projections": [], "meta": {}}
    try:
        f = PRED_DIR / f"{sport}_predictions.parquet"
        if f.exists(): result["predictions"] = pd.read_parquet(f)
        g = PRED_DIR / f"{sport}_game_projections.json"
        if g.exists(): result["game_projections"] = json.loads(g.read_text())
        m = PRED_DIR / f"{sport}_meta.json"
        if m.exists(): result["meta"] = json.loads(m.read_text())
    except Exception as e:
        print(f"  Warning: could not load {sport}: {e}")
    return result


def conf_color(conf: str) -> str:
    return {"Elite":"#c0392b","High":"#e67e22","Medium":"#2980b9","Low":"#6c757d"}.get(conf,"#6c757d")


def pct_bar(val: float, max_val: float, color: str) -> str:
    pct = min(int(val / max_val * 100), 100) if max_val > 0 else 0
    return (f'<div style="background:#e9ecef;border-radius:4px;height:8px;'
            f'width:100px;display:inline-block;vertical-align:middle;">'
            f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div></div>'
            f'&nbsp;<span style="font-size:12px;font-weight:700;">{val:.3f}</span>')


def build_html() -> str:
    nhl = load("nhl")
    mlb = load("mlb")
    nba = load("nba")

    # ── Styles ─────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sports Predictor — {TODAY_DISPLAY}</title>
</head>
<body style="margin:0;padding:0;background:#f4f6f9;font-family:'Segoe UI',Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f6f9;">
<tr><td align="center" style="padding:20px 10px;">
<table width="650" cellpadding="0" cellspacing="0" style="max-width:650px;width:100%;">

  <!-- Header -->
  <tr><td style="background:linear-gradient(135deg,#1a237e,#1565c0);border-radius:12px 12px 0 0;
    padding:28px 32px;text-align:center;">
    <div style="font-size:32px;margin-bottom:6px;">🏆</div>
    <h1 style="color:#fff;margin:0;font-size:22px;font-weight:700;">Sports Predictor</h1>
    <p style="color:#90caf9;margin:6px 0 0;font-size:14px;">{TODAY_DISPLAY}</p>
  </td></tr>

  <!-- Body -->
  <tr><td style="background:#fff;padding:0 32px 28px;border-radius:0 0 12px 12px;
    box-shadow:0 2px 12px rgba(0,0,0,.08);">
"""

    # ── NHL section ────────────────────────────────────────────────────────────
    html += _section_header("🏒", "NHL — Top Goalscorer Picks")
    if not nhl["predictions"].empty:
        df = nhl["predictions"].sort_values("goal_probability", ascending=False).head(10)
        html += '<table width="100%" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px;">'
        html += ('<tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">'
                 '<th align="left" style="padding:8px 6px;color:#495057;">#</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Player</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Team</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Opp</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Conf</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Goal Prob</th>'
                 '</tr>')
        for i, (_, row) in enumerate(df.iterrows(), 1):
            bg = "#fff" if i % 2 else "#f8f9fa"
            conf = str(row.get("confidence","Low"))
            prob = float(row.get("goal_probability",0))
            html += (f'<tr style="background:{bg};border-bottom:1px solid #dee2e6;">'
                     f'<td style="padding:7px 6px;color:#6c757d;font-weight:600;">#{i}</td>'
                     f'<td style="padding:7px 6px;font-weight:600;color:#212529;">{row.get("player_name","")}</td>'
                     f'<td style="padding:7px 6px;color:#1565c0;font-weight:600;">{row.get("team","")}</td>'
                     f'<td style="padding:7px 6px;color:#6c757d;">{row.get("opponent","")}</td>'
                     f'<td style="padding:7px 6px;"><span style="background:{conf_color(conf)};color:#fff;'
                     f'padding:2px 7px;border-radius:10px;font-size:11px;font-weight:700;">{conf}</span></td>'
                     f'<td style="padding:7px 6px;">{pct_bar(prob, 0.5, "#c0392b")}</td>'
                     f'</tr>')
        html += '</table>'
        # NHL game projections
        if nhl["game_projections"]:
            html += _game_proj_nhl(nhl["game_projections"])
    else:
        html += _no_data("No NHL games today or predictions not yet available.")

    # ── MLB section ────────────────────────────────────────────────────────────
    html += _section_header("⚾", "MLB — Top Batter Picks")
    if not mlb["predictions"].empty:
        df = mlb["predictions"].sort_values("proj_hrr", ascending=False).head(10) \
             if "proj_hrr" in mlb["predictions"].columns else \
             mlb["predictions"].sort_values("proj_hits", ascending=False).head(10)
        html += '<table width="100%" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px;">'
        html += ('<tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">'
                 '<th align="left" style="padding:8px 6px;color:#495057;">#</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Player</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Team</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Opp</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Conf</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">H+R+RBI</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">HR Prob</th>'
                 '</tr>')
        for i, (_, row) in enumerate(df.iterrows(), 1):
            bg = "#fff" if i % 2 else "#f8f9fa"
            conf = str(row.get("confidence","Low"))
            hrr  = float(row.get("proj_hrr", 0))
            hr   = float(row.get("proj_hr",  0))
            html += (f'<tr style="background:{bg};border-bottom:1px solid #dee2e6;">'
                     f'<td style="padding:7px 6px;color:#6c757d;font-weight:600;">#{i}</td>'
                     f'<td style="padding:7px 6px;font-weight:600;color:#212529;">{row.get("player_name","")}</td>'
                     f'<td style="padding:7px 6px;color:#1565c0;font-weight:600;">{row.get("team","")}</td>'
                     f'<td style="padding:7px 6px;color:#6c757d;">{row.get("opponent","")}</td>'
                     f'<td style="padding:7px 6px;"><span style="background:{conf_color(conf)};color:#fff;'
                     f'padding:2px 7px;border-radius:10px;font-size:11px;font-weight:700;">{conf}</span></td>'
                     f'<td style="padding:7px 6px;font-size:15px;font-weight:800;'
                     f'color:{"#f59e0b" if hrr>=2.0 else "#212529"};">{hrr:.2f}</td>'
                     f'<td style="padding:7px 6px;">{pct_bar(hr, 0.20, "#c0392b")}</td>'
                     f'</tr>')
        html += '</table>'
    else:
        html += _no_data("No MLB games today or predictions not yet available.")

    # ── NBA section ────────────────────────────────────────────────────────────
    html += _section_header("🏀", "NBA — Top Player Picks")
    if not nba["predictions"].empty:
        sort_col = "proj_pts" if "proj_pts" in nba["predictions"].columns else \
                   nba["predictions"].columns[0]
        df = nba["predictions"].sort_values(sort_col, ascending=False).head(10)
        html += '<table width="100%" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px;">'
        html += ('<tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">'
                 '<th align="left" style="padding:8px 6px;color:#495057;">#</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Player</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Team</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Opp</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Proj Pts</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Proj Reb</th>'
                 '<th align="left" style="padding:8px 6px;color:#495057;">Proj Ast</th>'
                 '</tr>')
        for i, (_, row) in enumerate(df.iterrows(), 1):
            bg = "#fff" if i % 2 else "#f8f9fa"
            html += (f'<tr style="background:{bg};border-bottom:1px solid #dee2e6;">'
                     f'<td style="padding:7px 6px;color:#6c757d;font-weight:600;">#{i}</td>'
                     f'<td style="padding:7px 6px;font-weight:600;color:#212529;">{row.get("player_name","")}</td>'
                     f'<td style="padding:7px 6px;color:#1565c0;font-weight:600;">{row.get("team","")}</td>'
                     f'<td style="padding:7px 6px;color:#6c757d;">{row.get("opponent","")}</td>'
                     f'<td style="padding:7px 6px;font-weight:700;color:#1565c0;">{float(row.get("proj_pts",0)):.1f}</td>'
                     f'<td style="padding:7px 6px;color:#212529;">{float(row.get("proj_reb",0)):.1f}</td>'
                     f'<td style="padding:7px 6px;color:#212529;">{float(row.get("proj_ast",0)):.1f}</td>'
                     f'</tr>')
        html += '</table>'
    else:
        html += _no_data("No NBA games today or predictions not yet available.")

    # ── Footer ─────────────────────────────────────────────────────────────────
    html += f"""
  <div style="margin-top:24px;padding-top:16px;border-top:1px solid #dee2e6;
    text-align:center;color:#adb5bd;font-size:11px;">
    Generated by Sports Predictor · {TODAY_DISPLAY}<br>
    Predictions are probabilistic estimates for entertainment purposes only.
  </div>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""
    return html


def _section_header(emoji: str, title: str) -> str:
    return (f'<div style="margin:24px 0 12px;padding-bottom:8px;'
            f'border-bottom:2px solid #1565c0;">'
            f'<h2 style="margin:0;font-size:16px;color:#1a237e;">'
            f'{emoji}&nbsp; {title}</h2></div>')


def _no_data(msg: str) -> str:
    return (f'<p style="color:#6c757d;font-size:13px;'
            f'padding:12px;background:#f8f9fa;border-radius:6px;">{msg}</p>')


def _game_proj_nhl(projs: list) -> str:
    if not projs: return ""
    html = '<div style="margin-top:12px;"><strong style="font-size:13px;color:#495057;">Game Projections</strong>'
    html += '<table width="100%" cellpadding="5" cellspacing="0" style="border-collapse:collapse;font-size:12px;margin-top:6px;">'
    html += ('<tr style="background:#f8f9fa;border-bottom:1px solid #dee2e6;">'
             '<th align="left" style="padding:6px;color:#495057;">Matchup</th>'
             '<th align="left" style="padding:6px;color:#495057;">Away ML</th>'
             '<th align="left" style="padding:6px;color:#495057;">Home ML</th>'
             '<th align="left" style="padding:6px;color:#495057;">Total</th>'
             '<th align="left" style="padding:6px;color:#495057;">Best Bet</th>'
             '</tr>')
    for p in projs:
        away = p.get("away_team",""); home = p.get("home_team","")
        rec  = p.get("recommendation",""); line = p.get("best_ou_line","")
        prob = p.get("best_ou_prob",0)
        rc   = "#c0392b" if rec=="OVER" else "#2980b9"
        html += (f'<tr style="border-bottom:1px solid #f0f0f0;">'
                 f'<td style="padding:6px;font-weight:600;">{away} @ {home}</td>'
                 f'<td style="padding:6px;">{p.get("away_ml_display","")}&nbsp;'
                 f'<span style="color:#6c757d;font-size:11px;">({p.get("away_win_prob",0):.0%})</span></td>'
                 f'<td style="padding:6px;">{p.get("home_ml_display","")}&nbsp;'
                 f'<span style="color:#6c757d;font-size:11px;">({p.get("home_win_prob",0):.0%})</span></td>'
                 f'<td style="padding:6px;">{p.get("proj_total",0):.1f}</td>'
                 f'<td style="padding:6px;"><span style="color:{rc};font-weight:700;">'
                 f'{rec} {line} ({prob:.0%})</span></td>'
                 f'</tr>')
    html += '</table></div>'
    return html


def send_email(html: str, sender: str, password: str, recipients: list):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🏆 Sports Predictor — {TODAY_DISPLAY}"
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    print(f"Sending to: {', '.join(recipients)}")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())
    print("✅ Email sent successfully")


if __name__ == "__main__":
    sender     = os.environ.get("EMAIL_SENDER", "")
    password   = os.environ.get("EMAIL_PASSWORD", "")
    recipients_raw = os.environ.get("EMAIL_RECIPIENTS", "")

    if not sender or not password or not recipients_raw:
        print("❌ Missing EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECIPIENTS env vars")
        print("   Set these as GitHub secrets and they will be injected automatically")
        sys.exit(1)

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    print(f"Building predictions email for {TODAY_DISPLAY}...")
    html = build_html()

    # Save a copy locally for debugging
    out = _ROOT / "data" / "cache" / "predictions" / "daily_email.html"
    out.write_text(html)
    print(f"HTML saved to {out}")

    send_email(html, sender, password, recipients)
