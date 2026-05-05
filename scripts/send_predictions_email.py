"""
scripts/send_predictions_email.py
Daily HTML email — top 10 per category, Elite picks only.

GitHub secrets needed:
    EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENTS
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

ET          = ZoneInfo("America/New_York")
PRED_DIR    = _ROOT / "data" / "cache" / "predictions"
TODAY_LONG  = datetime.now(ET).strftime("%A, %B %d, %Y")
TODAY_SHORT = datetime.now(ET).strftime("%b %d")
TOP_N       = 10


def load(sport):
    out = {"predictions":pd.DataFrame(),"pitcher_predictions":pd.DataFrame(),
           "game_projections":[],"meta":{}}
    try:
        for key,fname in [("predictions",f"{sport}_predictions.parquet"),
                          ("pitcher_predictions",f"{sport}_pitcher_predictions.parquet")]:
            f=PRED_DIR/fname
            if f.exists(): out[key]=pd.read_parquet(f)
        for key,fname in [("game_projections",f"{sport}_game_projections.json"),
                          ("meta",f"{sport}_meta.json")]:
            f=PRED_DIR/fname
            if f.exists(): out[key]=json.loads(f.read_text())
    except Exception as e:
        print(f"  Warning {sport}: {e}")
    return out


def sh(emoji,title,sub=""):
    s=f'<div style="color:#6c757d;font-size:12px;margin-top:2px;">{sub}</div>'if sub else""
    return(f'<div style="margin:24px 0 10px;padding:10px 14px;background:#f0f4ff;'
           f'border-left:4px solid #1565c0;border-radius:0 6px 6px 0;">'
           f'<h2 style="margin:0;font-size:15px;color:#1a237e;font-weight:700;">'
           f'{emoji}&nbsp;{title}</h2>{s}</div>')

def ch(title,sub=""):
    s=f'<div style="color:#6c757d;font-size:11px;">{sub}</div>'if sub else""
    return(f'<div style="margin:16px 0 6px;">'
           f'<strong style="font-size:13px;color:#495057;">{title}</strong>{s}</div>')

def to(): return'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-bottom:4px;">'
def tc(): return'</table>'

def hdr(*cols):
    c="".join(f'<th align="left" style="padding:6px 8px;color:#6c757d;font-size:10px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #dee2e6;white-space:nowrap;">{x}</th>'for x in cols)
    return f'<tr style="background:#f8f9fa;">{c}</tr>'

def row(i,*cells):
    bg="#fff"if i%2 else"#f9fafb"
    td="".join(f'<td style="padding:7px 8px;font-size:12px;border-bottom:1px solid #f0f0f0;">{c}</td>'for c in cells)
    return f'<tr style="background:{bg};">{td}</tr>'

def rk(i):
    c={1:"#c0392b",2:"#e67e22",3:"#f59e0b"}.get(i,"#6c757d")
    return f'<span style="color:{c};font-weight:700;">#{i}</span>'

def nm(n,sub=""):
    s=f'<div style="font-size:10px;color:#6c757d;margin-top:1px;">{sub}</div>'if sub else""
    return f'<strong style="font-size:13px;color:#212529;">{n}</strong>{s}'

def tm(t): return f'<span style="color:#1565c0;font-weight:700;">{t}</span>'

def vl(v,fmt=".2f",color="#212529",size="13px"):
    return f'<span style="color:{color};font-weight:700;font-size:{size};">{v:{fmt}}</span>'

def nd(msg):
    return f'<p style="color:#6c757d;font-size:12px;padding:10px 12px;background:#f8f9fa;border-radius:6px;margin:6px 0;">ℹ️ {msg}</p>'

def grow(i,matchup,c1,c2,c3,best):
    bg="#fff"if i%2 else"#f9fafb"
    tds="".join(f'<td style="padding:6px 8px;font-size:12px;border-bottom:1px solid #f0f0f0;">{x}</td>'for x in[matchup,c1,c2,c3,best])
    return f'<tr style="background:{bg};">{tds}</tr>'


def nhl_html(d):
    html=sh("🏒","NHL Picks","Elite goalscorer picks (goal probability ≥ 0.32)")
    df=d["predictions"]
    if df.empty: return html+nd("No NHL data.")
    elite=df[df["confidence"]=="Elite"].sort_values("goal_probability",ascending=False)
    if elite.empty: html+=nd("No Elite NHL picks today.")
    else:
        html+=f'<p style="color:#6c757d;font-size:12px;margin:0 0 8px;">{len(elite)} elite picks today</p>'
        html+=ch("⚡ Top Goal Scorers",f"Top {TOP_N} by goal probability")
        html+=to()+hdr("#","Player","Team","Opp","Goal Prob","Proj Pts","Proj SOG","Szn G/A")
        for i,(_, r) in enumerate(elite.head(TOP_N).iterrows(),1):
            prob=float(r.get("goal_probability",0))
            pc="#c0392b"if prob>=0.40 else"#e67e22"
            html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                      tm(r.get("team","")),r.get("opponent",""),
                      vl(prob,".3f",pc,"14px"),
                      vl(float(r.get("projected_points",0)),color="#8e44ad"),
                      vl(float(r.get("projected_sog",0)),".1f",color="#2980b9"),
                      f'{int(r.get("season_goals",0))}G/{int(r.get("season_assists",0))}A')
        html+=tc()
    if d["game_projections"]:
        html+=ch("🎰 Game Projections")
        html+=to()+hdr("Matchup","Away","Home","Total","Best Bet")
        for i,p in enumerate(d["game_projections"]):
            away=p.get("away_team","");home=p.get("home_team","")
            rec=p.get("recommendation","");line=p.get("best_ou_line","")
            prob=float(p.get("best_ou_prob",0))
            rc="#c0392b"if rec=="OVER"else"#2980b9"
            html+=grow(i,f'{away} @ {home}',
                f'{p.get("away_ml_display","")} ({float(p.get("away_win_prob",0)):.0%})',
                f'{p.get("home_ml_display","")} ({float(p.get("home_win_prob",0)):.0%})',
                f'{float(p.get("proj_total",0)):.1f} goals',
                f'<span style="color:{rc};font-weight:700;">{rec} {line} ({prob:.0%})</span>')
        html+=tc()
    return html


def mlb_html(d):
    html=sh("⚾","MLB Picks","Elite batter picks (hit probability ≥ 0.80)")
    df=d["predictions"]
    if df.empty: return html+nd("No MLB data.")
    elite=df[df["confidence"]=="Elite"]
    if elite.empty:
        html+=nd("No Elite MLB picks today — thresholds not met.")
    else:
        html+=f'<p style="color:#6c757d;font-size:12px;margin:0 0 8px;">{len(elite)} elite picks today</p>'
        # H+R+RBI
        html+=ch("🏆 H+R+RBI Leaders",f"Top {TOP_N} combined")
        top=elite.sort_values("proj_hrr",ascending=False).head(TOP_N)
        html+=to()+hdr("#","Player","Team","Opp","H+R+RBI","Proj H","Proj RBI","Proj R")
        for i,(_, r) in enumerate(top.iterrows(),1):
            hrr=float(r.get("proj_hrr",0))
            hc="#f59e0b"if hrr>=2.5 else"#c0392b"if hrr>=2.0 else"#212529"
            html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                      tm(r.get("team","")),r.get("opponent",""),
                      vl(hrr,".2f",hc,"14px"),
                      vl(float(r.get("proj_hits",0)),color="#27ae60"),
                      vl(float(r.get("proj_rbi",0)),color="#e67e22"),
                      vl(float(r.get("proj_runs",0)),color="#16a085"))
        html+=tc()
        # HR
        html+=ch("💣 HR Threats",f"Top {TOP_N} by home run probability")
        top_hr=elite.sort_values("proj_hr",ascending=False).head(TOP_N)
        html+=to()+hdr("#","Player","Team","Opp","HR Prob","Proj H","Proj TB","BvP")
        for i,(_, r) in enumerate(top_hr.iterrows(),1):
            hr=float(r.get("proj_hr",0))
            hrc="#c0392b"if hr>=0.10 else"#e67e22"if hr>=0.06 else"#2980b9"
            bab=int(r.get("bvp_ab",0));bhr=int(r.get("bvp_hr",0))
            html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                      tm(r.get("team","")),r.get("opponent",""),
                      vl(hr,".3f",hrc,"14px"),
                      vl(float(r.get("proj_hits",0)),color="#27ae60"),
                      vl(float(r.get("proj_tb",0)),color="#8e44ad"),
                      f'{bab}AB {bhr}HR'if bab>=5 else"—")
        html+=tc()
        # Contact / K avoidance
        html+=ch("🎯 Contact Leaders",f"Top {TOP_N} lowest K projection")
        top_k=elite.sort_values("proj_k",ascending=True).head(TOP_N)
        html+=to()+hdr("#","Player","Team","Opp","Proj K","Proj H","H+R+RBI","Opp ERA")
        for i,(_, r) in enumerate(top_k.iterrows(),1):
            k=float(r.get("proj_k",0));oera=float(r.get("opp_era",4.5))
            ec="#27ae60"if oera<=3.5 else"#e67e22"if oera<=4.5 else"#c0392b"
            html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                      tm(r.get("team","")),r.get("opponent",""),
                      vl(k,".2f","#c0392b"),
                      vl(float(r.get("proj_hits",0)),color="#27ae60"),
                      vl(float(r.get("proj_hrr",0)),color="#f59e0b"),
                      vl(oera,".2f",ec))
        html+=tc()
    # Pitchers
    pp=d.get("pitcher_predictions",pd.DataFrame())
    if not pp.empty:
        html+=ch("⚾ Starting Pitchers")
        html+=to()+hdr("Pitcher","Team","Opp","ERA","WHIP","K/9","Proj IP","Proj K","Proj ER","Quality")
        for i,(_, r) in enumerate(pp.iterrows(),1):
            era=float(r.get("era",4.5));ec="#27ae60"if era<=3.5 else"#e67e22"if era<=4.5 else"#c0392b"
            qual=str(r.get("quality","Average"))
            qc={"Ace":"#c0392b","Above Avg":"#e67e22","Average":"#2980b9","Below Avg":"#6c757d","Avoid":"#7f8c8d"}.get(qual,"#6c757d")
            html+=row(i,
                f'{r.get("pitcher_name","")} {"🏠"if r.get("is_home")else"✈️"}',
                tm(r.get("team","")),f'vs {r.get("opponent","")}',
                vl(era,".2f",ec),vl(float(r.get("whip",1.3)),color="#6c757d"),
                vl(float(r.get("k9",8.5)),".1f",color="#6c757d"),
                vl(float(r.get("proj_ip",5.5)),".1f",color="#2980b9"),
                vl(float(r.get("proj_k",5.0)),".1f",color="#27ae60"),
                vl(float(r.get("proj_er",2.5)),".2f",ec),
                f'<span style="color:{qc};font-weight:700;">{qual}</span>')
        html+=tc()
    if d["game_projections"]:
        html+=ch("🎰 Game Projections")
        html+=to()+hdr("Matchup","Away","Home","Total Runs","Best Bet")
        for i,p in enumerate(d["game_projections"]):
            away=p.get("away_team","");home=p.get("home_team","")
            rec=p.get("recommendation","");line=p.get("best_ou_line","")
            prob=float(p.get("best_ou_prob",0))
            rc="#c0392b"if rec=="OVER"else"#2980b9"
            html+=grow(i,f'{away} @ {home}',
                f'{p.get("away_ml_display","")} ({float(p.get("away_win_prob",0)):.0%})',
                f'{p.get("home_ml_display","")} ({float(p.get("home_win_prob",0)):.0%})',
                f'{float(p.get("total_proj_runs",0)):.1f} R',
                f'<span style="color:{rc};font-weight:700;">{rec} {line} ({prob:.0%})</span>')
        html+=tc()
    return html


def nba_html(d):
    html=sh("🏀","NBA Picks","Elite player picks (30+ projected points)")
    df=d["predictions"]
    if df.empty: return html+nd("No NBA data.")
    elite=df[df["confidence"]=="Elite"]
    if elite.empty:
        html+=nd("No Elite NBA picks today — thresholds not met.")
    else:
        html+=f'<p style="color:#6c757d;font-size:12px;margin:0 0 8px;">{len(elite)} elite picks today</p>'
        # Points
        html+=ch("🏆 Points Leaders",f"Top {TOP_N} by projected points")
        top=elite.sort_values("proj_pts",ascending=False).head(TOP_N)
        html+=to()+hdr("#","Player","Team","Opp","Proj Pts","Proj Reb","Proj Ast","Szn Avg")
        for i,(_, r) in enumerate(top.iterrows(),1):
            spts=float(r.get("season_pts",0));sreb=float(r.get("season_reb",0));sast=float(r.get("season_ast",0))
            html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                      tm(r.get("team","")),r.get("opponent",""),
                      vl(float(r.get("proj_pts",0)),".1f","#1565c0","14px"),
                      vl(float(r.get("proj_reb",0)),".1f",color="#16a085"),
                      vl(float(r.get("proj_ast",0)),".1f",color="#e67e22"),
                      f'{spts:.0f}P/{sreb:.0f}R/{sast:.0f}A')
        html+=tc()
        # Rebounds
        html+=ch("💪 Rebounding Leaders",f"Top {TOP_N} by projected rebounds")
        top_r=elite.sort_values("proj_reb",ascending=False).head(TOP_N)
        html+=to()+hdr("#","Player","Team","Opp","Proj Reb","Proj Pts","Proj Ast","DD Prob")
        for i,(_, r) in enumerate(top_r.iterrows(),1):
            dd=float(r.get("proj_dd",0));dc="#27ae60"if dd>=0.40 else"#6c757d"
            html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                      tm(r.get("team","")),r.get("opponent",""),
                      vl(float(r.get("proj_reb",0)),".1f","#16a085","14px"),
                      vl(float(r.get("proj_pts",0)),".1f",color="#1565c0"),
                      vl(float(r.get("proj_ast",0)),".1f",color="#e67e22"),
                      vl(dd,".0%",dc))
        html+=tc()
        # Assists
        html+=ch("🎯 Playmakers",f"Top {TOP_N} by projected assists")
        top_a=elite.sort_values("proj_ast",ascending=False).head(TOP_N)
        html+=to()+hdr("#","Player","Team","Opp","Proj Ast","Proj Pts","Proj 3PM","Proj Stocks")
        for i,(_, r) in enumerate(top_a.iterrows(),1):
            html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                      tm(r.get("team","")),r.get("opponent",""),
                      vl(float(r.get("proj_ast",0)),".1f","#e67e22","14px"),
                      vl(float(r.get("proj_pts",0)),".1f",color="#1565c0"),
                      vl(float(r.get("proj_fg3m",0)),".1f",color="#8e44ad"),
                      vl(float(r.get("proj_stocks",0)),".1f",color="#c0392b"))
        html+=tc()
        # DD candidates
        if "proj_dd" in elite.columns:
            dd_c=elite[elite["proj_dd"]>=0.35].sort_values("proj_dd",ascending=False)
            if not dd_c.empty:
                html+=ch("🔥 Double-Double Candidates","≥35% probability")
                html+=to()+hdr("#","Player","Team","Opp","DD Prob","Proj Pts","Proj Reb","Proj Ast")
                for i,(_, r) in enumerate(dd_c.head(TOP_N).iterrows(),1):
                    dd=float(r.get("proj_dd",0))
                    html+=row(i,rk(i),nm(r.get("player_name",""),r.get("game_label","")),
                              tm(r.get("team","")),r.get("opponent",""),
                              vl(dd,".0%","#27ae60","14px"),
                              vl(float(r.get("proj_pts",0)),".1f",color="#1565c0"),
                              vl(float(r.get("proj_reb",0)),".1f",color="#16a085"),
                              vl(float(r.get("proj_ast",0)),".1f",color="#e67e22"))
                html+=tc()
    if d["game_projections"]:
        html+=ch("🎰 Game Projections")
        html+=to()+hdr("Matchup","Away","Home","Total Pts","Spread")
        for i,p in enumerate(d["game_projections"]):
            away=p.get("away_team","");home=p.get("home_team","")
            html+=grow(i,f'{away} @ {home}',
                f'{p.get("away_ml_display","")} ({float(p.get("away_win_prob",0)):.0%})',
                f'{p.get("home_ml_display","")} ({float(p.get("home_win_prob",0)):.0%})',
                f'{float(p.get("total_proj_pts",0)):.1f} pts',
                f'{p.get("puck_line_away","")} ({float(p.get("away_cover_prob",0)):.0%})')
        html+=tc()
    return html


def build_html():
    nhl=load("nhl");mlb=load("mlb");nba=load("nba")
    n=sum(int((d["predictions"]["confidence"]=="Elite").sum())
          for d in[nhl,mlb,nba]
          if not d["predictions"].empty and"confidence"in d["predictions"].columns)
    html=f"""<!DOCTYPE html><html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sports Predictor — {TODAY_LONG}</title></head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:#212529;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f2f5;">
<tr><td align="center" style="padding:20px 10px;">
<table width="700" cellpadding="0" cellspacing="0"
  style="max-width:700px;width:100%;background:#fff;border-radius:12px;
  overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,.10);">
<tr><td style="background:linear-gradient(135deg,#1a237e,#1565c0 60%,#0288d1);
  padding:30px 32px;text-align:center;">
  <div style="font-size:38px;margin-bottom:6px;">🏆</div>
  <h1 style="color:#fff;margin:0;font-size:22px;font-weight:800;">Sports Predictor</h1>
  <p style="color:#90caf9;margin:5px 0 0;font-size:14px;">{TODAY_LONG}</p>
  <div style="margin-top:12px;display:inline-block;background:rgba(255,255,255,.15);
    border-radius:20px;padding:5px 16px;color:#fff;font-size:13px;font-weight:600;">
    {n} Elite picks · NHL · MLB · NBA</div>
</td></tr>
<tr><td style="padding:8px 28px 28px;">
"""
    html+=nhl_html(nhl)+mlb_html(mlb)+nba_html(nba)
    html+=f"""<div style="margin-top:28px;padding:16px;background:#f8f9fa;border-radius:8px;
  text-align:center;color:#adb5bd;font-size:11px;line-height:1.6;">
  Sports Predictor · {TODAY_LONG}<br>For entertainment only. Not financial or gambling advice.
</div></td></tr></table></td></tr></table></body></html>"""
    return html


def send_email(html,sender,password,recipients):
    msg=MIMEMultipart("alternative")
    msg["Subject"]=f"🏆 Sports Predictor — {TODAY_SHORT} Elite Picks"
    msg["From"]=sender; msg["To"]=", ".join(recipients)
    msg.attach(MIMEText(html,"html"))
    print(f"Sending to: {', '.join(recipients)}")
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
        s.login(sender,password); s.sendmail(sender,recipients,msg.as_string())
    print("✅ Email sent")


if __name__=="__main__":
    sender=os.environ.get("EMAIL_SENDER","")
    password=os.environ.get("EMAIL_PASSWORD","")
    rcpts=os.environ.get("EMAIL_RECIPIENTS","")
    if not all([sender,password,rcpts]):
        print("❌ Missing EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECIPIENTS"); sys.exit(1)
    recipients=[r.strip() for r in rcpts.split(",") if r.strip()]
    print(f"Building email for {TODAY_LONG}…")
    html=build_html()
    out=PRED_DIR/"daily_email.html"; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(html); print(f"Preview → {out}")
    send_email(html,sender,password,recipients)
