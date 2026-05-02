"""nba_tab.py — NBA tab UI renderer"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from app.auth import is_admin
import pandas as pd
import plotly.graph_objects as go
from nba_client import NBA_TEAM_NAMES
from nba_pipeline import NBAPipeline

ET = ZoneInfo("America/New_York")

def badge(conf):
    m={"Elite":"badge-elite","High":"badge-high","Medium":"badge-medium","Low":"badge-low"}
    return f'<span class="{m.get(str(conf),"badge-low")}">{conf}</span>'

def bar(val, max_val, colour, fmt=".1f"):
    pct = min(val/max_val*100,100) if max_val>0 else 0
    return (f'<div style="display:flex;align-items:center;gap:5px;">'
            f'<div style="flex:1;background:#1e2535;border-radius:5px;height:10px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{colour};border-radius:5px;"></div></div>'
            f'<span style="font-weight:700;color:#e8ecf4;min-width:34px;font-size:.88rem;">{val:{fmt}}</span></div>')

def render_nba(selected_date: str, force_retrain: bool):
    for k,v in {"nba_pipeline":None,"nba_preds":pd.DataFrame(),
                "nba_running":False,"nba_last_run":None,
                "nba_games":[],"nba_teams":[]}.items():
        if k not in st.session_state: st.session_state[k]=v


    # ── Auto-load from warm cache on first visit ──────────────────────────────
    if not st.session_state.get("nba_auto_loaded", False) and \
       st.session_state.nba_preds.empty:
        st.session_state["nba_auto_loaded"] = True
        st.session_state.nba_running = True
        st.rerun()

    if is_admin() and st.button("🏀 Load / Refresh NBA Predictions", type="primary",
                  use_container_width=True, key="nba_load"):
        st.session_state.nba_running = True

    if st.session_state.nba_running:
        st.session_state.nba_running = False
        pb=st.progress(0.0); stxt=st.empty()
        def upd(msg,f): pb.progress(min(f,1.0)); stxt.markdown(f"⚙️ **{msg}**")
        with st.spinner("Running NBA pipeline …"):
            try:
                pipe  = NBAPipeline()
                preds = pipe.run(force_retrain=force_retrain,
                                 status_callback=upd, date=selected_date)
                st.session_state.nba_pipeline = pipe
                st.session_state.nba_preds    = preds
                st.session_state.nba_games    = pipe.get_games()
                st.session_state.nba_teams    = pipe.get_teams_playing()
                st.session_state.nba_last_run = datetime.now(ET).strftime("%I:%M %p ET")
                pb.progress(1.0); stxt.markdown("✅ **Done!**"); time.sleep(0.6)
            except Exception as e:
                st.error(f"NBA Pipeline error: {e}"); st.exception(e)
        pb.empty(); stxt.empty(); st.rerun()

    preds    = st.session_state.nba_preds
    pipeline = st.session_state.nba_pipeline
    games    = st.session_state.nba_games

    if st.session_state.nba_last_run:
        st.caption(f"Last updated: {st.session_state.nba_last_run}")

    if preds.empty:
        st.markdown("""<div style="text-align:center;padding:2rem 0;">
          <div style="font-size:4rem;">🏀</div>
          <h3 style="color:#90caf9;">Click Load to fetch today's NBA games</h3>
        </div>""", unsafe_allow_html=True)
        return

    # ── Metrics ────────────────────────────────────────────────────────────────
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div class="label">Games Today</div><div class="value">{len(games)}</div><div class="sub">NBA matchups</div></div>',unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="label">Players</div><div class="value">{len(preds)}</div><div class="sub">with predictions</div></div>',unsafe_allow_html=True)
    with m3:
        top=preds.iloc[0] if not preds.empty else None
        st.markdown(f'<div class="metric-card"><div class="label">Top Scorer</div><div class="value" style="font-size:1.1rem;">{top["player_name"] if top is not None else "—"}</div><div class="sub">Proj {top["proj_pts"]:.1f} pts</div></div>',unsafe_allow_html=True)
    with m4:
        top3=preds.sort_values("proj_fg3m",ascending=False).iloc[0] if not preds.empty else None
        st.markdown(f'<div class="metric-card"><div class="label">Top 3PT Threat</div><div class="value" style="font-size:1.1rem;">{top3["player_name"] if top3 is not None else "—"}</div><div class="sub">Proj {top3["proj_fg3m"]:.1f} 3PM</div></div>',unsafe_allow_html=True)
    st.divider()

    # ── Game projections ────────────────────────────────────────────────────────
    gprojs = pipeline.game_proj if pipeline else []
    if gprojs:
        st.markdown("### 🎰 Game Projections")
        st.caption("Model-based estimates — not official lines. Entertainment only.")
        for row_start in range(0,len(gprojs),2):
            row_games=gprojs[row_start:row_start+2]
            cols=st.columns(len(row_games))
            for col,proj in zip(cols,row_games):
                away=proj.get("away_team",""); home=proj.get("home_team","")
                away_full=NBA_TEAM_NAMES.get(away,away); home_full=NBA_TEAM_NAMES.get(home,home)
                away_ml=proj.get("away_ml_display","N/A"); home_ml=proj.get("home_ml_display","N/A")
                away_prob=proj.get("away_win_prob",0.5); home_prob=proj.get("home_win_prob",0.5)
                away_p=proj.get("away_proj_pts",0); home_p=proj.get("home_proj_pts",0)
                total=proj.get("total_proj_pts",0); fav=proj.get("favourite",home)
                sl_home=proj.get("spread_line_home",""); sl_away=proj.get("spread_line_away","")
                hcvr=proj.get("home_cover_prob",0.5); acvr=proj.get("away_cover_prob",0.5)
                o215=proj.get("over_215",0); o225=proj.get("over_225",0); o235=proj.get("over_235",0)
                rec=proj.get("recommendation","OVER"); bl=proj.get("best_ou_line",225.5)
                bp=proj.get("best_ou_prob",0.5)
                ac="#e67e22" if away==fav else "#2980b9"
                hc="#e67e22" if home==fav else "#2980b9"
                rc="#c0392b" if rec=="OVER" else "#2980b9"
                o215c="#c0392b" if o215>0.5 else "#2980b9"; o215l="O" if o215>0.5 else "U"
                o225c="#c0392b" if o225>0.5 else "#2980b9"; o225l="O" if o225>0.5 else "U"
                o235c="#c0392b" if o235>0.5 else "#2980b9"; o235l="O" if o235>0.5 else "U"
                with col:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border:1px solid #2d3550;border-radius:12px;padding:1rem;margin-bottom:6px;">'
                        f'<div style="text-align:center;padding-bottom:8px;margin-bottom:10px;border-bottom:1px solid #2d3550;">'
                        f'<div style="font-size:1.1rem;font-weight:700;color:#e8ecf4;">{away} @ {home}</div>'
                        f'<div style="font-size:.72rem;color:#8892a4;">{away_full} vs {home_full}</div></div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">',unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#111827;border-radius:8px;padding:8px;">'
                        f'<div style="font-size:.65rem;color:#8892a4;text-transform:uppercase;font-weight:600;margin-bottom:6px;">Moneyline</div>'
                        f'<div style="margin-bottom:5px;"><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;font-weight:600;">{away}</span><span style="color:{ac};font-weight:700;">{away_ml}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(away_prob*100)}%;height:100%;background:{ac};border-radius:4px;"></div></div>'
                        f'<div style="font-size:.65rem;color:#8892a4;text-align:right;">{away_prob:.0%}</div></div>'
                        f'<div><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;font-weight:600;">{home} 🏠</span><span style="color:{hc};font-weight:700;">{home_ml}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(home_prob*100)}%;height:100%;background:{hc};border-radius:4px;"></div></div>'
                        f'<div style="font-size:.65rem;color:#8892a4;text-align:right;">{home_prob:.0%}</div></div></div>',unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#111827;border-radius:8px;padding:8px;">'
                        f'<div style="font-size:.65rem;color:#8892a4;text-transform:uppercase;font-weight:600;margin-bottom:6px;">Spread</div>'
                        f'<div style="margin-bottom:5px;"><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;">{home} {sl_home}</span><span style="color:#27ae60;font-weight:700;">{hcvr:.0%}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(hcvr*100)}%;height:100%;background:#27ae60;border-radius:4px;"></div></div></div>'
                        f'<div><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;">{away} {sl_away}</span><span style="color:#27ae60;font-weight:700;">{acvr:.0%}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(acvr*100)}%;height:100%;background:#27ae60;border-radius:4px;"></div></div></div>'
                        f'<div style="margin-top:5px;font-size:.65rem;color:#8892a4;">Proj: {away} {away_p:.0f} — {home} {home_p:.0f}</div></div>',unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#111827;border-radius:8px;padding:8px;">'
                        f'<div style="font-size:.65rem;color:#8892a4;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Total Pts</div>'
                        f'<div style="font-size:1.4rem;font-weight:700;color:#e8ecf4;text-align:center;">{total:.0f}</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;margin:5px 0;text-align:center;">'
                        f'<div style="background:#1e2535;border-radius:4px;padding:3px;"><div style="font-size:.6rem;color:#8892a4;">215</div>'
                        f'<div style="font-size:.7rem;font-weight:700;color:{o215c};">{o215l} {o215:.0%}</div></div>'
                        f'<div style="background:#1e2535;border-radius:4px;padding:3px;"><div style="font-size:.6rem;color:#8892a4;">225</div>'
                        f'<div style="font-size:.7rem;font-weight:700;color:{o225c};">{o225l} {o225:.0%}</div></div>'
                        f'<div style="background:#1e2535;border-radius:4px;padding:3px;"><div style="font-size:.6rem;color:#8892a4;">235</div>'
                        f'<div style="font-size:.7rem;font-weight:700;color:{o235c};">{o235l} {o235:.0%}</div></div></div>'
                        f'<div style="text-align:center;padding:3px;border-radius:5px;background:{rc}22;border:1px solid {rc};">'
                        f'<span style="color:{rc};font-weight:700;font-size:.7rem;">{rec} {bl:.0f} · {bp:.0%}</span></div></div>',unsafe_allow_html=True)
                    st.markdown('</div></div>',unsafe_allow_html=True)
        st.divider()

    # ── Filters ────────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Player Projections")
    teams = st.session_state.nba_teams
    game_labels = sorted(preds["game_label"].dropna().unique().tolist()) if "game_label" in preds.columns else []

    f1,f2,f3,f4,f5 = st.columns([2,2,1.5,1.8,1])
    with f1:
        st.markdown('<div class="filter-label">Team</div>',unsafe_allow_html=True)
        to=["🏀 All Teams"]+[f"{t} — {NBA_TEAM_NAMES.get(t,t)}" for t in teams]
        sl=st.selectbox("Team",to,index=0,label_visibility="collapsed",key="nba_team")
        ft=None if sl=="🏀 All Teams" else sl.split(" — ")[0]
    with f2:
        st.markdown('<div class="filter-label">Game</div>',unsafe_allow_html=True)
        go2=["🏀 All Games"]+game_labels
        sg=st.selectbox("Game",go2,index=0,label_visibility="collapsed",key="nba_game")
        fg=None if sg=="🏀 All Games" else sg
    with f3:
        st.markdown('<div class="filter-label">Sort By</div>',unsafe_allow_html=True)
        sm={"Proj Pts":"proj_pts","Proj Reb":"proj_reb","Proj Ast":"proj_ast",
            "Proj 3PM":"proj_fg3m","Proj Stocks":"proj_stocks"}
        ss=st.selectbox("Sort",list(sm.keys()),index=0,label_visibility="collapsed",key="nba_sort")
        sc=sm[ss]
    with f4:
        st.markdown('<div class="filter-label">Confidence</div>',unsafe_allow_html=True)
        sconf=st.selectbox("Conf",["All","Elite","High","Medium","Low"],index=0,label_visibility="collapsed",key="nba_conf")
    with f5:
        st.markdown('<div class="filter-label">Show</div>',unsafe_allow_html=True)
        top_n=st.number_input("N",5,100,25,5,label_visibility="collapsed",key="nba_n")

    filt=preds.copy()
    if ft: filt=filt[filt["team"]==ft]
    if fg: filt=filt[filt["game_label"]==fg]
    if sconf!="All" and "confidence" in filt.columns: filt=filt[filt["confidence"]==sconf]
    if sc in filt.columns: filt=filt.sort_values(sc,ascending=False)
    disp=filt.head(int(top_n))
    st.caption(f"Showing {len(disp)} of {len(filt)} players · sorted by {ss}")

    if disp.empty:
        st.info("No players match filters.")
        return

    grid="36px 1fr 60px 60px 90px 105px 95px 95px 95px 75px"
    hdr="#  |Player|Team|Opp|Conf|Proj Pts|Proj Reb|Proj Ast|Proj 3PM|Season".split("|")
    hstyle=(f"display:grid;grid-template-columns:{grid};gap:6px;padding:7px 12px;"
            "background:#1a1f2e;border-radius:8px;font-size:.68rem;color:#8892a4;"
            "text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;border:1px solid #2d3550;")
    st.markdown(f'<div style="{hstyle}">'+"".join(f"<div>{h}</div>" for h in hdr)+"</div>",unsafe_allow_html=True)

    for rank,(_,row) in enumerate(disp.iterrows(),1):
        name=row.get("player_name",""); team=row.get("team",""); opp=row.get("opponent","")
        conf=str(row.get("confidence","Low")); gp=int(row.get("gp",0))
        spts=float(row.get("season_pts",0)); sreb=float(row.get("season_reb",0)); sast=float(row.get("season_ast",0))
        pp=float(row.get("proj_pts",0)); pr=float(row.get("proj_reb",0))
        pa=float(row.get("proj_ast",0)); p3=float(row.get("proj_fg3m",0))
        row_bg="#0f1320" if rank%2==0 else "#111827"
        rc2="#e74c3c" if rank<=3 else "#8892a4"
        pc="#c0392b" if pp>=25 else "#e67e22" if pp>=18 else "#2980b9"
        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:6px;padding:8px 12px;'
            f'background:{row_bg};border-radius:8px;margin-bottom:2px;align-items:center;border:1px solid #1e2535;">'
            f'<div style="font-weight:700;color:{rc2};">#{rank}</div>'
            f'<div><div style="font-weight:600;color:#e8ecf4;font-size:.9rem;">{name}</div>'
            f'<div style="font-size:.66rem;color:#5a7fc4;">{row.get("game_label","")}</div></div>'
            f'<div style="font-weight:600;color:#7eb3ff;font-size:.85rem;">{team}</div>'
            f'<div style="color:#8892a4;font-size:.85rem;">{opp}</div>'
            f'<div>{badge(conf)}</div>'
            f'<div>{bar(pp,45.0,pc)}</div>'
            f'<div>{bar(pr,15.0,"#1a6b4a")}</div>'
            f'<div>{bar(pa,12.0,"#8e44ad")}</div>'
            f'<div>{bar(p3,5.0,"#e67e22")}</div>'
            f'<div style="font-size:.68rem;color:#8892a4;">{spts:.1f}p<br/>{sreb:.1f}r {sast:.1f}a</div>'
            f'</div>',unsafe_allow_html=True)

    st.divider()
    t1,t2=st.tabs(["📊 Charts","🔬 Feature Importance"])
    with t1:
        cdf=disp.head(20)
        if not cdf.empty:
            c1,c2=st.columns(2)
            with c1:
                fig=go.Figure(go.Bar(x=cdf["player_name"],y=cdf["proj_pts"].round(1),
                    marker_color=["#c0392b" if v>=25 else "#e67e22" if v>=18 else "#2980b9" for v in cdf["proj_pts"]],
                    text=[f"{v:.1f}" for v in cdf["proj_pts"]],textposition="outside"))
                fig.update_layout(title="Projected Points",xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117",paper_bgcolor="#0e1117",font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"),yaxis=dict(gridcolor="#1e2535"),
                    height=370,margin=dict(t=45,b=110))
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                fig2=go.Figure(go.Bar(x=cdf["player_name"],y=cdf["proj_fg3m"].round(1),
                    marker_color="#e67e22",text=[f"{v:.1f}" for v in cdf["proj_fg3m"]],
                    textposition="outside"))
                fig2.update_layout(title="Projected 3-Pointers Made",xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117",paper_bgcolor="#0e1117",font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"),yaxis=dict(gridcolor="#1e2535"),
                    height=370,margin=dict(t=45,b=110))
                st.plotly_chart(fig2,use_container_width=True)
    with t2:
        if pipeline:
            m=pipeline.models.get("pts")
            if m and m.is_trained:
                fi=m.feature_importance()
                if not fi.empty:
                    top=fi.head(15).sort_values("importance")
                    fig3=go.Figure(go.Bar(y=top["feature"],x=top["pct"],orientation="h",
                        marker_color="#2980b9",text=[f"{v:.1f}%" for v in top["pct"]],
                        textposition="outside"))
                    fig3.update_layout(title="Points Model — Feature Importance",
                        plot_bgcolor="#0e1117",paper_bgcolor="#0e1117",font=dict(color="#e8ecf4"),
                        xaxis=dict(gridcolor="#1e2535"),yaxis=dict(gridcolor="#1e2535"),
                        height=450,margin=dict(l=180))
                    st.plotly_chart(fig3,use_container_width=True)