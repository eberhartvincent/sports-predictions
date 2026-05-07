"""mlb_tab.py — MLB predictions UI with batter/pitcher toggle"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from app.auth import is_admin
import pandas as pd
import plotly.graph_objects as go

from mlb_pipeline import MLBPipeline
from config import MLB_TEAM_NAMES

ET = ZoneInfo("America/New_York")


def _badge(text, bg, fg="#fff"):
    return (f'<span style="background:{bg};color:{fg};padding:2px 7px;'
            f'border-radius:4px;font-size:.68rem;font-weight:700;">{text}</span>')

def _conf_badge(conf):
    colors = {"Elite":"#c0392b","High":"#e67e22","Medium":"#2980b9","Low":"#555"}
    return _badge(conf, colors.get(str(conf),"#555"))

def _qual_badge(qual):
    colors = {"Ace":"#c0392b","Above Avg":"#e67e22","Average":"#2980b9",
              "Below Avg":"#7f8c8d","Avoid":"#922b21"}
    return _badge(qual, colors.get(qual,"#555"))

def _bar(val, max_val, colour, fmt=".2f"):
    pct = min(val / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="flex:1;background:#1e2535;border-radius:5px;height:9px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{colour};border-radius:5px;"></div></div>'
        f'<span style="font-weight:700;color:#e8ecf4;min-width:36px;font-size:.88rem;">{val:{fmt}}</span></div>'
    )


def render_mlb(selected_date: str, force_retrain: bool):
    # ── Session state ──────────────────────────────────────────────────────────
    for k, v in {
        "mlb_pipeline": None, "mlb_preds": pd.DataFrame(),
        "mlb_pitcher_preds": pd.DataFrame(),
        "mlb_running": False, "mlb_last_run": None,
        "mlb_games": [], "mlb_teams": [],
        "mlb_view": "batters",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


    from app.prediction_store import load_predictions, last_updated, predictions_mtime, predictions_mtime

    # ── Load from pre-computed predictions (instant) ──────────────────────────
    _disk_mtime   = predictions_mtime("mlb")
    _session_mtime = st.session_state.get("_mlb_mtime")
    if st.session_state.mlb_preds.empty or (_disk_mtime and _disk_mtime != _session_mtime):
        stored = load_predictions("mlb")
        if not stored["predictions"].empty:
            st.session_state.mlb_preds         = stored["predictions"]
            st.session_state.mlb_pitcher_preds = stored["pitcher_predictions"]
            st.session_state.mlb_games         = stored["games"]
            st.session_state.mlb_teams         = sorted(
                stored["predictions"]["team"].dropna().unique().tolist()
            ) if "team" in stored["predictions"].columns else []
            st.session_state._mlb_game_proj    = stored["game_projections"]
            st.session_state.mlb_last_run      = last_updated("mlb") or "pre-computed"
            st.session_state._mlb_mtime        = _disk_mtime

    # ── Admin: refresh button ─────────────────────────────────────────────────
    if is_admin() and st.button("⚾ Refresh MLB Predictions", type="primary",
                  use_container_width=True, key="mlb_load"):
        st.session_state.mlb_running = True

    if st.session_state.mlb_running:
        st.session_state.mlb_running = False
        pb = st.progress(0.0); stxt = st.empty()
        def upd(msg, f):
            pb.progress(min(f,1.0)); stxt.markdown(f"⚙️ **{msg}**")
        with st.spinner("Running MLB pipeline …"):
            try:
                pipe  = MLBPipeline()
                preds = pipe.run(force_retrain=force_retrain,
                                 status_callback=upd, date=selected_date)
                st.session_state.mlb_pipeline      = pipe
                st.session_state.mlb_preds         = preds
                st.session_state.mlb_pitcher_preds = getattr(pipe,"pitcher_predictions",pd.DataFrame())
                st.session_state.mlb_games         = pipe.get_games()
                st.session_state.mlb_teams         = pipe.get_teams_playing()
                st.session_state._mlb_game_proj    = pipe.game_proj
                st.session_state.mlb_last_run      = datetime.now(ET).strftime("%I:%M %p ET")
                pb.progress(1.0); stxt.markdown("✅ **Done!**"); time.sleep(0.5)
            except Exception as e:
                st.error(f"MLB Pipeline error: {e}"); st.exception(e)
        pb.empty(); stxt.empty(); st.rerun()

    preds    = st.session_state.mlb_preds
    p_preds  = st.session_state.mlb_pitcher_preds
    pipeline = st.session_state.mlb_pipeline
    games    = st.session_state.mlb_games

    if st.session_state.mlb_last_run:
        st.caption(f"Last updated: {st.session_state.mlb_last_run}")

    if preds.empty and p_preds.empty:
        st.markdown("""<div style="text-align:center;padding:2rem 0;">
          <div style="font-size:4rem;">⚾</div>
          <h3 style="color:#90caf9;">Click Load to fetch today's MLB games</h3>
        </div>""", unsafe_allow_html=True)
        return

    # ── Summary bar ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="label">Games</div>'
                    f'<div class="value">{len(games)}</div>'
                    f'<div class="sub">today</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="label">Batters</div>'
                    f'<div class="value">{len(preds)}</div>'
                    f'<div class="sub">with projections</div></div>', unsafe_allow_html=True)
    with m3:
        top = preds.sort_values("proj_hits", ascending=False).iloc[0] if not preds.empty else None
        st.markdown(f'<div class="metric-card"><div class="label">Top Hitter</div>'
                    f'<div class="value" style="font-size:1.05rem;">{top["player_name"] if top is not None else "—"}</div>'
                    f'<div class="sub">{top["proj_hits"]:.2f} proj H</div></div>'
                    if top is not None else '<div class="metric-card"><div class="label">Top Hitter</div><div class="value">—</div></div>',
                    unsafe_allow_html=True)
    with m4:
        top_hr = preds.sort_values("proj_hr", ascending=False).iloc[0] if not preds.empty else None
        st.markdown(f'<div class="metric-card"><div class="label">Top HR Threat</div>'
                    f'<div class="value" style="font-size:1.05rem;">{top_hr["player_name"] if top_hr is not None else "—"}</div>'
                    f'<div class="sub">{top_hr["proj_hr"]:.3f} HR prob</div></div>'
                    if top_hr is not None else '<div class="metric-card"><div class="label">Top HR</div><div class="value">—</div></div>',
                    unsafe_allow_html=True)
    st.divider()

    # ── Game projection cards ─────────────────────────────────────────────────
    gprojs = st.session_state.get("_mlb_game_proj",
             pipeline.game_proj if pipeline else [])
    if gprojs:
        for row_start in range(0, len(gprojs), 3):
            cols = st.columns(min(3, len(gprojs) - row_start))
            for col, proj in zip(cols, gprojs[row_start:row_start+3]):
                away, home = proj["away_team"], proj["home_team"]
                game_info  = next((g for g in games
                                   if str(g.get("game_id",""))==str(proj.get("game_id",""))), {})
                away_p = game_info.get("away_pitcher_name","TBD")
                home_p = game_info.get("home_pitcher_name","TBD")
                fav    = proj["favourite"]
                wx     = proj.get("weather","")
                rec    = proj.get("recommendation","OVER")
                rc     = "#c0392b" if rec=="OVER" else "#2980b9"
                with col:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border:1px solid #2d3550;'
                        f'border-radius:10px;padding:0.85rem;margin-bottom:6px;">'
                        f'<div style="font-size:1rem;font-weight:700;color:#e8ecf4;text-align:center;">'
                        f'{away} @ {home}</div>'
                        f'<div style="font-size:.65rem;color:#5a7fc4;text-align:center;margin-bottom:6px;">'
                        f'{away_p} vs {home_p}</div>'
                        + (f'<div style="font-size:.63rem;color:#e67e22;text-align:center;'
                           f'margin-bottom:6px;">🌤 {wx}</div>' if wx else '')
                        + f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:.72rem;">'
                        f'<div style="background:#111827;border-radius:6px;padding:5px;text-align:center;">'
                        f'<div style="color:#8892a4;">Away win</div>'
                        f'<div style="color:{"#e67e22" if away==fav else "#e8ecf4"};font-weight:700;">'
                        f'{proj["away_win_prob"]:.0%} {proj["away_ml_display"]}</div></div>'
                        f'<div style="background:#111827;border-radius:6px;padding:5px;text-align:center;">'
                        f'<div style="color:#8892a4;">Home win</div>'
                        f'<div style="color:{"#e67e22" if home==fav else "#e8ecf4"};font-weight:700;">'
                        f'{proj["home_win_prob"]:.0%} {proj["home_ml_display"]}</div></div>'
                        f'<div style="background:#111827;border-radius:6px;padding:5px;text-align:center;">'
                        f'<div style="color:#8892a4;">Total proj</div>'
                        f'<div style="color:#e8ecf4;font-weight:700;">{proj["total_proj_runs"]:.1f} R</div></div>'
                        f'<div style="background:{rc}22;border:1px solid {rc};border-radius:6px;'
                        f'padding:5px;text-align:center;">'
                        f'<div style="color:#8892a4;">Best bet</div>'
                        f'<div style="color:{rc};font-weight:700;">{rec} {proj["best_ou_line"]} '
                        f'({proj["best_ou_prob"]:.0%})</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True)
        st.divider()

    # ── Batter / Pitcher toggle ───────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🏏  Batter Projections", use_container_width=True,
                     type="primary" if st.session_state.mlb_view=="batters" else "secondary"):
            st.session_state.mlb_view = "batters"; st.rerun()
    with c2:
        if st.button("⚾  Pitcher Projections", use_container_width=True,
                     type="primary" if st.session_state.mlb_view=="pitchers" else "secondary"):
            st.session_state.mlb_view = "pitchers"; st.rerun()

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.mlb_view == "batters":
        _render_batters(preds, games, pipeline)
    else:
        _render_pitchers(p_preds, games)


# ── Batter table ──────────────────────────────────────────────────────────────

def _render_batters(preds, games, pipeline):
    if preds.empty:
        st.info("No batter projections available.")
        return

    st.markdown("### 🏏 Batter Projections")

    teams       = sorted(preds["team"].dropna().unique().tolist()) if "team" in preds.columns else []
    game_labels = sorted(preds["game_label"].dropna().unique().tolist()) if "game_label" in preds.columns else []

    f1, f2, f3, f4, f5 = st.columns([2,2,1.5,1.8,1])
    with f1:
        team_opts = ["All Teams"] + [f"{t} — {MLB_TEAM_NAMES.get(t,t)}" for t in teams]
        sel_team  = st.selectbox("Team", team_opts, key="mlb_b_team", label_visibility="collapsed")
        ft = None if sel_team=="All Teams" else sel_team.split(" — ")[0]
    with f2:
        game_opts = ["All Games"] + game_labels
        sel_game  = st.selectbox("Game", game_opts, key="mlb_b_game", label_visibility="collapsed")
        fg = None if sel_game=="All Games" else sel_game
    with f3:
        sort_map = {"H+R+RBI":"proj_hrr","Proj H":"proj_hits","Proj HR":"proj_hr",
                    "Proj RBI":"proj_rbi","Proj R":"proj_runs","Proj TB":"proj_tb","Proj K":"proj_k"}
        sel_sort = st.selectbox("Sort", list(sort_map.keys()), key="mlb_b_sort", label_visibility="collapsed")
        sc = sort_map[sel_sort]
    with f4:
        sel_conf = st.selectbox("Confidence", ["All","Elite","High","Medium","Low"],
                                key="mlb_b_conf", label_visibility="collapsed")
    with f5:
        top_n = st.number_input("N", 5, 150, 30, 5, key="mlb_b_n", label_visibility="collapsed")

    filt = preds.copy()
    if ft:  filt = filt[filt["team"]==ft]
    if fg:  filt = filt[filt["game_label"]==fg]
    if sel_conf != "All" and "confidence" in filt.columns:
        filt = filt[filt["confidence"]==sel_conf]
    if sc in filt.columns:
        filt = filt.sort_values(sc, ascending=False)
    disp = filt.head(int(top_n))
    st.caption(f"Showing {len(disp)} of {len(filt)} batters · sorted by {sel_sort}")

    if disp.empty:
        st.info("No batters match filters."); return

    # Column layout: # | Player | Team | Opp | Conf | H | HR | RBI | R | TB | Season/BvP
    grid   = "28px 1fr 52px 52px 72px 95px 80px 80px 80px 80px 100px"
    hdrs   = "#|Player|Team|Opp|Conf|H+R+RBI|Proj H|Proj HR|Proj RBI|Proj R|Season/BvP"
    hstyle = (f"display:grid;grid-template-columns:{grid};gap:5px;padding:6px 12px;"
              "background:#1a1f2e;border-radius:8px;font-size:.62rem;color:#8892a4;"
              "text-transform:uppercase;margin-bottom:4px;border:1px solid #2d3550;")
    st.markdown(f'<div style="{hstyle}">' +
                "".join(f"<div>{h}</div>" for h in hdrs.split("|")) + "</div>",
                unsafe_allow_html=True)

    for rank, (_, row) in enumerate(disp.iterrows(), 1):
        name    = row.get("player_name","")
        team    = row.get("team","")
        opp     = row.get("opponent","")
        conf    = str(row.get("confidence","Low"))
        gp      = int(row.get("gp",0))
        savg    = float(row.get("season_avg",0))
        shr     = int(row.get("season_hr",0))
        sh      = int(row.get("season_hits",0))
        ph      = float(row.get("proj_hits",0))
        phr     = float(row.get("proj_hr",0))
        prbi    = float(row.get("proj_rbi",0))
        pruns   = float(row.get("proj_runs",0))
        ptb     = float(row.get("proj_tb",0))
        phrr    = float(row.get("proj_hrr", ph + pruns + prbi))
        wx      = str(row.get("weather",""))
        bvp_ab  = int(row.get("bvp_ab",0))
        bvp_hr  = int(row.get("bvp_hr",0))
        p_hand  = str(row.get("pitcher_hand","R"))
        b_hand  = str(row.get("batter_hand","R"))

        bg      = "#0f1320" if rank%2==0 else "#111827"
        rc      = "#e74c3c" if rank<=3 else "#8892a4"
        hc      = "#27ae60" if ph>=0.90 else "#f39c12" if ph>=0.70 else "#2980b9"
        hrcolor = "#c0392b" if phr>=0.08 else "#e67e22" if phr>=0.04 else "#7f8c8d"

        platoon = ""
        if (b_hand=="L" and p_hand=="R") or (b_hand=="R" and p_hand=="L"):
            platoon = ' <span style="background:#1a5c3a;color:#4ade80;padding:1px 4px;border-radius:3px;font-size:.55rem;">ADV</span>'

        wx_icon = "🌤" if wx and "out" in wx else "🌬" if wx and ("in " in wx or "cross" in wx) else ""
        bvp_str = f"{bvp_ab}AB {bvp_hr}HR" if bvp_ab>=5 else "—"
        bvp_col = "#f59e0b" if bvp_ab>=20 else "#8892a4"

        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:5px;padding:7px 12px;'
            f'background:{bg};border-radius:8px;margin-bottom:2px;align-items:center;'
            f'border:1px solid #1e2535;">'
            f'<div style="font-weight:700;color:{rc};font-size:.82rem;">#{rank}</div>'
            f'<div>'
            f'  <div style="font-weight:600;color:#e8ecf4;font-size:.85rem;">{name}{platoon}</div>'
            f'  <div style="font-size:.6rem;color:#5a7fc4;">{row.get("game_label","")}'
            + (f' {wx_icon} <span style="color:#e67e22;">{wx}</span>' if wx else '')
            + f'</div></div>'
            f'<div style="color:#7eb3ff;font-size:.8rem;font-weight:600;">{team}</div>'
            f'<div style="color:#8892a4;font-size:.8rem;">{opp}</div>'
            f'<div>{_conf_badge(conf)}</div>'
            f'<div style="text-align:center;">'
            f'  <div style="font-size:1.1rem;font-weight:800;'
            f'color:{"#f1c40f" if phrr>=2.5 else "#e8ecf4"};">{phrr:.2f}</div>'
            f'  <div style="font-size:.58rem;color:#8892a4;">H+R+RBI</div>'
            f'</div>'
            f'<div>{_bar(ph, 2.0, hc)}</div>'
            f'<div>{_bar(phr, 0.20, hrcolor, ".3f")}</div>'
            f'<div>{_bar(prbi, 2.5, "#e67e22", ".2f")}</div>'
            f'<div>{_bar(pruns, 2.0, "#16a085", ".2f")}</div>'
            f'<div style="font-size:.62rem;line-height:1.5;">'
            f'  <span style="color:#8892a4;">.{int(savg*1000):03d} · {sh}H · {shr}HR</span><br/>'
            f'  <span style="color:{bvp_col};">vs P: {bvp_str}</span><br/>'
            f'  <span style="color:#6b7280;">{gp}G played</span>'
            f'</div></div>',
            unsafe_allow_html=True)

    st.divider()

    # Quick charts
    with st.expander("📊 Charts", expanded=False):
        top20 = disp.head(20)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=top20["player_name"],
                                   y=top20["proj_hits"].round(2),
                                   marker_color="#27ae60",
                                   text=[f"{v:.2f}" for v in top20["proj_hits"]],
                                   textposition="outside"))
            fig.update_layout(title="Projected Hits", xaxis_tickangle=-40, height=350,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8ecf4"),
                margin=dict(t=40,b=100))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Bar(x=top20["player_name"],
                                    y=top20["proj_hr"].round(3),
                                    marker_color="#c0392b",
                                    text=[f"{v:.3f}" for v in top20["proj_hr"]],
                                    textposition="outside"))
            fig2.update_layout(title="HR Probability", xaxis_tickangle=-40, height=350,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8ecf4"),
                margin=dict(t=40,b=100))
            st.plotly_chart(fig2, use_container_width=True)


# ── Pitcher table ─────────────────────────────────────────────────────────────

def _render_pitchers(p_preds, games):
    st.markdown("### ⚾ Starting Pitcher Projections")

    if p_preds.empty:
        st.info("No pitcher projections — run predictions first.")
        return

    # Filters
    f1, f2, f3 = st.columns([2, 2, 1.5])
    with f1:
        teams    = sorted(p_preds["team"].dropna().unique().tolist())
        team_opts= ["All Teams"] + [f"{t} — {MLB_TEAM_NAMES.get(t,t)}" for t in teams]
        sel_team = st.selectbox("Team", team_opts, key="mlb_p_team", label_visibility="collapsed")
        ft       = None if sel_team=="All Teams" else sel_team.split(" — ")[0]
    with f2:
        game_labels = sorted(p_preds["game_label"].dropna().unique().tolist())
        game_opts   = ["All Games"] + game_labels
        sel_game    = st.selectbox("Game", game_opts, key="mlb_p_game", label_visibility="collapsed")
        fg          = None if sel_game=="All Games" else sel_game
    with f3:
        sort_opts = {"ERA (low→high)":"era","Proj K":"proj_k","Proj IP":"proj_ip"}
        sel_sort  = st.selectbox("Sort", list(sort_opts.keys()), key="mlb_p_sort", label_visibility="collapsed")
        asc       = sel_sort == "ERA (low→high)"
        sc        = sort_opts[sel_sort]

    filt = p_preds.copy()
    if ft: filt = filt[filt["team"]==ft]
    if fg: filt = filt[filt["game_label"]==fg]
    filt = filt.sort_values(sc, ascending=asc).reset_index(drop=True)

    st.caption(f"Showing {len(filt)} starting pitchers · sorted by {sel_sort}")

    grid   = "28px 1fr 55px 55px 70px 80px 75px 75px 75px 80px"
    hdrs   = "#|Pitcher|Team|Opp|Quality|ERA/WHIP|Proj IP|Proj K|Proj ER|Weather"
    hstyle = (f"display:grid;grid-template-columns:{grid};gap:5px;padding:6px 12px;"
              "background:#1a1f2e;border-radius:8px;font-size:.62rem;color:#8892a4;"
              "text-transform:uppercase;margin-bottom:4px;border:1px solid #2d3550;")
    st.markdown(f'<div style="{hstyle}">' +
                "".join(f"<div>{h}</div>" for h in hdrs.split("|")) + "</div>",
                unsafe_allow_html=True)

    for rank, (_, row) in enumerate(filt.iterrows(), 1):
        name   = row.get("pitcher_name","")
        team   = row.get("team","")
        opp    = row.get("opponent","")
        qual   = row.get("quality","Average")
        era    = float(row.get("era",4.5))
        whip   = float(row.get("whip",1.3))
        k9     = float(row.get("k9",8.5))
        pip    = float(row.get("proj_ip",5.5))
        pk     = float(row.get("proj_k",5.0))
        per    = float(row.get("proj_er",2.5))
        wx     = str(row.get("weather",""))
        gl     = row.get("game_label","")
        gp     = int(row.get("gp",0))
        is_home= bool(row.get("is_home",False))

        bg  = "#0f1320" if rank%2==0 else "#111827"
        rc  = "#e74c3c" if rank<=3 else "#8892a4"
        erac= "#27ae60" if era<=3.5 else "#f39c12" if era<=4.5 else "#c0392b"
        erl = "🏠" if is_home else "✈️"

        wx_icon = "🌤" if wx and "out" in wx else "🌬" if wx and ("in " in wx or "cross" in wx) else ""

        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:5px;padding:7px 12px;'
            f'background:{bg};border-radius:8px;margin-bottom:2px;align-items:center;'
            f'border:1px solid #1e2535;">'
            f'<div style="font-weight:700;color:{rc};font-size:.82rem;">#{rank}</div>'
            f'<div>'
            f'  <div style="font-weight:600;color:#e8ecf4;font-size:.85rem;">{name} {erl}</div>'
            f'  <div style="font-size:.6rem;color:#5a7fc4;">{gl}</div>'
            f'  <div style="font-size:.6rem;color:#6b7280;">{gp}G this season</div>'
            f'</div>'
            f'<div style="color:#7eb3ff;font-size:.8rem;font-weight:600;">{team}</div>'
            f'<div style="color:#8892a4;font-size:.8rem;">vs {opp}</div>'
            f'<div>{_qual_badge(qual)}</div>'
            f'<div style="font-size:.75rem;">'
            f'  <span style="color:{erac};font-weight:700;">ERA {era:.2f}</span><br/>'
            f'  <span style="color:#8892a4;">WHIP {whip:.2f} · K/9 {k9:.1f}</span>'
            f'</div>'
            f'<div>{_bar(pip, 9.0, "#2980b9", ".1f")}</div>'
            f'<div>{_bar(pk,  12.0, "#27ae60", ".1f")}</div>'
            f'<div style="font-size:.82rem;font-weight:700;color:{erac}">{per:.2f}</div>'
            f'<div style="font-size:.65rem;color:#e67e22;">'
            + (f'{wx_icon} {wx}' if wx else '—')
            + f'</div></div>',
            unsafe_allow_html=True)

    st.divider()
    with st.expander("📊 Pitcher Charts", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=filt["pitcher_name"], y=filt["proj_k"],
                                   marker_color="#27ae60",
                                   text=[f"{v:.1f}" for v in filt["proj_k"]],
                                   textposition="outside"))
            fig.update_layout(title="Projected Strikeouts", xaxis_tickangle=-40, height=350,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8ecf4"),
                margin=dict(t=40,b=100))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Bar(x=filt["pitcher_name"], y=filt["era"],
                                    marker_color=["#27ae60" if e<=3.5 else "#e67e22" if e<=4.5
                                                  else "#c0392b" for e in filt["era"]],
                                    text=[f"{v:.2f}" for v in filt["era"]],
                                    textposition="outside"))
            fig2.update_layout(title="Season ERA (lower = better)", xaxis_tickangle=-40, height=350,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8ecf4"),
                margin=dict(t=40,b=100))
            st.plotly_chart(fig2, use_container_width=True)