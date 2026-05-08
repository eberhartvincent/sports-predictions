"""app/tabs/nhl_tab.py — NHL predictions tab"""
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def _badge(conf):
    m = {"Elite":"badge-elite","High":"badge-high","Medium":"badge-medium","Low":"badge-low"}
    return f'<span class="{m.get(str(conf),"badge-low")}">{conf}</span>'


def _bar(val, max_val, colour, fmt=".2f"):
    pct = min(val/max_val*100, 100) if max_val > 0 else 0
    return (f'<div style="display:flex;align-items:center;gap:5px;">'
            f'<div style="flex:1;background:#1e2535;border-radius:5px;height:10px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{colour};border-radius:5px;"></div></div>'
            f'<span style="font-weight:700;color:#e8ecf4;min-width:36px;font-size:.88rem;">{val:{fmt}}</span></div>')

def _apply_prob_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply physics-based probability ceiling to loaded predictions.
    A player cannot score more goals than their shot volume × sh% allows.
    Runs on every load so stale parquet files get corrected immediately.
    """
    if df.empty or "goal_probability" not in df.columns:
        return df

    LEAGUE_AVG_SH  = 0.104
    BEST_CASE_RATE = 0.130
    K_SHOTS        = 150

    df = df.copy()
    ceilings = []
    for _, row in df.iterrows():
        shots_pg    = float(row.get("season_shots_pg",
                      row.get("rolling_5g_shots", 0)) or 0)
        gp          = max(int(row.get("gp_season", row.get("gp", 1))), 1)
        season_goals= int(row.get("season_goals", 0))
        total_shots = shots_pg * gp
        raw_sh      = season_goals / max(total_shots, 1)
        w           = total_shots / (total_shots + K_SHOTS)
        reg_sh      = w * raw_sh + (1 - w) * LEAGUE_AVG_SH
        # Scale multiplier with sample — 0-goal players with < 15 GP get hard cap
        sample_mult = min(1.5, 1.0 + 0.5 * min(total_shots / 100, 1.0))
        ceiling     = shots_pg * min(reg_sh * sample_mult, BEST_CASE_RATE)
        if season_goals == 0 and gp < 15:
            ceiling = min(ceiling, 0.08)
        ceilings.append(max(0.02, min(ceiling, 0.65)))

    import numpy as np
    raw_probs  = df["goal_probability"].values.astype(float)
    ceil_arr   = np.array(ceilings)
    clipped    = np.minimum(raw_probs, ceil_arr)
    final      = 0.90 * clipped + 0.10 * np.minimum(raw_probs, 0.65)
    df["goal_probability"] = np.round(final, 4)

    def _conf(p):
        if p >= 0.32: return "Elite"
        if p >= 0.22: return "High"
        if p >= 0.14: return "Medium"
        return "Low"
    df["confidence"] = df["goal_probability"].apply(_conf)
    return df.sort_values("goal_probability", ascending=False).reset_index(drop=True)




def render_nhl(selected_date_str):
    from config import NHL_TEAMS, CURRENT_SEASON
    from core.pipelines.nhl_pipeline import NHLPipeline
    from app.auth import is_admin

    ET = ZoneInfo("America/New_York")

    # ── Session state defaults ────────────────────────────────────────────────
    for k, v in {
        "nhl_pipeline": None, "nhl_predictions": pd.DataFrame(),
        "nhl_last_run": None, "nhl_running": False,
        "nhl_teams": [], "nhl_games": [],
        "nhl_auto_loaded": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    from app.prediction_store import load_predictions, last_updated, predictions_mtime
    from pathlib import Path as _Path, predictions_mtime

    # ── Load from pre-computed predictions first (instant) ────────────────────
    # warm_cache.py saves predictions daily — just read the files.
    # Admin can force a fresh run; viewers always see the saved output.
    # Reload if empty OR if session state has a different date than the saved file
    # If admin selected a past date, load from history parquet
    _selected = date_str  # passed in from main.py
    _today    = datetime.now(ET).strftime("%Y-%m-%d") if _selected else None
    _is_today = (_selected is None or _selected == _today)
    _hist_file = _Path("data/cache/predictions/history") / f"nhl_{_selected}.parquet" if _selected and not _is_today else None

    _disk_mtime   = predictions_mtime("nhl")
    _session_mtime = st.session_state.get("_nhl_mtime")
    _session_date  = st.session_state.get("_nhl_date")
    if st.session_state.nhl_predictions.empty or (_selected != _session_date) or (_is_today and _disk_mtime and _disk_mtime != _session_mtime):
        # Load from history for past dates, today's parquet for today
        _hist = Path("data/cache/predictions/history") / f"nhl_{date_str}.parquet" \
                if date_str and date_str != datetime.now(ET).strftime("%Y-%m-%d") else None
        if _hist and _hist.exists():
            import pandas as _pd
            stored = dict(load_predictions("nhl"))
            stored["predictions"] = _pd.read_parquet(_hist)
        else:
            stored = load_predictions("nhl")
        if not stored["predictions"].empty:
            st.session_state.nhl_predictions  = _apply_prob_ceiling(stored["predictions"])
            st.session_state.nhl_games        = stored["games"]
            st.session_state.nhl_teams        = sorted(
                stored["predictions"]["team"].dropna().unique().tolist()
            ) if "team" in stored["predictions"].columns else []
            st.session_state.nhl_pipeline     = None   # no live pipeline object needed
            st.session_state._nhl_game_proj   = stored["game_projections"]
            st.session_state._nhl_metrics     = stored["metrics"]
            st.session_state.nhl_last_run     = last_updated("nhl") or "pre-computed"
            st.session_state._nhl_mtime        = _disk_mtime

    # ── Admin: refresh button to re-run the full pipeline ─────────────────────
    if is_admin():
        if st.button("🏒 Refresh NHL Predictions", type="primary",
                     use_container_width=True, key="nhl_load"):
            st.session_state.nhl_running = True

    # ── Pipeline execution (admin refresh only) ───────────────────────────────
    if st.session_state.get("nhl_running", False):
        st.session_state.nhl_running = False
        pb   = st.progress(0.0)
        stxt = st.empty()

        def upd(msg, frac):
            pb.progress(min(frac, 1.0))
            stxt.markdown(f"⚙️ **{msg}**")

        with st.spinner("Running NHL pipeline …"):
            try:
                pipe  = NHLPipeline()
                preds = pipe.run(
                    force_retrain=force_retrain,
                    status_callback=upd,
                    date=selected_date_str,
                )
                st.session_state.nhl_pipeline    = pipe
                st.session_state.nhl_predictions = _apply_prob_ceiling(preds)
                st.session_state.nhl_last_run    = datetime.now(ET).strftime("%I:%M %p ET")
                st.session_state.nhl_games       = pipe.get_games_today()
                st.session_state.nhl_teams       = (
                    sorted(preds["team"].dropna().unique().tolist())
                    if not preds.empty and "team" in preds.columns else []
                )
                st.session_state._nhl_game_proj  = pipe.game_projections
                st.session_state._nhl_metrics    = pipe.model_metrics
                pb.progress(1.0); stxt.markdown("✅ **Done!**"); time.sleep(0.5)
            except Exception as e:
                st.error(f"NHL Pipeline error: {e}"); st.exception(e)
        pb.empty(); stxt.empty()
        st.rerun()

    preds    = st.session_state.nhl_predictions
    pipeline = st.session_state.nhl_pipeline
    games    = st.session_state.nhl_games

    if st.session_state.nhl_last_run:
        st.caption(f"Last updated: {st.session_state.nhl_last_run}")

    # ── Empty state ───────────────────────────────────────────────────────────
    if preds.empty:
        st.markdown("""<div style="text-align:center;padding:2rem 0;">
          <div style="font-size:4rem;">🏒</div>
          <h3 style="color:#90caf9;">Click Load to fetch today's NHL predictions</h3>
        </div>""", unsafe_allow_html=True)
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="label">Games Today</div>'
                    f'<div class="value">{len(games)}</div>'
                    f'<div class="sub">NHL matchups</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="label">Players Analysed</div>'
                    f'<div class="value">{len(preds)}</div>'
                    f'<div class="sub">skaters with predictions</div></div>', unsafe_allow_html=True)
    with m3:
        _metrics = st.session_state.get('_nhl_metrics', pipeline.model_metrics if pipeline and pipeline.model_metrics else {})
        auc = _metrics.get("train_auc", 0)
        st.markdown(f'<div class="metric-card"><div class="label">Model CV-AUC</div>'
                    f'<div class="value">{auc:.3f}</div>'
                    f'<div class="sub">cross-validated</div></div>', unsafe_allow_html=True)
    with m4:
        top_prob = float(preds["goal_probability"].max()) if not preds.empty else 0
        top_name = preds.iloc[0]["player_name"] if not preds.empty else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Top Scorer Prob</div>'
                    f'<div class="value">{top_prob:.2f}</div>'
                    f'<div class="sub">{top_name}</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── Today's matchups ──────────────────────────────────────────────────────
    if games:
        st.markdown("### 📅 Today's Matchups")
        cols = st.columns(min(len(games), 4))
        for i, g in enumerate(games):
            away, home = g.get("away_team",""), g.get("home_team","")
            try:
                etz  = timezone(timedelta(hours=-5))
                dt   = datetime.fromisoformat(g.get("start_time_utc","").replace("Z","+00:00"))
                tstr = dt.astimezone(etz).strftime("%-I:%M %p ET")
            except Exception:
                tstr = "TBD"
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div style="background:#1a1f2e;border:1px solid #2d3550;border-radius:10px;'
                    f'padding:.8rem;text-align:center;margin-bottom:.5rem;">'
                    f'<div style="font-size:1.05rem;font-weight:700;color:#e8ecf4;">{away} @ {home}</div>'
                    f'<div style="font-size:.7rem;color:#5a7fc4;margin-top:2px;">'
                    f'{NHL_TEAMS.get(away,away)} vs {NHL_TEAMS.get(home,home)}</div>'
                    f'<div style="font-size:.72rem;color:#8892a4;">{tstr}</div>'
                    f'</div>', unsafe_allow_html=True)
        st.divider()

    # ── Game projections ──────────────────────────────────────────────────────
    game_projections = st.session_state.get('_nhl_game_proj', pipeline.game_projections if pipeline else [])
    if game_projections:
        st.markdown("### 🎰 Game Projections")
        st.caption("Model-based estimates — not official betting lines. For entertainment purposes.")
        for row_start in range(0, len(game_projections), 2):
            row_games = game_projections[row_start:row_start+2]
            cols = st.columns(len(row_games))
            for col, proj in zip(cols, row_games):
                away      = proj["away_team"]; home = proj["home_team"]
                away_full = NHL_TEAMS.get(away, away); home_full = NHL_TEAMS.get(home, home)
                away_ml   = proj.get("away_ml_display","N/A")
                home_ml   = proj.get("home_ml_display","N/A")
                away_prob = proj.get("away_win_prob", 0.5)
                home_prob = proj.get("home_win_prob", 0.5)
                away_g    = proj.get("away_proj_goals", 0)
                home_g    = proj.get("home_proj_goals", 0)
                total     = proj.get("proj_total", 0)
                fav       = proj.get("favourite", home)
                puck_home = proj.get("puck_line_home","-1.5")
                puck_away = proj.get("puck_line_away","+1.5")
                h_cov     = proj.get("home_cover_prob", 0.5)
                a_cov     = proj.get("away_cover_prob", 0.5)
                over_55   = proj.get("over_5_5_prob", 0)
                over_60   = proj.get("over_6_0_prob", 0)
                over_65   = proj.get("over_6_5_prob", 0)
                rec       = proj.get("recommendation","OVER")
                best_line = proj.get("best_ou_line", 5.5)
                best_prob = proj.get("best_ou_prob", 0.5)

                ac  = "#e67e22" if away==fav else "#2980b9"
                hc  = "#e67e22" if home==fav else "#2980b9"
                rc  = "#c0392b" if rec=="OVER" else "#2980b9"
                o55c= "#c0392b" if over_55>0.5 else "#2980b9"; o55l="O" if over_55>0.5 else "U"
                o60c= "#c0392b" if over_60>0.5 else "#2980b9"; o60l="O" if over_60>0.5 else "U"
                o65c= "#c0392b" if over_65>0.5 else "#2980b9"; o65l="O" if over_65>0.5 else "U"

                with col:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border:1px solid #2d3550;border-radius:12px;padding:1rem;margin-bottom:6px;">'
                        f'<div style="text-align:center;padding-bottom:8px;margin-bottom:10px;border-bottom:1px solid #2d3550;">'
                        f'<div style="font-size:1.1rem;font-weight:700;color:#e8ecf4;">{away} @ {home}</div>'
                        f'<div style="font-size:.72rem;color:#8892a4;">{away_full} vs {home_full}</div></div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">',
                        unsafe_allow_html=True)

                    # Moneyline
                    st.markdown(
                        f'<div style="background:#111827;border-radius:8px;padding:8px;">'
                        f'<div style="font-size:.65rem;color:#8892a4;text-transform:uppercase;font-weight:600;margin-bottom:6px;">Moneyline</div>'
                        f'<div style="margin-bottom:5px;"><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;font-weight:600;">{away}</span>'
                        f'<span style="color:{ac};font-weight:700;">{away_ml}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(away_prob*100)}%;height:100%;background:{ac};border-radius:4px;"></div></div>'
                        f'<div style="font-size:.65rem;color:#8892a4;text-align:right;">{away_prob:.0%}</div></div>'
                        f'<div><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;font-weight:600;">{home} 🏠</span>'
                        f'<span style="color:{hc};font-weight:700;">{home_ml}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(home_prob*100)}%;height:100%;background:{hc};border-radius:4px;"></div></div>'
                        f'<div style="font-size:.65rem;color:#8892a4;text-align:right;">{home_prob:.0%}</div></div></div>',
                        unsafe_allow_html=True)

                    # Puck line
                    st.markdown(
                        f'<div style="background:#111827;border-radius:8px;padding:8px;">'
                        f'<div style="font-size:.65rem;color:#8892a4;text-transform:uppercase;font-weight:600;margin-bottom:6px;">Puck Line</div>'
                        f'<div style="margin-bottom:5px;"><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;">{home} {puck_home}</span>'
                        f'<span style="color:#27ae60;font-weight:700;">{h_cov:.0%}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(h_cov*100)}%;height:100%;background:#27ae60;border-radius:4px;"></div></div></div>'
                        f'<div><div style="display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:2px;">'
                        f'<span style="color:#e8ecf4;">{away} {puck_away}</span>'
                        f'<span style="color:#27ae60;font-weight:700;">{a_cov:.0%}</span></div>'
                        f'<div style="background:#1e2535;border-radius:4px;height:8px;overflow:hidden;">'
                        f'<div style="width:{int(a_cov*100)}%;height:100%;background:#27ae60;border-radius:4px;"></div></div></div>'
                        f'<div style="margin-top:5px;font-size:.65rem;color:#8892a4;">Proj: {away} {away_g:.1f} — {home} {home_g:.1f}</div>'
                        f'</div>', unsafe_allow_html=True)

                    # Totals
                    st.markdown(
                        f'<div style="background:#111827;border-radius:8px;padding:8px;">'
                        f'<div style="font-size:.65rem;color:#8892a4;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Total Goals</div>'
                        f'<div style="font-size:1.4rem;font-weight:700;color:#e8ecf4;text-align:center;">{total:.1f}</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;margin:5px 0;text-align:center;">'
                        f'<div style="background:#1e2535;border-radius:4px;padding:3px;">'
                        f'<div style="font-size:.6rem;color:#8892a4;">5.5</div>'
                        f'<div style="font-size:.7rem;font-weight:700;color:{o55c};">{o55l} {over_55:.0%}</div></div>'
                        f'<div style="background:#1e2535;border-radius:4px;padding:3px;">'
                        f'<div style="font-size:.6rem;color:#8892a4;">6.0</div>'
                        f'<div style="font-size:.7rem;font-weight:700;color:{o60c};">{o60l} {over_60:.0%}</div></div>'
                        f'<div style="background:#1e2535;border-radius:4px;padding:3px;">'
                        f'<div style="font-size:.6rem;color:#8892a4;">6.5</div>'
                        f'<div style="font-size:.7rem;font-weight:700;color:{o65c};">{o65l} {over_65:.0%}</div></div></div>'
                        f'<div style="text-align:center;padding:3px;border-radius:5px;background:{rc}22;border:1px solid {rc};">'
                        f'<span style="color:{rc};font-weight:700;font-size:.7rem;">{rec} {best_line} · {best_prob:.0%}</span>'
                        f'</div></div>', unsafe_allow_html=True)

                    st.markdown('</div></div>', unsafe_allow_html=True)
        st.divider()

    # ── Player projections — filters ──────────────────────────────────────────
    st.markdown("### 🎯 Player Projections")
    teams_playing = st.session_state.nhl_teams
    game_labels   = sorted(preds["game_label"].dropna().unique().tolist()) \
                    if "game_label" in preds.columns else []

    fc1,fc2,fc3,fc4,fc5,fc6 = st.columns([2,2,1.4,1.4,1.6,1])
    with fc1:
        st.markdown('<div class="filter-label">Team</div>', unsafe_allow_html=True)
        team_opts = ["🌍 All Teams"] + [f"{t} — {NHL_TEAMS.get(t,t)}" for t in teams_playing]
        sel_team  = st.selectbox("Team", team_opts, index=0, label_visibility="collapsed", key="nhl_f_team")
        flt_team  = None if sel_team=="🌍 All Teams" else sel_team.split(" — ")[0]
    with fc2:
        st.markdown('<div class="filter-label">Game</div>', unsafe_allow_html=True)
        sel_game = st.selectbox("Game", ["🏒 All Games"]+game_labels, index=0,
                                label_visibility="collapsed", key="nhl_f_game")
        flt_game = None if sel_game=="🏒 All Games" else sel_game
    with fc3:
        st.markdown('<div class="filter-label">Position</div>', unsafe_allow_html=True)
        sel_pos = st.selectbox("Position", ["All","Forwards","Defence"],
                               index=0, label_visibility="collapsed", key="nhl_f_pos")
    with fc4:
        st.markdown('<div class="filter-label">Confidence</div>', unsafe_allow_html=True)
        sel_conf = st.selectbox("Confidence", ["All Tiers","Elite","High","Medium","Low"],
                                index=0, label_visibility="collapsed", key="nhl_f_conf")
        flt_conf = None if sel_conf=="All Tiers" else sel_conf
    with fc5:
        st.markdown('<div class="filter-label">Sort By</div>', unsafe_allow_html=True)
        sort_opts = {"Proj Goals":"goal_probability","Proj Assists":"projected_assists",
                     "Proj Points":"projected_points","Proj SOG":"projected_sog"}
        sel_sort  = st.selectbox("Sort", list(sort_opts.keys()), index=0,
                                 label_visibility="collapsed", key="nhl_f_sort")
        sort_col  = sort_opts[sel_sort]
        # Map sort to per-category confidence column
        nhl_conf_map = {
            "goal_probability":   "conf_goals",
            "projected_sog":      "conf_sog",
            "projected_assists":  "confidence",
            "projected_points":   "confidence",
        }
        active_conf_col = nhl_conf_map.get(sort_col, "confidence")
    with fc6:
        st.markdown('<div class="filter-label">Show</div>', unsafe_allow_html=True)
        top_n = st.number_input("Show", min_value=5, max_value=100, value=25,
                                step=5, label_visibility="collapsed", key="nhl_f_n")

    # Apply filters
    filt = preds.copy()
    if flt_team: filt = filt[filt["team"].astype(str).str.strip()==flt_team]
    if flt_game: filt = filt[filt["game_label"]==flt_game]
    if sel_pos=="Forwards" and "position" in filt.columns:
        filt = filt[~filt["position"].astype(str).str.upper().isin(["D","G"])]
    elif sel_pos=="Defence" and "position" in filt.columns:
        filt = filt[filt["position"].astype(str).str.upper()=="D"]
    if flt_conf:
        cf = active_conf_col if active_conf_col in filt.columns else "confidence"
        filt = filt[filt[cf].astype(str)==flt_conf]
    if sort_col in filt.columns:
        filt = filt.sort_values(sort_col, ascending=False)
    disp = filt.head(int(top_n))

    parts = [p for p in [flt_team, flt_game,
                          sel_pos if sel_pos!="All" else None, flt_conf] if p]
    st.caption(f"Showing {len(disp)} of {len(filt)} players"
               + (f" — {' · '.join(parts)}" if parts else "")
               + f" · sorted by {sel_sort}")

    if disp.empty:
        st.info("No players match the current filters.")
        return

    # ── Player table ──────────────────────────────────────────────────────────
    grid = "36px 1fr 64px 64px 90px 125px 105px 105px 105px 72px"
    hstyle = (f"display:grid;grid-template-columns:{grid};gap:6px;padding:7px 12px;"
              "background:#1a1f2e;border-radius:8px;font-size:.68rem;color:#8892a4;"
              "text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;"
              "border:1px solid #2d3550;")
    hdrs = ["#","Player","Team","Opp","Conf",
            f"{'Proj Goals ▼' if sel_sort=='Proj Goals' else 'Proj Goals'}",
            f"{'Proj Ast ▼' if sel_sort=='Proj Assists' else 'Proj Ast'}",
            f"{'Proj Pts ▼' if sel_sort=='Proj Points' else 'Proj Pts'}",
            f"{'Proj SOG ▼' if sel_sort=='Proj SOG' else 'Proj SOG'}",
            "Season"]
    st.markdown(f'<div style="{hstyle}">'
                + "".join(f"<div>{h}</div>" for h in hdrs)
                + "</div>", unsafe_allow_html=True)

    for rank, (_, row) in enumerate(disp.iterrows(), 1):
        name     = row.get("player_name","")
        team     = row.get("team","")
        opp      = row.get("opponent","")
        prob     = float(row.get("goal_probability",0))
        conf     = str(row.get("confidence","Low"))
        gp       = int(row.get("gp_season",0))
        goals    = int(row.get("season_goals",0))
        assists  = int(row.get("season_assists",0))
        proj_sog = float(row.get("projected_sog",0))
        proj_pts = float(row.get("projected_points",0))
        proj_ast = float(row.get("projected_assists",0))
        game_lbl = row.get("game_label",f"{team} vs {opp}")
        pos      = str(row.get("position","")).upper()

        g_col    = "#c0392b" if prob>=0.35 else "#e67e22" if prob>=0.25 else "#2980b9" if prob>=0.15 else "#3d5a80"
        p_col    = "#8e44ad" if proj_pts>=0.70 else "#6c3483" if proj_pts>=0.45 else "#4a235a"
        a_col    = "#16a085" if proj_ast>=0.50 else "#1a6b4a"
        row_bg   = "#0f1320" if rank%2==0 else "#111827"
        rc2      = "#e74c3c" if rank<=3 else "#8892a4"
        pos_b    = (f'<span style="font-size:.6rem;background:#1e3a8a;color:#90caf9;'
                    f'padding:1px 5px;border-radius:4px;">{pos}</span> ' if pos else "")

        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:6px;padding:8px 12px;'
            f'background:{row_bg};border-radius:8px;margin-bottom:2px;'
            f'align-items:center;border:1px solid #1e2535;">'
            f'<div style="font-weight:700;color:{rc2};font-size:.9rem;">#{rank}</div>'
            f'<div><div style="font-weight:600;color:#e8ecf4;font-size:.9rem;">{pos_b}{name}</div>'
            f'<div style="font-size:.66rem;color:#5a7fc4;">{game_lbl}</div></div>'
            f'<div style="font-weight:600;color:#7eb3ff;font-size:.85rem;">{team}</div>'
            f'<div style="color:#8892a4;font-size:.85rem;">{opp}</div>'
            f'<div>{_badge(conf)}</div>'
            f'<div>{_bar(prob,     0.50, g_col)}</div>'
            f'<div>{_bar(proj_ast, 0.80, a_col)}</div>'
            f'<div>{_bar(proj_pts, 1.0,  p_col)}</div>'
            f'<div>{_bar(proj_sog, 5.0,  "#1a6b4a", ".1f")}</div>'
            f'<div style="font-size:.68rem;color:#8892a4;">{goals}G/{assists}A<br/>{gp}GP</div>'
            f'</div>', unsafe_allow_html=True)

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Charts","🔬 Feature Importance","📈 Model Metrics"])

    with tab1:
        cdf = disp.head(20).copy()
        if not cdf.empty:
            cdf["g_col"] = cdf["goal_probability"].apply(
                lambda p: "#c0392b" if p>=0.35 else "#e67e22" if p>=0.25 else "#2980b9")
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(x=cdf["player_name"], y=cdf["goal_probability"].round(3),
                    marker_color=cdf["g_col"],
                    text=[f"{v:.2f}" for v in cdf["goal_probability"]], textposition="outside"))
                fig.update_layout(title="Projected Goal Probability", xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"), yaxis=dict(gridcolor="#1e2535"),
                    height=380, margin=dict(t=45,b=110))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure(go.Bar(x=cdf["player_name"],
                    y=cdf.get("projected_sog", pd.Series([0]*len(cdf))),
                    marker_color="#1a6b4a",
                    text=[f"{v:.1f}" for v in cdf.get("projected_sog", [0]*len(cdf))],
                    textposition="outside"))
                fig2.update_layout(title="Projected Shots on Goal", xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"), yaxis=dict(gridcolor="#1e2535"),
                    height=380, margin=dict(t=45,b=110))
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if pipeline and pipeline.model and pipeline.model.is_trained:
            fi = pipeline.model.feature_importance()
            if not fi.empty:
                top = fi.head(20).sort_values("importance")
                fig3 = go.Figure(go.Bar(y=top["feature"], x=top["importance_pct"],
                    orientation="h", marker_color="#2980b9",
                    text=[f"{v:.1f}%" for v in top["importance_pct"]], textposition="outside"))
                fig3.update_layout(title="Top 20 Feature Importances",
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"), yaxis=dict(gridcolor="#1e2535"),
                    height=500, margin=dict(l=200))
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Run the pipeline to see feature importance.")

    with tab3:
        if pipeline and pipeline.model_metrics:
            m = pipeline.model_metrics
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("CV-AUC",      f"{m.get('train_auc',0):.3f}")
            with c2: st.metric("Brier Score",  f"{m.get('brier_score',0):.4f}")
            with c3: st.metric("Samples",      f"{m.get('n_samples',0):,}")
            with c4: st.metric("Features",     f"{m.get('n_features',0)}")
            st.markdown(f"**Goal rate in training:** {m.get('goal_rate',0):.1%} &nbsp;|&nbsp; "
                        f"**Season:** {CURRENT_SEASON[:4]}-{CURRENT_SEASON[4:]}")
            # Admin-only: show retrain button inside metrics tab
            if is_admin():
                st.divider()
                st.markdown("**Admin:** Use 'Force model retrain' in the sidebar, "
                            "then click Load/Refresh above to trigger a fresh retrain.")
        else:
            st.info("Run the pipeline to see model metrics.")

    st.divider()
    st.markdown("<div style='text-align:center;color:#555;font-size:.75rem;'>"
                "Data from NHL API · For entertainment only · Predictions are probabilistic estimates"
                "</div>", unsafe_allow_html=True)
