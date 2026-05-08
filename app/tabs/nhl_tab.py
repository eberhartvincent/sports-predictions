"""nhl_tab.py — NHL tab UI renderer"""

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.auth import is_admin
from app.prediction_store import load_predictions, last_updated, predictions_mtime
from config import NHL_TEAMS, CURRENT_SEASON
from nhl_pipeline import NHLPipeline

ET = ZoneInfo("America/New_York")


# ── Helpers ───────────────────────────────────────────────────────────────────

def confidence_badge(conf):
    m = {"Elite": "badge-elite", "High": "badge-high",
         "Medium": "badge-medium", "Low": "badge-low"}
    return f'<span class="{m.get(str(conf), "badge-low")}">{conf}</span>'

<<<<<<< HEAD

def make_bar(val, max_val, colour, fmt=".2f"):
    pct = min(val / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="flex:1;background:#1e2535;border-radius:5px;height:10px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{colour};border-radius:5px;"></div>'
        f'</div>'
        f'<span style="font-weight:700;color:#e8ecf4;min-width:36px;font-size:.88rem;">'
        f'{val:{fmt}}</span></div>'
    )


# ── Session state init ────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "nhl_predictions": pd.DataFrame(),
        "nhl_pipeline":    None,
        "nhl_last_run":    None,
        "_nhl_mtime":      None,
        "_nhl_date":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_nhl(date_str: str):
    _init_state()

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    is_today  = (date_str is None or date_str == today_str)
    selected  = date_str or today_str

    # ── Auto-load predictions ─────────────────────────────────────────────────
    hist_file = (
        Path("data/cache/predictions/history") / f"nhl_{selected}.parquet"
        if not is_today else None
    )

    disk_mtime    = predictions_mtime("nhl")
    session_mtime = st.session_state.get("_nhl_mtime")
    session_date  = st.session_state.get("_nhl_date")

    need_reload = (
        st.session_state.nhl_predictions.empty
        or session_date != selected
        or (is_today and disk_mtime and disk_mtime != session_mtime)
    )

    if need_reload:
        if not is_today and hist_file and hist_file.exists():
=======
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
    from pathlib import Path as _Path

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
>>>>>>> 3121e961f582ee3232ca419619d5a552ccea5d9e
            try:
                st.session_state.nhl_predictions = pd.read_parquet(hist_file)
                st.session_state["_nhl_date"]    = selected
                st.session_state["_nhl_mtime"]   = None
            except Exception as e:
                st.warning(f"Could not load history for {selected}: {e}")
                st.session_state.nhl_predictions = pd.DataFrame()
        elif is_today:
            df = load_predictions("nhl")
            if df is not None and not df.empty:
                st.session_state.nhl_predictions = df
                st.session_state["_nhl_mtime"]   = disk_mtime
                st.session_state["_nhl_date"]    = selected
        else:
            st.session_state.nhl_predictions = pd.DataFrame()
            st.session_state["_nhl_date"] = selected

    preds    = st.session_state.nhl_predictions
    pipeline = st.session_state.nhl_pipeline

    # ── Header ────────────────────────────────────────────────────────────────
    lu = last_updated("nhl")
    col_h, col_meta = st.columns([3, 1])
    with col_h:
        date_label = "Today" if is_today else selected
        st.markdown(
            f'<div class="section-header">🏒 NHL — {date_label}</div>',
            unsafe_allow_html=True,
        )
    with col_meta:
        if lu:
            st.caption(f"Updated: {lu}")
        if not is_today and (hist_file is None or not hist_file.exists()):
            st.caption("No history available for this date.")

    # ── Admin: run pipeline ───────────────────────────────────────────────────
    if is_admin() and is_today:
        with st.expander("⚙️ Admin — Run Pipeline", expanded=False):
            if st.button("▶ Run NHL Pipeline", key="nhl_run_btn"):
                with st.spinner("Running NHL pipeline…"):
                    try:
                        t0 = time.time()
                        p  = NHLPipeline(selected)
                        p.run()
                        st.session_state.nhl_pipeline = p
                        df2 = load_predictions("nhl")
                        if df2 is not None and not df2.empty:
                            st.session_state.nhl_predictions = df2
                            st.session_state["_nhl_mtime"]   = predictions_mtime("nhl")
                            st.session_state["_nhl_date"]    = selected
                        st.success(f"Done in {time.time()-t0:.1f}s")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

    # ── No data state ─────────────────────────────────────────────────────────
    if preds.empty:
        if is_today:
            st.info("No NHL predictions loaded yet. "
                    + ("Run the pipeline above." if is_admin() else
                       "Predictions are generated automatically each morning."))
        else:
            st.info(f"No saved predictions found for {selected}.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    teams_in_preds = sorted(preds["team"].dropna().unique().tolist()) if "team" in preds.columns else []
    games_in_preds = sorted(preds["game_label"].dropna().unique().tolist()) if "game_label" in preds.columns else []

    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
    with fc1:
        sel_team = st.selectbox("Filter by team", ["All"] + teams_in_preds, key="nhl_team_filter")
    with fc2:
        sel_game = st.selectbox("Filter by game", ["All"] + games_in_preds, key="nhl_game_filter")
    with fc3:
        conf_opts = ["All", "Elite", "High", "Medium", "Low"]
        sel_conf  = st.selectbox("Min confidence", conf_opts, key="nhl_conf_filter")
    with fc4:
        top_n = st.selectbox("Show top", [25, 50, 100, 200], key="nhl_topn")

    disp = preds.copy()
    if sel_team != "All":
        disp = disp[disp["team"] == sel_team]
    if sel_game != "All":
        disp = disp[disp["game_label"] == sel_game]
    if sel_conf != "All":
        conf_order = {"Elite": 4, "High": 3, "Medium": 2, "Low": 1}
        min_rank   = conf_order.get(sel_conf, 1)
        disp = disp[disp["confidence"].map(lambda c: conf_order.get(str(c), 0)) >= min_rank]

    disp = disp.sort_values("goal_probability", ascending=False).head(top_n)

    if disp.empty:
        st.warning("No players match the selected filters.")
        return

    # ── Player table ──────────────────────────────────────────────────────────
    st.markdown(f"**{len(disp)} players** shown")

    grid   = "36px 1fr 55px 55px 85px 100px 90px 90px 90px 70px"
    hdrs   = ["#", "Player", "Team", "Opp", "Conf",
              "Goal Prob", "Proj SOG", "Proj Pts", "Proj Ast", "Season"]
    hstyle = (
        f"display:grid;grid-template-columns:{grid};gap:6px;padding:7px 12px;"
        "background:#1a1f2e;border-radius:8px;font-size:.68rem;color:#8892a4;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;"
        "border:1px solid #2d3550;"
    )
    st.markdown(
        f'<div style="{hstyle}">' + "".join(f"<div>{h}</div>" for h in hdrs) + "</div>",
        unsafe_allow_html=True,
    )

    for rank, (_, row) in enumerate(disp.iterrows(), 1):
        name  = row.get("player_name", "")
        team  = row.get("team", "")
        opp   = row.get("opponent", "")
        conf  = str(row.get("confidence", "Low"))
        gp    = int(row.get("gp", 0))
        goals = int(row.get("season_goals", 0))
        shots = int(row.get("season_shots", 0))
        prob  = float(row.get("goal_probability", 0))
        sog   = float(row.get("projected_sog", 0))
        pts   = float(row.get("projected_points", 0))
        ast_  = float(row.get("projected_assists", 0))

        row_bg = "#0f1320" if rank % 2 == 0 else "#111827"
        rc2    = "#e74c3c" if rank <= 3 else "#8892a4"
        g_col  = "#c0392b" if prob >= 0.35 else "#e67e22" if prob >= 0.25 else "#2980b9"
        p_col  = "#27ae60" if pts >= 0.5 else "#2980b9"

        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:6px;'
            f'padding:8px 12px;background:{row_bg};border-radius:8px;'
            f'margin-bottom:2px;align-items:center;border:1px solid #1e2535;">'
            f'<div style="font-weight:700;color:{rc2};">#{rank}</div>'
            f'<div>'
            f'  <div style="font-weight:600;color:#e8ecf4;font-size:.9rem;">{name}</div>'
            f'  <div style="font-size:.66rem;color:#5a7fc4;">{row.get("game_label","")}</div>'
            f'</div>'
            f'<div style="font-weight:600;color:#7eb3ff;font-size:.85rem;">{team}</div>'
            f'<div style="color:#8892a4;font-size:.85rem;">{opp}</div>'
            f'<div>{confidence_badge(conf)}</div>'
            f'<div>{make_bar(prob, 0.50, g_col)}</div>'
            f'<div>{make_bar(sog,  5.0,  "#1a6b4a", ".1f")}</div>'
            f'<div>{make_bar(pts,  1.0,  p_col)}</div>'
            f'<div>{make_bar(ast_, 1.0,  "#8e44ad")}</div>'
            f'<div style="font-size:.68rem;color:#8892a4;line-height:1.3;">'
            f'  {goals}G/{shots}S<br/>{gp}GP'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔬 Feature Importance", "📈 Model Metrics"])

    with tab1:
        cdf = disp.head(20).copy()
        if not cdf.empty:
            cdf["g_colour"] = cdf["goal_probability"].apply(
                lambda p: "#c0392b" if p >= 0.35 else "#e67e22" if p >= 0.25 else "#2980b9"
            )
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=cdf["player_name"], y=cdf["goal_probability"].round(3),
                    marker_color=cdf["g_colour"],
                    text=[f"{v:.2f}" for v in cdf["goal_probability"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Goal Prob: %{y:.3f}<extra></extra>",
                ))
                fig.update_layout(
                    title="Goal Probability", xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"),
                    yaxis=dict(gridcolor="#1e2535"),
                    height=420, margin=dict(t=50, b=120),
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "projected_sog" in cdf.columns:
                    fig2 = go.Figure(go.Bar(
                        x=cdf["player_name"], y=cdf["projected_sog"].round(1),
                        marker_color="#1a6b4a",
                        text=[f"{v:.1f}" for v in cdf["projected_sog"]],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Proj SOG: %{y:.1f}<extra></extra>",
                    ))
                    fig2.update_layout(
                        title="Projected Shots on Goal", xaxis_tickangle=-40,
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font=dict(color="#e8ecf4"),
                        xaxis=dict(gridcolor="#1e2535"),
                        yaxis=dict(gridcolor="#1e2535"),
                        height=420, margin=dict(t=50, b=120),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if pipeline:
            m = getattr(pipeline, "model", None)
            if m and hasattr(m, "feature_importance"):
                fi = m.feature_importance()
                if not fi.empty:
                    top = fi.head(15).sort_values("importance")
                    fig3 = go.Figure(go.Bar(
                        y=top["feature"], x=top["pct"], orientation="h",
                        marker_color="#2980b9",
                        text=[f"{v:.1f}%" for v in top["pct"]],
                        textposition="outside",
                    ))
                    fig3.update_layout(
                        title="Top 15 Feature Importances",
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font=dict(color="#e8ecf4"),
                        xaxis=dict(gridcolor="#1e2535"),
                        yaxis=dict(gridcolor="#1e2535"),
                        height=450, margin=dict(l=180, t=45),
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Feature importances not available.")
        else:
            st.info("Run the pipeline to see feature importances.")

    with tab3:
        if pipeline:
            m = getattr(pipeline, "model", None)
            if m and hasattr(m, "metrics") and m.metrics:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("CV-AUC",   f"{m.metrics.get('cv_auc', 0):.3f}")
                mc2.metric("Train AUC", f"{m.metrics.get('train_auc', 0):.3f}")
                mc3.metric("Samples",   f"{m.metrics.get('n_samples', 0):,}")
            else:
                st.info("No model metrics available.")
        else:
            st.info("Run the pipeline to see model metrics.")

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:.75rem;'>"
        "Data from NHL API · For entertainment only · "
        "Predictions are probabilistic estimates"
        "</div>",
        unsafe_allow_html=True,
    )
