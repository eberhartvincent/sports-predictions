"""
main.py — Sports Predictor
Run:  streamlit run main.py
"""

import compat  # noqa — must be first

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

from app.auth import require_auth, is_admin, current_user, logout

st.set_page_config(
    page_title="Sports Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
  .main{background-color:#0e1117}.stApp{font-family:'Inter',sans-serif}
  .metric-card{background:linear-gradient(135deg,#1a1f2e,#252b3b);border:1px solid #2d3550;
    border-radius:12px;padding:1rem 1.4rem;text-align:center;margin-bottom:.5rem}
  .metric-card .label{font-size:.75rem;color:#8892a4;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
  .metric-card .value{font-size:1.8rem;font-weight:700;color:#e8ecf4}
  .metric-card .sub{font-size:.75rem;color:#5a7fc4}
  .badge-elite{background:#c0392b;color:#fff;padding:2px 8px;border-radius:12px;font-size:.7rem;font-weight:600}
  .badge-high{background:#e67e22;color:#fff;padding:2px 8px;border-radius:12px;font-size:.7rem;font-weight:600}
  .badge-medium{background:#2980b9;color:#fff;padding:2px 8px;border-radius:12px;font-size:.7rem;font-weight:600}
  .badge-low{background:#555;color:#ccc;padding:2px 8px;border-radius:12px;font-size:.7rem;font-weight:600}
  .app-header{background:linear-gradient(90deg,#1a237e,#0d47a1 50%,#1565c0);
    border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;border:1px solid #1e3a8a}
  .app-header h1{color:#fff;margin:0;font-size:2rem}
  .app-header p{color:#90caf9;margin:.3rem 0 0}
  .filter-label{font-size:.72rem;color:#8892a4;text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px}
  #MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
</style>""", unsafe_allow_html=True)

require_auth()

for k, v in {
    "nhl_pipeline":None,"nhl_predictions":pd.DataFrame(),
    "nhl_last_run":None,"nhl_running":False,
    "nhl_teams":[],"nhl_games":[],"selected_date":None,
}.items():
    if k not in st.session_state: st.session_state[k] = v

ET = ZoneInfo("America/New_York")

with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:.5rem 0 1rem">
      <div style="font-size:2.5rem">🏆</div>
      <div style="font-size:1.15rem;font-weight:700;color:#e8ecf4">Sports Predictor</div>
      <div style="font-size:.72rem;color:#8892a4">NHL · MLB · NBA</div>
    </div>""", unsafe_allow_html=True)

    user  = current_user()
    admin = is_admin()
    bc    = "#c0392b" if admin else "#2980b9"
    rl    = "Admin"   if admin else "Viewer"
    st.markdown(f'<div style="background:#1a1f2e;border:1px solid #2d3550;border-radius:8px;'
                f'padding:8px 12px;margin-bottom:12px;display:flex;align-items:center;gap:8px;">'
                f'<span style="font-size:1.4rem">👤</span><div>'
                f'<div style="color:#e8ecf4;font-weight:600;font-size:.9rem">{user}</div>'
                f'<span style="background:{bc};color:#fff;padding:1px 7px;border-radius:10px;'
                f'font-size:.62rem;font-weight:700">{rl}</span></div></div>',
                unsafe_allow_html=True)

    if st.button("🚪 Sign Out", use_container_width=True): logout()

    st.divider()

    if admin:
        st.markdown("### 📅 Date")
        today_et    = datetime.now(ET).date()
        picked_date = st.date_input("Date",
            value=st.session_state.selected_date or today_et,
            min_value=datetime(2024,10,1).date(), max_value=today_et,
            label_visibility="collapsed")

        if picked_date != st.session_state.selected_date:
            st.session_state.selected_date    = picked_date
            # Reset all sports so they reload for the new date
            for _k in ["nhl_predictions","nhl_pipeline","nhl_games","nhl_teams",
                       "nhl_auto_loaded","nhl_last_run",
                       "mlb_preds","mlb_pipeline","mlb_games","mlb_teams","mlb_last_run",
                       "nba_preds","nba_pipeline","nba_games","nba_teams","nba_last_run"]:
                if _k in st.session_state:
                    st.session_state[_k] = pd.DataFrame() if "pred" in _k or _k=="mlb_preds" or _k=="nba_preds" else (
                        None if "pipeline" in _k or "last_run" in _k else
                        False if _k=="nhl_auto_loaded" else [])

        st.caption(f"Showing **{'Today' if picked_date==today_et else picked_date.strftime('%b %d, %Y')}**")
        st.divider()
    else:
        today_et    = datetime.now(ET).date()
        picked_date = today_et
        if st.session_state.selected_date != today_et:
            st.session_state.selected_date = today_et

    force_retrain = False
    if admin:
        st.markdown("### ⚙️ Admin Controls")
        force_retrain = st.checkbox("Force model retrain", value=False,
                                    help="Retrain all models from scratch")
        if st.button("🔄 Warm Cache Now", use_container_width=True):
            with st.spinner("Running cache warm (~5 min)…"):
                import subprocess, sys
                r = subprocess.run([sys.executable, "scripts/warm_cache.py"],
                                   capture_output=True, text=True, cwd=".")
            st.success("Done!") if r.returncode==0 else st.error(r.stderr[-400:])
    else:
        st.markdown("### ℹ️ About")
        st.markdown("XGBoost predictions: rolling stats, pitcher matchups, "
                    "park factors, weather & lineup context.")

    if st.session_state.nhl_last_run:
        st.caption(f"NHL updated: {st.session_state.nhl_last_run}")

picked = st.session_state.selected_date
date_display = picked.strftime("%B %d, %Y") if picked else datetime.now(ET).strftime("%B %d, %Y")
st.markdown(f"""<div class="app-header">
  <h1>🏆 Sports Predictor</h1>
  <p>Machine-learning projections for {date_display} · NHL · MLB · NBA</p>
</div>""", unsafe_allow_html=True)

selected_date_str = picked.strftime("%Y-%m-%d") if picked else None

from app.tabs.mlb_tab import render_mlb
from app.tabs.nba_tab import render_nba
from app.tabs.nhl_tab import render_nhl

tab_nhl, tab_mlb, tab_nba = st.tabs([
    "🏒  NHL — Goalscorer",
    "⚾  MLB — Batter & Pitcher",
    "🏀  NBA — Player Props",
])

with tab_mlb: render_mlb(selected_date_str, force_retrain)
with tab_nba: render_nba(selected_date_str, force_retrain)
with tab_nhl: render_nhl(selected_date_str, force_retrain)
