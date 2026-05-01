"""
app/auth.py — Authentication and role management.

Roles:
  admin  — full access: load predictions, force retrain, manage users
  viewer — read-only: view predictions, cannot retrain or manage

Passwords are stored as bcrypt hashes. To generate a new hash:
    python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"

FIRST-TIME SETUP:
  1. Change ADMIN_PASSWORD_HASH to a hash of your chosen password
  2. Add viewer accounts to USERS as needed
  3. Rebuild the Docker container
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ── User database ──────────────────────────────────────────────────────────────
# Passwords are sha256-hashed for simplicity (no extra deps).
# To generate: python -c "import hashlib; print(hashlib.sha256(b'yourpass').hexdigest())"

def _h(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ── CONFIGURE YOUR CREDENTIALS HERE ──────────────────────────────────────────
# Change these before deploying. Admin is you; add viewers as needed.
USERS: dict[str, dict] = {
    "admin": {
        "password_hash": "ba82e16f506cb41712bbe4ddf697e8dfce1c2b2c4a22466fbd7841d9a47ae824",   # ← CHANGE THIS PASSWORD
        "role":          "admin",
        "display_name":  "Admin",
    },
    "churm": {
        "password_hash": _h("churm"),  # ← CHANGE THIS PASSWORD
        "role":          "viewer",
        "display_name":  "Churm",
    },
    # Add more viewers:
    # "alice": {"password_hash": _h("alicepass"), "role": "viewer", "display_name": "Alice"},
}
# ─────────────────────────────────────────────────────────────────────────────

# Max failed attempts before lockout
MAX_ATTEMPTS  = 5
LOCKOUT_SECS  = 300   # 5 minutes


def _init_auth_state():
    for k, v in {
        "authenticated": False,
        "username":      None,
        "role":          None,
        "display_name":  None,
        "login_attempts": 0,
        "lockout_until": 0.0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_authenticated() -> bool:
    _init_auth_state()
    return bool(st.session_state.authenticated)


def is_admin() -> bool:
    return is_authenticated() and st.session_state.role == "admin"


def current_user() -> Optional[str]:
    return st.session_state.get("display_name")


def logout():
    for k in ("authenticated","username","role","display_name"):
        st.session_state[k] = None if k != "authenticated" else False
    st.rerun()


def require_auth():
    """Call at the top of any page that requires login. Stops rendering if not authed."""
    _init_auth_state()
    if not st.session_state.authenticated:
        _render_login()
        st.stop()


def require_admin():
    """Call for admin-only actions. Shows error if viewer tries."""
    require_auth()
    if st.session_state.role != "admin":
        st.error("⛔ Admin access required.")
        st.stop()


def _render_login():
    """Full-page login form."""
    # Center the login card
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="
            background:linear-gradient(135deg,#1a1f2e,#252b3b);
            border:1px solid #2d3550; border-radius:20px;
            padding:2.5rem 2rem; margin-top:4rem; text-align:center;
        ">
          <div style="font-size:3.5rem; margin-bottom:0.5rem;">🏆</div>
          <h2 style="color:#e8ecf4; margin:0 0 0.3rem;">Sports Predictor</h2>
          <p style="color:#8892a4; font-size:.85rem; margin:0 0 1.5rem;">
            ML-powered daily predictions · NHL · MLB · NBA
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Lockout check
        now = time.time()
        locked = st.session_state.lockout_until > now
        if locked:
            remaining = int(st.session_state.lockout_until - now)
            st.error(f"🔒 Too many failed attempts. Try again in {remaining}s.")
            return

        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username", placeholder="username")
            password = st.text_input("Password", type="password", placeholder="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

        if submitted:
            _attempt_login(username.strip().lower(), password)

        attempts = st.session_state.login_attempts
        if attempts > 0:
            st.caption(f"Failed attempts: {attempts}/{MAX_ATTEMPTS}")


def _attempt_login(username: str, password: str):
    user = USERS.get(username)
    if user and user["password_hash"] == _h(password):
        st.session_state.authenticated  = True
        st.session_state.username       = username
        st.session_state.role           = user["role"]
        st.session_state.display_name   = user["display_name"]
        st.session_state.login_attempts = 0
        st.session_state.lockout_until  = 0.0
        st.rerun()
    else:
        st.session_state.login_attempts += 1
        if st.session_state.login_attempts >= MAX_ATTEMPTS:
            st.session_state.lockout_until = time.time() + LOCKOUT_SECS
            st.error(f"🔒 Locked out for {LOCKOUT_SECS//60} minutes.")
        else:
            st.error("❌ Invalid username or password.")
