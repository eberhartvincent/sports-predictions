#!/bin/bash
# home_machine_setup.sh
# Run this ONCE on your home machine to get everything running.
# After this, updates happen automatically — you never need to touch it again.

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
info() { echo -e "${YELLOW}  → $1${NC}"; }
err()  { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

echo ""
echo "🏆  Sports Predictor — Home Machine Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Check Docker is installed ──────────────────────────────────────────────
command -v docker &>/dev/null || err "Docker is not installed. Install Docker Desktop from https://docker.com"
command -v docker-compose &>/dev/null || docker compose version &>/dev/null || \
    err "Docker Compose not found. It comes with Docker Desktop."
ok "Docker found: $(docker --version)"

# ── Get GitHub username ────────────────────────────────────────────────────
echo ""
read -p "  Enter your GitHub username: " GITHUB_USER
[ -z "$GITHUB_USER" ] && err "GitHub username cannot be empty"
echo ""

# ── Log in to GitHub Container Registry ───────────────────────────────────
info "Logging in to ghcr.io..."
echo ""
echo "  You need a GitHub Personal Access Token (PAT) with 'read:packages' scope."
echo "  Create one at: https://github.com/settings/tokens/new"
echo "  Select: read:packages"
echo ""
read -s -p "  Paste your GitHub PAT (input hidden): " GH_PAT
echo ""
echo "$GH_PAT" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin
ok "Logged in to ghcr.io"

# ── Write .env file ────────────────────────────────────────────────────────
info "Writing .env file..."
cat > .env << ENVEOF
GITHUB_USER=${GITHUB_USER}
ENVEOF
ok ".env file written"

# ── Pull the image ─────────────────────────────────────────────────────────
info "Pulling latest image from ghcr.io/${GITHUB_USER}/sports-predictor..."
docker pull ghcr.io/${GITHUB_USER}/sports-predictor:latest
ok "Image pulled"

# ── Start the containers ───────────────────────────────────────────────────
info "Starting Sports Predictor + Watchtower..."
docker-compose up -d
ok "Containers started"

# ── Get local IP ──────────────────────────────────────────────────────────
if command -v ipconfig &>/dev/null; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "your-machine-ip")
elif command -v hostname &>/dev/null; then
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "your-machine-ip")
else
    LOCAL_IP="your-machine-ip"
fi

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅  Setup complete!${NC}"
echo ""
echo "  App is running at:"
echo -e "    ${YELLOW}http://localhost:8501${NC}          (this machine)"
echo -e "    ${YELLOW}http://${LOCAL_IP}:8501${NC}   (any device on your network)"
echo ""
echo "  Updates happen automatically:"
echo "    • GitHub Actions runs daily at 11 AM ET"
echo "    • Builds a fresh image with today's predictions"
echo "    • Watchtower detects the new image within 5 minutes"
echo "    • Container restarts with fresh data — zero action needed from you"
echo ""
echo "  Useful commands:"
echo "    docker-compose ps                    # check status"
echo "    docker-compose logs -f               # live logs"
echo "    docker-compose pull && docker-compose up -d  # manual update"
echo ""
