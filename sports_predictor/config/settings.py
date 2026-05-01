"""
config.py — Central configuration for all sports pipelines.

This is the ONE file you need to edit between seasons, to tune projections,
or to adjust API/cache behaviour. All other modules import from here.

Sections:
  1. NHL settings
  2. MLB settings
  3. NBA settings
  4. Model settings (shared)
  5. Cache & API settings
  6. Projection tuning
  7. UI settings
  8. Team reference data (read-only — only change if leagues expand/relocate)
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. NHL
# ══════════════════════════════════════════════════════════════════════════════

# Season identifier used by the NHL API  — update each October
# Format: "YYYYYYYY"  e.g. 2024-25 season → "20242025"
NHL_CURRENT_SEASON = "20252026"
CURRENT_SEASON     = NHL_CURRENT_SEASON   # alias kept for legacy imports

# Game type:  2 = regular season,  3 = playoffs
NHL_SEASON_TYPE = 2
SEASON_TYPE     = NHL_SEASON_TYPE         # alias

# Minimum games played before a player is included in predictions
NHL_MIN_GP = 5
MIN_GP     = NHL_MIN_GP   # alias for legacy imports

# Players inactive for longer than this many days are treated as injured
NHL_INACTIVITY_DAYS = 30

# Cache directory for NHL API responses
NHL_CACHE_DIR = "data/cache/nhl"
CACHE_DIR     = NHL_CACHE_DIR             # alias

# NHL API base URL (no trailing slash)
NHL_API_BASE = "https://api-web.nhle.com/v1"

# Cache TTLs (minutes) for each NHL data type
NHL_TTL_SCHEDULE  = 60
NHL_TTL_ROSTER    = 120
NHL_TTL_GAME_LOGS = 120
NHL_TTL_GOALIES   = 240
NHL_TTL_INJURIES  = 60
NHL_TTL_LEADERS   = 240


# ══════════════════════════════════════════════════════════════════════════════
# 2. MLB
# ══════════════════════════════════════════════════════════════════════════════

# Season year string used by the MLB Stats API — update each March
MLB_SEASON = "2026"

# Minimum games played before a batter is included in predictions
MLB_MIN_GP = 10

# Batters inactive longer than this many days are skipped
# Use a large value (200) to cover the full Oct→Mar offseason gap
MLB_INACTIVITY_DAYS = 30

# Cache directory for MLB API responses
MLB_CACHE_DIR = "data/cache/mlb"

# MLB Stats API base URL
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Cache TTLs (minutes)
MLB_TTL_SCHEDULE   = 120
MLB_TTL_ROSTER     = 120
MLB_TTL_GAME_LOGS  = 120
MLB_TTL_PITCHERS   = 60

# Default pitcher quality when no probable pitcher data is available
MLB_DEFAULT_ERA    = 4.20
MLB_DEFAULT_WHIP   = 1.30
MLB_DEFAULT_K9     = 8.5

# Number of recent starts to average for opposing pitcher quality
MLB_PITCHER_RECENT_STARTS = 5

# League-average runs per team per game (used for regression to mean)
MLB_LEAGUE_AVG_RUNS = 4.5

# Home field advantage in runs
MLB_HOME_ADVANTAGE_RUNS = 0.1

# How much to regress team run projections toward league average (0–1)
# Higher = more regression, lower = trust the model more
MLB_REGRESSION_WEIGHT = 0.40

# Confidence tier thresholds (hit probability)
MLB_CONF_ELITE  = 0.65
MLB_CONF_HIGH   = 0.55
MLB_CONF_MEDIUM = 0.45


# ══════════════════════════════════════════════════════════════════════════════
# 3. NBA
# ══════════════════════════════════════════════════════════════════════════════

# Season string used by ESPN API — update each October
# Format: "YYYY-YY"  e.g. 2025-26 season → "2025-26"
NBA_SEASON = "2025-26"

# Minimum games played before a player is included in predictions
NBA_MIN_GP = 10

# Players averaging fewer minutes per game than this are excluded (deep bench)
NBA_MIN_MINUTES_AVG = 8

# Players inactive longer than this many days during the season are skipped
# Only applied when the season is active (latest log within 14 days of today)
NBA_INACTIVITY_DAYS = 14

# Cache directory for NBA (ESPN) API responses
NBA_CACHE_DIR = "data/cache/nba"

# ESPN NBA API base URL
NBA_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

# Cache TTLs (minutes)
NBA_TTL_SCHEDULE = 60
NBA_TTL_ROSTER   = 180
NBA_TTL_GAME_LOGS = 120

# League-average points per team per game (used for regression to mean)
NBA_LEAGUE_AVG_PTS = 112.0

# Home court advantage in points
NBA_HOME_ADVANTAGE_PTS = 2.0

# How much to regress team point projections toward league average (0–1)
NBA_REGRESSION_WEIGHT = 0.35

# Normal distribution σ for NBA score spread (used in O/U and spread math)
NBA_SPREAD_SIGMA = 11.0

# Confidence tier thresholds (projected points)
NBA_CONF_ELITE  = 25
NBA_CONF_HIGH   = 18
NBA_CONF_MEDIUM = 12


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model settings (shared across all sports)
# ══════════════════════════════════════════════════════════════════════════════

# XGBoost hyperparameters — applies to all SportModel instances
MODEL_N_ESTIMATORS     = 300
MODEL_MAX_DEPTH        = 4
MODEL_LEARNING_RATE    = 0.05
MODEL_SUBSAMPLE        = 0.8
MODEL_COLSAMPLE_BYTREE = 0.8
MODEL_RANDOM_STATE     = 42

# Minimum training samples required before fitting a model
MODEL_MIN_SAMPLES = 100

# Directory for saved model files
MODEL_CACHE_DIR = "data/cache/model"

# Rolling window weights for projection blending: last 3g / 5g / 10g / season
ROLL_WEIGHT_3G     = 0.40
ROLL_WEIGHT_5G     = 0.30
ROLL_WEIGHT_10G    = 0.20
ROLL_WEIGHT_SEASON = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cache & API settings
# ══════════════════════════════════════════════════════════════════════════════

# HTTP request timeout in seconds
API_TIMEOUT_SECONDS = 15

# Number of retry attempts for failed API calls
API_RETRIES = 3

# Delay between game-log requests to avoid rate-limiting (seconds)
NHL_REQUEST_DELAY = 0.25
MLB_REQUEST_DELAY = 0.20
NBA_REQUEST_DELAY = 0.60

# Standard request headers (User-Agent used for all sports APIs)
REQUEST_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


# ══════════════════════════════════════════════════════════════════════════════
# 6. NHL projection tuning
# ══════════════════════════════════════════════════════════════════════════════

# League-average goals per team per game (used for regression to mean)
NHL_LEAGUE_AVG_GOALS = 2.85

# Home-ice goal advantage
NHL_HOME_ADVANTAGE_GOALS = 0.08

# How much to regress team goal projections toward league average (0–1)
NHL_REGRESSION_WEIGHT = 0.35

# Player signal blend weight in final xG (remainder comes from team model)
# e.g. 0.20 = 80% team model + 20% individual player signal
NHL_PLAYER_SIGNAL_WEIGHT = 0.20

# Hard floor/ceiling for projected goals per team per game
NHL_XG_MIN = 1.5
NHL_XG_MAX = 4.5

# NHL confidence tier thresholds (goal probability)
NHL_CONF_ELITE  = 0.35
NHL_CONF_HIGH   = 0.25
NHL_CONF_MEDIUM = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# 7. UI settings
# ══════════════════════════════════════════════════════════════════════════════

# How many players to show by default in the prediction table
UI_DEFAULT_SHOW_N = 25

# Earliest date selectable in the date picker
UI_DATE_MIN = "2024-10-01"

# App title shown in browser tab
UI_APP_TITLE = "Multi-Sport Predictor"


# ── Legacy NHL model paths (used by model_trainer.py) ─────────────────────────
MODEL_DIR      = MODEL_CACHE_DIR
MODEL_PATH     = f"{MODEL_CACHE_DIR}/nhl_goalscorer_model.joblib"
SCALER_PATH    = f"{MODEL_CACHE_DIR}/nhl_goalscorer_scaler.joblib"
FEATURES_PATH  = f"{MODEL_CACHE_DIR}/nhl_goalscorer_features.joblib"
METRICS_PATH   = f"{MODEL_CACHE_DIR}/nhl_goalscorer_metrics.joblib"

# ── Rolling windows for NHL feature engineering ────────────────────────────────
ROLLING_WINDOWS = [3, 5, 10]

# ── NST (Natural Stat Trick) scraper settings ──────────────────────────────────
NST_BASE       = "https://www.naturalstattrick.com"
NST_PLAYER_URL = f"{NST_BASE}/playerteams.php"
REQUEST_DELAY  = NHL_REQUEST_DELAY   # alias

NST_TEAM_MAP = {
    "ANA":"ANA","ARI":"ARI","BOS":"BOS","BUF":"BUF","CGY":"CGY",
    "CAR":"CAR","CHI":"CHI","COL":"COL","CBJ":"CBJ","DAL":"DAL",
    "DET":"DET","EDM":"EDM","FLA":"FLA","LAK":"L.A","MIN":"MIN",
    "MTL":"MTL","NSH":"NSH","NJD":"N.J","NYI":"NYI","NYR":"NYR",
    "OTT":"OTT","PHI":"PHI","PIT":"PIT","SJS":"S.J","STL":"STL",
    "TBL":"T.B","TOR":"TOR","UTA":"UTA","VAN":"VAN","VGK":"VGK",
    "WSH":"WSH","WPG":"WPG","SEA":"SEA",
}
# ══════════════════════════════════════════════════════════════════════════════
# These rarely change — only update if teams relocate, expand, or rebrand.

NHL_TEAMS = {
    "ANA": "Anaheim Ducks",       "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",       "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames",      "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks",  "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets","DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",   "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",      "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils",
    "NYI": "New York Islanders",  "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",     "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",     "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",   "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
    "SEA": "Seattle Kraken",
}

MLB_TEAMS = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CWS", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KC",  108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SD",  137: "SF",  136: "SEA",
    138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 120: "WSH",
}

MLB_TEAM_NAMES = {
    "ARI":"Arizona Diamondbacks", "ATL":"Atlanta Braves",
    "BAL":"Baltimore Orioles",    "BOS":"Boston Red Sox",
    "CHC":"Chicago Cubs",         "CWS":"Chicago White Sox",
    "CIN":"Cincinnati Reds",      "CLE":"Cleveland Guardians",
    "COL":"Colorado Rockies",     "DET":"Detroit Tigers",
    "HOU":"Houston Astros",       "KC": "Kansas City Royals",
    "LAA":"Los Angeles Angels",   "LAD":"Los Angeles Dodgers",
    "MIA":"Miami Marlins",        "MIL":"Milwaukee Brewers",
    "MIN":"Minnesota Twins",      "NYM":"New York Mets",
    "NYY":"New York Yankees",     "OAK":"Oakland Athletics",
    "PHI":"Philadelphia Phillies","PIT":"Pittsburgh Pirates",
    "SD": "San Diego Padres",     "SF": "San Francisco Giants",
    "SEA":"Seattle Mariners",     "STL":"St. Louis Cardinals",
    "TB": "Tampa Bay Rays",       "TEX":"Texas Rangers",
    "TOR":"Toronto Blue Jays",    "WSH":"Washington Nationals",
}

NBA_ESPN_TEAMS = {
    1:"ATL",  2:"BOS",  3:"NOP",  4:"CHI",  5:"CLE",  6:"DAL",  7:"DEN",
    8:"DET",  9:"GSW", 10:"HOU", 11:"IND", 12:"LAC", 13:"LAL", 14:"MIA",
    15:"MIL", 16:"MIN", 17:"BKN", 18:"NYK", 19:"ORL", 20:"PHI", 21:"PHX",
    22:"POR", 23:"SAC", 24:"SAS", 25:"OKC", 26:"UTA", 27:"MEM", 28:"WSH",
    29:"TOR", 30:"CHA",
}

NBA_TEAM_NAMES = {
    "ATL":"Atlanta Hawks",         "BOS":"Boston Celtics",
    "NOP":"New Orleans Pelicans",  "CHI":"Chicago Bulls",
    "CLE":"Cleveland Cavaliers",   "DAL":"Dallas Mavericks",
    "DEN":"Denver Nuggets",        "DET":"Detroit Pistons",
    "GSW":"Golden State Warriors", "HOU":"Houston Rockets",
    "IND":"Indiana Pacers",        "LAC":"LA Clippers",
    "LAL":"Los Angeles Lakers",    "MIA":"Miami Heat",
    "MIL":"Milwaukee Bucks",       "MIN":"Minnesota Timberwolves",
    "BKN":"Brooklyn Nets",         "NYK":"New York Knicks",
    "ORL":"Orlando Magic",         "PHI":"Philadelphia 76ers",
    "PHX":"Phoenix Suns",          "POR":"Portland Trail Blazers",
    "SAC":"Sacramento Kings",      "SAS":"San Antonio Spurs",
    "OKC":"Oklahoma City Thunder", "UTA":"Utah Jazz",
    "MEM":"Memphis Grizzlies",     "WSH":"Washington Wizards",
    "TOR":"Toronto Raptors",       "CHA":"Charlotte Hornets",
}
