# sports-predictions
This app will run a daily model to predict the outcomes of sporting events.

## Getting started
Clone the repo and start from the `sports_predictor` directory.

1. Set your password — open app/auth.py, find the USERS dict, and replace the hash:

```bash
python3 -c "import hashlib; print(hashlib.sha256(b'YOUR_PASSWORD').hexdigest())"
```

Paste the result into "password_hash" for the admin user.

2. Run with Docker:

```bash
cd sports_predictor
docker-compose up -d --build
```

Open http://localhost:8501 on that machine, or http://<machine-ip>:8501 from any device on your network. First boot will warm the caches automatically (takes ~5 minutes), after that it loads instantly.
