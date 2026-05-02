# config/__init__.py
# Re-exports everything from settings.py so that
# `from config import X` works identically to `from config.settings import X`
from config.settings import *  # noqa: F401, F403
