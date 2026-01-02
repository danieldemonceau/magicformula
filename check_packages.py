#!/usr/bin/env python3
"""Quick check for required packages."""

import importlib.util

REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "yfinance": "yfinance",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic_settings",
    "dateutil": "dateutil",
    "requests": "requests",
    "aiohttp": "aiohttp",
}

missing = [
    name for name, module in REQUIRED.items()
    if importlib.util.find_spec(module) is None
]

if missing:
    print(f"Missing: {', '.join(missing)}")
    exit(1)
else:
    print("All packages installed")
    exit(0)

