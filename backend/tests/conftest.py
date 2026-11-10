"""
Pytest configuration and environment setup for Verity backend tests.
"""

import os
import sys
from pathlib import Path

# Ensure tests run hermetically without requiring external network or live Qdrant
os.environ["QDRANT_URL"] = ""
os.environ.setdefault("EMBEDDING_BACKEND", "fastembed")

# Ensure project root is at the head of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

