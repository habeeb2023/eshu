"""
conftest.py — pytest root configuration.
Sets PYTHONPATH so `app` and `ui` packages resolve from project root.
"""
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))
