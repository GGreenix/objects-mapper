#!/usr/bin/env python3
"""Entry point: ``python main.py [--replay video.mp4] [--config config.yaml]``."""

import sys

from field_mapper.app import run

if __name__ == "__main__":
    sys.exit(run())
