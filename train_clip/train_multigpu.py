"""Compatibility entry point; GPU selection now comes from the YAML config."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_clip.train import main

if __name__ == '__main__':
    main()
