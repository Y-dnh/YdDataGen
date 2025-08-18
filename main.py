import argparse
import sys
from pathlib import Path
import logging
from typing import Dict

from src.config import CONFIG
from src.download import VideoDownloader
from src.extract_frames import extract_frames

logger = logging.getLogger(__name__)


result = VideoDownloader().download()
result = extract_frames(result)

print(result)
