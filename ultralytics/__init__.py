# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.50"

import os
import sys
from pathlib import Path

# This repository uses a read-only Conda environment. Load binary wheels that
# are installed in the project-local package directory before importing cv2.
_LOCAL_PACKAGES = Path(__file__).resolve().parents[1] / ".python_packages"
if _LOCAL_PACKAGES.is_dir():
    _local_package_path = str(_LOCAL_PACKAGES)
    if _local_package_path not in sys.path:
        sys.path.insert(0, _local_package_path)

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
