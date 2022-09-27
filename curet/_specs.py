from pathlib import Path
from typing import Final

# NOTE: index starts from 1

CURET_NUM_CLASSES: Final = 61

CURET_SAMPLES_PER_CLASS: Final = 205

CURET_SAMPLE_URL = "https://www.cs.columbia.edu/CAVE/exclude/curet/data/zips/sample{:02}.zip"

CURET_SAMPLE_FILENAME: Final[Path] = "{}-{}-{}.bmp"

CURET_SAMPLE_DIRNAME: Final[Path] = "sample{}"