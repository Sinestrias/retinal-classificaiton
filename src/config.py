from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    IMG_DIR: Path = DATA_DIR / "images"
    CSV_PATH: Path = DATA_DIR / "label_images.csv"
    OUT_DIR: Path = PROJECT_ROOT / "outputs"

    CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
    NUM_CLASSES: int = 8

    # ✅ giảm size để tăng tốc rất nhiều
    IMG_SIZE: int = 224

    # ✅ tăng batch (nếu GPU đủ VRAM; nếu out-of-memory thì hạ về 16)
    BATCH_SIZE: int = 32

    SEED: int = 42
    VAL_RATIO: float = 0.15

    EPOCHS_STAGE1: int = 8
    EPOCHS_STAGE2: int = 12
    LR_STAGE1: float = 1e-3
    LR_STAGE2: float = 1e-5

    # ✅ bật nếu có GPU NVIDIA (nếu bạn đang train CPU thì bật cũng không giúp)
    USE_MIXED_PRECISION: bool = True

    # ✅ cache ra disk để tăng tốc đọc ảnh (ổn định hơn cache RAM)
    CACHE_TO_DISK: bool = True
    CACHE_DIR: Path = PROJECT_ROOT / ".cache_tfdata"
