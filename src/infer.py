import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import ast
from pathlib import Path
from .config import CFG


def preprocess(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CFG.IMG_SIZE, CFG.IMG_SIZE))
    x = img.astype("float32") / 255.0
    return np.expand_dims(x, 0)


def load_true_label_from_csv(image_path: str):
    """
    Đọc nhãn gốc (one-hot 8 lớp) từ CSV theo filename.
    Trả về np.array shape (8,) hoặc None nếu không tìm thấy.
    """
    if not CFG.CSV_PATH.exists():
        return None

    filename = Path(image_path).name  # chỉ lấy tên file
    df = pd.read_csv(CFG.CSV_PATH)

    row = df[df["images"].astype(str) == str(filename)]
    if len(row) == 0:
        return None

    t = row.iloc[0]["target"]
    try:
        arr = ast.literal_eval(t) if isinstance(t, str) else t
        if not isinstance(arr, (list, tuple)) or len(arr) != CFG.NUM_CLASSES:
            return None
        arr = [0 if int(v) == 0 else 1 for v in arr]
        return np.array(arr, dtype=np.int32)
    except Exception:
        return None


def main(image_path: str, thr: float = 0.5):
    model_path = CFG.OUT_DIR / "best_retina_multilabel.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Hãy chạy: python -m src.train"
        )

    model = tf.keras.models.load_model(model_path, compile=False)

    x = preprocess(image_path)
    p = model.predict(x, verbose=0)[0]

    # ===== OUTPUT GỐC (GIỮ NGUYÊN) =====
    print("Image:", image_path)
    for i, c in enumerate(CFG.CLASSES):
        print(f"{c}: {p[i]:.4f} -> {'POS' if p[i] >= thr else 'neg'}")

    # ===== PHẦN THÊM: SO SÁNH VỚI NHÃN GỐC =====
    y_true = load_true_label_from_csv(image_path)

    if y_true is None:
        print("\n[GT] Không tìm thấy nhãn gốc trong CSV cho ảnh này.")
        return

    y_pred = (p >= thr).astype(np.int32)

    true_labels = [CFG.CLASSES[i] for i in range(CFG.NUM_CLASSES) if y_true[i] == 1]
    pred_labels = [CFG.CLASSES[i] for i in range(CFG.NUM_CLASSES) if y_pred[i] == 1]

    print("\n[GT] Ground-truth labels:", true_labels if true_labels else ["(none)"])
    print("[PR] Predicted labels   :", pred_labels if pred_labels else ["(none)"])

    print("[CMP] Per-class comparison (TP / FP / FN / TN):")
    for i, c in enumerate(CFG.CLASSES):
        yt = int(y_true[i])
        yp = int(y_pred[i])
        if yt == 1 and yp == 1:
            s = "TP"
        elif yt == 0 and yp == 1:
            s = "FP"
        elif yt == 1 and yp == 0:
            s = "FN"
        else:
            s = "TN"
        print(f"  {c}: {s}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.infer path/to/image.jpg [threshold]")
        raise SystemExit(1)

    img_path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.5
    main(img_path, thr)
