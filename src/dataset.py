import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from .config import CFG

AUTOTUNE = tf.data.AUTOTUNE

def best_multilabel_split(df):
    def score_split(tr, va):
        tr_rate = tr[CFG.CLASSES].mean().values
        va_rate = va[CFG.CLASSES].mean().values
        return float(np.abs(tr_rate - va_rate).mean())

    best = None
    best_score = 1e9
    for s in range(40):
        tr, va = train_test_split(df, test_size=CFG.VAL_RATIO, random_state=CFG.SEED+s, shuffle=True)
        sc = score_split(tr, va)
        if sc < best_score:
            best_score = sc
            best = (tr, va)

    tr, va = best
    print("Best split mean abs label-rate diff:", best_score)
    return tr, va

def compute_pos_weight(train_df):
    y = train_df[CFG.CLASSES].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return ((neg + 1e-6) / (pos + 1e-6)).astype(np.float32)

def build_augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augment")

def make_tfds(df, augmenter=None, training=True, cache_name="train"):
    paths = [str(CFG.IMG_DIR / f) for f in df["images"].astype(str).tolist()]
    labels = df[CFG.CLASSES].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(4096, seed=CFG.SEED, reshuffle_each_iteration=True)

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (CFG.IMG_SIZE, CFG.IMG_SIZE))
        return img

    def _map(p, y):
        x = load_image(p)
        if training and augmenter is not None:
            x = augmenter(x, training=True)
        return x, y

    # ✅ tăng parallelism để đỡ nghẽn CPU
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)

    # ✅ cache (ra disk) trước khi batch/prefetch để giảm I/O
    if getattr(CFG, "CACHE_TO_DISK", False):
        CFG.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(CFG.CACHE_DIR / f"{cache_name}.cache"))

    ds = ds.batch(CFG.BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds
