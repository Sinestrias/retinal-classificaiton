import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from .config import CFG
from .data_prep import load_and_clean_df
from .dataset import best_multilabel_split, compute_pos_weight, make_tfds, build_augmenter
from .losses import make_weighted_bce
from .model import build_model

def setup_runtime():
    # Check GPU
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)

    # memory growth (nếu có GPU)
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("Memory growth set failed:", e)

    # mixed precision
    if CFG.USE_MIXED_PRECISION and gpus:
        from keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision policy:", mixed_precision.global_policy())
    else:
        print("Mixed precision OFF (no GPU or disabled).")

def main():
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_runtime()

    df = load_and_clean_df()
    train_df, val_df = best_multilabel_split(df)

    augmenter = build_augmenter()
    train_ds = make_tfds(train_df, augmenter=augmenter, training=True,  cache_name="train")
    val_ds   = make_tfds(val_df,   augmenter=None,     training=False, cache_name="val")

    pos_weight = compute_pos_weight(train_df)
    loss_fn = make_weighted_bce(pos_weight)

    model, base = build_model()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
        tf.keras.metrics.AUC(name="auc", multi_label=True),
    ]

    best_path = CFG.OUT_DIR / "best_retina_multilabel.keras"
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(best_path), monitor="val_auc", mode="max", save_best_only=True),
    ]

    # Stage 1
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(CFG.LR_STAGE1), loss=loss_fn, metrics=metrics)
    print("🚀 Stage 1 (freeze)")
    model.fit(train_ds, validation_data=val_ds, epochs=CFG.EPOCHS_STAGE1, callbacks=callbacks)

    # Stage 2
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    train_ds_s2 = make_tfds(train_df, augmenter=None, training=True, cache_name="train_noaug")
    model.compile(optimizer=tf.keras.optimizers.Adam(CFG.LR_STAGE2), loss=loss_fn, metrics=metrics)
    print("🔓 Stage 2 (fine-tune)")
    model.fit(train_ds, validation_data=val_ds, epochs=CFG.EPOCHS_STAGE2, callbacks=callbacks)

    print("✅ Done. Best model:", best_path)

if __name__ == "__main__":
    main()
