import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB0
from .config import CFG

def build_model():
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(CFG.IMG_SIZE, CFG.IMG_SIZE, 3)
    )
    base.trainable = False

    inputs = layers.Input(shape=(CFG.IMG_SIZE, CFG.IMG_SIZE, 3))
    x = inputs * 255.0
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # ✅ nếu mixed precision thì output float32 để ổn định loss/metrics
    out_dtype = "float32" if CFG.USE_MIXED_PRECISION else None
    outputs = layers.Dense(CFG.NUM_CLASSES, activation="sigmoid", dtype=out_dtype)(x)

    model = models.Model(inputs, outputs)
    return model, base
