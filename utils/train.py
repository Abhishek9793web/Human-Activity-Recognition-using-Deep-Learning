import os
import tensorflow as tf

def train_model(model, X_train, y_train, X_val, y_val, artifacts_dir="artifacts", epochs=20, batch_size=64):
    os.makedirs(artifacts_dir, exist_ok=True)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(artifacts_dir, "best_model.h5"), save_best_only=True)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return model, history