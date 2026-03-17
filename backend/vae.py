"""
2.
vae.py
------
Variational Autoencoder (VAE) for learning normal network traffic.
Trained ONLY on benign samples — anomalies are detected via
reconstruction error + KL divergence (combined VAE loss).

Usage:
    python vae.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
FIG_DIR    = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LATENT_DIM  = 8
EPOCHS      = 60
BATCH_SIZE  = 256
VAL_SPLIT   = 0.1
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# SAMPLING LAYER  (reparameterisation trick)
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Samples z = mu + eps * sigma  where eps ~ N(0, I)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch  = tf.shape(z_mean)[0]
        dim    = tf.shape(z_mean)[1]
        eps    = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ─────────────────────────────────────────────
# BUILD VAE
# ─────────────────────────────────────────────
def build_vae(input_dim: int):
    # ── Encoder ──
    enc_input  = keras.Input(shape=(input_dim,), name="encoder_input")
    x          = layers.Dense(64, activation="relu")(enc_input)
    x          = layers.BatchNormalization()(x)
    x          = layers.Dense(32, activation="relu")(x)
    x          = layers.BatchNormalization()(x)
    x          = layers.Dense(16, activation="relu")(x)
    z_mean     = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var  = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z          = Sampling(name="z")([z_mean, z_log_var])

    encoder = Model(enc_input, [z_mean, z_log_var, z], name="encoder")

    # ── Decoder ──
    dec_input  = keras.Input(shape=(LATENT_DIM,), name="decoder_input")
    x          = layers.Dense(16, activation="relu")(dec_input)
    x          = layers.BatchNormalization()(x)
    x          = layers.Dense(32, activation="relu")(x)
    x          = layers.BatchNormalization()(x)
    x          = layers.Dense(64, activation="relu")(x)
    dec_output = layers.Dense(input_dim, activation="linear")(x)

    decoder = Model(dec_input, dec_output, name="decoder")

    return encoder, decoder


# ─────────────────────────────────────────────
# VAE MODEL  (custom train step)
# ─────────────────────────────────────────────
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder       = encoder
        self.decoder       = decoder
        self.total_loss_tracker  = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker  = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker     = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction       = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction       = self.decoder(z, training=False)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(data - reconstruction), axis=1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        total_loss = recon_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        return self.decoder(z)


# ─────────────────────────────────────────────
# RECONSTRUCTION ERROR
# ─────────────────────────────────────────────
def compute_reconstruction_error(vae: VAE, X: np.ndarray) -> np.ndarray:
    """Per-sample mean squared reconstruction error."""
    X_reconstructed = vae.predict(X, batch_size=512, verbose=0)
    errors = np.mean(np.square(X - X_reconstructed), axis=1)
    return errors


# ─────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Total VAE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["recon_loss"],     label="Train Recon")
    axes[1].plot(history.history["val_recon_loss"], label="Val Recon")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "vae_training.png")
    plt.savefig(path)
    plt.close()
    print(f"  Training plot saved → {path}")


# ─────────────────────────────────────────────
# PLOT RECONSTRUCTION ERROR DISTRIBUTION
# ─────────────────────────────────────────────
def plot_error_distribution(errors_normal, errors_attack):
    plt.figure(figsize=(10, 5))
    plt.hist(errors_normal, bins=100, alpha=0.6,
             color="steelblue", label="Normal",  density=True)
    plt.hist(errors_attack, bins=100, alpha=0.6,
             color="crimson",   label="Attack",  density=True)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("VAE Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "vae_error_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Error distribution plot saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def train_vae():
    print("\n━━━ Loading processed data ━━━")
    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"), allow_pickle=True)
    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"),  allow_pickle=True)

    input_dim = X_train.shape[1]
    print(f"  Input dim: {input_dim}")

    # ── Train on NORMAL traffic only ──
    X_train_normal = X_train[y_train == "normal"]
    print(f"  Normal training samples: {len(X_train_normal)}")

    print("\n━━━ Building VAE ━━━")
    encoder, decoder = build_vae(input_dim)
    encoder.summary()

    vae = VAE(encoder, decoder, name="vae")
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    print("\n━━━ Training VAE ━━━")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        )
    ]

    history = vae.fit(
        X_train_normal, X_train_normal,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    print("\n━━━ Saving models ━━━")
    encoder.save(os.path.join(MODELS_DIR, "vae_encoder.keras"))
    decoder.save(os.path.join(MODELS_DIR, "vae_decoder.keras"))
    print(f"  Encoder → {MODELS_DIR}/vae_encoder.keras")
    print(f"  Decoder → {MODELS_DIR}/vae_decoder.keras")

    print("\n━━━ Plotting training history ━━━")
    plot_history(history)

    print("\n━━━ Computing reconstruction errors ━━━")
    errors_normal = compute_reconstruction_error(vae, X_train_normal)

    X_train_attack = X_train[y_train != "normal"]
    errors_attack  = compute_reconstruction_error(vae, X_train_attack)

    print(f"  Normal  error — mean: {errors_normal.mean():.4f}  "
          f"std: {errors_normal.std():.4f}  "
          f"95th pct: {np.percentile(errors_normal, 95):.4f}")
    print(f"  Attack  error — mean: {errors_attack.mean():.4f}  "
          f"std: {errors_attack.std():.4f}")

    plot_error_distribution(errors_normal, errors_attack)

    # Save errors for threshold.py to use
    np.save(os.path.join(PROC_DIR, "train_normal_errors.npy"), errors_normal)
    np.save(os.path.join(PROC_DIR, "train_attack_errors.npy"), errors_attack)

    # Save full train errors + labels (for threshold optimisation)
    all_errors = compute_reconstruction_error(vae, X_train)
    np.save(os.path.join(PROC_DIR, "train_all_errors.npy"),   all_errors)
    np.save(os.path.join(PROC_DIR, "train_binary_labels.npy"),
            (y_train != "normal").astype(int))

    print("\n✅  VAE training complete!\n")
    return vae, encoder, decoder


if __name__ == "__main__":
    train_vae()