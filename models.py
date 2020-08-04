import random
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import mse
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

from util import config_to_instance


def make_reconstruction_loss(n_features):
    def reconstruction_loss(input_and_mask, y_pred, weights=None):
        X_values = input_and_mask[:, :n_features]
        # X_values.name = "$X_values"

        missing_mask = input_and_mask[:, n_features:]
        # missing_mask.name = "$missing_mask"
        observed_mask = 1 - missing_mask
        # observed_mask.name = "$observed_mask"

        X_values_observed = X_values * observed_mask
        # X_values_observed.name = "$X_values_observed"

        pred_observed = y_pred * observed_mask

        if weights is not None:
            return weighted_mse(
                y_true=X_values_observed, y_pred=pred_observed, weighted=weights
            )
        return mse(y_true=X_values_observed, y_pred=pred_observed)

    return reconstruction_loss


def weighted_mse(X_true, X_pred, weights):
    return np.linalg.norm(np.multiply(self.weights, (X_true - X_pred)))


def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)


class Autoencoder:
    def __init__(
        self,
        data,
        recurrent_weight=0.75,
        optimizer="adam",
        dropout_probability=0.4,
        layer_sizes=[2000, 500],
        hidden_activation=leaky_relu,
        output_activation="sigmoid",
        init="glorot_uniform",
        l1_penalty=0,
        l2_penalty=0,
        normalize=False,
        norm_factor=None,
        n_reads=None,
        exp_rate=2,
        thresh=100,
        training=False,
        val_mask=None,
        val_data=None,
    ):
        self.data = data.copy()
        self.recurrent_weight = recurrent_weight
        self.optimizer = config_to_instance(optimizer)
        self.dropout_probability = dropout_probability
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_sizes = layer_sizes
        self.init = init
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.training = training
        self.weights = None
        self.val_mask = val_mask
        self.val_data = val_data
        if self.val_data is not None and self.val_mask is None:
            print(
                "Validation mask has not been provided, validating on the whole data."
            )
            self.val_mask = ~np.isnan(self.val_data)
        if normalize:
            print(f"centering data")
            x = self.data.T[1]
            mean = np.array(
                [np.nan_to_num(np.mean(x[~np.isnan(x)])) for x in self.data.T]
            )
            nan_sum = np.sum(np.isnan(mean))
            assert not nan_sum, f"{nan_sum} NaN values encountered in mean"
            self.data -= mean
            if self.val_data is not None:
                self.val_data -= mean
        if norm_factor:
            print(f"normalizing data by {norm_factor}")
            self.data /= norm_factor
            if self.val_data is not None:
                self.val_data /= norm_factor
        if n_reads is not None:
            print("Weighting samples based on the number of reads.")
            self.weights = self.get_weights(n_reads, exp_rate, thresh)

    def get_weights(self, n_reads, exp_rate, thresh):
        nr = n_reads[n_reads > 0]
        nr = np.minimum(nr, thresh)
        nr = np.log(nr)
        maxr = np.max(nr)
        minr = np.min(nr)
        weights = np.exp(exp_rate * (nr - maxr) / (maxr - minr))
        weights_arr = np.zeros(n_reads.shape)
        weights_arr[n_reads > 0] = weights
        return weights_arr

    def _get_hidden_layer_sizes(self):
        n_dims = self.data.shape[1]
        return self.layer_sizes
        """
        return [
            min(self.layer_sizes[0], 8 * n_dims),
            min(self.layer_sizes[1], 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]
        """

    def _create_model(self):

        hidden_layer_sizes = self._get_hidden_layer_sizes()
        first_layer_size = hidden_layer_sizes[0]
        n_dims = self.data.shape[1]

        inp = Input((2 * self.data.shape[-1]))
        x = Dense(
            first_layer_size,
            input_dim=2 * n_dims,
            activation=self.hidden_activation,
            kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
            kernel_initializer=self.init,
        )(inp)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_probability)(x, training=self.training)

        for layer_size in hidden_layer_sizes[1:]:
            x = Dense(
                layer_size,
                activation=self.hidden_activation,
                kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                kernel_initializer=self.init,
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_probability)(x, training=self.training)

        output = Dense(
            n_dims,
            # activation=self.output_activation,
            kernel_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
            kernel_initializer=self.init,
        )(x)

        loss_function = make_reconstruction_loss(n_dims)

        model = Model(inp, output, name="autoencoder")
        model.compile(optimizer=self.optimizer, loss=loss_function)
        return model

    def fill(self, missing_mask):
        self.data[missing_mask] = -1

    def _create_missing_mask(self):
        if self.data.dtype != "f" and self.data.dtype != "d":
            self.data = self.data.astype(float)

        return np.isnan(self.data)

    def _train_epoch(self, model, missing_mask, batch_size):
        input_with_mask = np.hstack([self.data, missing_mask])
        n_samples = len(input_with_mask)
        n_batches = int(np.ceil(n_samples / batch_size))
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = input_with_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            model.train_on_batch(batch_data, batch_data)
        return model.predict(input_with_mask)

    def run(
        self,
        logs_dir,
        batch_size=256,
        n_epochs=500,
        log_every=50,
        save_every=100,
        patience=None,
        max_val=None,
    ):
        print(f"Running training for {n_epochs} epochs.")
        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        self.model = self._create_model()
        observed_mask = ~missing_mask
        if patience:
            print(
                f"Will stop training if val mae does not improve for {patience} epochs."
            )
        early_counts = 0
        best_val = np.inf
        for epoch in range(n_epochs):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mae = masked_mae(
                X_true=self.data, X_pred=X_pred, mask=observed_mask
            )
            if self.val_data is not None:
                val_mae = masked_mae(
                    X_true=self.val_data, X_pred=X_pred, mask=self.val_mask
                )
                if val_mae < best_val:
                    best_val = val_mae
                    early_counts = 0
                elif max_val and best_val > max_val:
                    pass
                else:
                    early_counts += 1
                    if patience and early_counts == patience:
                        print(
                            f"early stopping at epoch {epoch}: validation mae did not improve for {patience} epochs"
                        )
                        print(
                            f"best validation score: {best_val}, final validation score: {val_mae}"
                        )
                        break
            if epoch % log_every == 0:
                mesg = f"epoch {epoch}\n observed mae: {observed_mae}"
                if self.val_data is not None:
                    mesg += f", val mae: {val_mae}"
                print(mesg)
            if epoch and epoch % save_every == 0:
                dst_path = os.path.join(logs_dir, f"o{epoch}.npy")
                print(f"saving into {dst_path}")
                np.save(dst_path, X_pred)
            old_weight = 1.0 - self.recurrent_weight
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing
        np.save(os.path.join(logs_dir, "output.npy"), self.data)
        self.model.save_weights(os.path.join(logs_dir, "model.h5"))
