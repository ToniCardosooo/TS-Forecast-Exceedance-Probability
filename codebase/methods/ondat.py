import re

import torch
import torch.nn as nn
import numpy as np
from neuralforecast.models import NHITS

from codebase.methods.mbb import TimeSeriesBootstrap


class OnDAT_NHITS(NHITS):
    """
    On-the-fly Data Augmentation for Time Series
    """

    def __init__(self,
                 period,
                 moving_blocks,
                 on_train: bool,
                 on_valid: bool,
                 log_on_mbb: bool = True,
                 **kwargs):

        self.boot = TimeSeriesBootstrap(log=log_on_mbb,
                                        period=period,
                                        moving_blocks=moving_blocks)
        self.on_train = on_train
        self.on_valid = on_valid
        self.log_on_mbb = log_on_mbb

        super().__init__(**kwargs)

    def __repr__(self):

        if self.on_train:
            if self.on_valid:
                conf = 'OnDAT'
            else:
                conf = 'OnDAT(Tr)'
        else:
            if self.on_valid:
                conf = 'OnDAT(Vl)'
            else:
                raise ValueError('Need either self.on_train or self.on_valid')

        if not self.boot.moving_blocks:
            conf = re.sub('OnDAT', 'OnDAT(fixed)', conf)

        if not self.log_on_mbb:
            conf = 'OnDAT(no_log)'

        return conf

    def _create_windows(self, batch, step, w_idxs=None):
        # Parse common data
        window_size = self.input_size + self.h
        temporal_cols = batch["temporal_cols"]
        temporal = batch["temporal"]

        if step == "train":
            if self.val_size + self.test_size > 0:
                cutoff = -self.val_size - self.test_size
                temporal = temporal[:, :, :cutoff]

                if self.on_train:
                    temporal = self.boot.transform_temporal_batch(temporal)

            temporal = self.padder_train(temporal)
            if temporal.shape[-1] < window_size:
                raise Exception(
                    "Time series is too short for training, consider setting a smaller input size or set start_padding_enabled=True"
                )
            windows = temporal.unfold(
                dimension=-1, size=window_size, step=self.step_size
            )

            # [B, C, Ws, L+H] 0, 1, 2, 3
            # -> [B * Ws, L+H, C] 0, 2, 3, 1
            windows_per_serie = windows.shape[2]
            windows = windows.permute(0, 2, 3, 1).contiguous()
            windows = windows.reshape(-1, window_size, len(temporal_cols))

            # Sample and Available conditions
            available_idx = temporal_cols.get_loc("available_mask")
            available_condition = windows[:, : self.input_size, available_idx]
            available_condition = torch.sum(available_condition, axis=1)
            final_condition = available_condition > 0
            if self.h > 0:
                sample_condition = windows[:, self.input_size:, available_idx]
                sample_condition = torch.sum(sample_condition, axis=1)
                final_condition = (sample_condition > 0) & (available_condition > 0)
            windows = windows[final_condition]

            # Parse Static data to match windows
            # [B, S_in] -> [B, Ws, S_in] -> [B*Ws, S_in]
            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)
            if static is not None:
                static = torch.repeat_interleave(
                    static, repeats=windows_per_serie, dim=0
                )
                static = static[final_condition]

            # Protection of empty windows
            if final_condition.sum() == 0:
                raise Exception("No windows available for training")

            # Sample windows
            n_windows = len(windows)
            if self.windows_batch_size is not None:
                w_idxs = np.random.choice(
                    n_windows,
                    size=self.windows_batch_size,
                    replace=(n_windows < self.windows_batch_size),
                )
                windows = windows[w_idxs]

                if static is not None:
                    static = static[w_idxs]

            # think about interaction available * sample mask
            # [B, C, Ws, L+H]
            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )
            return windows_batch

        elif step in ["predict", "val"]:
            if step == "predict":
                initial_input = temporal.shape[-1] - self.test_size
                if (
                        initial_input <= self.input_size
                ):  # There is not enough data to predict first timestamp
                    padder_left = nn.ConstantPad1d(
                        padding=(self.input_size - initial_input, 0), value=0
                    )
                    temporal = padder_left(temporal)
                predict_step_size = self.predict_step_size
                cutoff = -self.input_size - self.test_size
                temporal = temporal[:, :, cutoff:]

            elif step == "val":
                predict_step_size = self.step_size
                cutoff = -self.input_size - self.val_size - self.test_size
                if self.test_size > 0:
                    temporal = batch["temporal"][:, :, cutoff: -self.test_size]
                else:
                    temporal = batch["temporal"][:, :, cutoff:]

                if self.on_valid:
                    temporal = self.boot.transform_temporal_batch(temporal, augment=False)

                if temporal.shape[-1] < window_size:
                    initial_input = temporal.shape[-1] - self.val_size
                    padder_left = nn.ConstantPad1d(
                        padding=(self.input_size - initial_input, 0), value=0
                    )
                    temporal = padder_left(temporal)

            if (
                    (step == "predict")
                    and (self.test_size == 0)
                    and (len(self.futr_exog_list) == 0)
            ):
                padder_right = nn.ConstantPad1d(padding=(0, self.h), value=0)
                temporal = padder_right(temporal)

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=predict_step_size
            )

            # [batch, channels, windows, window_size] 0, 1, 2, 3
            # -> [batch * windows, window_size, channels] 0, 2, 3, 1
            windows_per_serie = windows.shape[2]
            windows = windows.permute(0, 2, 3, 1).contiguous()
            windows = windows.reshape(-1, window_size, len(temporal_cols))

            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)
            if static is not None:
                static = torch.repeat_interleave(
                    static, repeats=windows_per_serie, dim=0
                )

            # Sample windows for batched prediction
            if w_idxs is not None:
                windows = windows[w_idxs]
                if static is not None:
                    static = static[w_idxs]

            windows_batch = dict(
                temporal=windows,
                temporal_cols=temporal_cols,
                static=static,
                static_cols=static_cols,
            )
            return windows_batch
        else:
            raise ValueError(f"Unknown step {step}")
