import copy

import torch
import pandas as pd
import numpy as np
from arch.bootstrap import MovingBlockBootstrap
from statsmodels.tsa.api import STL

from codebase.methods.log import LogTransformation


class TimeSeriesBootstrap:
    def __init__(self,
                 log: bool,
                 period: int,
                 moving_blocks: bool):

        self.log = log
        self.period = period
        self.moving_blocks = moving_blocks

    def bootstrap_ts(self, ts: np.ndarray):

        if not isinstance(ts, np.ndarray):
            ts = torch.tensor(ts)

        if self.log:
            ts_log = LogTransformation.transform(ts)
        else:
            ts_log = ts

        stl = STL(ts_log, period=self.period).fit()

        if self.moving_blocks:
            bs_resid = self.moving_block_bs(stl.resid, self.period)
        else:
            bs_resid = pd.Series(stl.resid).sample(len(stl.resid)).values

        bs_series = stl.trend + stl.seasonal + bs_resid

        if self.log:
            bs_series = LogTransformation.inverse_transform(bs_series)

        return bs_series

    def transform_tensor(self, ts: torch.tensor):
        arr = ts[0, :][ts[1, :] > 0]
        # print(arr)

        try:
            bs_arr = self.bootstrap_ts(arr)
        except ValueError:
            bs_arr = arr

        bs_t = torch.tensor(bs_arr, dtype=ts[0, :].dtype)

        ts[0, :][ts[1, :] > 0] = bs_t

        return ts

    def transform_temporal_batch(self, temporal, augment: bool = True):
        temporal_ = copy.deepcopy(temporal)
        temporal_o = copy.deepcopy(temporal)

        for i, ts in enumerate(temporal_):
            temporal_[i] = self.transform_tensor(ts)

        if augment:
            temporal_ = torch.concat([temporal_,
                                      temporal_o])

        return temporal_

    # def from_dataset(self, dataset: TimeSeriesDataset):
    #
    #     ts_df = dataset.temporal[:, 0].numpy()
    #     ga = GroupedArray(ts_df, dataset.indptr)
    #
    #     bootstrap_sample = []
    #
    #     for i in range(dataset.n_groups):
    #         ts, ts_idx = ga.take([i])
    #
    #         ts_bs = self.bootstrap_ts(ts)
    #
    #         bootstrap_sample.append(ts_bs)
    #
    #     bootstrap_arr = np.concatenate(bootstrap_sample)
    #
    #     temporal_bs = np.vstack([bootstrap_arr, dataset.temporal[:, 1].numpy()]).transpose()
    #     temporal_bs = torch.tensor(temporal_bs, dtype=torch.float)
    #
    #     return temporal_bs

    def bootstrap_df(self, dataset: pd.DataFrame):
        """
        :param dataset: (time_index, group_id, value)
        """

        grouped_df = dataset.groupby("unique_id")

        bootstrapped_list = []
        for g, df in grouped_df:
            s = df['y'].values

            if self.log:
                ts_log = LogTransformation.transform(s)
            else:
                ts_log = s

            try:
                stl = STL(ts_log, period=self.period).fit()

                if self.moving_blocks:
                    bs_resid = self.moving_block_bs(stl.resid, self.period)
                else:
                    bs_resid = pd.Series(stl.resid).sample(len(stl.resid)).values

                bs_series = stl.trend + stl.seasonal + bs_resid
            except ValueError:
                bs_series = ts_log

            if self.log:
                bs_series = LogTransformation.inverse_transform(bs_series)

            df['y'] = bs_series
            df = df.reset_index(drop=True)

            bootstrapped_list.append(df)

        bootstrapped_df = pd.concat(bootstrapped_list).reset_index(drop=True)
        bootstrapped_df['unique_id'] = bootstrapped_df['unique_id'].apply(lambda x: f'BS_{x}')

        return bootstrapped_df

    def apriori_augment_df(self, df, horizon):
        train_df, test_df = self.train_test_split(df, horizon)
        synth_tr_df = self.bootstrap_df(train_df)
        # augmented_tr_df = pd.concat([synth_tr_df, train_df]).reset_index(drop=True)

        # placeholder
        synth_test_df = test_df.copy()
        synth_test_df['unique_id'] = test_df['unique_id'].apply(lambda x: f'BS_{x}')

        synth_df = pd.concat([synth_tr_df, synth_test_df]).reset_index(drop=True)

        augmented_df = pd.concat([synth_df, df]).reset_index(drop=True)
        augmented_df = augmented_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        return augmented_df

    @staticmethod
    def moving_block_bs(x: pd.Series, w: int):
        mbb = MovingBlockBootstrap(block_size=w, x=x)

        xp = mbb.bootstrap(1)
        xp_data = list(xp)
        bs = xp_data[0][1]['x']

        return bs

    @staticmethod
    def train_test_split(df: pd.DataFrame, horizon: int):
        df_by_unq = df.groupby('unique_id')

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values('ds')

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df
