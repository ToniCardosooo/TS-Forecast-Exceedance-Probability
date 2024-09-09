from pprint import pprint

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset, dataset_names

from codebase.load_data.base import LoadDataset

pprint(dataset_names)


class GluontsDataset(LoadDataset):
    _DATASET_NAME = 'GLUONTS'
    DATASET_NAME = _DATASET_NAME

    horizons_map = {
        'car_parts_without_missing': 12,
        #'airpassengers': 12,
        #'exchange_rate_nips': 12,
        'electricity_weekly': 12,
        #'nn5_weekly': 12,
        #'m1_quarterly': 4,
        #'m1_monthly': 12,
    }

    frequency_map = {
        'car_parts_without_missing': 12,
        #'airpassengers': 12,
        #'exchange_rate_nips': 365,
        'electricity_weekly': 52,
        #'nn5_weekly': 52,
        #'m1_quarterly': 4,
        #'m1_monthly': 12,
    }

    lag_map = {
        'car_parts_without_missing': 24,
        #'airpassengers': 24,
        #'exchange_rate_nips': 24,
        'electricity_weekly': 24,
        #'nn5_weekly': 24,
        #'m1_quarterly': 24,
        #'m1_monthly': 24,
    }

    frequency_pd = {
        'car_parts_without_missing': 'M',
        #'airpassengers': 'M',
        #'exchange_rate_nips': 'D',
        'electricity_weekly': 'W',
        #'nn5_weekly': 'W',
        #'m1_quarterly': 'Q',
        #'m1_monthly': 'M',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group):
        # group = 'solar_weekly'
        dataset = get_dataset(group, regenerate=True)
        train_list = dataset.train

        df_list = []
        for i, series in enumerate(train_list):
            s = pd.Series(
                series["target"],
                index=pd.date_range(
                    start=series["start"].to_timestamp(),
                    freq=series["start"].freq,
                    periods=len(series["target"]),
                ),
            )

            s_df = s.reset_index()
            s_df.columns = ['ds', 'y']
            s_df['unique_id'] = f'ID{i}'

            df_list.append(s_df)

        df = pd.concat(df_list).reset_index(drop=True)
        df = df[['unique_id', 'ds', 'y']]

        GluontsDataset.DATASET_NAME = GluontsDataset._DATASET_NAME
        GluontsDataset.DATASET_NAME += "_"+group
        return df
