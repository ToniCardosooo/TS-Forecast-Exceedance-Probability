from pprint import pprint

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset, dataset_names

from codebase.load_data.base import LoadDataset

pprint(dataset_names)


class GluontsDataset(LoadDataset):
    DATASET_NAME = 'GLUONTS'

    horizons_map = {
        'nn5_weekly': 12,
        'electricity_weekly': 12,
        'm1_quarterly': 3,
        'm1_monthly': 8,
    }

    frequency_map = {
        'nn5_weekly': 52,
        'electricity_weekly': 52,
        'm1_quarterly': 4,
        'm1_monthly': 12,
    }

    context_length = {
        'nn5_weekly': 52,
        'electricity_weekly': 52,
        'm1_quarterly': 8,
        'm1_monthly': 24,
    }

    frequency_pd = {
        'nn5_weekly': 'W',
        'electricity_weekly': 'W',
        'm1_quarterly': 'Q',
        'm1_monthly': 'M',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group):
        # group = 'solar_weekly'
        dataset = get_dataset(group, regenerate=False)
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

        return df
