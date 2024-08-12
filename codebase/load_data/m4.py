import pandas as pd
from datasetsforecast.m4 import M4

from codebase.load_data.base import LoadDataset


class M4Dataset(LoadDataset):
    DATASET_NAME = 'M4'

    horizons_map = {
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
    }

    frequency_map = {
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 52,
    }

    context_length = {
        'Quarterly': 10,
        'Monthly': 24,
        'Weekly': 26,
    }

    frequency_pd = {
        'Quarterly': 'Q',
        'Monthly': 'M',
        'Weekly': 'W',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group):
        ds, *_ = M4.load(cls.DATASET_PATH, group=group)
        ds['ds'] = ds['ds'].astype(int)

        if group == 'Quarterly':
            ds = ds.query('unique_id!="Q23425"').reset_index(drop=True)

        unq_periods = ds['ds'].sort_values().unique()

        dates = pd.date_range(end='2024-03-01',
                              periods=len(unq_periods),
                              freq=cls.frequency_pd[group])

        new_ds = {k: v for k, v in zip(unq_periods, dates)}

        ds['ds'] = ds['ds'].map(new_ds)

        return ds
