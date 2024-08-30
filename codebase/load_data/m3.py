from datasetsforecast.m3 import M3

from codebase.load_data.base import LoadDataset


class M3Dataset(LoadDataset):
    _DATASET_NAME = 'M3'
    DATASET_NAME = _DATASET_NAME

    horizons_map = {
        'Quarterly': 6,
        'Monthly': 18
    }

    frequency_map = {
        'Quarterly': 4,
        'Monthly': 12
    }

    context_length = {
        'Quarterly': 8,
        'Monthly': 24
    }

    frequency_pd = {
        'Quarterly': 'Q',
        'Monthly': 'M'
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group):
        ds, *_ = M3.load(cls.DATASET_PATH, group=group)

        M3Dataset.DATASET_NAME = M3Dataset._DATASET_NAME
        M3Dataset.DATASET_NAME += "_"+group
        
        return ds
