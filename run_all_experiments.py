import os

from neuralforecast.losses.pytorch import MQLoss, DistributionLoss

from codebase.load_data.m3 import M3Dataset # Monthly, Quarterly
from codebase.load_data.tourism import TourismDataset # Monthly, Quarterly
from codebase.load_data.gluonts import GluontsDataset # m1_monthly, m1_quarterly, electricity_weekly

from empirical_analysis.train import run_experiment

SCALER = 'standard'
THR_PERCENTILES = [90, 95, 99]

DATASETS = [M3Dataset, TourismDataset, GluontsDataset]
LOSSES = [MQLoss, DistributionLoss]

for dataset in DATASETS:
    for group in dataset.frequency_map.keys():
        for loss_fn in LOSSES:
            os.chdir("/home/tonicardoso/Desktop/TS-Forecast-Exceedance-Probability")
            run_experiment(dataset, group, THR_PERCENTILES, loss_fn, SCALER)

 