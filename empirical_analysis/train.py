import pandas as pd
import os

from datasetsforecast.m3 import M3

from neuralforecast.auto import AutoMLP, AutoNHITS, AutoLSTM, AutoGRU # to include DeepAR
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from utilsforecast.losses import smape

from utils import preprocess_dataset, train_and_predict, get_global_auc_logloss

DATASET = M3
LOSS = DistributionLoss
LEVEL_LIST = [70, 80, 90]
LOSS_KWARGS = {'level': LEVEL_LIST} if LOSS == MQLoss else {'distribution': 'Normal', 'level': LEVEL_LIST, 'return_params': True}
THR_PERCENTILE = 95
HORIZON = 12
LAG = 24 
SCALER = 'standard'


if __name__ == '__main__':
    dataset_name = DATASET.__name__

    df, _, _ = DATASET.load(directory='./', group='Monthly')
    df['ds'] = pd.to_datetime(df['ds'])
    train_df, test_df = preprocess_dataset(df, HORIZON, THR_PERCENTILE)
    
    try:
        os.makedirs(f"./{dataset_name}_{THR_PERCENTILE}_{LOSS.__name__}")
    except OSError:
        print(f"Directory {dataset_name}_{THR_PERCENTILE}_{LOSS.__name__} already exists!")

    os.chdir(f"./{dataset_name}_{THR_PERCENTILE}_{LOSS.__name__}")

    model_classes = [AutoMLP, AutoNHITS, AutoLSTM, AutoGRU] # to include DeepAR but it doesnt use MQLoss
    models = {}

    for model_class in model_classes:

        config = model_class.get_default_config(h=HORIZON, backend='ray')
        config["input_size"] = LAG
        config["max_steps"] = 1500  ### train on 1500 max steps
        config["val_check_steps"] = 100
        config["random_seed"] = tune.randint(1, 10)

        # start_padding_enabled=True for NHITS and MLP
        if model_class in [AutoNHITS, AutoMLP]:
            config["start_padding_enabled"] = True

        models[model_class.__name__] = model_class(
            h=HORIZON,
            loss=LOSS(**LOSS_KWARGS),
            config=config,
            search_alg=HyperOptSearch(),
            backend='ray',
            num_samples=20 ### train on 20 samples
        )

    # Train models and save predictions
    pred_df = train_and_predict(models, train_df, test_df, horizon=HORIZON, scaler=SCALER, percentile_training_levels=LEVEL_LIST, test_percentile=THR_PERCENTILE)
    pred_df.to_csv(f"pred_df_{dataset_name}_{THR_PERCENTILE}_{LOSS.__name__}.csv", index=True)

    smape_names = [model_name+'-median' for model_name in models.keys()] + ['SeasonalNaive']
    smape_df = smape(pred_df, models=smape_names, id_col='unique_id', target_col='y_true')
    smape_df.to_csv(f"smape_{dataset_name}_{THR_PERCENTILE}_{LOSS.__name__}.csv", index=False)

    get_global_auc_logloss(pred_df, list(models.keys())+['SeasonalNaive'], filename=f"auc_logloss_{dataset_name}_{THR_PERCENTILE}_{LOSS.__name__}.csv")
