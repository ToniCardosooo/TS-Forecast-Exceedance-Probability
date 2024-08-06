import pandas as pd
import os

from datasetsforecast.m3 import M3

from neuralforecast.auto import AutoMLP, AutoNHITS, AutoLSTM, AutoGRU, AutoDeepAR
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from utilsforecast.losses import smape

from utils import preprocess_dataset, train_and_numerical_forecast, predict_exceedance_from_percentiles, predict_exceedance_from_params, get_global_auc_logloss

DATASET = M3
GROUP = 'Monthly'
THR_PERCENTILE = [90, 95, 99]
LOSS = MQLoss

LEVEL_LIST = [80, 90, 98]
LOSS_KWARGS = {'level': LEVEL_LIST} if LOSS == MQLoss else {'distribution': 'Normal', 'level': LEVEL_LIST, 'return_params': True}
HORIZON = 12
LAG = 24 
SCALER = 'standard'


if __name__ == '__main__':
    dataset_name = DATASET.__name__

    df, _, _ = DATASET.load(directory='./', group=GROUP)
    df['ds'] = pd.to_datetime(df['ds'])
    train_df, test_df = preprocess_dataset(df, HORIZON, THR_PERCENTILE)

    experiment_id = f"{GROUP}_{LOSS.__name__}"
    
    try:
        os.makedirs(f"./{dataset_name}_{experiment_id}")
    except OSError:
        print(f"Directory {dataset_name}_{experiment_id} already exists!")

    os.chdir(f"./{dataset_name}_{experiment_id}")

    model_classes = [AutoMLP, AutoNHITS, AutoLSTM, AutoGRU] # AutoMLP, AutoNHITS, AutoLSTM, AutoGRU
    if LOSS == DistributionLoss:
        model_classes.append(AutoDeepAR)

    models = {}

    for model_class in model_classes:
        
        # Hyperparameter tunning
        config = model_class.get_default_config(h=HORIZON, backend='ray')
        config["input_size"] = LAG
        config["max_steps"] = 1500  ### train on 1500 max steps
        config["val_check_steps"] = 300 ### 300 steps interval for validation
        config["random_seed"] = tune.randint(1, 10)
        config["enable_checkpointing"] = True

        # start_padding_enabled=True for NHITS and MLP
        if model_class in [AutoMLP, AutoNHITS]:
            config["start_padding_enabled"] = True

        # DeepAR does not return parameters due to Monte Carlo sampling
        if model_class == AutoDeepAR:
            LOSS_KWARGS['return_params'] = False

        # Model hyperparameters
        model_kwargs = {
            'h': HORIZON,
            'loss': LOSS(**LOSS_KWARGS),
            'config': config,
            'search_alg': HyperOptSearch(),
            'backend': 'ray',
            'num_samples': 20 ### train on 20 samples
        }

        # DeepAR MQLoss validation loss doesnt use 3 quatile levels by default, it's needed to set them
        if model_class == AutoDeepAR:
            model_kwargs['valid_loss'] = MQLoss(level=LEVEL_LIST)
        
        # Define model
        models[model_class.__name__] = model_class(**model_kwargs)

    # Train models and save numerical forecast predictions
    forecast_df = train_and_numerical_forecast(models, train_df, test_df, horizon=HORIZON, group=GROUP, scaler=SCALER, percentile_training_levels=LEVEL_LIST)
    forecast_df.to_csv(f"forecast_df_{dataset_name}_{experiment_id}.csv", index=False)

    # Get SMAPE values
    models_names = ['SeasonalNaive'] + list(models.keys())
    smape_df = smape(forecast_df, models=models_names, id_col='unique_id', target_col='y_true')
    smape_df.to_csv(f"smape_{dataset_name}_{experiment_id}.csv", index=False)

    # Predict exceedance events, save them and perform AUC and Log Loss metrics on the results
    # From percentiles
    exceedance_percentiles_df = predict_exceedance_from_percentiles(forecast_df, test_df, models_names, THR_PERCENTILE, f"exceedance_percentiles_{experiment_id}.csv")
    get_global_auc_logloss(exceedance_percentiles_df, test_df, models_names, THR_PERCENTILE, filename=f"auc_logloss_percentiles_{dataset_name}_{experiment_id}.csv")

    # From distribution parameters
    if "AutoDeepAR" in list(models.keys()):
        models_names = list(models.keys()) 
        models_names.remove("AutoDeepAR") # Does not include Seasonal Naive nor AutoDeepAR
        exceedance_params_df = predict_exceedance_from_params(forecast_df, test_df, models_names, THR_PERCENTILE, f"exceedance_params_{experiment_id}.csv")
        get_global_auc_logloss(exceedance_params_df, test_df, models_names, THR_PERCENTILE, filename=f"auc_logloss_params_{dataset_name}_{experiment_id}.csv")


