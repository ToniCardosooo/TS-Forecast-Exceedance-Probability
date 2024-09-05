import pandas as pd
import os

from neuralforecast.auto import AutoMLP, AutoNHITS, AutoLSTM, AutoGRU, AutoDeepAR
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from utilsforecast.losses import smape

from empirical_analysis.utils import (
    preprocess_dataset, 
    train_and_numerical_forecast, 
    predict_exceedance_from_percentiles, 
    predict_exceedance_from_params, 
    get_global_auc_logloss
)

def run_experiment(dataset, group, thr_percentiles, loss_fn, scaler):

    horizon = dataset.horizons_map[group]

    level_list = [100 - 2*(100-thr) for thr in thr_percentiles]
    loss_kwargs = {'level': level_list} if loss_fn == MQLoss else {'distribution': 'Normal', 'level': level_list, 'return_params': True}

    df = dataset.load_data(group=group)

    train_df, test_df = preprocess_dataset(df, horizon, thr_percentiles)

    experiment_id = f"{dataset.DATASET_NAME}_{loss_fn.__name__}"
    
    try:
        os.makedirs(f"./{experiment_id}")
    except OSError:
        print(f"Directory {experiment_id} already exists!")

    os.chdir(f"./{experiment_id}")

    model_classes = [AutoMLP, AutoNHITS, AutoLSTM, AutoGRU] # AutoMLP, AutoNHITS, AutoLSTM, AutoGRU
    if loss_fn == DistributionLoss:
        model_classes.append(AutoDeepAR)

    models = {}

    for model_class in model_classes:
        
        # Hyperparameter tunning
        config = model_class.get_default_config(h=horizon, backend='ray')
        config["input_size"] = dataset.lag_map[group]
        config["max_steps"] = 1500  ### train on 1500 max steps
        config["val_check_steps"] = 300 ### 300 steps interval for validation
        config["random_seed"] = tune.randint(1, 100)
        config["enable_checkpointing"] = True

        # start_padding_enabled=True for NHITS and MLP
        if model_class in [AutoMLP, AutoNHITS]:
            config["start_padding_enabled"] = True

        # DeepAR does not return parameters due to Monte Carlo sampling
        if model_class == AutoDeepAR:
            loss_kwargs['return_params'] = False

        # Model hyperparameters
        model_kwargs = {
            'h': horizon,
            'loss': loss_fn(**loss_kwargs),
            'config': config,
            'search_alg': HyperOptSearch(n_initial_points=10),
            'backend': 'ray',
            'num_samples': 40 ### train on 20 samples
        }

        # DeepAR MQLoss validation loss doesnt use 3 quatile levels by default, it's needed to set them
        if model_class == AutoDeepAR:
            model_kwargs['valid_loss'] = MQLoss(level=level_list)
        
        # Define model
        models[model_class.__name__] = model_class(**model_kwargs)

    # Train models and save numerical forecast predictions
    if f"forecast_df_{experiment_id}.csv" not in os.listdir():
        forecast_df = train_and_numerical_forecast(models, train_df, test_df, horizon=horizon, dataset=dataset, group=group, scaler=scaler, percentile_training_levels=level_list)
        forecast_df.to_csv(f"forecast_df_{experiment_id}.csv", index=False)
    else:
        print("Forecast values already exist!")
        forecast_df = pd.read_csv(f"forecast_df_{experiment_id}.csv")

    # Get SMAPE values
    models_names = ['SeasonalNaive'] + list(models.keys())
    
    if loss_fn == MQLoss:
        models_names = ['SeasonalNaive'] + [model_name+"-median" for model_name in models.keys()]

    smape_df = smape(forecast_df, models=models_names, id_col='unique_id', target_col='y_true')
    smape_df.to_csv(f"smape_{experiment_id}.csv", index=False)

    if loss_fn == MQLoss:
        models_names = ['SeasonalNaive'] + list(models.keys())

    # Predict exceedance events, save them and perform AUC and Log Loss metrics on the results
    # From percentiles
    exceedance_percentiles_df = predict_exceedance_from_percentiles(forecast_df, test_df, models_names, thr_percentiles, f"exceedance_percentiles_{experiment_id}.csv")
    get_global_auc_logloss(exceedance_percentiles_df, test_df, models_names, thr_percentiles, filename=f"auc_logloss_percentiles_{experiment_id}.csv")

    # From distribution parameters
    if "AutoDeepAR" in list(models.keys()):
        models_names = list(models.keys()) 
        models_names.remove("AutoDeepAR") # Does not include Seasonal Naive nor AutoDeepAR
        exceedance_params_df = predict_exceedance_from_params(forecast_df, test_df, models_names, thr_percentiles, f"exceedance_params_{experiment_id}.csv")
        get_global_auc_logloss(exceedance_params_df, test_df, models_names, thr_percentiles, filename=f"auc_logloss_params_{experiment_id}.csv")



if __name__ == '__main__':

    from codebase.load_data.m3 import M3Dataset # Monthly, Quarterly
    from codebase.load_data.tourism import TourismDataset # Monthly, Quarterly
    from codebase.load_data.gluonts import GluontsDataset # m1_monthly, m1_quarterly, electricity_weekly

    DATASET = M3Dataset
    GROUP = 'Monthly'
    THR_PERCENTILES = [90, 95, 99]
    LOSS = DistributionLoss

    HORIZON = 12
    LAG = 24 
    SCALER = 'standard'

    run_experiment(
        dataset=DATASET,
        group=GROUP,
        thr_percentiles=THR_PERCENTILES,
        loss_fn=LOSS,
        scaler=SCALER
    )