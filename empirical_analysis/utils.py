import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, log_loss

from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast

def preprocess_dataset(df, horizon, percentile):
    # obtain percentile value based on the train data
    df[f'{percentile}th_percentile'] = df.groupby('unique_id')['y'].transform(lambda x: x.iloc[:-horizon].quantile(percentile/100))
    # add classification if timeseries value exceeds the percentile
    df['y_above_percentile'] = (df['y'] >= df[f'{percentile}th_percentile']).astype(int)
    # use the last "horizon" timesteps as test timesteps
    test_df = df.groupby('unique_id').tail(horizon)
    train_df = df.drop(test_df.index)
    return train_df, test_df


def prob_and_class(row_pred, row_test, model_name, percentile):
    # check whether MQLoss or DistributionLoss was used by verifying there isnt/is a column with "loc"

    # DistributionLoss
    if any(["loc" in col for col in row_pred.index]):
        mean = row_pred[f'{model_name}-loc']
        std = row_pred[f'{model_name}-scale']
        
        probability = 1 - norm.cdf(row_test[f'{percentile}th_percentile'], loc=mean, scale=std)
        classification = 1 if probability >= 0.5 else 0
        return probability, classification
    
    # MQLoss
    else:
        # x_high = mean + std * z_high
        # x_low = mean + std * z_low
        z_low, z_high = norm.ppf([1-percentile/100, percentile/100])

        conf_int_length = percentile - (100-percentile)

        std = (row_pred[f'{model_name}-hi-{conf_int_length}'] - row_pred[f'{model_name}-lo-{conf_int_length}']) / (z_high - z_low)
        mean = row_pred[f'{model_name}-hi-{conf_int_length}'] - z_high * std
        
        probability = 1 - norm.cdf(row_test[f'{percentile}th_percentile'], loc=mean, scale=std)
        classification = 1 if probability >= 0.5 else 0
        return probability, classification


"""
models = {
    "NHITS": NHITS(...),
    "GRU": GRU(...),
    ...
}
"""

def train_and_predict(models, train_df, test_df, horizon, scaler, percentile_training_levels, test_percentile):
    # Seasonal Naive as a baseline
    snaive = SeasonalNaive(season_length=12)  # Quarterly -> 4
    sf = StatsForecast(models=[snaive], freq='M')
    pred_sf = sf.forecast(df=train_df, h=horizon, level=percentile_training_levels)
    pred_sf['y_true'] = test_df['y'].to_list()
    pred_sf['y_true_above_thr'] = test_df['y_above_percentile'].to_list()

    # Train the other models
    nf = NeuralForecast(models=list(models.values()), freq='M', local_scaler_type=scaler)
    nf.fit(train_df, val_size=horizon, verbose=False)

    # Save best hyperparameters for each model
    if "Auto" in nf.models[0].__class__.__name__:
        for i in range(len(models.keys())):
            model_name = nf.models[i].__class__.__name__
            nf.models[i].results.get_dataframe().to_csv(f"hyperparams_{model_name}.csv", index=False)

    pred = nf.predict(test_df, verbose=False)

    # Put all predictions together
    # Classify the forecast as above/below the threshold
    pred['y_true'] = test_df['y'].to_list()
    pred['y_true_above_thr'] = test_df['y_above_percentile'].to_list()
    pred['SeasonalNaive'] = pred_sf['SeasonalNaive']
    
    for model_name in ['SeasonalNaive'] + list(models.keys()):
        above_thr_prob = []
        above_thr_class = []

        for i in range(pred.shape[0]):
            row_pred = pred.iloc[i, :] if model_name != 'SeasonalNaive' else pred_sf.iloc[i, :]
            row_test = test_df.iloc[i, :]
            probability, classification = prob_and_class(row_pred, row_test, model_name, test_percentile)
            above_thr_prob.append(probability)
            above_thr_class.append(classification)

        pred[f'{model_name}_forecast_above_thr_prob'] = above_thr_prob
        pred[f'{model_name}_forecast_above_thr'] = above_thr_class

    pred['unique_id'] = pred.index
    return pred

def get_global_auc_logloss(pred_df, models_names, filename="auc_logloss.csv"):
    auc_logloss_df = []
    
    y_true_above_thr = pred_df['y_true_above_thr'].tolist() + [0,1]
    for model_name in models_names:
        model_forecast_above_thr = pred_df[f'{model_name}_forecast_above_thr'].tolist() + [0,1]
        model_auc = roc_auc_score(y_true_above_thr, model_forecast_above_thr)
        model_logloss = log_loss(y_true_above_thr, model_forecast_above_thr)
        auc_logloss_df.append({
            'Model': model_name,
            'AUC': model_auc,
            'LogLoss': model_logloss
        })
    
    auc_logloss_df = pd.DataFrame(data=auc_logloss_df)
    auc_logloss_df.to_csv(filename, index=False)