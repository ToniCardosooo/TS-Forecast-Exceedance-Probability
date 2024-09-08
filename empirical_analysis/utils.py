import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, log_loss

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast

def preprocess_dataset(df, horizon, percentiles):
    for percentile in percentiles:
        # obtain percentile value based on the train data
        df[f'y_percentile_{percentile}'] = df.groupby('unique_id')['y'].transform(lambda x: x.iloc[:-horizon].quantile(percentile/100))
        # add classification if timeseries value exceeds the percentile
        df[f'y_above_percentile_{percentile}'] = (df['y'] >= df[f'y_percentile_{percentile}']).astype(int)
        
    # use the last "horizon" timesteps as test timesteps
    test_df = df.groupby('unique_id').tail(horizon)
    train_df = df.drop(test_df.index)
    return train_df, test_df


def predict_exceedance_from_percentiles(pred_df, test_df, models_names, percentiles, filename="exceedance_percentiles.csv"):
    exceedance_df = pd.DataFrame()

    for percentile in percentiles:

        z_low, z_high = norm.ppf([1-percentile/100, percentile/100])
        conf_int_length = percentile - (100-percentile)

        for model_name in models_names:
            model_exceedance_preds = []
            model_exceedance_prob_preds = []

            for i in range(pred_df.shape[0]):
                row_pred = pred_df.iloc[i, :]
                row_test = test_df.iloc[i, :]

                std = (row_pred[f'{model_name}-hi-{conf_int_length}'] - row_pred[f'{model_name}-lo-{conf_int_length}']) / (z_high - z_low)
                mean = row_pred[f'{model_name}-hi-{conf_int_length}'] - z_high * std
                
                probability = 1 - norm.cdf(row_test[f'y_percentile_{percentile}'], loc=mean, scale=std)
                classification = 1 if probability >= 0.5 else 0

                model_exceedance_prob_preds.append(probability)
                model_exceedance_preds.append(classification)
            
            exceedance_df[f'{model_name}_{percentile}'] = model_exceedance_preds
            exceedance_df[f'{model_name}_prob_{percentile}'] = model_exceedance_prob_preds
    
    exceedance_df.to_csv(filename, index=False)
    return exceedance_df


def predict_exceedance_from_params(pred_df, test_df, models_names, percentiles, filename="exceedance_params.csv"):
    exceedance_df = pd.DataFrame()

    for percentile in percentiles:

        for model_name in models_names:
            model_exceedance_preds = []
            model_exceedance_prob_preds = []

            for i in range(pred_df.shape[0]):
                row_pred = pred_df.iloc[i, :]
                row_test = test_df.iloc[i, :]

                mean = row_pred[f'{model_name}-loc']
                std = row_pred[f'{model_name}-scale']
                
                probability = 1 - norm.cdf(row_test[f'y_percentile_{percentile}'], loc=mean, scale=std)
                classification = 1 if probability >= 0.5 else 0

                model_exceedance_prob_preds.append(probability)
                model_exceedance_preds.append(classification)
            
            exceedance_df[f'{model_name}_{percentile}'] = model_exceedance_preds
            exceedance_df[f'{model_name}_prob_{percentile}'] = model_exceedance_prob_preds
    
    exceedance_df.to_csv(filename, index=False)
    return exceedance_df


def train_and_numerical_forecast(stats_models, neural_models, train_df, test_df, horizon, dataset, group, scaler, percentile_training_levels):
    # Run statistical models
    sf = StatsForecast(models=list(stats_models.values()), freq=dataset.frequency_pd[group], n_jobs=-1)
    pred_sf = sf.forecast(df=train_df, h=horizon, level=percentile_training_levels)
    
    # Train Deep Learning models
    nf = NeuralForecast(models=list(neural_models.values()), freq=dataset.frequency_pd[group], local_scaler_type=scaler)
    nf.fit(train_df, val_size=horizon, verbose=False)
    pred_nf = nf.predict(verbose=False)

    # Save best hyperparameters for each model
    if "Auto" in nf.models[0].__class__.__name__:
        for i in range(len(neural_models.keys())):
            model_name = nf.models[i].__class__.__name__
            nf.models[i].results.get_dataframe().to_csv(f"hyperparams_{model_name}.csv", index=False)

    # Merge statistical models' results with the other models results
    pred_df = pd.merge(pred_sf, pred_nf, how='inner', on=['unique_id','ds'])

    # Add true values to the dataframe
    pred_df['y_true'] = test_df['y'].to_list()

    pred_df = pred_df.reset_index()
    return pred_df


def get_global_auc_logloss(pred_df, test_df, models_names, test_percentiles, filename="auc_logloss.csv"):
    auc_logloss_df = []
    
    test_df = test_df.reset_index()

    for percentile in test_percentiles:

        y_true_above_thr = test_df[f'y_above_percentile_{percentile}']

        for model_name in models_names:

            model_forecast_above_thr = pred_df[f'{model_name}_prob_{percentile}']
            
            y_true_above_thr_filter = y_true_above_thr[model_forecast_above_thr.notna()].tolist() + [0,1]
            model_forecast_above_thr = model_forecast_above_thr.dropna(ignore_index=True).tolist() + [0,1]

            model_auc = roc_auc_score(y_true_above_thr_filter, model_forecast_above_thr)
            model_logloss = log_loss(y_true_above_thr_filter, model_forecast_above_thr)
            auc_logloss_df.append({
                'Model_Percentile': f'{model_name}_{percentile}',
                'AUC': model_auc,
                'LogLoss': model_logloss
            })
        
    auc_logloss_df = pd.DataFrame(data=auc_logloss_df)
    auc_logloss_df.to_csv(filename, index=False)