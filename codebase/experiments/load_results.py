import os
import re

import pandas as pd


class ResultsReader:
    RESULTS_DIR = 'assets/results'
    # RESULTS_DIR = '/Users/vcerq/Desktop/assets/results'

    @classmethod
    def read_results(cls):
        files = cls.get_all_files()

        results_l = []
        for file in files:
            print(file)
            if not bool(re.search('_results.csv$', file)):
                continue

            cv_ = pd.read_csv(f'{cls.RESULTS_DIR}/{file}')
            group = re.sub('_results.csv$', '', file)
            cv_['group'] = group
            cv_['freq'] = group.split('_')[-1].title()

            results_l.append(cv_)

        results_df = pd.concat(results_l).reset_index(drop=True)

        return results_df

    @classmethod
    def get_all_files(cls):
        files = os.listdir(cls.RESULTS_DIR)
        return files


class AblationResultsReader(ResultsReader):

    @classmethod
    def read_results(cls):
        files = cls.get_all_files()

        results_l = []
        for file in files:
            print(file)
            if bool(re.search('_ablation.csv', file)):
                cv_ = pd.read_csv(f'{cls.RESULTS_DIR}/{file}')
                group = re.sub('_ablation.csv$', '', file)
                cv_['group'] = group
                cv_['freq'] = group.split('_')[-1].title()

                results_l.append(cv_)

        results_df = pd.concat(results_l).reset_index(drop=True)

        return results_df


class AmplifierResultsReader(ResultsReader):
    RESULTS_DIR = 'assets/results'

    @classmethod
    def read_results(cls):
        files = cls.get_all_files()

        results_l = []
        for file in files:
            print(file)
            if bool(re.search('_amplifier.csv', file)):
                cv_ = pd.read_csv(f'{cls.RESULTS_DIR}/{file}')
                group = re.sub('_amplifier.csv$', '', file)
                cv_['group'] = group
                cv_['freq'] = group.split('_')[-1].title()

                results_l.append(cv_)

        results_df = pd.concat(results_l).reset_index(drop=True)

        return results_df


class LogReader:
    # LOGS_DIR = 'assets/logs'
    LOGS_DIR = '/Users/vcerq/Desktop/assets/logs'

    MODELS_NAME_MAP = {
        'NHITS': 'NHITS',
        'DA': 'DA+Standard',
        'OnDAT': 'NH_OnDAT_A3',
    }

    DATAGROUP_PAIRS = [
        'Gluonts_m1_quarterly',
        'Gluonts_m1_monthly',
        'M3_Monthly',
        'M3_Quarterly',
        'Tourism_Monthly',
        'Tourism_Quarterly',
        'M4_Monthly',
        'M4_Quarterly',
    ]

    @classmethod
    def read_results(cls):

        val_losses = {}
        for ds in cls.DATAGROUP_PAIRS:
            # ds = 'Gluonts_m1_quarterly'
            loss_dict = {}
            for mod, modn in cls.MODELS_NAME_MAP.items():
                path = f'{cls.LOGS_DIR}/{mod}_{ds}/version_0/metrics.csv'

                try:
                    loss = pd.read_csv(path)
                except FileNotFoundError:
                    continue

                final_vl_loss = loss['valid_loss'].dropna().values[-1]

                loss_dict[modn] = final_vl_loss

            val_losses[ds] = loss_dict

        val_df = pd.DataFrame(val_losses).T

        return val_df
