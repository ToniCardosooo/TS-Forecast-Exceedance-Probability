import pandas as pd
import plotnine as p9

from neuralforecast.losses.numpy import smape


class ResultAnalysis:
    METADATA = ['freq', 'group', 'ds', 'cutoff', 'unique_id', 'dataset', 'y']

    def __init__(self, cv: pd.DataFrame):
        self.cv = cv

    @classmethod
    def calc_error(cls, cv: pd.DataFrame):
        methods = cls.get_methods(cv)

        error = {}
        for m in methods:
            err = smape(y=cv['y'], y_hat=cv[m])
            error[m] = err

        error = pd.Series(error)

        return error

    @classmethod
    def get_methods(cls, cv: pd.DataFrame):
        methods = cv.drop(columns=cls.METADATA).columns.tolist()

        return methods

    @staticmethod
    def get_p9_theme():
        my_theme = p9.theme_538(base_family='Palatino', base_size=12) + \
                   p9.theme(plot_margin=.025,
                            axis_text_y=p9.element_text(size=10),
                            panel_background=p9.element_rect(fill='white'),
                            plot_background=p9.element_rect(fill='white'),
                            strip_background=p9.element_rect(fill='white'),
                            legend_background=p9.element_rect(fill='white'),
                            axis_text_x=p9.element_text(size=10))

        return my_theme

    @classmethod
    def calc_error_by(cls, cv, by='group'):
        # 'group' or 'unique_id'

        cv_grouped = cv.groupby(by)

        dataset_error = {}
        for g, df in cv_grouped:
            dataset_error[g] = cls.calc_error(df)

        df = pd.concat(dataset_error, axis=1).T

        return df
