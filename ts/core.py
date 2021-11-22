import numpy as np
import pandas as pd
import functools
import pickle as pkl
import os.path

from global_vars import engine_ali
from datetime import datetime

from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from pmdarima.arima import auto_arima
from pmdarima.preprocessing import FourierFeaturizer

from itertools import starmap

from .auto import auto_arima_lite


def get_latest_factor_premium(factors=None, fillna_value=0, factor_tbl="processed_factor_premium_monthly"):
    print(f"Fill na with {fillna_value}")
    with engine_ali.connect() as conn:
        prem = pd.read_sql(f"SELECT * FROM {factor_tbl}", conn).drop("len", axis=1)

    prem["period_end"] = pd.to_datetime(prem["period_end"], format="%Y-%m-%d")
    prem = prem.dropna(subset=["group", "period_end"])

    if factors is None:
        factors = prem.select_dtypes("float").columns.tolist()
        print(f"Model for factors {','.join(factors)}.")
    else:
        ignored_factors = list(set(prem.select_dtypes('float').columns.tolist()) - set(factors))
        print(f"Model for factors {','.join(factors)} only. Skip {','.join(ignored_factors)}.")
    prem[factors] = prem.groupby("group")[factors].fillna(fillna_value).copy()
    prem = prem.melt(
        id_vars=["period_end", "group"],
        value_vars=factors,
        var_name="factor",
        value_name="premium")
    prem = prem.sort_values(by=["group", "period_end"]).reset_index(drop=True)
    return prem


class MarketFactorARIMA:
    def __init__(
            self,
            market_name,
            factor_name,
            out_of_sample_size=12,
            test_size=3,
            num_splits=3,
            # num_splits=1,
            season_size=12
        ):
        self.market_name = market_name
        self.factor_name = factor_name
        self.out_of_sample_size = out_of_sample_size
        self.test_size = test_size
        self.num_splits = num_splits
        self.season_size = season_size
    
    def _get_best_model_params(self, ts, exogeneous, trace=False):
        auto_arima_default = functools.partial(
            auto_arima_lite,
            m=self.season_size,
            out_of_sample_size=self.out_of_sample_size,
            suppress_warnings=True,
            information_criterion="oob",
            return_valid_fits=True,
            trace=trace
        )
        return auto_arima_default(ts, X=exogeneous)

    def train(self, dates, premium, fourier=False, trace=False):
        dates_train = dates[:-self.test_size]
        train_val_splits = ExpandingWindowSplitter(
            initial_window=dates_train.size-self.out_of_sample_size*self.num_splits,
            step_length=self.out_of_sample_size,
            fh=np.arange(1, self.out_of_sample_size+1)
        ).split(dates_train)
        train_val_splits = list(map(np.concatenate, train_val_splits))
        
        if fourier:
            self.ff = FourierFeaturizer(m=self.season_size, k=min(3, self.season_size//2))
            data = map(lambda idx: (premium.iloc[idx], self.ff.fit_transform(premium.iloc[idx])[1]),
                   train_val_splits)
        else:
            self.ff = None
            data = map(lambda idx: (premium.iloc[idx], None), train_val_splits)

        if trace:
            data = map(lambda args: (*args, trace), data)

        models = [x for lst in starmap(self._get_best_model_params, data) for x in lst]
        
        val_res = pd.DataFrame.from_records(models, columns=["order", "seasonal", "score", "with_intercept"])
        self.val_res = val_res.copy()
        val_res = val_res.groupby(["order", "seasonal", "with_intercept"]).agg(["count", "mean"])
        val_res.columns = val_res.columns.get_level_values(1)
        val_res = val_res.groupby("count")["mean"].idxmin()
        best_order, best_seasonal, best_with_intercept = val_res.reset_index().set_index("mean").idxmax()[0]
        self.best = ARIMA(
            order=best_order,
            seasonal_order=best_seasonal,
            sp=self.season_size,
            with_intercept=best_with_intercept,
            suppress_warnings=True)

        if self.ff:
            self.best.fit(
                premium[:-self.out_of_sample_size-self.test_size],
                X=self.ff.fit_transform(premium[:-self.out_of_sample_size-self.test_size])[1]
            )
        else:
            self.best.fit(premium[:-self.out_of_sample_size-self.test_size])
        
        return self.best

    def save(self, save_dir):
        if self.best is None:
            raise Exception(f"There is no best model for ({self.market_name}, {self.factor_name})")
        fname_parts = [
            self.market_name.replace(' ', '_'),
            self.factor_name.replace(' ', '_'),
            f"{round(datetime.now().timestamp())}"
        ]
        fname = "-".join(fname_parts)
        full_path = os.path.join(save_dir, fname)
        with open(full_path, "wb") as f:
            pkl.dump(self, f)
        print(f"Model saved in {full_path}")      

if __name__ == "__main__":
    premium_df = get_latest_factor_premium()
    premium_df_gp = premium_df.groupby(["group", "factor"])[["period_end", "premium"]]

    i = 1

    for name, df in premium_df_gp:
        try:
            model = MarketFactorARIMA(name[0], name[1])
            model.train(df["period_end"], df["premium"])
            model.save("outputs/models")
        except Exception as e:
            print(f"{name[0]}-{name[1]}:", e)
    