import numpy as np
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def arima_forecast(train, order=(1, 1, 1), horizon=1):
    """
    Forecast using ARIMA.
    """

    warnings.filterwarnings("ignore")

    model = ARIMA(
        train,
        order=order
    )

    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=horizon)

    return np.array(forecast)


def ets_forecast(train, horizon=1, trend="add", seasonal=None, seasonal_periods=None):
    """
    Forecast using Exponential Smoothing.
    """

    warnings.filterwarnings("ignore")

    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated"
    )

    fitted_model = model.fit(optimized=True)

    forecast = fitted_model.forecast(horizon)

    return np.array(forecast)