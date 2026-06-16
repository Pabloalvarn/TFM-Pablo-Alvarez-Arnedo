import numpy as np


def naive_forecast(train, horizon=1):

    last_value = train.iloc[-1]

    return np.repeat(
        last_value,
        horizon
    )


def seasonal_naive_forecast(
    train,
    season_length=4,
    horizon=1
):

    seasonal_value = train.iloc[-season_length]

    return np.repeat(
        seasonal_value,
        horizon
    )


def moving_average_forecast(
    train,
    window=4,
    horizon=1
):

    mean_value = train.iloc[-window:].mean()

    return np.repeat(
        mean_value,
        horizon
    )