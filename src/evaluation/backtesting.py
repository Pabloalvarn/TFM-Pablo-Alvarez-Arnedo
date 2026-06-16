import pandas as pd


def rolling_window_split(
    data,
    train_size,
    horizon=1,
    step=1
):
    """
    Rolling window backtesting.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset ordenado temporalmente.

    train_size : int
        Tamaño ventana entrenamiento.

    horizon : int
        Horizonte predicción.

    step : int
        Desplazamiento entre iteraciones.

    Yields
    ------
    train_idx, test_idx
    """

    n_samples = len(data)

    for start in range(
        0,
        n_samples - train_size - horizon + 1,
        step
    ):

        train_end = start + train_size
        test_end = train_end + horizon

        train_idx = list(range(start, train_end))
        test_idx = list(range(train_end, test_end))

        yield train_idx, test_idx


def expanding_window_split(
    data,
    initial_train_size,
    horizon=1,
    step=1
):
    """
    Expanding window backtesting.
    """

    n_samples = len(data)

    for train_end in range(
        initial_train_size,
        n_samples - horizon + 1,
        step
    ):

        test_end = train_end + horizon

        train_idx = list(range(0, train_end))
        test_idx = list(range(train_end, test_end))

        yield train_idx, test_idx