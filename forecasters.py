import pandas as pd
import torch
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tweediegp.intermittent_gp import intermittentGP

# the generic method for a gaussian process
def gp(train_y, train_x, test_x, likelihood="negbin", kernel=None, num_inducing_points=None):

    # fix parameters to pass to the model
    if num_inducing_points is not None:
        if num_inducing_points >= len(train_y):
            num_inducing_points=None
    scaling = None if likelihood=="negbin" else "mean"

    # instanciate the model
    model = intermittentGP(
        likelihood=likelihood,
        kernel=kernel,
        scaling=scaling,
        num_inducing_points=num_inducing_points,
        min_iter=50,
        max_iter=100,
        n_samples=20000
    )

    # try to fit and predict with it, otherwise report failure
    for i in range(5):
        try:
            model.build(train_x, train_y)
            model.fit(train_x, train_y)
            _, samples = model.predict(test_x)
            return torch.round(samples).clip(min=0)
        except:
            pass
    raise ValueError()


# this function runs an ets
def ets(train_y, observed, h):
    train_y = torch.where(observed, train_y, torch.nan).detach().numpy()
    dates = pd.date_range(pd.Timestamp('2021-04-12'), periods=len(train_y), freq="W")
    train_y = pd.Series(train_y, index=dates).interpolate(method='time')
    try:
        ets = ExponentialSmoothing(train_y, seasonal_periods=52)
    except:
        ets = ExponentialSmoothing(train_y)
    fitted = ets.fit(optimized=True)
    forecast = fitted.forecast(h)
    sigma = np.std(fitted.resid) 
    samples = torch.tensor([forecast.values + np.random.normal(0, sigma, size=h) for _ in range(5000)])
    return samples


# this function returns values from the empirical distribution
def empirical(train_y, observed, h):
    train_y = train_y[observed]
    return train_y[torch.randint(0, len(train_y)), (5000, h)]

# this function is our baeline model
def baseline(train_y, observed, h):
    if torch.all(train_y[observed]) > 0.:
        return ets(train_y, observed, h)
    else:
        return empirical(train_y, observed, h)
