import pandas as pd
import torch
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import rpy2.robjects.conversion as rconv
ro.r('library(smooth)')
ro.r('library(forecast)')

from reconciliation import temporal_recon_ts
from tweediegp.intermittent_gp import intermittentGP


# the generic method for a Gaussian Process
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


# a GP with negative binomial likeihoo
def negbin_gp(train_y, train_x, test_x):
    return gp(train_y, train_x, test_x, likelihood="negbin")


# GP with gaussian likeilhood and periodic kernel
def exact_periodic_gp(train_y, train_x, test_x):
        return gp(train_y, train_x, test_x, likelihood="gaussian", kernel="periodic")


# GP with gaussian likelihood
def exact_gp(train_y, train_x, test_x):
        return gp(train_y, train_x, test_x, likelihood="gaussian")


# the model to be used on bottom time series 
def bottom_gp(train_y, observed, h):
    if torch.all(train_y[observed] > 0.):
        fn = exact_periodic_gp
    else:
        fn = negbin_gp
    recon_samples, _ = temporal_recon_ts(fn, train_y, observed, 3)
    return torch.tensor(recon_samples[:,:h])


# apply temporal reconciliation on bottom_gp
def negbingp_recon(train_y, observed, h):
    recon_samples, _ = temporal_recon_ts(negbin_gp, train_y, observed, 3)
    return torch.tensor(recon_samples[:,:h])


# apply temporal reconciliation on exact_gp
def exact_gp_recon(train_y, observed, h):
    recon_samples, _ = temporal_recon_ts(exact_gp, train_y, observed, 3)
    return torch.tensor(recon_samples[:,:h])


# apply temporal reconciliation on exact_periodic_gp
def exact_periodic_gp_recon(train_y, observed, h):
    recon_samples, _ = temporal_recon_ts(exact_periodic_gp, train_y, observed, 3)
    return torch.tensor(recon_samples[:,:h])


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
    return train_y[torch.randint(0, len(train_y), (5000, h))]


# this function is our baeline model
def baseline(train_y, observed, h):
    if torch.all(train_y[observed]) > 0.:
        return ets(train_y, observed, h)
    else:
        return empirical(train_y, observed, h)

# run the ADAM model from R
def adam_R(train_y, observed, h):

    # convert the inputs to numpy if required
    if isinstance(train_y, torch.Tensor):
        train_y = train_y.detach().numpy()
    if isinstance(observed, torch.Tensor):
        observed = observed.detach().numpy()
    train_y = train_y[observed]

    # pass them to the R global environment
    converter = rconv.Converter('numpy')
    converter += numpy2ri.converter
    with rconv.localconverter(converter):
        ro.globalenv['train_y'] = ro.FloatVector(train_y)
        ro.globalenv['h'] = ro.IntVector([h])
        
        # run the R code with the model function
        ro.r('''
             formula = ifelse(any(train_y <= 0), "XXN", "ZZN")
             model <- adam(train_y, model=formula)
             pred = forecast(model, h=h, interval="simulated", scenarios=TRUE, nsim=20000)
             samples = pmax(round(t(pred$scenarios)), 0)
            ''')
        return torch.tensor(ro.r('samples'))
    

# apply temporal reconciliation to R ADAM model
def adam_R_recon(train_y, observed, h):

    # define the function and apply it
    def fn(train_y, train_x, test_x):
        return adam_R(train_y, ~torch.isnan(train_x), len(test_x))
    recon_samples, _ = temporal_recon_ts(fn, torch.tensor(train_y), torch.tensor(observed), 3)
    return recon_samples[:,:h] 

# run the ETS model from R
def ets_R(train_y, observed, h):

    # convert the inputs to numpy if required
    if isinstance(train_y, torch.Tensor):
        train_y = train_y.detach().numpy()
    if isinstance(observed, torch.Tensor):
        observed = observed.detach().numpy()
    train_y = train_y[observed]

    # pass them to the R global environment
    converter = rconv.Converter('numpy')
    converter += numpy2ri.converter
    with rconv.localconverter(converter):
        ro.globalenv['train_y'] = ro.FloatVector(train_y)
        ro.globalenv['h'] = ro.IntVector([h])
        
        # run the R code with the model function
        ro.r('''
             formula = ifelse(any(train_y <= 0), "XXN", "ZZN")
             model = es(train_y, model=formula, h=h)
             pred = forecast(model, h=h, nsim = 20000, interval="simulated", scenarios=TRUE)
             samples = pmax(round(t(pred$scenarios)), 0)
            ''') 
        return torch.tensor(ro.r('samples'))
    

# apply temporal reconciliation to R ETS model
def ets_R_recon(train_y, observed, h):
    
    # define the function and apply it
    def fn(train_y, train_x, test_x):
        return ets_R(train_y, ~torch.isnan(train_x), len(test_x))
    recon_samples, _ = temporal_recon_ts(fn, torch.tensor(train_y), torch.tensor(observed), 3)
    return recon_samples[:,:h] 

# run the ARIMA model from R
def arima_R(train_y, observed, h):

    # convert the inputs to numpy if required
    if isinstance(train_y, torch.Tensor):
        train_y = train_y.detach().numpy()
    if isinstance(observed, torch.Tensor):
        observed = observed.detach().numpy()
    train_y = train_y[observed]

    # pass them to the R global environment
    converter = rconv.Converter('numpy')
    converter += numpy2ri.converter
    with rconv.localconverter(converter):
        ro.globalenv['train_y'] = ro.FloatVector(train_y)
        ro.globalenv['h'] = ro.IntVector([h])
        
        # run the R code with the model function
        ro.r('''
             model <- auto.arima(train_y, seasonal=FALSE)
             samples = pmax(round(t(replicate(20000, simulate(model, nsim = h)))), 0)
            ''')
        samples = torch.tensor(ro.r('samples'))
        if samples.shape[0] ==1:
            samples = samples.t()
        return samples


# apply temporal reconciliation to R ARIMA model
def arima_R_recon(train_y, observed, h):
    
    # define the function and apply it
    def fn(train_y, train_x, test_x):
        return arima_R(train_y, ~torch.isnan(train_x), len(test_x))
    recon_samples, _ = temporal_recon_ts(fn, torch.tensor(train_y), torch.tensor(observed), 3)
    return recon_samples[:,:h] 