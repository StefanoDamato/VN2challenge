import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.model.deepar.module import DeepARModel
from gluonts.torch.distributions import NegativeBinomialOutput


from utils import import_raw_data, make_dataloaders, EarlyStopper
from dlinear import DLinearModel
from tweediegp.intermittent_gp import intermittentGP

# compute the quantile loss
def quantile_loss(forecast, actual, quantile):
    return 2*(actual - forecast)*(quantile - (actual < forecast))

def validate_ql_DeepAR(context_length, num_layers, hidden_size, scaling, emb_dim, emb_idx, rm_spikes):

    q = 5/6
    h = 3
    raw_datasets = import_raw_data()

    ql_simple, ql_cumulative = [], []
    for T in [153, 154, 155, 156, 157]:

        loaders = make_dataloaders(raw_datasets, T, h, rm_spikes=rm_spikes)
        train_loader, validation_loader, test_loader, _ = loaders

        model = DeepARModel(
            freq="W",
            prediction_length=h,
            context_length=context_length,
            num_feat_dynamic_real=1,
            num_feat_static_cat=len(emb_idx),
            num_feat_static_real=1,
            cardinality = [[111, 47, 26, 6, 3, 3, 67, 297][i] for i in emb_idx],#[master[col].nunique() for col in master.columns],
            embedding_dimension = [(5, 4, 3, 2, 1, 1, 4, 10)[i]*emb_dim for i in emb_idx], #[(min(50, (cat_card+1)//2)) for cat_card in cardinality],
            num_layers = num_layers,
            hidden_size= hidden_size,
            dropout_rate = 0.2,
            distr_output=NegativeBinomialOutput(),
            scaling=scaling,
            lags_seq=get_lags_for_frequency("W", lag_ub=T-context_length-3*h),
            num_parallel_samples = 200
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        early_stopper = EarlyStopper(patience=15)

        for epoch in tqdm(range(150)):

            train_loss, validation_loss = 0., 0.

            model.train()
            for i, batch in enumerate(train_loader):
                
                optimizer.zero_grad()

                loss = model.loss(
                    feat_static_cat = batch[FieldName.FEAT_STATIC_CAT][:,emb_idx],
                    feat_static_real = torch.zeros([len(batch["past_"+FieldName.FEAT_DYNAMIC_REAL]), 1]),
                    past_time_feat = batch["past_"+FieldName.FEAT_DYNAMIC_REAL],
                    past_target = batch["past_"+FieldName.TARGET],
                    past_observed_values = batch["past_"+FieldName.OBSERVED_VALUES],
                    future_time_feat = batch["future_"+FieldName.FEAT_DYNAMIC_REAL],
                    future_target = batch["future_"+FieldName.TARGET],
                    future_observed_values = batch["future_"+FieldName.OBSERVED_VALUES]
                    ).mean()
                
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            train_loss = train_loss/(i+1)

            for i, batch in enumerate(validation_loader):

                loss = model.loss(
                    feat_static_cat = batch[FieldName.FEAT_STATIC_CAT][:,emb_idx],
                    feat_static_real = torch.zeros([len(batch["past_"+FieldName.FEAT_DYNAMIC_REAL]), 1]),
                    past_time_feat = batch["past_"+FieldName.FEAT_DYNAMIC_REAL],
                    past_target = batch["past_"+FieldName.TARGET],
                    past_observed_values = batch["past_"+FieldName.OBSERVED_VALUES],
                    future_time_feat = batch["future_"+FieldName.FEAT_DYNAMIC_REAL],
                    future_target = batch["future_"+FieldName.TARGET],
                    future_observed_values = batch["future_"+FieldName.OBSERVED_VALUES],
                    ).mean()
                
                validation_loss += loss.item()

            validation_loss = validation_loss/(i+1)

            if early_stopper(validation_loss, model):
                model = early_stopper.best_model
                break

        forecast_samples, actuals = [], []

        model.eval()
        for batch in test_loader:

            pred = model.forward(
                feat_static_cat = batch[FieldName.FEAT_STATIC_CAT][:,emb_idx],
                feat_static_real = torch.zeros([len(batch["past_"+FieldName.FEAT_DYNAMIC_REAL]), 1]),
                past_time_feat = batch["past_"+FieldName.FEAT_DYNAMIC_REAL],
                past_target = batch["past_"+FieldName.TARGET],
                past_observed_values = batch["past_"+FieldName.OBSERVED_VALUES],
                future_time_feat = batch["future_"+FieldName.FEAT_DYNAMIC_REAL]
                )
            
            actuals.append(batch["future_"+FieldName.TARGET].detach().numpy())
            forecast_samples.append(pred.detach().numpy())

        forecast_samples, actuals = np.vstack(forecast_samples), np.vstack(actuals)

        quantile_loss_simple = quantile_loss(np.quantile(forecast_samples, q, axis=1), actuals, q)
        quantile_loss_cumulative = quantile_loss(np.quantile(np.sum(forecast_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q)

        ql_simple.append(np.mean(quantile_loss_simple))
        ql_cumulative.append(np.mean(quantile_loss_cumulative))
        
    return np.mean(ql_simple), np.mean(ql_cumulative)


def validate_ql_intermittentGP(model_params, rm_unobserved, rm_spikes):

    q = 5/6
    h = 3
    raw_datasets = import_raw_data()

    non_spikes = torch.tensor((raw_datasets[0].mean() <= 4).values)

    ql_simple, ql_cumulative = [], []

    for T in [153, 154, 155, 156, 157]:

        loaders = make_dataloaders(raw_datasets, T, h, batch_size=1)
        _, _, test_loader, _ = loaders
        
        actuals, forecast_samples = [], []

        for ts in tqdm(test_loader):

            train_y, test_y = ts['past_target'].squeeze(), ts['future_target'].squeeze()
            observed = ts['past_observed_values'].squeeze()
            x = torch.arange(len(train_y) + len(test_y))/52
            train_x, test_x = x[:len(train_y)], x[len(train_y):]

            slicing_idx = torch.tensor(True).repeat(len(train_y))
            if rm_spikes:
                slicing_idx = torch.logical_and(slicing_idx, non_spikes[:len(train_y)])
            if rm_unobserved:
                slicing_idx = torch.logical_and(slicing_idx, observed)
            train_x, train_y = train_x[slicing_idx], train_y[slicing_idx]

            likelihood = model_params["likelihood"]
            kernel = model_params["kernel"]
            scaling = model_params["scaling"]
            num_inducing_points = min(model_params["num_inducing_points"], len(train_x)) if model_params["num_inducing_points"] is not None else None
            min_iter = model_params["min_iter"]
            max_iter = model_params["max_iter"]

            counter = 0
            while counter >= 0 and counter < 20:
                counter += 1
                try:
                    model = intermittentGP(likelihood=likelihood, kernel=kernel, scaling=scaling, min_iter=min_iter, max_iter= max_iter, num_inducing_points=num_inducing_points)
                    model.build(train_x, train_y)
                    model.fit(train_x, train_y)
                    _, samples = model.predict(test_x)
                    counter = -1
                except:
                    pass
            if counter == 20:
                return np.nan, np.nan

            actuals.append(test_y.numpy())
            forecast_samples.append(samples.unsqueeze(0).detach().numpy())
    
        forecast_samples, actuals = np.vstack(forecast_samples), np.vstack(actuals)

        quantile_loss_simple = quantile_loss(np.quantile(forecast_samples, q, axis=1), actuals, q)
        quantile_loss_cumulative = quantile_loss(np.quantile(np.sum(forecast_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q)

        ql_simple.append(np.mean(quantile_loss_simple))
        ql_cumulative.append(np.mean(quantile_loss_cumulative))

    return np.mean(ql_simple), np.mean(ql_cumulative)


def validate_ql_DLinear(context_length, hidden_dimension, kernel_size, scaling, time_feat, emb_dim, emb_idx, rm_spikes):

    q = 5/6
    h = 3
    raw_datasets = import_raw_data()

    ql_simple, ql_cumulative = [], []

    for T in [153, 154, 155, 156, 157]:

        loaders = make_dataloaders(raw_datasets, T, h, rm_spikes=rm_spikes)
        train_loader, validation_loader, test_loader, _ = loaders

        model = DLinearModel(
            freq="W",
            prediction_length=h,
            context_length=context_length,
            num_feat_dynamic_real=1,
            num_feat_static_cat=len(emb_idx),
            num_feat_static_real=1,
            cardinality = [[111, 47, 26, 6, 3, 3, 67, 297][i] for i in emb_idx],
            embedding_dimension = [(5, 4, 3, 2, 1, 1, 4, 10)[i]*emb_dim for i in emb_idx], 
            hidden_dimension = hidden_dimension,
            kernel_size = kernel_size,
            dropout_rate = 0.2,
            distr_output=NegativeBinomialOutput(),
            scaling=scaling
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        early_stopper = EarlyStopper(patience=15)

        for epoch in tqdm(range(150)):

            train_loss, validation_loss = 0., 0.

            model.train()
            for i, batch in enumerate(train_loader):
                
                optimizer.zero_grad()

                loss = model.loss(
                    feat_static_cat = batch[FieldName.FEAT_STATIC_CAT][:,emb_idx],
                    feat_static_real = torch.zeros([len(batch["past_"+FieldName.FEAT_DYNAMIC_REAL]), 1]),
                    past_time_feat = batch["past_"+FieldName.FEAT_DYNAMIC_REAL] if time_feat else torch.zeros(batch["past_"+FieldName.FEAT_DYNAMIC_REAL].shape),
                    past_target = batch["past_"+FieldName.TARGET],
                    past_observed_values = batch["past_"+FieldName.OBSERVED_VALUES],
                    future_time_feat = batch["future_"+FieldName.FEAT_DYNAMIC_REAL] if time_feat else torch.zeros(batch["future_"+FieldName.FEAT_DYNAMIC_REAL].shape),
                    future_target = batch["future_"+FieldName.TARGET],
                    future_observed_values = batch["future_"+FieldName.OBSERVED_VALUES]
                    ).mean()
                
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss = train_loss/(i+1)

            for i, batch in enumerate(validation_loader):

                loss = model.loss(
                    feat_static_cat = batch[FieldName.FEAT_STATIC_CAT][:,emb_idx],
                    feat_static_real = torch.zeros([len(batch["past_"+FieldName.FEAT_DYNAMIC_REAL]), 1]),
                    past_time_feat = batch["past_"+FieldName.FEAT_DYNAMIC_REAL] if time_feat else torch.zeros(batch["past_"+FieldName.FEAT_DYNAMIC_REAL].shape),
                    past_target = batch["past_"+FieldName.TARGET],
                    past_observed_values = batch["past_"+FieldName.OBSERVED_VALUES],
                    future_time_feat = batch["future_"+FieldName.FEAT_DYNAMIC_REAL] if time_feat else torch.zeros(batch["future_"+FieldName.FEAT_DYNAMIC_REAL].shape),
                    future_target = batch["future_"+FieldName.TARGET],
                    future_observed_values = batch["future_"+FieldName.OBSERVED_VALUES],
                    ).mean()
                
                validation_loss += loss.item()

            validation_loss = validation_loss/(i+1)

            if early_stopper(validation_loss, model):
                model = early_stopper.best_model
                break

        forecast_samples, actuals = [], []

        model.eval()
        for batch in test_loader:

            distr_args, loc, scale = model.forward(
                feat_static_cat = batch[FieldName.FEAT_STATIC_CAT][:,emb_idx],
                feat_static_real = torch.zeros([len(batch["past_"+FieldName.FEAT_DYNAMIC_REAL]), 1]),
                past_time_feat = batch["past_"+FieldName.FEAT_DYNAMIC_REAL] if time_feat else torch.zeros(batch["past_"+FieldName.FEAT_DYNAMIC_REAL].shape),
                past_target = batch["past_"+FieldName.TARGET],
                past_observed_values = batch["past_"+FieldName.OBSERVED_VALUES],
                future_time_feat = batch["future_"+FieldName.FEAT_DYNAMIC_REAL] if time_feat else torch.zeros(batch["future_"+FieldName.FEAT_DYNAMIC_REAL].shape)
                )
            
            distribution = model.distr_output.distribution(distr_args, loc=loc, scale=scale)
            pred = distribution.sample(torch.Size([10000])).swapaxes(0, 1)
            
            actuals.append(batch["future_"+FieldName.TARGET].detach().numpy())
            forecast_samples.append(pred.detach().numpy())

        forecast_samples, actuals = np.vstack(forecast_samples), np.vstack(actuals)

        quantile_loss_simple = quantile_loss(np.quantile(forecast_samples, q, axis=1), actuals, q)
        quantile_loss_cumulative = quantile_loss(np.quantile(np.sum(forecast_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q)

        ql_simple.append(np.mean(quantile_loss_simple))
        ql_cumulative.append(np.mean(quantile_loss_cumulative))
        
    return np.mean(ql_simple), np.mean(ql_cumulative)


def validate_smooth_ts(datasets, model_fn):

    # set hyperparams of the experiment
    quant = [0.5, 0.6, 0.7, 0.8, 5/6]
    h = 3

    # iterate across differnt time horizons
    validation_results = {}
    for T in [datasets[0].shape[1] - k for k in range(4, -1, -1)]:

        # build the iterator forr running the model
        loaders = make_dataloaders(datasets, T, h, batch_size=1)
        _, _, test_loader, _ = loaders
        actuals, forecast_samples = [], []

        # iterate across t.s.
        for ts in tqdm(test_loader):

            # import the past of the t.s. (y) and the time input (x) if required
            train_y, test_y = ts['past_target'].squeeze(), ts['future_target'].squeeze()
            observed = ts['past_observed_values'].squeeze()

            # skip non-smooth t.s.
            if not torch.all(train_y[observed] > 0):
                continue
            samples = model_fn(train_y, observed, h)

            # convert samples as a (#samples x h) array or tensor
            samples = model_fn(train_y, observed, h)
            if samples.ndim == 2:
                if isinstance(samples, torch.Tensor):
                    samples = samples.unsqueeze(0).detach().numpy()
                elif isinstance(samples, np.ndarray):
                    samples = samples[None, :]
                else:
                    raise TypeError("Invalid format (required torch.tensor or np.ndarray)")
            else:
                raise TypeError("Invalid dimension (reqired 2-dimensional item)")

            # store actual values and forecast samples
            actuals.append(test_y.numpy())
            forecast_samples.append(samples)
    
        # compute the quantile loss
        forecast_samples, actuals = np.vstack(forecast_samples), np.vstack(actuals)
        quantile_loss_simple = {
            q:quantile_loss(np.quantile(forecast_samples, q, axis=1), actuals, q).mean() for q in quant
        }
        quantile_loss_cumulative = {
            q:quantile_loss(np.quantile(np.sum(forecast_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q).mean() for q in quant
        }
        
        # save the results into a dictionary
        window_results = {
            "N_smooth":len(actuals),
            "actuals":actuals,
            "forecast_samples":forecast_samples,
            "quantile_loss_simple":quantile_loss_simple,
            "quantile_loss_cumulative":quantile_loss_cumulative,
            "quantile_loss_cumulative":quantile_loss_cumulative
        }
        validation_results[T] = window_results
    return validation_results

# validate any local model
def validate_local_fn(datasets, model_fn):

    # set hyperparams of the experiment
    quant = [.5, .6, .7, .8, 5/6]
    h = 3

    # iterate across differnt time horizons
    validation_results = {}
    for T in [datasets[0].shape[1] - k for k in range(4, -1, -1)]:

        # build the iterator forr running the model
        loaders = make_dataloaders(datasets, T, h, batch_size=1)
        _, _, test_loader, _ = loaders
        actuals, forecast_samples = [], []

        # iterate across t.s.
        for ts in tqdm(test_loader):

            # import the past of the t.s. (y) and make forecasts
            train_y, test_y = ts['past_target'].squeeze(), ts['future_target'].squeeze()
            observed = ts['past_observed_values'].squeeze()
            samples = model_fn(train_y, observed, h)

            # convert samples as a (#samples x h) array or tensor
            if samples.ndim == 2:
                if isinstance(samples, torch.Tensor):
                    samples = samples.unsqueeze(0).detach().numpy()
                elif isinstance(samples, np.ndarray):
                    samples = samples[None, :]
                else:
                    raise TypeError("Invalid format (required torch.tensor or np.ndarray)")
            else:
                raise TypeError("Invalid dimension (reqired 2-dimensional item)")

            # store actual values and forecast samples
            actuals.append(test_y.numpy())
            forecast_samples.append(samples)
    
        # compute the quantile loss
        forecast_samples, actuals = np.vstack(forecast_samples), np.vstack(actuals)
        quantile_loss_simple = {
            q:quantile_loss(np.quantile(forecast_samples, q, axis=1), actuals, q).mean() for q in quant
        }
        quantile_loss_cumulative = {
            q:quantile_loss(np.quantile(np.sum(forecast_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q).mean() for q in quant
        }
        quantile_loss(np.quantile(np.sum(forecast_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q).mean()


        # save the results into a dictionary
        window_results = {
            "forecast_samples":forecast_samples,
            "actuals":actuals,
            "quantile_loss_simple":quantile_loss_simple,
            "quantile_loss_cumulative":quantile_loss_cumulative
        }
        validation_results[T] = window_results
    
    return validation_results





