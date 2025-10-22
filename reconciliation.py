import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import import_raw_data, make_dataloaders
from validation import quantile_loss
from tweediegp.intermittent_gp import intermittentGP
from bayesreconpy.reconc_buis import reconc_buis

# impute the missing values with GPs
def impute_GP(y, observed, hot_start=True, n_samples=1000, kernel=None):

    # move everything to torch and slice to get GP inputs 
    y, observed, x = torch.tensor(y), torch.tensor(observed), torch.arange(len(y))/52
    if hot_start:
        start = torch.where(observed)[0][0].item()
        x, y, observed = x[start:], y[start:], observed[start:]
    missing_x, observed_x, observed_y = x[~observed], x[observed], y[observed]

    # impute the value for the missing x with a GP
    counter = 0
    while counter < 5:
        counter += 1
        try:
            model = intermittentGP(likelihood="negbin", 
                                            kernel=kernel, 
                                            scaling=None, 
                                            min_iter=50, 
                                            max_iter=100, 
                                            num_inducing_points=min(100, len(observed_y)),
                                            tolerance = 1,
                                            n_samples = n_samples)
            model.build(observed_x, observed_y)
            model.fit(observed_x, observed_y)
            meanpred, samples = model.predict(missing_x)
            break
        except:
            meanpred = torch.full((len(missing_x),), torch.nan)
            samples = torch.full((n_samples, len(missing_x)), torch.nan)

    # move everything back to numpy and prepare the arrays
    y, observed = y.numpy(), observed.numpy()
    meanpred, samples = meanpred.detach().numpy(), samples.detach().numpy()
    imputed_meanpred, imputed_samples = y, np.tile(y, (n_samples,1))

    # fill them with predicted values 
    imputed_meanpred[~observed] = meanpred
    imputed_samples[:,~observed] = samples
    imputed_observed = np.repeat(True, len(observed))

    # in case of hot start prepend some zeros
    if hot_start and start>0:
        imputed_meanpred = np.concatenate((np.zeros(start), imputed_meanpred))
        imputed_samples = np.concatenate((np.zeros((n_samples, start)), imputed_samples), axis=1)
        imputed_observed = np.concatenate((np.repeat(False, start), imputed_observed))

    # the mean is not integer! return it, and samples and observed indices
    return imputed_meanpred, imputed_samples, imputed_observed


# make dataframe with imputed data instead of missing
def make_imputed_data(sales, in_stock, hot_start, n_samples=None):
    
    # initialise stuctures to store values
    sales_imputed, is_imputed = [], []
    samples = {}

    # iterate over t.s. imputing with GPs
    for idx in tqdm(sales.index):
        y = np.array(sales.loc[idx].values)
        observed = np.array(in_stock.loc[idx].values[:sales.shape[1]])

        # save imputed values
        imputed_meanpred, imputed_samples, imputed_observed = impute_GP(y,observed, hot_start, 2 if n_samples is None else n_samples)
        sales_imputed.append(pd.Series(np.round(imputed_meanpred), index=sales.columns, name=idx))
        is_imputed.append(pd.Series(np.logical_and(~observed, imputed_observed), index=sales.columns, name=idx))
        samples[idx] = imputed_samples

    # make dataframes imputed with the mean (with rounding)
    sales_imputed = pd.DataFrame(sales_imputed)
    is_imputed = pd.DataFrame(is_imputed)

    # iterate over samples, making a dataframes for each
    sales_samples = []
    if n_samples is not None:
        for i in range(n_samples):
            imputed_samples = []
            for idx in sales.index:
                imputed_samples.append(pd.Series(samples[idx][i,], index=sales.columns, name=idx))
            sales_samples.append(pd.DataFrame(imputed_samples, index=sales.index))
       
    # return them as 2 dataframes and a list of sample dataframes
    return sales_imputed, is_imputed, sales_samples


# create a hierachy for generating the forecasts
def matrix_from_columns(master, hierarchy=1):

    # select which of the two hierachies to use
    assert type(hierarchy) in [int, list]
    if type(hierarchy) == int:
        if hierarchy == 1:
            columns = ["Product", "ProductGroup", "Division",	"Department",	"DepartmentGroup"]
        elif hierarchy == 2:
            columns = ["Store", "Format"]
    elif type(hierarchy) == list:
        columns = hierarchy

    # iterate through each value of each aggregation level
    A = []
    for col in columns:
        for j in sorted(master[col].unique()):
            idx = master[col] == j
            A.append(np.array(idx.values, np.int32)) 
    return np.array(A)

# build a hierarchy  starting from an aggregation matrix A
def aggregate_data(sales, in_stock, master, A, is_imputed = None, imputed_prop = 0.2):

    # iterate through each value of each aggregation level
    sales_aggr, in_stock_aggr, master_aggr = [], [], []
    for a in A:
        a = a.astype(bool)
        ts = sales[a].sum(axis=0)
        sales_aggr.append(ts)

        # select the aggregated observed values
        obs = in_stock[a].all(axis=0)
        if is_imputed is not None:
            obs = np.logical_or(obs, 
                                np.array((sales[a]*is_imputed[a]).sum(axis=0)) < np.array(imputed_prop*(sales[a]*in_stock[a]).sum(axis=0)))
        
        in_stock_aggr.append(obs)

        # extract master features, when unique across bottoms
        uniques = [master[a][col].unique() for col in master.columns]
        feats = pd.Series([uniques[i].item() if len(uniques[i]) == 1 else -1 for i in range(len(uniques))], index=master.columns)                     
        master_aggr.append(feats)

    # build the dataframes and return them
    sales_aggr = pd.DataFrame(sales_aggr)
    in_stock_aggr = pd.DataFrame(in_stock_aggr)
    master_aggr = pd.DataFrame(master_aggr)
    return sales_aggr, in_stock_aggr, master_aggr




# reconcile the forecasts on the hierachy of the dataset
def valid_sectional_reconciliation(datasets_bottom, datasets_aggr, fn_bottom, fn_aggr, A, prop=.2):

    # define specifics of the experiment
    q = 5/6
    h = 3
    assert A.shape[0] == len(datasets_aggr[0])
    assert A.shape[1] == len(datasets_bottom[0])

    # merge the dataframes (aggregated go fist, bottom second)
    merged_datasets = [
        pd.concat([
            aggr.set_index(aggr.index.to_flat_index()),
            data.set_index(data.index.to_flat_index())
            ], 
            axis=0).reset_index(drop=True)
            for data, aggr in zip(
                datasets_bottom,
                datasets_aggr
                )
    ]

    # iterate across different training sets
    validation_results = {}
    for T in [datasets_bottom[0].shape[1] - k for k in range(4, -1, -1)]:

        # load the data
        loaders = make_dataloaders(merged_datasets, T, h, batch_size=1)
        _, _, test_loader, _ = loaders
        
        # initialize the arrays for storing true values and samples
        actuals, forecast_samples, hierarchical_samples = [], [], []

        # iterate over t.s. of the dataset, using different forecasters
        for i, ts in tqdm(enumerate(test_loader)):
            if i < len(A):
                fn = fn_aggr
            else:
                fn = fn_bottom

            # select the values and provide base forecasts
            train_y, test_y = ts['past_target'].squeeze(), ts['future_target'].squeeze()
            observed = ts['past_observed_values'].squeeze()
            samples = fn(train_y, observed, h)
            
            # include observed values and samples
            hierarchical_samples.append(samples.unsqueeze(0).detach().numpy())
            if i >= len(A):
                actuals.append(test_y.numpy())
                forecast_samples.append(samples.unsqueeze(0).detach().numpy())

        # stack together previously computed values
        forecast_samples, actuals, hierarchical_samples = (
            np.vstack(forecast_samples), 
            np.vstack(actuals), 
            np.vstack(hierarchical_samples)
        )

        # compute the quantile loss
        quantile_loss_simple_base = quantile_loss(np.quantile(forecast_samples, q, axis=1), actuals, q).mean()
        quantile_loss_cumulative_base = quantile_loss(np.quantile(np.sum(forecast_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q).mean()

        # apply BUIS hierarchically
        reconciled_samples = []
        n_base = len(hierarchical_samples)
        for hor in range(h):
            bottom_samples = reconc_buis(A=A, 
                            base_forecasts=[hierarchical_samples[ns,:,hor] for ns in range(n_base)],
                            in_type=["samples"]*n_base,
                            distr='discrete', 
                            num_samples=hierarchical_samples.shape[1])['bottom_reconciled_samples']
            reconciled_samples.append(bottom_samples)

        # aggregate reconciled samples and compute the quantile loss
        reconciled_samples = np.dstack(reconciled_samples)
        quantile_loss_simple_recon = quantile_loss(np.quantile(reconciled_samples, q, axis=1), actuals, q).mean()
        quantile_loss_cumulative_recon = quantile_loss(np.quantile(np.sum(reconciled_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q).mean()
        
        # save the results into a dictionary
        window_results = {
            "forecast_samples":forecast_samples,
            "actuals":actuals,
            "hierarchical_samples":hierarchical_samples,
            "reconciled_samples":reconciled_samples,
            "quantile_loss_simple_base":quantile_loss_simple_base,
            "quantile_loss_simple_recon":quantile_loss_simple_recon,
            "quantile_loss_cumulative_base":quantile_loss_cumulative_base,
            "quantile_loss_cumulative_recon":quantile_loss_cumulative_recon
        }
        validation_results[T] = window_results
    return validation_results


# aggregate the time series, with padding (keeping nans)
def temporal_aggr(x, k, fun):
    pad_len = (-len(x)) % k
    if pad_len > 0:
        pad = torch.full(((-len(x)) % k,), torch.nan)
        x = torch.cat((pad, x))
    x = x.view(-1, k)
    if fun == "mean":
        return torch.mean(x, dim=1)
    elif fun == "sum":
        return torch.sum(x, dim=1)
    else:
        raise ValueError("fun must be either 'mean' or 'sum'")
    

# perform temporal reconciliation on a ts
def temporal_recon_ts(fn, train_y, observed, h_recon):
    
    # instanciate inputs, replacing unobsrved with nans
    x = torch.arange(len(train_y) + h_recon)/52
    train_x, test_x = x[:len(train_y)], x[len(train_y):]
    train_x_full = torch.where(observed, train_x, torch.nan)
    train_y_full = torch.where(observed, train_y, torch.nan)     

    # provide base_forecasts
    base_bottom_samples = fn(
        train_y_full[~torch.isnan(train_y_full)],
        train_x_full[~torch.isnan(train_x_full)], 
        test_x
        ).detach().numpy()
            
    # generate aggregated forecasts
    train_x_aggr = temporal_aggr(train_x_full, h_recon, "mean")
    train_y_aggr = temporal_aggr(train_y_full, h_recon, "sum")
    base_aggr_samples = fn(
        train_y_aggr[~torch.isnan(train_y_aggr)],
        train_x_aggr[~torch.isnan(train_x_aggr)], 
        temporal_aggr(test_x, h_recon, "mean")
    ).detach().numpy()

    # build the hierarchy (aggregated level 1st) and reconcile it
    hier_samples = np.concat((base_aggr_samples, base_bottom_samples), axis=1)
    recon_bottom_samples = reconc_buis(A=np.ones((1,h_recon)),
                                       base_forecasts=[hier_samples[:,j] for j in range(1+h_recon)],
                                       in_type=['samples']*(1+h_recon),
                                       distr='discrete', 
                                       num_samples=hier_samples.shape[0])['bottom_reconciled_samples'].T
    return recon_bottom_samples, base_bottom_samples

# reconcile on the temporal leve (with a single aggregation levl)
def valid_temporal_reconciliation(datasets, fn, h_recon):

    # define specifics of the experiment
    q = 5/6
    h = 3
    assert h_recon >= h

    # iterate across different training sets
    validation_results = {}
    for T in [datasets[0].shape[1] - k for k in range(4, -1, -1)]:

        # load the data
        loaders = make_dataloaders(datasets, T, h, batch_size=1)
        _, _, test_loader, _ = loaders
        
        # initialize the arrays for storing true values and samples
        actuals, base_samples, recon_samples = [], [], []

        # iterate over t.s. of the dataset
        for ts in tqdm(test_loader):

            # take the values and reconcile them
            train_y, test_y = ts['past_target'].squeeze(), ts['future_target']
            observed = ts['past_observed_values'].squeeze()
            recon_bottom_samples, base_bottom_samples = temporal_recon_ts(fn, train_y, observed, h_recon)
            
            # include observed values and samples to lists
            actuals.append(test_y.numpy())
            base_samples.append(base_bottom_samples[None,:,:h])
            recon_samples.append(recon_bottom_samples[None,:,:h])

        # stack together previously computed values
        actuals, base_samples, recon_samples = (
            np.vstack(actuals), 
            np.vstack(base_samples),
            np.vstack(recon_samples)
        )

        # compute the quantile loss
        quantile_loss_simple_base = quantile_loss(np.quantile(base_samples, q, axis=1), actuals, q).mean()
        quantile_loss_cumulative_base = quantile_loss(np.quantile(np.sum(base_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q).mean()
        quantile_loss_simple_recon = quantile_loss(np.quantile(recon_samples, q, axis=1), actuals, q).mean()
        quantile_loss_cumulative_recon = quantile_loss(np.quantile(np.sum(recon_samples, axis=2), q, axis=1), np.sum(actuals, axis=1), q).mean()
        
        # save the results into a dictionary
        window_results = {
            "actuals":actuals,
            "base_samples":base_samples,
            "recon_samples":recon_samples,
            "quantile_loss_simple_base":quantile_loss_simple_base,
            "quantile_loss_simple_recon":quantile_loss_simple_recon,
            "quantile_loss_cumulative_base":quantile_loss_cumulative_base,
            "quantile_loss_cumulative_recon":quantile_loss_cumulative_recon
        }
        validation_results[T] = window_results

    return validation_results