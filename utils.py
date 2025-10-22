import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader, InferenceDataLoader
from gluonts.torch.batchify import batchify
from gluonts.transform import AddAgeFeature, Chain, InstanceSplitter
from gluonts.transform.sampler import ExpectedNumInstanceSampler, PredictionSplitSampler


def optimal_forecast(forecast_samples: np.ndarray, s0: int, w1: int, w2: int):
    """
    Compute the optimal forecast simulating the inventory policy of the competition.

    Args:
        forecast_samples (np.ndarray):
            Forecast sample paths with shape Nx3 (number of samples x steps-ahead).
        s0 (int):
            The inventory available from the beginning ("End Inventory").
        w1 (int):
            The stock in transit for the next week ("In Transit W+1").
        w2 (int):
            The stock in transit for two weeks ahead ("In Transit W+2").

    Returns:
    np.ndarray
        The array with the forecast that minimizes the loss.
    """
    
    # compute the stock at time 1 (end of week)
    s1 = (s0 + w1 - forecast_samples[:,0]).clip(min=0)

    # compute the stock at time 2 (end of week)
    s2 = (s1 + w2 - forecast_samples[:,1]).clip(min=0)

    # compute how much stock is needed to match the simulated demand
    stock_shortage = (forecast_samples[:,2] - s2).clip(min=0)

    # return the quantile 
    return np.quantile(stock_shortage, 5/6)


# save the forecasts into a csv
def save_forecasts(forecast_samples: np.ndarray):

    # import the indices and use them in a pandas df
    sales, _, _ = import_raw_data()
    forecast_df = pd.DataFrame(forecast_samples, index=sales.index).astype(int)
    forecast_df.to_csv("order " + datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".csv")


# import the datasets as pandas dataframes
def import_raw_data(path = os.path.join(os.getcwd(), "data"), week=None, code_master = True):
    
    # import th week of interest
    if week is None:
        pattern = re.compile(r"Week (\d+).*Sales\.csv$")
        salesfile = max([file for file in os.listdir(path) if file.endswith("Sales.csv")], 
                        key=lambda x: int(pattern.search(x).group(1)))
    else:
        salesfile = next(file for file in os.listdir(path) 
                         if file.startswith(f"Week {week}") and file.endswith("Sales.csv"))

    # sales contains the time series
    sales = pd.read_csv(os.path.join(path, salesfile))

    # sales contains the time series
    sales = pd.read_csv(os.path.join(path, salesfile))
    sales.set_index(['Store', "Product"], inplace=True)

    # in_stock has a mask for observed values
    in_stock = pd.read_csv(os.path.join(path, 'Week 0 - In Stock.csv'))
    in_stock.set_index(['Store', "Product"], inplace=True)

    # master has static covariates, which can be re-indexed them from zero
    master = pd.read_csv(os.path.join(os.getcwd(), 'data', 'Week 0 - Master.csv'))
    master.set_index(['Store', "Product"], inplace=True)
    master['Store'] = master.index.get_level_values('Store')
    master['Product'] = master.index.get_level_values('Product')
    master = master.apply(lambda col: pd.Categorical(col).codes) if code_master else master

    # return 3 datasets
    return sales, in_stock, master

# remove large demand spikes (across all datasets)
def remove_spikes(sales, in_stock, threshold = 4.):
    
    # remove all data for times where they globally get over the threshold
    in_stock = in_stock.iloc[:,:sales.shape[1]]
    in_stock.iloc[:,(np.nanmean(np.where(in_stock, sales, np.nan), axis=0) > threshold)] = False
    return in_stock
    

# make gluonts loaders for training models and cross validating
def make_dataloaders(raw_datasets, T, h, batch_size = 32, rm_spikes = False):

    # import the data
    sales, in_stock, master = raw_datasets
    non_spikes = (sales.mean() <= 4)

    # specific slicing informations
    datasets = {'train' : [], 'validation' : [], "test" : [], "competition" : []}
    ts_len = {'train' : T-2*h, 'validation' : T-h, 'test' : T, 'competition' : sales.shape[1]}

    # make the dataset storing each t.s. as a dictionary
    for idx in sales.index:
        for key in datasets.keys():
            datasets[key].append(
                {
                    FieldName.ITEM_ID : idx,
                    FieldName.START : sales.columns[0],
                    FieldName.TARGET : sales.loc[idx].values[:ts_len[key]],
                    FieldName.OBSERVED_VALUES : np.logical_and(
                        in_stock.loc[idx][:ts_len[key]].values, 
                        non_spikes[:ts_len[key]].values) if rm_spikes else in_stock.loc[idx][:ts_len[key]].values,
                    FieldName.FEAT_STATIC_CAT : master.loc[idx].values
                }
            )

    # build each dataset for a list of ts
    for key in datasets.keys():
        datasets[key] = ListDataset(datasets[key], freq="W")

    # store the transformations to apply when sampling
    transforms = {}
    samplers = {
        "train": ExpectedNumInstanceSampler(num_instances=1, min_future=h),
        "validation" : PredictionSplitSampler(min_future=h),
        "test" : PredictionSplitSampler(min_future=h),
        "competition" : PredictionSplitSampler()
    }

    # make the transformation concatenating operations
    for key in datasets.keys():
        transforms[key] = Chain([

            # add a real dynamic feature of log(timestamp)
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_DYNAMIC_REAL,
                pred_length=h,
            ),

            # split the t.s. into past and future values
            InstanceSplitter(
                target_field=FieldName.TARGET,
                past_length=ts_len[key] - (h if key!="competition" else 0),
                future_length=h,
                instance_sampler=samplers[key],
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                is_pad_field=FieldName.IS_PAD,
                time_series_fields=[FieldName.FEAT_DYNAMIC_REAL, FieldName.OBSERVED_VALUES]
            )
        ])

    # make the dataloaders, one for each phase
    train_loader = TrainDataLoader(
        dataset=datasets['train'],
        transform=transforms['train'], 
        batch_size=batch_size,
        num_batches_per_epoch=30,
        stack_fn = batchify
    )
    validation_loader = ValidationDataLoader(
        dataset=datasets['validation'],
        transform=transforms['validation'], 
        batch_size=batch_size,
        stack_fn = batchify
    )
    test_loader = InferenceDataLoader(
        dataset=datasets['test'],
        transform=transforms['test'],  
        batch_size=batch_size,
        stack_fn=batchify
    )
    competition_loader = InferenceDataLoader(
        dataset=datasets['competition'],
        transform=transforms['competition'], 
        batch_size=batch_size,
        stack_fn=batchify
    )

    # return them as a tuple
    return train_loader, validation_loader, test_loader, competition_loader


# create an early stopper class
class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.best_model = None
        self.counter = 0

    def __call__(self, current_loss, current_model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_model = current_model
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
