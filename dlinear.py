
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
    NegativeBinomialOutput,
)
from gluonts.torch.scaler import Scaler, MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.model.simple_feedforward import make_linear_layer
from gluonts.torch.util import (
    lagged_sequence_values,
    repeat_along_dim,
    take_last,
    unsqueeze_expand,
    weighted_average
)
from gluonts.itertools import prod
from gluonts.model import Input, InputSpec

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, ...].repeat(1, (self.kernel_size) // 2, 1)
        end = x[:, -1:, ...].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearModel(nn.Module):
    """
    Module implementing a feed-forward model form the paper
    https://arxiv.org/pdf/2205.13504.pdf extended for probabilistic
    forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dimension
        Size of last hidden layers in the feed-forward network.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: Optional[List[int]] = None,
        hidden_dimension: int = 40,
        kernel_size: int = 25,
        dropout_rate: float = 0.1,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        # assert num_feat_dynamic_real > 0
        # assert num_feat_static_real > 0
        # assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert (
            embedding_dimension is None
            or len(embedding_dimension) == num_feat_static_cat
        )

        self.prediction_length = prediction_length
        self.context_length = context_length

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        
        self.hidden_dimension = hidden_dimension
        self.decomposition = SeriesDecomp(kernel_size)

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.linear_seasonal = make_linear_layer(
            context_length, prediction_length * hidden_dimension
        )
        self.linear_trend = make_linear_layer(
            context_length, prediction_length * hidden_dimension
        )
        self.linear_feat = make_linear_layer(
            self._number_of_features, prediction_length * hidden_dimension,
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.distr_output = distr_output
        self.args_proj = self.distr_output.get_args_proj(hidden_dimension)

    @property
    def _number_of_static_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_static_real
            + 1  # the log(scale)
        )
    
    @property
    def _number_of_time_features(self) -> int:
        return (
            self.num_feat_static_real
            * (self.context_length + self.prediction_length)
        )
    
    @property
    def _number_of_features(self) -> int:
        return (
            self._number_of_static_features
            + self._number_of_time_features
        )


    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat),
                    dtype=torch.long,
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real),
                    dtype=torch.float,
                ),
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self.context_length),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:

        context = past_target[..., -self.context_length :]
        observed_context = past_observed_values[..., -self.context_length :]

        input, loc, scale = self.scaler(context, observed_context)

        # future_length = future_time_feat.shape[-2]
        # if future_length > 1:
        #     assert future_target is not None
        #     input = torch.cat(
        #         (input, future_target[..., : future_length - 1] / scale),
        #         dim=-1,
        #     )

    
        # prior_input = past_target[..., : -self.context_length] / scale

        # lags = lagged_sequence_values(
        #     self.lags_seq, prior_input, input, dim=-1
        # )

        time_feat = torch.cat(
            (
                take_last(past_time_feat, dim=-2, num=self.context_length),
                future_time_feat,
            ),
            dim=-2,
        )
        time_feat = time_feat.view(time_feat.shape[0], -1)

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=-1,
        )

        features = torch.cat((static_feat, time_feat), dim=-1)

        res, trend = self.decomposition(input.unsqueeze(-1))
        seasonal_output = self.linear_seasonal(res.squeeze(-1))
        trend_output = self.linear_trend(trend.squeeze(-1))
        feat_output = self.linear_feat(features)
        nn_out = seasonal_output + trend_output + feat_output
        nn_out = self.dropout(nn_out)

        distr_args = self.args_proj(
            nn_out.reshape(-1, self.prediction_length, self.hidden_dimension)
        )
        return distr_args, loc, scale

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        distr_args, loc, scale = self.forward(
            feat_static_cat = feat_static_cat,
            feat_static_real = feat_static_real,
            past_time_feat = past_time_feat,
            past_target = past_target,
            past_observed_values = past_observed_values,
            future_time_feat = future_time_feat
        )
        loss = self.distr_output.loss(
            target=future_target, distr_args=distr_args, loc=loc, scale=scale
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
    
