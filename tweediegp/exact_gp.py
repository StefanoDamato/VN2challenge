import json
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.kernels import ScaleKernel, RBFKernel, SpectralMixtureKernel, LinearKernel, PeriodicKernel

from .gp_models import GP
from .intermittent_gp import EarlyStopper


class eGP(ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, kernel = None, priors = {}):
        super(eGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(lengthscale_prior=priors.get('lengthscale', None)), 
                                        outputscale_prior=priors.get('outputscale', None))

        assert(kernel in [None, 'sm', 'linear', 'periodic'])
        if kernel == 'sm':
            smk = SpectralMixtureKernel(2)
            try:
                smk.initialize_from_data_empspect(train_x, train_y.to(torch.float32))
            except:
                smk.initialize_from_data(train_x, train_y.to(torch.float32))
            self.covar_module += smk
        elif kernel == 'linear':
            self.covar_module += ScaleKernel(LinearKernel())
        elif kernel == 'periodic':
            pk = ScaleKernel(PeriodicKernel())
            pk.base_kernel._set_period_length(1.)
            pk.base_kernel.raw_period_length.requires_grad = False
            self.covar_module += pk


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class exactGP:
    """
    This class creates a Gaussian Process for forecasting intermittent time series.

    Args:
        kernel (Union(None, str)): the kernel used to induce the covariance matrix of the Gaussian Process.
        scaling (Union(None, str)): the kind of scaling to be applied to the data.
        scale_factor (float): the value the data is scaled to.
        num_inducing_points (Union(None, int)): the amount of inducing points got a sparse GP.
        init_inducing_points (Union(None, str)): the scheme for the initializiation of inucing points.
        n_samples (int): the amount of samples generated in predction.
        max_iter (int): number of maximum training iterations.
        min_iter (int): number of minimum training iterations.
        tolerance (int): the toleramce of the early stopping.
        delta (float): the delta of the eraly stopping.
    """

    def __init__(self, 
                 kernel = None, 
                 scaling = None, 
                 scale_factor = 1., 
                 num_inducing_points = None,
                 init_inducing_points = None, 
                 n_samples = 5e04,
                 max_iter = 100, 
                 min_iter = 25, 
                 tolerance = 1, 
                 delta = 1e-04,
                 priors = False):
        
        assert kernel in [None, 'sm', 'linear', 'periodic']
        assert scaling in [True, False]

        if priors is not False:
            def dict_to_prior(dict):
                class_name = dict['class_name']
                params = {key: torch.tensor(value) for key, value in dict['params'].items()}
                try:
                    prior = getattr(gpytorch.priors, class_name + 'Prior')
                except:
                    prior = globals()[class_name + 'Prior']
                return prior(**params)     
            with open(priors, 'r') as file:
                priors = {name: dict_to_prior(prior_dict) for name, prior_dict in json.load(file).items()}
        else:
            priors = {}

        self.kernel = kernel
        self.scaling = scaling
        self.scale_factor = scale_factor
        self.num_inducing_points = num_inducing_points
        self.init_inducing_points = init_inducing_points
        self.n_samples = n_samples
        self.max_iter = max_iter 
        self.min_iter = min_iter
        self.tolerance = tolerance
        self.delta = delta
        self.priors = priors


    def build(self, train_x, train_y):
        """
        Build the GP saving the likelihood and and the process itself (they are GPyTorch instances).

        Args:
            train_x (Union(None, torch.Tensor): the timestamps of training.
            train_y (Union(None, torch.Tensor)): the trainig values.
        """  

        self._likelihood = GaussianLikelihood()

        if self.num_inducing_points is not None:
            self._gp = GP(train_x, train_y, self.kernel, self.num_inducing_points, self.init_inducing_points, self.priors)
        else:
            self._gp = eGP(train_x, train_y, self._likelihood, self.kernel, self.priors)
        
        self._early_stopper = EarlyStopper(self.tolerance, self.delta)


    def fit(self, train_x, train_y):
        """
        Trains the variational GP, using early stopping.

        Args:
            train_x (torch.Tensor): the timestamps of training.
            train_y (torch.Tensor): the trainig values.
        """

        if self.scaling:
            self._scale = torch.mean(train_y)
        else:
            self._scale = None
        
        if self._scale and (self._scale == 0 or torch.isnan(self._scale)):
            self._scale = None
        
        if self._scale:
            self._scale /= self.scale_factor
            train_y = train_y - self._scale

        self._early_stopper = EarlyStopper(self.tolerance, self.delta)

        if isinstance(self._gp, ApproximateGP):
            mll = gpytorch.mlls.VariationalELBO(self._likelihood, self._gp, num_data=train_y.size(0))
        elif isinstance(self._gp, ExactGP):
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._gp)
        else:
            raise TypeError("GP object must be either ApproximateGP or ExactGP")
        
        optimizer = torch.optim.Adam(mll.parameters(), lr = 0.1)
        self._stop = None

        for iter in range(self.max_iter):
            self._gp.train(), self._likelihood.train(), mll.train()
            optimizer.zero_grad()

            try:
                pred = self._gp(train_x)
                loss = -mll(pred, train_y)
            except Exception as exc:
                self._stop = str(exc.__class__.__name__)
                break

            loss.backward()
            optimizer.step()

            if iter > (self.min_iter) - self._early_stopper.tolerance - 1:
                self._early_stopper.update(loss.item(), self._gp, self._likelihood)
                if self._early_stopper.is_saturated():
                    self._stop = "Variational loss converged"
                    self._gp, self._likelihood = self._early_stopper.best_models
                    break
        
        if self._stop is  None and iter == (self.max_iter)-1:
            self._stop = "Max iterations reached"
        self._iter = iter

    def predict(self, test_x):
        """
        Predicts future values using the appropriate non-gaussian likelihood.

        Args:
            test_x (torch.Tensor): the timestamps of test.

        Reuturns:
            torch.Tensor: the predicted mean.
            torch.tensor: a 2-dimensional tensor of predictive samples.
        """
        
        self._gp.eval(), self._likelihood.eval()
        posterior_gp_test = self._gp(test_x)

        mean_pred = posterior_gp_test.loc.detach()
        samples_pred = self._likelihood(posterior_gp_test.sample(torch.Size([int(self.n_samples)]))).sample()

        if self._scale:
            mean_pred = mean_pred + self._scale 
            samples_pred = samples_pred + self._scale
        
        return mean_pred, samples_pred
