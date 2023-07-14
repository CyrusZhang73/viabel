import pickle

import jax.numpy as anp
import numpy as np
import bridgestan as bs
import pytest
from jax.scipy.stats import norm
from jax.test_util import check_vjp, check_grads
from functools import partial
from jax import vjp
from viabel import models


def _test_model(m, x, supports_tempering, supports_constrain):
    m_vjp = partial(vjp, m)
    check_vjp(m, m_vjp, (x,))
    check_grads(m, (x,), order=1, modes=("rev"))
    check_vjp(m, x[0])
    assert supports_tempering == m.supports_tempering
    if supports_tempering:  # pragma: no cover
        m.set_inverse_temperature(.5)
    else:
        with pytest.raises(NotImplementedError):
            m.set_inverse_temperature(.5)
    if supports_constrain:
        m.constrain(x[0]) == supports_constrain
    else:
        with pytest.raises(NotImplementedError):
            m.constrain(x[0])


test_model = """data {
  int<lower=0> N;   // number of observations
  matrix[N, 2] x;   // predictor matrix
  vector[N] y;      // outcome vector
  real<lower=1> df; // degrees of freedom
}

parameters {
  vector[2] beta;       // coefficients for predictors
}

model {
  beta ~ normal(0, 10);
  y ~ student_t(df, x * beta, 1);  // likelihood
}"""


def test_Model():
    mean = np.array([1., -1.])[np.newaxis, :]
    stdev = np.array([2., 5.])[np.newaxis, :]

    def log_p(x):
        return anp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    model = models.Model(log_p)
    x = 4 * np.random.randn(10, 2)
    _test_model(model, x, False, False)


def test_StanModel():
    compiled_model_file = 'robust_reg_model.pkl'
    try:
        with open(compiled_model_file, 'rb') as f:
            regression_model = pickle.load(f)
    except BaseException:  # pragma: no cover
        regression_model = pystan.StanModel(model_code=test_model,
                                            model_name='regression_model')
        with open('robust_reg_model.pkl', 'wb') as f:
            pickle.dump(regression_model, f)
    np.random.seed(5039)
    stan = "gaussian.stan"
    data = "gaussian.data.json"
    fit = bs.StanModel.from_stan_file(stan, data)
    model = models.StanModel(fit)
    x = np.random.random(model.param_unc_num())
    _,grad_expected = fit.log_density_gradient(x)
    _, vjpfun = vjp(model, x)
    grad = vjpfun(1.0)
    grad_actual = np.asarray(grad[0], dtype=np.float32)
    return np.testing.assert_allclose(grad_actual, grad_expected)
    #_test_model(model, x, False, dict(beta=x[0]))
