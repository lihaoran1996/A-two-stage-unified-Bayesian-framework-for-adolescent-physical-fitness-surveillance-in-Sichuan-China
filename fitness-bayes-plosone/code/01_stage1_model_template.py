"""
Stage 1 Bayesian CFA model template.

This script shows how the Stage 1 Bayesian confirmatory factor analysis (CFA)
model was specified and fitted. The restricted educational administrative data
are not included in this repository.

Authorized users should prepare the required input arrays before running this
script.
"""

import numpy as np
import pymc as pm
import arviz as az


# =============================================================================
# 1. Observed items and latent factors
# =============================================================================

item_cols = [
    "BMI",
    "Vital Capacity",
    "50 m Sprint",
    "Jump Rope",
    "800 m Run",
    "1000 m Run",
    "Shuttle Run (50×8)",
    "Sit-ups",
    "Standing Long Jump",
    "Pull-ups",
    "Sit-and-Reach",
]

factor_names = [
    "Strength",
    "Speed",
    "Endurance",
    "Flexibility",
]


# =============================================================================
# 2. Prior structure for the CFA loading matrix
# =============================================================================

def build_loading_prior(item_cols, factor_names):
    """
    Construct the theory-driven prior mean and standard deviation matrices
    for the CFA loading matrix.

    The prior structure includes:
    1. primary item-factor loadings: Normal(1, 0.2^2);
    2. weak cross-loadings: Normal(0, 0.1^2);
    3. global indicators, BMI and vital capacity: Normal(0, 1^2) on all factors.
    """

    n_items = len(item_cols)
    n_factors = len(factor_names)

    item_index = {item: j for j, item in enumerate(item_cols)}
    factor_index = {factor: k for k, factor in enumerate(factor_names)}

    # Default weak cross-loading prior
    mu_prior = np.zeros((n_items, n_factors))
    sigma_prior = np.ones((n_items, n_factors)) * 0.1

    def set_primary_loading(items, factor_name):
        k = factor_index[factor_name]
        for item in items:
            if item in item_index:
                j = item_index[item]
                mu_prior[j, k] = 1.0
                sigma_prior[j, k] = 0.2

    # Primary loading structure
    set_primary_loading(
        ["Sit-ups", "Standing Long Jump", "Pull-ups"],
        "Strength",
    )

    set_primary_loading(
        ["50 m Sprint", "Jump Rope"],
        "Speed",
    )

    set_primary_loading(
        ["800 m Run", "1000 m Run", "Shuttle Run (50×8)"],
        "Endurance",
    )

    set_primary_loading(
        ["Sit-and-Reach"],
        "Flexibility",
    )

    # BMI and vital capacity are treated as global health-related indicators
    # and are allowed to load freely on all latent factors.
    for item in ["BMI", "Vital Capacity"]:
        if item in item_index:
            j = item_index[item]
            mu_prior[j, :] = 0.0
            sigma_prior[j, :] = 1.0

    return mu_prior, sigma_prior


mu_prior, sigma_prior = build_loading_prior(
    item_cols=item_cols,
    factor_names=factor_names,
)


# =============================================================================
# 3. Required input arrays from restricted data
# =============================================================================

# The following arrays must be prepared from the restricted source data.
# They are intentionally not included in this public repository.

# y_obs:
#     Observed item scores on the original 0--100 scale.
#     Shape: (n_observations,)
#
# cell_idx:
#     Integer index of the city-stage-year cell for each observed score.
#     Shape: (n_observations,)
#
# item_idx:
#     Integer index of the observed test item for each observed score.
#     Shape: (n_observations,)

# Example:
# y_obs = np.array([...], dtype=float)
# cell_idx = np.array([...], dtype=int)
# item_idx = np.array([...], dtype=int)

# In the original analysis:
# n_cells = 21 cities × 5 educational stages × 6 years = 630

# Uncomment and replace the following lines when authorized data are available.
# y_obs = ...
# cell_idx = ...
# item_idx = ...


# =============================================================================
# 4. Basic input checks
# =============================================================================

n_cells = int(np.max(cell_idx)) + 1
n_items = len(item_cols)
n_factors = len(factor_names)

assert mu_prior.shape == (n_items, n_factors)
assert sigma_prior.shape == (n_items, n_factors)
assert len(y_obs) == len(cell_idx) == len(item_idx)


# =============================================================================
# 5. Model coordinates
# =============================================================================

coords = {
    "cell": np.arange(n_cells),
    "item": item_cols,
    "factor": factor_names,
}


# =============================================================================
# 6. Stage 1 Bayesian CFA model
# =============================================================================

with pm.Model(coords=coords) as stage1_cfa_model:

    # Factor loading matrix
    Lambda = pm.Normal(
        "Lambda",
        mu=mu_prior,
        sigma=sigma_prior,
        dims=("item", "factor"),
    )

    # Latent factor scores for city-stage-year cells
    F = pm.Normal(
        "F",
        mu=0.0,
        sigma=1.0,
        dims=("cell", "factor"),
    )

    # Item-specific intercepts
    alpha = pm.Normal(
        "alpha",
        mu=70.0,
        sigma=20.0,
        dims="item",
    )

    # Item-specific residual standard deviations
    sigma_y = pm.HalfNormal(
        "sigma_y",
        sigma=10.0,
        dims="item",
    )

    # Observation model
    F_obs = F[cell_idx]
    Lambda_obs = Lambda[item_idx]

    mu_obs = alpha[item_idx] + pm.math.sum(F_obs * Lambda_obs, axis=1)

    y_like = pm.Normal(
        "y_like",
        mu=mu_obs,
        sigma=sigma_y[item_idx],
        observed=y_obs,
    )

    # Posterior sampling
    stage1_trace = pm.sample(
        draws=4000,
        tune=4000,
        chains=4,
        cores=4,
        target_accept=0.90,
        random_seed=42,
        return_inferencedata=True,
    )


# =============================================================================
# 7. Optional posterior predictive sampling
# =============================================================================

with stage1_cfa_model:
    stage1_trace = pm.sample_posterior_predictive(
        stage1_trace,
        var_names=["y_like"],
        extend_inferencedata=True,
        random_seed=42,
    )


# =============================================================================
# 8. Optional save
# =============================================================================

# The full trace may contain observed data, posterior predictive draws, and
# cell-level latent factor scores. It should be reviewed and cleaned before
# any public release if the source data are restricted.

# stage1_trace.to_netcdf("cfa_step1_trace.nc")