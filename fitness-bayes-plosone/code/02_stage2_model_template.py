# =============================================================================
# Step 2: Bayesian spatiotemporal hierarchical model
# =============================================================================

import numpy as np
import pymc as pm
import arviz as az


# =============================================================================
# 1. Required input arrays from restricted data
# =============================================================================

# The following arrays must be prepared from the restricted source data.
# They are intentionally not included in this public repository.

# y_by_factor:
#     Dictionary of aggregated factor-level outcomes.
#     Each entry contains the city-stage-year cell-level factor scores for one
#     latent fitness factor.
#
#     Expected keys:
#         "strength", "speed", "endurance", "flexibility"
#
#     Example:
#         y_by_factor["strength"]
#
#     Shape for each factor:
#         (n_observations,)
#
# city_idx:
#     Integer city index for each city-stage-year observation.
#     Values must range from 0 to n_cities - 1.
#     Shape: (n_observations,)
#
# year_idx:
#     Integer year index for each city-stage-year observation.
#     Values must range from 0 to n_years - 1.
#     Shape: (n_observations,)
#
# grade_idx:
#     Integer educational-stage index for each city-stage-year observation.
#     Values must range from 0 to n_grades - 1.
#     Shape: (n_observations,)
#
# X:
#     Standardized city-year covariate matrix aligned with the observations.
#     All covariates should be standardized before being passed to this script.
#     Shape: (n_observations, n_covariates)
#
# Q_spatial:
#     Positive-definite CAR precision matrix for the city-level spatial random
#     effects. It should be constructed from the administrative adjacency matrix
#     before running this script.
#     Shape: (n_cities, n_cities)
#
# city_names:
#     City labels used for PyMC coordinates.
#     Length: n_cities
#
# year_labels:
#     Year labels used for PyMC coordinates.
#     Length: n_years
#
# grade_labels:
#     Educational-stage labels used for PyMC coordinates.
#     Length: n_grades
#
# covariate_names:
#     Covariate labels used for PyMC coordinates.
#     Length: n_covariates
#
# In the original analysis:
#     n_cities = 21
#     n_grades = 5
#     n_years = 6
#     n_observations = 21 × 5 × 6 = 630
#
# Example placeholders:
# y_by_factor = {
#     "strength": np.array([...], dtype=float),
#     "speed": np.array([...], dtype=float),
#     "endurance": np.array([...], dtype=float),
#     "flexibility": np.array([...], dtype=float),
# }
#
# city_idx = np.array([...], dtype=int)
# year_idx = np.array([...], dtype=int)
# grade_idx = np.array([...], dtype=int)
# X = np.array([...], dtype=float)
# Q_spatial = np.array([...], dtype=float)
#
# city_names = [...]
# year_labels = [...]
# grade_labels = [...]
# covariate_names = [...]


# =============================================================================
# 2. Model-building and sampling function
# =============================================================================

def run_step2_model(
    y,
    city_idx,
    year_idx,
    grade_idx,
    X,
    Q_spatial,
    city_names,
    year_labels,
    grade_labels,
    covariate_names,
    draws=1000,
    tune=1000,
    chains=4,
    cores=4,
    target_accept=0.95,
    random_seed=2024,
):
    """
    Fit the Stage 2 Bayesian spatiotemporal hierarchical model for one latent
    fitness factor.

    Model specification
    -------------------
    For observation r belonging to city i, educational stage g, and year t,

        y_r ~ Normal(mu_r, sigma_obs)

        mu_r = alpha
               + x_r' beta
               + delta_g
               + S_i
               + T_t

    where
        alpha     is the global intercept,
        beta      is the vector of covariate effects,
        delta_g   is the educational-stage fixed effect with a sum-to-zero constraint,
        S_i       is the city-level CAR spatial random effect,
        T_t       is the year-level RW1 temporal random effect.

    Parameters
    ----------
    y : array-like, shape (n_observations,)
        Aggregated factor-level outcome for one fitness factor.

    city_idx : array-like, shape (n_observations,)
        City index for each observation.

    year_idx : array-like, shape (n_observations,)
        Year index for each observation.

    grade_idx : array-like, shape (n_observations,)
        Educational-stage index for each observation.

    X : array-like, shape (n_observations, n_covariates)
        Standardized covariate matrix.

    Q_spatial : array-like, shape (n_cities, n_cities)
        Positive-definite CAR precision matrix.

    city_names, year_labels, grade_labels, covariate_names : list
        Coordinate labels for the PyMC model.

    draws, tune, chains, cores, target_accept, random_seed :
        NUTS sampling configuration.

    Returns
    -------
    model : pm.Model
        Compiled PyMC model.

    trace : arviz.InferenceData
        Posterior samples.
    """

    y = np.asarray(y, dtype=float)
    city_idx = np.asarray(city_idx, dtype=int)
    year_idx = np.asarray(year_idx, dtype=int)
    grade_idx = np.asarray(grade_idx, dtype=int)
    X = np.asarray(X, dtype=float)
    Q_spatial = np.asarray(Q_spatial, dtype=float)

    n_observations = y.shape[0]
    n_cities = len(city_names)
    n_years = len(year_labels)
    n_grades = len(grade_labels)
    n_covariates = len(covariate_names)

    if X.shape != (n_observations, n_covariates):
        raise ValueError(
            "X must have shape (n_observations, n_covariates). "
            f"Expected {(n_observations, n_covariates)}, got {X.shape}."
        )

    if Q_spatial.shape != (n_cities, n_cities):
        raise ValueError(
            "Q_spatial must have shape (n_cities, n_cities). "
            f"Expected {(n_cities, n_cities)}, got {Q_spatial.shape}."
        )

    coords = {
        "obs": np.arange(n_observations),
        "city": city_names,
        "year": year_labels,
        "grade": grade_labels,
        "covariate": covariate_names,
    }

    with pm.Model(coords=coords) as model:

        # ---------------------------------------------------------------------
        # Fixed effects
        # ---------------------------------------------------------------------

        intercept = pm.Normal(
            "intercept",
            mu=0.0,
            sigma=2.0,
        )

        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=2.0,
            dims="covariate",
        )

        covariate_effect = pm.math.dot(X, beta)

        # ---------------------------------------------------------------------
        # Educational-stage effects
        # ---------------------------------------------------------------------

        delta_raw = pm.Normal(
            "delta_raw",
            mu=0.0,
            sigma=1.0,
            dims="grade",
        )

        delta = pm.Deterministic(
            "delta",
            delta_raw - delta_raw.mean(),
            dims="grade",
        )

        # ---------------------------------------------------------------------
        # Spatial random effects
        # ---------------------------------------------------------------------

        tau_spatial = pm.Gamma(
            "tau_spatial",
            alpha=2.0,
            beta=1.0,
        )

        S_raw = pm.MvNormal(
            "S_spatial",
            mu=np.zeros(n_cities),
            tau=tau_spatial * Q_spatial,
            dims="city",
        )

        S_c = pm.Deterministic(
            "S_c",
            S_raw - S_raw.mean(),
            dims="city",
        )

        # ---------------------------------------------------------------------
        # Temporal random effects
        # ---------------------------------------------------------------------

        sigma_time = pm.HalfNormal(
            "sigma_time",
            sigma=0.5,
        )

        T_t = pm.GaussianRandomWalk(
            "T_t",
            sigma=sigma_time,
            init_dist=pm.Normal.dist(mu=0.0, sigma=1.0),
            dims="year",
        )

        # ---------------------------------------------------------------------
        # Linear predictor
        # ---------------------------------------------------------------------

        mu = pm.Deterministic(
            "mu",
            intercept
            + covariate_effect
            + delta[grade_idx]
            + S_c[city_idx]
            + T_t[year_idx],
            dims="obs",
        )

        # ---------------------------------------------------------------------
        # Observation model
        # ---------------------------------------------------------------------

        sigma_obs = pm.HalfNormal(
            "sigma_obs",
            sigma=1.0,
        )

        pm.Normal(
            "y_like",
            mu=mu,
            sigma=sigma_obs,
            observed=y,
            dims="obs",
        )

        # ---------------------------------------------------------------------
        # Posterior sampling
        # ---------------------------------------------------------------------

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return model, trace


# =============================================================================
# 3. Fit separate models for the four latent fitness factors
# =============================================================================

factor_order = [
    "strength",
    "speed",
    "endurance",
    "flexibility",
]

models_step2 = {}
traces_step2 = {}

for factor_name in factor_order:
    model, trace = run_step2_model(
        y=y_by_factor[factor_name],
        city_idx=city_idx,
        year_idx=year_idx,
        grade_idx=grade_idx,
        X=X,
        Q_spatial=Q_spatial,
        city_names=city_names,
        year_labels=year_labels,
        grade_labels=grade_labels,
        covariate_names=covariate_names,
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=2024,
    )

    models_step2[factor_name] = model
    traces_step2[factor_name] = trace

    # Uncomment the following line to save posterior samples.
    # az.to_netcdf(trace, f"trace_step2_{factor_name}.nc")


# =============================================================================
# 4. Optional posterior summaries
# =============================================================================

# Example:
# az.summary(
#     traces_step2["strength"],
#     var_names=[
#         "intercept",
#         "beta",
#         "delta",
#         "tau_spatial",
#         "sigma_time",
#         "sigma_obs",
#     ],
# )