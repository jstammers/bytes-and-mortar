"""Bayesian hierarchical model for UK property price prediction.

This marimo notebook explores a PyMC-based Bayesian hierarchical regression
model as an alternative to the linear and XGBoost models in the main training
pipeline.

Hierarchy:
    price_i ~ LogNormal(mu_i, sigma)
    mu_i = alpha[district_i] + beta_floor * floor_area_i + beta_eff * energy_eff_i
    alpha[d] ~ Normal(mu_alpha, sigma_alpha)   # district-level random intercepts
    mu_alpha ~ Normal(12, 1)                   # prior centred on log(~£160k)
    sigma_alpha ~ HalfNormal(1)
    beta_floor ~ Normal(0, 0.01)
    beta_eff ~ Normal(0, 0.01)
    sigma ~ HalfNormal(0.5)

The LogNormal likelihood directly models the right-skew of prices without
needing an explicit log transformation of y — the hierarchical prior on
district intercepts provides partial pooling, borrowing strength from
data-rich districts for data-sparse ones.

Install dependencies:
    just install-notebooks

Run the notebook:
    uv run marimo edit notebooks/pymc_property_price.py
"""

import marimo

__generated_with = "0.6.0"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import warnings

    import arviz as az
    import marimo as mo
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    return az, mo, np, pd, warnings


@app.cell
def _config(mo):
    mo.md(
        """
        # Bayesian Hierarchical Property Price Model

        This notebook fits a **PyMC hierarchical log-normal regression** on the
        processed UK property dataset.  The model uses district-level random
        intercepts to pool information across geography — especially useful for
        districts with few transactions.

        > **Dependencies**: install with `just install-notebooks`
        > **Data**: run `just run` first to generate `data/processed/uk_property_sales.parquet`
        """
    )
    return


@app.cell
def _data_controls(mo):
    nrows_slider = mo.ui.slider(
        start=500, stop=10_000, step=500, value=2_000, label="Rows to sample"
    )
    test_year_picker = mo.ui.dropdown(
        options=["2022", "2023", "2024"], value="2024", label="Holdout year"
    )
    return nrows_slider, test_year_picker


@app.cell
def _load_data(mo, nrows_slider, pd):
    from src.data.config import PROCESSED_DATA_PATH

    try:
        df_full = pd.read_parquet(PROCESSED_DATA_PATH)
        df = df_full.sample(n=min(nrows_slider.value, len(df_full)), random_state=42)
        mo.md(
            f"Loaded **{len(df):,}** rows from `{PROCESSED_DATA_PATH.name}` "
            f"({df_full.shape[1]} columns)"
        )
    except FileNotFoundError:
        mo.md(
            "⚠️  **Data not found.** Run `just run` to generate the processed dataset, "
            "then restart this notebook."
        )
        df = None

    return df, df_full


@app.cell
def _feature_prep(df, mo, np, pd, test_year_picker):
    """Prepare features using the shared FeatureConfig API."""
    from src.features.build_features import FeatureConfig, MissingStrategy

    if df is None:
        mo.stop(True, mo.md("⚠️  No data loaded."))

    feat_config = FeatureConfig(
        numeric_features=["total_floor_area", "current_energy_efficiency"],
        categorical_features=["property_type"],
        target_encode_features=[],  # handled manually for PyMC
        missing_strategy=MissingStrategy.drop,
    )

    # Keep only rows with required columns
    required = feat_config.numeric_features + ["price", "district", "date_of_transfer"]
    df_clean = df.dropna(subset=required).copy()

    # Temporal split
    test_year = int(test_year_picker.value)
    train_mask = pd.to_datetime(df_clean["date_of_transfer"]).dt.year < test_year
    train_df = df_clean[train_mask].copy()
    test_df = df_clean[~train_mask].copy()

    # Encode districts as integer indices (PyMC needs integer coords)
    districts = train_df["district"].unique().tolist()
    district_idx = {d: i for i, d in enumerate(districts)}
    train_df["district_idx"] = train_df["district"].map(district_idx)
    # Test rows with unseen districts get index -1 (excluded from evaluation)
    test_df["district_idx"] = test_df["district"].map(district_idx).fillna(-1).astype(int)

    log_price_train = np.log(train_df["price"].values.astype(float))
    floor_area_train = (train_df["total_floor_area"].values - train_df["total_floor_area"].mean()) / train_df["total_floor_area"].std()
    energy_eff_train = (train_df["current_energy_efficiency"].values - train_df["current_energy_efficiency"].mean()) / train_df["current_energy_efficiency"].std()

    n_districts = len(districts)

    mo.md(
        f"""
        ### Data summary
        | | |
        |---|---|
        | Train rows | {len(train_df):,} |
        | Test rows  | {len(test_df):,} |
        | Districts  | {n_districts} |
        | Test year  | {test_year} |
        """
    )

    return (
        district_idx,
        districts,
        energy_eff_train,
        floor_area_train,
        log_price_train,
        n_districts,
        test_df,
        test_year,
        train_df,
    )


@app.cell
def _sampling_controls(mo):
    draws_slider = mo.ui.slider(start=200, stop=2000, step=200, value=500, label="MCMC draws")
    tune_slider = mo.ui.slider(start=200, stop=1000, step=200, value=500, label="Tuning steps")
    chains_slider = mo.ui.slider(start=1, stop=4, step=1, value=2, label="Chains")
    run_button = mo.ui.run_button(label="▶  Fit model")
    return chains_slider, draws_slider, run_button, tune_slider


@app.cell
def _model_and_sampling(
    chains_slider,
    district_idx,
    draws_slider,
    energy_eff_train,
    floor_area_train,
    log_price_train,
    mo,
    n_districts,
    np,
    run_button,
    train_df,
    tune_slider,
):
    mo.stop(not run_button.value, mo.md("Press **▶ Fit model** to run MCMC sampling."))

    import pymc as pm

    with pm.Model() as model:
        # --- Hyperpriors (district-level pooling) ---
        mu_alpha = pm.Normal("mu_alpha", mu=np.log(300_000), sigma=1.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)

        # --- District random intercepts (partial pooling) ---
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_districts)

        # --- Feature coefficients ---
        beta_floor = pm.Normal("beta_floor", mu=0, sigma=0.3)
        beta_energy = pm.Normal("beta_energy", mu=0, sigma=0.3)

        # --- Observation noise ---
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        # --- Linear predictor ---
        district_idxs = train_df["district_idx"].values
        mu = alpha[district_idxs] + beta_floor * floor_area_train + beta_energy * energy_eff_train

        # --- Likelihood: LogNormal for price (always positive, right-skewed) ---
        pm.Normal("log_price_obs", mu=mu, sigma=sigma, observed=log_price_train)

        # --- Sampling ---
        idata = pm.sample(
            draws=draws_slider.value,
            tune=tune_slider.value,
            chains=chains_slider.value,
            target_accept=0.9,
            return_inferencedata=True,
            progressbar=True,
        )

    mo.md("✅ MCMC sampling complete.")
    return idata, model


@app.cell
def _diagnostics(az, idata, mo):
    """MCMC convergence diagnostics."""
    summary = az.summary(idata, var_names=["mu_alpha", "sigma_alpha", "beta_floor", "beta_energy", "sigma"])
    rhat_ok = (summary["r_hat"] < 1.05).all()

    mo.md(
        f"""
        ### Convergence diagnostics

        **R-hat < 1.05 for all parameters:** {'✅ Yes' if rhat_ok else '⚠️  No — consider more tuning steps or chains'}

        {summary.to_html()}
        """
    )
    return rhat_ok, summary


@app.cell
def _posterior_plots(az, idata, mo):
    """Posterior distributions for global parameters."""
    import matplotlib.pyplot as plt

    fig, axes = az.plot_posterior(
        idata,
        var_names=["mu_alpha", "sigma_alpha", "beta_floor", "beta_energy", "sigma"],
        figsize=(14, 6),
    )
    plt.tight_layout()
    return (mo.as_html(fig),)


@app.cell
def _district_intercepts(az, districts, idata, mo, np):
    """Visualise district-level random intercepts (partial pooling effect)."""
    import matplotlib.pyplot as plt

    alpha_samples = idata.posterior["alpha"].values  # (chains, draws, n_districts)
    alpha_mean = alpha_samples.reshape(-1, len(districts)).mean(axis=0)
    alpha_hdi = az.hdi(idata, var_names=["alpha"])["alpha"].values

    sorted_idx = np.argsort(alpha_mean)
    top_n = min(30, len(districts))

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, alpha_mean[sorted_idx[-top_n:]], color="steelblue", alpha=0.7, label="Posterior mean")
    ax.errorbar(
        alpha_mean[sorted_idx[-top_n:]],
        y_pos,
        xerr=[
            alpha_mean[sorted_idx[-top_n:]] - alpha_hdi[sorted_idx[-top_n:], 0],
            alpha_hdi[sorted_idx[-top_n:], 1] - alpha_mean[sorted_idx[-top_n:]],
        ],
        fmt="none",
        color="navy",
        capsize=3,
        label="94% HDI",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([districts[i] for i in sorted_idx[-top_n:]], fontsize=8)
    ax.set_xlabel("Log-price intercept (higher = more expensive district)")
    ax.set_title(f"Top {top_n} district intercepts (partial pooling)")
    ax.legend()
    plt.tight_layout()

    mo.md(
        f"""
        ### District-level random intercepts

        Higher values indicate more expensive districts on average.
        The partial pooling shrinks data-sparse districts towards the global mean.

        {mo.as_html(fig)}
        """
    )
    return alpha_hdi, alpha_mean, sorted_idx


@app.cell
def _test_evaluation(district_idx, idata, mo, np, test_df):
    """Posterior predictive evaluation on the held-out test set."""
    from src.models.evaluate import compute_metrics

    # Only evaluate on test rows with known districts
    test_known = test_df[test_df["district_idx"] >= 0].copy()
    if len(test_known) == 0:
        mo.md("⚠️  No test rows with known districts — skipping evaluation.")
    else:
        # Posterior mean prediction: exp(alpha[d] + beta*X)
        alpha_post_mean = idata.posterior["alpha"].values.reshape(-1, idata.posterior.dims["alpha_dim_0"]).mean(axis=0)
        beta_floor_mean = float(idata.posterior["beta_floor"].values.mean())
        beta_energy_mean = float(idata.posterior["beta_energy"].values.mean())

        floor_mean = test_df["total_floor_area"].mean()
        floor_std = test_df["total_floor_area"].std()
        energy_mean = test_df["current_energy_efficiency"].mean()
        energy_std = test_df["current_energy_efficiency"].std()

        floor_scaled = (test_known["total_floor_area"].values - floor_mean) / floor_std
        energy_scaled = (test_known["current_energy_efficiency"].values - energy_mean) / energy_std

        log_pred = (
            alpha_post_mean[test_known["district_idx"].values]
            + beta_floor_mean * floor_scaled
            + beta_energy_mean * energy_scaled
        )
        pred_prices = np.exp(log_pred)
        true_prices = test_known["price"].values.astype(float)

        metrics = compute_metrics(true_prices, pred_prices)

        mo.md(
            f"""
            ### Test set evaluation ({len(test_known):,} rows)

            {metrics.summary("Bayesian hierarchical")}

            > These metrics are directly comparable to those logged by
            > `bytes-and-mortar train run` for linear and XGBoost models.
            """
        )
    return metrics, pred_prices, test_known, true_prices


@app.cell
def _sklearn_wrapper_note(mo):
    mo.md(
        """
        ---
        ### Path to sklearn integration

        To integrate this Bayesian model into the main training pipeline, wrap the
        fitted PyMC model in a sklearn-compatible class:

        ```python
        from sklearn.base import BaseEstimator, RegressorMixin

        class BayesianPropertyModel(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                # Run pm.sample() here, store idata
                ...
            def predict(self, X):
                # Return posterior mean predictions
                ...
        ```

        Then pass an instance to `build_best_pipeline()` in `train_model.py` and
        extend `ModelType` with a `bayesian` variant.
        """
    )
    return


if __name__ == "__main__":
    app.run()
