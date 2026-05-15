"""Marimo notebook: explore the trained property-price model.

Three sections:

1. HPI vs regional house prices — how districts track the UK HPI and where
   the model predicts well/poorly (MdAPE choropleth + 24-month forecast).
2. House prices over time — national property-type trends and spatial
   deviation from the national average for a chosen year.
3. Interactive valuation — pick a region + property and see a residual-
   bootstrap predictive distribution of sale price.

Setup
-----
- Install deps:                 `just install-notebooks`
- Ensure data is processed:     `just run`
- Ensure a model is trained:    `just train`
- Launch:                       `just notebook`

On first run, ONS Local Authority District boundaries are downloaded into
``data/geo/lad_boundaries.geojson`` (cached thereafter).
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl

    warnings.filterwarnings("ignore")
    return mo, np, pd, pl, plt


@app.cell
def _intro(mo):
    mo.md("""
    # Property model exploration

    Diagnose the trained Perpetual GBM:
    1. How do regional house prices track the **HPI** — and where does the model err most?
    2. How have prices for **each property type** moved across the country, and where do
       they sell above / below the national average?
    3. **Score one property** interactively and see the predictive distribution.
    """)
    return


@app.cell
def _config():
    from src.data.config import (
        HPI_REGIONAL_DATA_PATH,
        MODELS_DIR,
        PROCESSED_DATA_PATH,
        PROCESSED_DIR,
    )

    CACHE_DIR = PROCESSED_DIR / "notebook_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = MODELS_DIR / "perpetual_best.joblib"
    HOLDOUT_YEARS = [2024]
    return (
        CACHE_DIR,
        HOLDOUT_YEARS,
        HPI_REGIONAL_DATA_PATH,
        MODEL_PATH,
        PROCESSED_DATA_PATH,
    )


@app.cell
def _load_holdout(
    CACHE_DIR,
    HOLDOUT_YEARS,
    HPI_REGIONAL_DATA_PATH,
    PROCESSED_DATA_PATH,
    mo,
    pd,
    pl,
):
    """Lazy-scan the processed parquet and materialise only the 2024 holdout.

    HPI features (averageprice, index) are joined here — they are stored
    separately in uk_hpi_regional.parquet and not embedded in the main sales
    parquet.  The trained pipeline was fitted with these features via
    join_hpi_to_sales(), so they must be present before calling pipeline.predict.

    Result is cached to data/processed/notebook_cache/holdout_<years>_hpi.parquet.
    """
    from src.models.property_price.features import join_hpi_to_sales

    cache_key = "_".join(str(y) for y in HOLDOUT_YEARS)
    holdout_cache = CACHE_DIR / f"holdout_{cache_key}_hpi.parquet"

    if not PROCESSED_DATA_PATH.exists():
        mo.md(
            f"**Data not found** at `{PROCESSED_DATA_PATH}`. "
            "Run `just run` first to generate the processed dataset."
        ).callout(kind="warn")
        holdout = None
    elif holdout_cache.exists():
        holdout = pd.read_parquet(holdout_cache)
    else:
        lf = pl.scan_parquet(PROCESSED_DATA_PATH).filter(
            pl.col("date_of_transfer").dt.year().is_in(HOLDOUT_YEARS)
        )
        lf = join_hpi_to_sales(lf, HPI_REGIONAL_DATA_PATH)
        collected = lf.collect()
        assert isinstance(collected, pl.DataFrame)
        holdout = collected.to_pandas()
        holdout.to_parquet(holdout_cache)

    if holdout is not None:
        mo.md(
            f"Loaded **{len(holdout):,}** holdout rows "
            f"(years: {HOLDOUT_YEARS}) across "
            f"**{holdout['district'].nunique():,}** districts."
        )
    return cache_key, holdout


@app.cell
def _load_model(MODEL_PATH, mo):
    from src.models.property_price.predict import load_pipeline

    try:
        pipeline = load_pipeline(MODEL_PATH)
        mo.md(f"Loaded model from `{MODEL_PATH.name}`.")
    except FileNotFoundError:
        pipeline = None
        mo.md("**Model not found.** Run `just train` first to fit a Perpetual GBM.").callout(
            kind="warn"
        )
    return (pipeline,)


@app.cell
def _holdout_predictions(CACHE_DIR, cache_key, holdout, np, pd, pipeline):
    """Score the holdout slice with the trained pipeline and cache predictions."""
    pred_cache = CACHE_DIR / f"holdout_preds_{cache_key}_hpi.parquet"
    if holdout is None or pipeline is None:
        scored = None
    elif pred_cache.exists():
        scored = pd.read_parquet(pred_cache)
    else:
        from src.models.property_price.features import FEATURE_SCHEMA

        feature_cols = [c for c in FEATURE_SCHEMA if c in holdout.columns]
        X = holdout[feature_cols]
        y_pred_log = pipeline.predict(X)
        scored = holdout[
            ["date_of_transfer", "district", "county", "property_type", "price"]
        ].copy()
        scored["y_pred"] = np.expm1(y_pred_log)
        scored["y_log"] = np.log1p(holdout["price"].to_numpy())
        scored["y_pred_log"] = y_pred_log
        scored["residual_log"] = scored["y_log"] - scored["y_pred_log"]
        scored["abs_pct_error"] = (scored["y_pred"] - scored["price"]).abs() / scored["price"]
        scored.to_parquet(pred_cache)
    return (scored,)


@app.cell
def _load_geometries(mo):
    from src.viz.maps import load_lad_geometries

    try:
        geometries = load_lad_geometries()
        geo_status = mo.md(f"Loaded **{len(geometries)}** LAD polygons for choropleth maps.")
    except Exception as exc:
        geometries = None
        geo_status = mo.md(
            f"Could not load LAD boundaries: `{exc}`. "
            "Set the `LAD_GEOJSON_URL` env var or commit a geojson at "
            "`data/geo/lad_boundaries.geojson`."
        ).callout(kind="warn")
    geo_status
    return (geometries,)


@app.cell
def _section_1_header(mo):
    mo.md("""
    ## 1. HPI vs regional house prices

    How do district-level **actual** sale prices compare to the **UK HPI**,
    and where does the model predict well or poorly?
    """)
    return


@app.cell
def _section_1_aggregates(HPI_REGIONAL_DATA_PATH, holdout, mo, pl):
    if holdout is None:
        actual_by_district = None
        hpi_by_district = None
    else:
        actual_by_district = (
            holdout.groupby("district", as_index=False)
            .agg(median_price=("price", "median"), n_sales=("price", "size"))
            .sort_values("median_price", ascending=False)
        )

        if HPI_REGIONAL_DATA_PATH.exists():
            hpi_df = pl.scan_parquet(HPI_REGIONAL_DATA_PATH).collect().to_pandas()
            hpi_df = hpi_df.sort_values("date")
            hpi_by_district = hpi_df.groupby("district", as_index=False).agg(
                hpi_averageprice=("averageprice", "last")
            )
        else:
            hpi_by_district = None
            mo.md(
                f"HPI regional file missing at `{HPI_REGIONAL_DATA_PATH}` — "
                "run `just run` to regenerate."
            ).callout(kind="warn")
    return actual_by_district, hpi_by_district


@app.cell
def _section_1_actual_vs_hpi_maps(
    actual_by_district,
    geometries,
    hpi_by_district,
    mo,
):
    from src.viz.maps import choropleth

    if actual_by_district is None or geometries is None:
        side_by_side = mo.md("_Skipped: holdout or geometries unavailable._")
    else:
        actual_map = choropleth(
            dict(
                zip(
                    actual_by_district["district"], actual_by_district["median_price"], strict=False
                )
            ),
            title="Median sale price (holdout)",
            legend_name="GBP",
            geometries=geometries,
        )
        hpi_map = (
            choropleth(
                dict(
                    zip(
                        hpi_by_district["district"],
                        hpi_by_district["hpi_averageprice"],
                        strict=False,
                    )
                ),
                title="HPI average price (latest)",
                legend_name="GBP",
                geometries=geometries,
            )
            if hpi_by_district is not None
            else None
        )
        side_by_side = mo.hstack(
            [
                mo.vstack(
                    [
                        mo.md("**Actual median sale price**"),
                        mo.iframe(actual_map._repr_html_(), height="500px"),
                    ]
                ),
                mo.vstack(
                    [
                        mo.md("**HPI average price**"),
                        mo.iframe(hpi_map._repr_html_(), height="500px"),
                    ]
                )
                if hpi_map is not None
                else mo.md("_HPI map unavailable._"),
            ]
        )
    side_by_side
    return (choropleth,)


@app.cell
def _section_1_regional_mdape(geometries, mo, scored):
    """MdAPE per district from the model on the 2024 holdout."""
    if scored is None or geometries is None:
        regional_mdape = None
        mdape_map = mo.md("_Skipped: predictions or geometries unavailable._")
    else:
        regional = (
            scored.groupby("district", as_index=False)
            .agg(mdape=("abs_pct_error", "median"), n=("abs_pct_error", "size"))
            .query("n >= 30")
            .sort_values("mdape")
        )
        regional_mdape = regional

        from src.viz.maps import choropleth

        fmap = choropleth(
            dict(zip(regional["district"], regional["mdape"], strict=False)),
            title="MdAPE by district",
            legend_name="MdAPE (lower is better)",
            fill_color="RdYlGn_r",
            geometries=geometries,
            bins=6,
        )
        mdape_map = mo.iframe(fmap._repr_html_(), height="600px")
    mdape_map
    return choropleth, regional_mdape


@app.cell
def _section_1_leaderboards(mo, regional_mdape):
    if regional_mdape is None:
        leaderboard = mo.md("_No regional metrics to display._")
    else:
        best = regional_mdape.head(20)
        worst = regional_mdape.tail(20).iloc[::-1]
        leaderboard = mo.hstack(
            [
                mo.vstack([mo.md("**Best-predicted 20 districts**"), mo.ui.table(best)]),
                mo.vstack([mo.md("**Worst-predicted 20 districts**"), mo.ui.table(worst)]),
            ]
        )
    leaderboard
    return


@app.cell
def _section_1_forecast_controls(holdout, mo):
    if holdout is None:
        district_picker = None
    else:
        districts = sorted(holdout["district"].dropna().unique().tolist())
        district_picker = mo.ui.dropdown(
            options=districts,
            value=districts[0] if districts else None,
            label="Forecast district",
        )
    district_picker
    return (district_picker,)


@app.cell
def _section_1_forecast(HPI_REGIONAL_DATA_PATH, district_picker, mo, pl, plt):
    """Run the EnsembleForecaster on the chosen district's HPI series."""
    if district_picker is None or district_picker.value is None:
        forecast_fig = mo.md("_Pick a district to forecast its HPI._")
    elif not HPI_REGIONAL_DATA_PATH.exists():
        forecast_fig = mo.md("_HPI regional file missing._").callout(kind="warn")
    else:
        from src.models.hpi_forecast.forecasters import EnsembleForecaster

        district = district_picker.value
        hpi_df = (
            pl.scan_parquet(HPI_REGIONAL_DATA_PATH)
            .filter(pl.col("district") == district)
            .sort("date")
            .collect()
            .to_pandas()
        )
        if len(hpi_df) < 36:
            forecast_fig = mo.md(
                f"_Not enough HPI history for `{district}` ({len(hpi_df)} months)._"
            )
        else:
            forecaster = EnsembleForecaster()
            forecaster.fit_from_frame(hpi_df, value_col="averageprice", date_col="date")
            result = forecaster.forecast(steps=24, alpha=0.05)
            fc_df = result.to_frame()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hpi_df["date"], hpi_df["averageprice"], label="Actual", color="C0")
            ax.plot(fc_df["date"], fc_df["forecast"], label="Forecast", color="C1")
            ax.fill_between(
                fc_df["date"],
                fc_df["lower"],
                fc_df["upper"],
                color="C1",
                alpha=0.2,
                label="95% PI",
            )
            ax.set_title(f"HPI forecast — {district}")
            ax.set_ylabel("Average price (GBP)")
            ax.legend()
            plt.tight_layout()
            forecast_fig = mo.mpl.interactive(fig)
    forecast_fig
    return


@app.cell
def _section_2_header(mo):
    mo.md("""
    ## 2. Property types over time

    National trend by property type, then how each type's median deviates from
    the national average across the country in a chosen year.
    """)
    return


@app.cell
def _section_2_load_typed(CACHE_DIR, PROCESSED_DATA_PATH, pd, pl):
    """National median price by year x property_type, aggregated at scan time."""
    cache = CACHE_DIR / "national_price_by_type_year.parquet"
    if not PROCESSED_DATA_PATH.exists():
        national = None
    elif cache.exists():
        national = pd.read_parquet(cache)
    else:
        national = (
            pl.scan_parquet(PROCESSED_DATA_PATH)
            .filter(pl.col("property_type").is_in(["D", "S", "T", "F"]))
            .group_by(["year", "property_type"])
            .agg(
                pl.col("price").median().alias("median_price"),
                pl.len().alias("n"),
            )
            .sort(["year", "property_type"])
            .collect()
            .to_pandas()
        )
        national.to_parquet(cache)
    return (national,)


@app.cell
def _section_2_national_trend(mo, national, plt):
    if national is None:
        national_fig = mo.md("_Processed dataset not available._")
    else:
        labels = {"D": "Detached", "S": "Semi", "T": "Terraced", "F": "Flat"}
        fig, ax = plt.subplots(figsize=(10, 4))
        for pt, group in national.groupby("property_type"):
            ax.plot(group["year"], group["median_price"], label=labels.get(pt, pt))
        ax.set_xlabel("Year")
        ax.set_ylabel("Median sale price (GBP)")
        ax.set_title("National median sale price by property type")
        ax.legend()
        plt.tight_layout()
        national_fig = mo.mpl.interactive(fig)
    national_fig
    return


@app.cell
def _section_2_deviation_controls(mo, national):
    if national is None:
        year_slider = None
        type_picker = None
        controls = mo.md("")
    else:
        years = sorted(national["year"].unique().tolist())
        year_slider = mo.ui.slider(
            start=min(years),
            stop=max(years),
            value=max(years),
            step=1,
            label="Year",
        )
        type_picker = mo.ui.dropdown(
            options=["D", "S", "T", "F"], value="D", label="Property type (D/S/T/F)"
        )
        controls = mo.hstack([year_slider, type_picker])
    controls
    return type_picker, year_slider


@app.cell
def _section_2_deviation_map(
    CACHE_DIR,
    PROCESSED_DATA_PATH,
    choropleth,
    geometries,
    mo,
    pd,
    pl,
    type_picker,
    year_slider,
):
    """Signed % deviation from the national median, per district, for one year + type."""
    if (
        geometries is None
        or year_slider is None
        or type_picker is None
        or not PROCESSED_DATA_PATH.exists()
    ):
        deviation_map = mo.md("_Skipped: controls or data unavailable._")
    else:
        year = int(year_slider.value)
        pt = type_picker.value
        cache = CACHE_DIR / f"district_deviation_{year}_{pt}.parquet"
        if cache.exists():
            dev = pd.read_parquet(cache)
        else:
            dev = (
                pl.scan_parquet(PROCESSED_DATA_PATH)
                .filter((pl.col("year") == year) & (pl.col("property_type") == pt))
                .group_by("district")
                .agg(
                    pl.col("price").median().alias("median_price"),
                    pl.len().alias("n"),
                )
                .filter(pl.col("n") >= 5)
                .collect()
                .to_pandas()
            )
            national_median = dev["median_price"].median()
            dev["pct_dev"] = (dev["median_price"] - national_median) / national_median
            dev.to_parquet(cache)

        fmap2 = choropleth(
            dict(zip(dev["district"], dev["pct_dev"], strict=False)),
            title=f"{pt} price deviation from national median, {year}",
            legend_name="% deviation",
            fill_color="RdBu_r",
            geometries=geometries,
            bins=7,
        )
        deviation_map = mo.iframe(fmap2._repr_html_(), height="600px")
    deviation_map
    return


@app.cell
def _section_2_top_bottom(
    CACHE_DIR,
    PROCESSED_DATA_PATH,
    mo,
    pd,
    pl,
    type_picker,
    year_slider,
):
    """Top-10 above / bottom-10 below the national median for the chosen year + type."""
    if year_slider is None or type_picker is None or not PROCESSED_DATA_PATH.exists():
        leaderboards = mo.md("")
    else:
        year = int(year_slider.value)
        pt = type_picker.value
        cache = CACHE_DIR / f"district_deviation_{year}_{pt}.parquet"
        if cache.exists():
            dev = pd.read_parquet(cache).sort_values("pct_dev", ascending=False)
        else:
            dev = (
                pl.scan_parquet(PROCESSED_DATA_PATH)
                .filter((pl.col("year") == year) & (pl.col("property_type") == pt))
                .group_by("district")
                .agg(
                    pl.col("price").median().alias("median_price"),
                    pl.len().alias("n"),
                )
                .filter(pl.col("n") >= 5)
                .collect()
                .to_pandas()
            )
            national_median = dev["median_price"].median()
            dev["pct_dev"] = (dev["median_price"] - national_median) / national_median
            dev = dev.sort_values("pct_dev", ascending=False)
        top = dev.head(10)
        bottom = dev.tail(10).iloc[::-1]
        leaderboards = mo.hstack(
            [
                mo.vstack([mo.md("**Top 10 above national median**"), mo.ui.table(top)]),
                mo.vstack(
                    [
                        mo.md("**Bottom 10 below national median**"),
                        mo.ui.table(bottom),
                    ]
                ),
            ]
        )
    leaderboards
    return


@app.cell
def _section_3_header(mo):
    mo.md("""
    ## 3. Interactive valuation

    Specify a property below and the notebook scores it with the trained
    Perpetual GBM, then bootstraps log-space residuals from comparable
    2024 sales to estimate a distribution of likely sale prices.
    """)
    return


@app.cell
def _section_3_inputs(holdout, mo):
    if holdout is None:
        ui = None
    else:
        districts = sorted(holdout["district"].dropna().unique().tolist())
        counties = sorted(holdout["county"].dropna().unique().tolist())
        energy_ratings = ["A", "B", "C", "D", "E", "F", "G"]
        age_bands = sorted(x for x in holdout["construction_age_band"].dropna().unique().tolist())
        ui = mo.ui.dictionary(
            {
                "district": mo.ui.dropdown(options=districts, value=districts[0], label="District"),
                "county": mo.ui.dropdown(options=counties, value=counties[0], label="County"),
                "property_type": mo.ui.dropdown(
                    options=["D", "S", "T", "F"], value="D", label="Property type"
                ),
                "old_new": mo.ui.dropdown(options=["Y", "N"], value="N", label="New build?"),
                "duration": mo.ui.dropdown(options=["F", "L"], value="F", label="Tenure"),
                "current_energy_rating": mo.ui.dropdown(
                    options=energy_ratings, value="D", label="EPC rating"
                ),
                "construction_age_band": mo.ui.dropdown(
                    options=age_bands or ["unknown"],
                    value=age_bands[0] if age_bands else "unknown",
                    label="Age band",
                ),
                "total_floor_area": mo.ui.slider(
                    start=30, stop=500, step=5, value=100, label="Floor area (m²)"
                ),
                "number_habitable_rooms": mo.ui.slider(
                    start=1, stop=15, step=1, value=4, label="Habitable rooms"
                ),
                "current_energy_efficiency": mo.ui.slider(
                    start=20, stop=100, step=1, value=65, label="Energy efficiency"
                ),
                "year": mo.ui.slider(start=2020, stop=2026, step=1, value=2024, label="Year"),
                "month": mo.ui.slider(start=1, stop=12, step=1, value=6, label="Month"),
            }
        )
    ui
    return (ui,)


@app.cell
def _section_3_build_input_row(holdout, pd, ui):
    """Translate the UI dictionary into a one-row dataframe matching FEATURE_SCHEMA."""
    if ui is None or holdout is None:
        single_row = None
    else:
        values = {k: c.value for k, c in ui.value.items()}
        district_rows = holdout[holdout["district"] == values["district"]]
        if district_rows.empty:
            single_row = None
        else:
            seed = district_rows.iloc[0]
            row = {
                **values,
                "postcode_outward": seed.get("postcode_outward"),
                "averageprice": seed.get("averageprice"),
                "index": seed.get("index"),
            }
            single_row = pd.DataFrame([row])
    return (single_row,)


@app.cell
def _section_3_bootstrap(mo, np, pipeline, scored, single_row, ui):
    """Residual bootstrap stratified by district + property_type (with fallback)."""
    if pipeline is None or scored is None or single_row is None or ui is None:
        bootstrap_panel = mo.md("_Awaiting model + holdout + inputs._")
        samples = None
        point = None
        scope = None
    else:
        from src.models.property_price.features import FEATURE_SCHEMA
        from src.models.property_price.predict import bootstrap_distribution, predict_price

        feature_cols = [c for c in FEATURE_SCHEMA if c in single_row.columns]
        X = single_row[feature_cols]
        district = ui.value["district"].value
        ptype = ui.value["property_type"].value

        pool = scored[(scored["district"] == district) & (scored["property_type"] == ptype)][
            "residual_log"
        ].to_numpy()
        scope = "district + type"
        if pool.size < 30:
            pool = scored[scored["property_type"] == ptype]["residual_log"].to_numpy()
            scope = "property type"
        if pool.size < 30:
            pool = scored["residual_log"].to_numpy()
            scope = "all holdout"

        samples = bootstrap_distribution(pipeline, X, pool, n=5000, seed=0)
        point = float(predict_price(pipeline, X)[0])

        lo, med, hi = np.quantile(samples, [0.05, 0.5, 0.95])
        bootstrap_panel = mo.md(
            f"""
            **Point estimate**: £{point:,.0f}
            **Predictive median**: £{med:,.0f}
            **90% interval**: £{lo:,.0f} – £{hi:,.0f}
            Residual pool: {pool.size:,} samples ({scope})
            """
        )
    bootstrap_panel
    return point, samples


@app.cell
def _section_3_distribution_plot(mo, np, plt, point, samples):
    if samples is None or point is None:
        dist_fig = mo.md("")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(samples, bins=60, color="C0", alpha=0.7, edgecolor="white")
        for q, ls, label in [
            (0.05, "--", "5%"),
            (0.5, "-", "median"),
            (0.95, "--", "95%"),
        ]:
            ax.axvline(np.quantile(samples, q), color="black", linestyle=ls, label=label)
        ax.axvline(point, color="C3", linestyle="-", label="point")
        ax.set_xlabel("Predicted sale price (GBP)")
        ax.set_ylabel("Frequency")
        ax.set_title("Bootstrapped predictive distribution")
        ax.legend()
        plt.tight_layout()
        dist_fig = mo.mpl.interactive(fig)
    dist_fig
    return


@app.cell
def _section_3_comparables(holdout, mo, ui):
    """Twenty most recent comparable 2024 sales (same district + type)."""
    if holdout is None or ui is None:
        comps_panel = mo.md("")
    else:
        district = ui.value["district"].value
        ptype = ui.value["property_type"].value
        comps = (
            holdout[(holdout["district"] == district) & (holdout["property_type"] == ptype)]
            .sort_values("date_of_transfer", ascending=False)
            .head(20)
        )
        if comps.empty:
            comps_panel = mo.md(f"_No comparable sales in {district} ({ptype})._")
        else:
            cols = [
                "date_of_transfer",
                "postcode",
                "property_type",
                "total_floor_area",
                "number_habitable_rooms",
                "current_energy_rating",
                "price",
            ]
            cols = [c for c in cols if c in comps.columns]
            comps_panel = mo.ui.table(comps[cols].reset_index(drop=True))
    comps_panel
    return


if __name__ == "__main__":
    app.run()
