# HVAC ROM-Degradation Suite

Deployable Streamlit software for reduced-order HVAC energy modelling with equipment degradation, maintenance strategy comparison, climate scenario analysis, validation upload, benchmark/sensitivity reporting, zone-specific tables, export-ready figures, and CatBoost surrogate modelling.

## Architecture

The software keeps the numerical model in **`hvac_v3_engine.py`**. The Streamlit app imports:

```python
from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model, train_surrogate_models
```

The UI and `report_addons.py` do not replace the main HVAC equations. They provide upload handling, validation comparison, benchmark tables, zone-level allocation tables, and export packaging.

## Files

| File | Purpose |
|---|---|
| `hvac_v3_engine.py` | Main scientific engine: building/HVAC dataclasses, weather normalization, degradation, scenario simulation, exports, CatBoost training |
| `streamlit_app.py` | Main deployable Streamlit interface |
| `report_addons.py` | Post-processing helpers for validation, benchmark, fuel breakdown, comfort, site data, internal gains, zone analysis, and ZIP export |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark theme deployment defaults |
| `run_example.py` | Non-UI quick run for testing deployment |
| `examples/sample_daily_weather.csv` | Example CSV weather file |

## Install and run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Weather uploads

The app supports:

1. EPW upload directly through Streamlit.
2. CSV upload directly through Streamlit.
3. EPW file path.
4. CSV file path.
5. Synthetic weather.

Accepted daily CSV columns:

```text
day_of_year,T_mean_C,T_max_C,RH_mean_pct,GHI_mean_Wm2
```

The CSV reader also accepts common hourly columns such as `Date/Time`, `T_amb_C`, `RH_pct`, and `GHI_Wm2`.

## Main outputs

The engine preserves:

- Daily ML dataset CSV
- Annual CSV
- Summary CSV
- Excel report
- PDF report
- Journal-ready figures
- Baseline no-degradation layer

The upgraded app adds detailed sheets:

- `fuel_breakdown.csv`
- `comfort.csv`
- `site_data.csv`
- `internal_gains.csv`
- `validation_template.csv`
- `validation_comparison.csv` after validation upload
- `benchmark_summary.csv`
- `zone_analysis.csv`
- `kpi_summary.csv`
- `detailed_outputs.xlsx`
- ZIP bundle export

## Run a quick non-UI example

```bash
python run_example.py
```

This writes a small one-year demonstration to `example_run/`.

## Deployment notes

For Streamlit Community Cloud, upload this folder to a GitHub repository and set the main file to:

```text
streamlit_app.py
```

For reproducible research, keep `hvac_v3_engine.py` versioned because it is the single source of truth for model equations and scenario calculations.
