from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from hvac_v3_engine import (
    BuildingSpec,
    HVACConfig,
    HVAC_PRESETS,
    SCENARIOS,
    SEVERITY_LEVELS,
    CLIMATE_LEVELS,
    run_scenario_model,
    train_surrogate_models,
)
from report_addons import (
    read_weather_upload,
    build_detailed_tables,
    save_detailed_outputs,
    load_validation_file,
    build_validation_comparison,
    create_zip_from_folder,
    find_result_paths,
)

st.set_page_config(page_title="HVAC ROM-Degradation Suite", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {background: linear-gradient(180deg, #07101f 0%, #101729 55%, #151827 100%);} 
.block-container {padding-top: 1.15rem; padding-bottom: 2.2rem; max-width: 1320px;}
h1, h2, h3, h4, h5, h6, p, label, span, div {color: #eaf0fb;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);} 
div[data-baseweb="tab-list"] {gap: 0.55rem; border-bottom: 1px solid rgba(255,255,255,0.10); padding-bottom: 0.25rem;}
button[data-baseweb="tab"] {background: rgba(255,255,255,0.035) !important; border-radius: 14px 14px 0 0 !important; padding: 0.8rem 1.0rem !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.07) !important;}
button[data-baseweb="tab"][aria-selected="true"] {color: #ff686b !important; border-bottom: 2px solid #ff686b !important; background: rgba(255,255,255,0.075) !important;}
div[data-testid="stExpander"] {border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; background: rgba(255,255,255,0.035); margin-bottom: 0.9rem;}
div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 0.5rem 0.7rem;}
div[data-testid="stDataFrame"] {border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; overflow: hidden;}
div.stButton > button {border-radius: 14px !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.18) !important;}
.small-muted {color:#aeb8ce; font-size:0.94rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div style="padding: 0.55rem 0 1.0rem 0;">
      <div style="font-size: 2.7rem; font-weight: 850; letter-spacing: -0.035em; color: #f6f8fc;">
        HVAC ROM-Degradation Suite
      </div>
      <div class="small-muted" style="max-width: 980px; margin-top:0.35rem;">
        Reduced-order HVAC energy, degradation, maintenance-strategy, climate-scenario, validation, reporting, and CatBoost surrogate modelling platform.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def default_zone_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"zone_name": "Lecture_01", "zone_type": "Lecture", "area_m2": 200.0, "occ_density": 0.12, "term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
        {"zone_name": "Office_01", "zone_type": "Office", "area_m2": 120.0, "occ_density": 0.06, "term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
        {"zone_name": "Lab_01", "zone_type": "Lab", "area_m2": 180.0, "occ_density": 0.08, "term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
        {"zone_name": "Corridor", "zone_type": "Corridor", "area_m2": 100.0, "occ_density": 0.01, "term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
        {"zone_name": "Service_01", "zone_type": "Service", "area_m2": 80.0, "occ_density": 0.02, "term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
    ])


def download_file_button(path: str | Path, label: str, key: str | None = None):
    path = Path(path)
    if path.exists() and path.is_file():
        with path.open("rb") as f:
            st.download_button(label, f.read(), file_name=path.name, key=key or f"dl_{path.name}")


# ---------------- Setup ----------------
with st.popover("⚙️ Setup", use_container_width=False):
    st.markdown("### Building identity")
    building_type = st.text_input("Building type", value=st.session_state.get("building_type", "Educational / University building"))
    location = st.text_input("Location / weather label", value=st.session_state.get("location", "User-defined"))

    st.markdown("### Geometry")
    area_m2 = st.number_input("Conditioned area (m²)", min_value=100.0, value=float(st.session_state.get("area_m2", 5000.0)), step=100.0)
    floors = st.number_input("Floors", min_value=1, value=int(st.session_state.get("floors", 4)), step=1)
    n_spaces = st.number_input("Number of spaces", min_value=1, value=int(st.session_state.get("n_spaces", 40)), step=1)

    st.markdown("### Envelope")
    wall_u = st.number_input("Wall U-value (W/m²K)", min_value=0.05, value=float(st.session_state.get("wall_u", 0.6)), step=0.05)
    roof_u = st.number_input("Roof U-value (W/m²K)", min_value=0.05, value=float(st.session_state.get("roof_u", 0.35)), step=0.05)
    window_u = st.number_input("Window U-value (W/m²K)", min_value=0.1, value=float(st.session_state.get("window_u", 2.7)), step=0.1)
    shgc = st.number_input("SHGC", min_value=0.05, max_value=0.95, value=float(st.session_state.get("shgc", 0.35)), step=0.01)
    glazing_ratio = st.number_input("Glazing ratio", min_value=0.01, max_value=0.95, value=float(st.session_state.get("glazing_ratio", 0.30)), step=0.01)
    infiltration_ach = st.number_input("Infiltration (ACH)", min_value=0.0, value=float(st.session_state.get("infiltration_ach", 0.5)), step=0.1)

    st.markdown("### Internal loads")
    occupancy_density = st.number_input("General occupancy density (person/m²)", min_value=0.001, value=float(st.session_state.get("occupancy_density", 0.08)), step=0.01)
    lighting_w_m2 = st.number_input("Lighting power density (W/m²)", min_value=0.0, value=float(st.session_state.get("lighting_w_m2", 10.0)), step=1.0)
    equipment_w_m2 = st.number_input("Equipment power density (W/m²)", min_value=0.0, value=float(st.session_state.get("equipment_w_m2", 8.0)), step=1.0)
    sensible_w_per_person = st.number_input("Sensible heat per person (W)", min_value=20.0, value=float(st.session_state.get("sensible_w_per_person", 75.0)), step=5.0)

    st.markdown("### HVAC sizing and degradation")
    hvac_system_type = st.selectbox("HVAC system type", list(HVAC_PRESETS.keys()), index=list(HVAC_PRESETS.keys()).index(st.session_state.get("hvac_system_type", "Chiller_AHU")))
    airflow_m3h_m2 = st.number_input("Airflow intensity (m³/h·m²)", min_value=0.1, value=float(st.session_state.get("airflow_m3h_m2", 4.0)), step=0.1)
    cooling_w_m2 = st.number_input("Cooling design intensity (W/m²)", min_value=1.0, value=float(st.session_state.get("cooling_w_m2", 100.0)), step=5.0)
    heating_w_m2 = st.number_input("Heating design intensity (W/m²)", min_value=1.0, value=float(st.session_state.get("heating_w_m2", 55.0)), step=5.0)
    years = st.number_input("Simulation years", min_value=1, max_value=50, value=int(st.session_state.get("years", 20)), step=1)

    cop_aging_rate = st.number_input("COP aging rate", min_value=0.0001, value=float(st.session_state.get("cop_aging_rate", 0.005)), step=0.001, format="%.4f")
    rf_star = st.number_input("RF* fouling asymptote", min_value=1e-6, value=float(st.session_state.get("rf_star", 2e-4)), format="%.6f")
    b_foul = st.number_input("Fouling growth constant B", min_value=0.001, value=float(st.session_state.get("b_foul", 0.015)), step=0.001, format="%.3f")
    dust_rate = st.number_input("Dust accumulation rate", min_value=0.1, value=float(st.session_state.get("dust_rate", 1.2)), step=0.1)
    k_clog = st.number_input("Clogging coefficient", min_value=0.1, value=float(st.session_state.get("k_clog", 6.0)), step=0.1)
    deg_trigger = st.number_input("Degradation trigger", min_value=0.1, max_value=1.5, value=float(st.session_state.get("deg_trigger", 0.55)), step=0.01)
    degradation_model = st.selectbox("Degradation model", ["physics", "linear_ts", "exponential_ts"], index=["physics", "linear_ts", "exponential_ts"].index(st.session_state.get("degradation_model", "physics")), format_func=lambda x: {"physics":"Physics-based fouling/clogging", "linear_ts":"Linear time-series", "exponential_ts":"Exponential time-series"}[x])
    linear_deg_per_day = st.number_input("Linear degradation slope per day", min_value=0.000001, value=float(st.session_state.get("linear_deg_per_day", 0.00012)), step=0.00001, format="%.6f")
    exp_deg_rate_per_day = st.number_input("Exponential degradation rate per day", min_value=0.000001, value=float(st.session_state.get("exp_deg_rate_per_day", 0.00018)), step=0.00001, format="%.6f")

bldg = BuildingSpec(
    building_type=building_type, location=location, conditioned_area_m2=float(area_m2), floors=int(floors), n_spaces=int(n_spaces),
    occupancy_density_p_m2=float(occupancy_density), lighting_w_m2=float(lighting_w_m2), equipment_w_m2=float(equipment_w_m2),
    airflow_m3h_m2=float(airflow_m3h_m2), infiltration_ach=float(infiltration_ach), sensible_w_per_person=float(sensible_w_per_person),
    cooling_intensity_w_m2=float(cooling_w_m2), heating_intensity_w_m2=float(heating_w_m2),
    wall_u=float(wall_u), roof_u=float(roof_u), window_u=float(window_u), shgc=float(shgc), glazing_ratio=float(glazing_ratio),
)
cfg = HVACConfig(
    years=int(years), hvac_system_type=hvac_system_type, COP_AGING_RATE=float(cop_aging_rate), RF_STAR=float(rf_star), B_FOUL=float(b_foul),
    DUST_RATE=float(dust_rate), K_CLOG=float(k_clog), DEG_TRIGGER=float(deg_trigger), degradation_model=degradation_model,
    LINEAR_DEG_PER_DAY=float(linear_deg_per_day), EXP_DEG_RATE_PER_DAY=float(exp_deg_rate_per_day),
)

st.markdown("<div class='small-muted'>Use the tabs below for model execution, direct weather upload, validation, sensitivity/benchmarking, zone tables, surrogate modelling, and exports.</div>", unsafe_allow_html=True)
tabs = st.tabs(["Scenario Modeling", "Extra UI Tools", "KPI Charts", "Surrogate Train / Predict", "Exports", "Guide"])

with tabs[0]:
    st.subheader("Scenario modeling")
    c1, c2, c3, c4 = st.columns(4)
    axis_mode = c1.selectbox("Modeling level", ["one_severity", "one_strategy", "two_axis", "three_axis"], format_func=lambda x: {"one_severity":"One-axis severity", "one_strategy":"One-axis strategy S0–S3", "two_axis":"Strategy × severity", "three_axis":"Strategy × severity × climate"}[x])
    fixed_strategy = c2.selectbox("Strategy selection", list(SCENARIOS.keys()), index=3, format_func=lambda x: f"{x} — {SCENARIOS[x]}")
    fixed_severity = c3.selectbox("Fixed severity", list(SEVERITY_LEVELS.keys()), index=1)
    fixed_climate = c4.selectbox("Fixed climate", list(CLIMATE_LEVELS.keys()), index=0)

    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    weather_mode_ui = c1.selectbox("Weather source", ["synthetic", "upload_csv_epw", "epw_path", "csv_path"], format_func=lambda x: {"synthetic":"Synthetic daily weather", "upload_csv_epw":"Upload CSV/EPW directly", "epw_path":"EPW path", "csv_path":"CSV path"}[x])
    random_state = int(c2.number_input("Random state", min_value=1, value=42, step=1))
    out_dir = c3.text_input("Output folder", "v3_run")

    uploaded_weather = None
    weather_df = None
    epw_path = None
    csv_path = None
    if weather_mode_ui == "upload_csv_epw":
        uploaded_weather = st.file_uploader("Upload weather file (.csv or .epw)", type=["csv", "epw", "txt"])
        if uploaded_weather is not None:
            try:
                weather_df = read_weather_upload(uploaded_weather)
                st.session_state["uploaded_weather_df"] = weather_df
                st.success(f"Weather upload parsed successfully: {len(weather_df)} daily records")
                st.dataframe(weather_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Weather upload error: {e}")
    elif weather_mode_ui == "epw_path":
        epw_path = st.text_input("EPW file path", "")
    elif weather_mode_ui == "csv_path":
        csv_path = st.text_input("CSV weather file path", "")

    include_baseline_layer = st.checkbox("Export baseline no-degradation layer", value=True)
    use_zone_occ = st.checkbox("Use zone-specific occupancy input", value=False)
    zone_df = None
    if use_zone_occ:
        zone_df = st.data_editor(default_zone_table(), num_rows="dynamic", use_container_width=True)

    if st.button("Run selected model", type="primary"):
        try:
            if weather_mode_ui == "upload_csv_epw":
                weather_df = st.session_state.get("uploaded_weather_df")
                if weather_df is None:
                    st.warning("Upload a CSV/EPW weather file first.")
                    st.stop()
                engine_weather_mode = "uploaded"
            elif weather_mode_ui == "epw_path":
                engine_weather_mode = "epw"
            elif weather_mode_ui == "csv_path":
                engine_weather_mode = "csv"
            else:
                engine_weather_mode = "synthetic"

            result = run_scenario_model(
                output_dir=out_dir,
                axis_mode=axis_mode,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode,
                epw_path=epw_path if epw_path else None,
                csv_path=csv_path if csv_path else None,
                weather_df=weather_df,
                fixed_strategy=fixed_strategy,
                fixed_severity=fixed_severity,
                fixed_climate=fixed_climate,
                zone_df=zone_df,
                random_state=random_state,
                include_baseline_layer=include_baseline_layer,
                degradation_model=degradation_model,
            )
            st.session_state["last_result"] = result
            st.session_state["last_result_dir"] = out_dir
            st.session_state["last_zone_df"] = zone_df

            tables = build_detailed_tables(out_dir, bldg=bldg, cfg=cfg, zone_df=zone_df)
            detailed_paths = save_detailed_outputs(out_dir, tables)
            st.session_state["last_detailed_paths"] = detailed_paths
            st.success("Model run and detailed outputs finished.")
            st.json({**result, "extra_detailed_outputs": detailed_paths})
            summary_path = Path(result["summary_csv"])
            if summary_path.exists():
                st.dataframe(pd.read_csv(summary_path), use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[1]:
    st.subheader("Extra UI tools: validation, benchmark sensitivity, zone tables, upload handling")
    target_folder = st.text_input("Result folder for extra tools", st.session_state.get("last_result_dir", "v3_run"), key="extra_folder")
    paths = find_result_paths(target_folder)
    if paths["summary"].exists():
        summary_df = pd.read_csv(paths["summary"])
        st.markdown("### Validation upload")
        vfile = st.file_uploader("Upload validation CSV from DesignBuilder, EnergyPlus, measured data, or published reference", type=["csv"], key="validation_file")
        if vfile is not None:
            validation_df = load_validation_file(vfile)
            comparison = build_validation_comparison(summary_df, validation_df, source_name=Path(vfile.name).stem)
            comparison_path = Path(target_folder) / "validation_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            st.dataframe(comparison, use_container_width=True)
            download_file_button(comparison_path, "Download validation_comparison.csv")

        st.markdown("### Benchmark / sensitivity summary")
        if (Path(target_folder) / "benchmark_summary.csv").exists():
            bench = pd.read_csv(Path(target_folder) / "benchmark_summary.csv")
        else:
            tables = build_detailed_tables(target_folder, bldg=bldg, cfg=cfg, zone_df=st.session_state.get("last_zone_df"))
            save_detailed_outputs(target_folder, tables)
            bench = tables["benchmark_summary"]
        st.dataframe(bench, use_container_width=True)
        if len(bench) and "energy_delta_pct" in bench.columns:
            st.bar_chart(bench.set_index("scenario_combo_3axis")["energy_delta_pct"])

        st.markdown("### Zone analysis")
        zone_path = Path(target_folder) / "zone_analysis.csv"
        if zone_path.exists():
            zdf = pd.read_csv(zone_path)
            st.dataframe(zdf.head(300), use_container_width=True)
            download_file_button(zone_path, "Download zone_analysis.csv")
        else:
            st.info("Run the model with zone-specific occupancy enabled to generate zone analysis.")
    else:
        st.info("Run a model first, or type an existing result folder.")

with tabs[2]:
    st.subheader("KPI charts")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "v3_run"), key="kpi_folder"))
    if folder.exists():
        paths = find_result_paths(folder)
        if paths["summary"].exists():
            kpi = pd.read_csv(paths["summary"])
            st.dataframe(kpi, use_container_width=True)
            for metric in ["Total Energy MWh", "Mean Degradation Index", "Mean Comfort Deviation C", "Total CO2 tonne"]:
                if metric in kpi.columns:
                    st.line_chart(kpi.set_index("scenario_combo_3axis")[metric])
        figs = folder / "figures"
        if figs.exists():
            img_files = sorted(figs.glob("*.png"))[:24]
            cols = st.columns(2)
            for i, img in enumerate(img_files):
                with cols[i % 2]:
                    st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No result folder found yet.")

with tabs[3]:
    st.subheader("Train CatBoost surrogate")
    dataset_path = st.text_input("Input dataset CSV", str(Path(st.session_state.get("last_result_dir", "v3_run")) / "matrix_ml_dataset.csv"))
    surrogate_out = st.text_input("Surrogate output folder", "v3_surrogate")
    n_iter_search = int(st.number_input("CatBoost search iterations", min_value=2, value=6, step=1))
    shap_sample = int(st.number_input("SHAP sample size", min_value=100, value=1000, step=100))
    if st.button("Train CatBoost surrogate"):
        try:
            result = train_surrogate_models(dataset_path, surrogate_out, n_iter_search, shap_sample, int(42))
            st.success("Surrogate training finished.")
            st.json(result)
            p = Path(result["metrics_csv"])
            if p.exists():
                st.dataframe(pd.read_csv(p), use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[4]:
    st.subheader("Exports and results")
    folder = Path(st.text_input("Folder to inspect/export", st.session_state.get("last_result_dir", "v3_run"), key="export_folder"))
    if folder.exists():
        csvs = sorted(folder.glob("*.csv"))
        st.write(f"CSV files found: {len(csvs)}")
        for csvf in csvs[:16]:
            with st.expander(csvf.name):
                try:
                    st.dataframe(pd.read_csv(csvf).head(80), use_container_width=True)
                except Exception as e:
                    st.warning(str(e))
                download_file_button(csvf, f"Download {csvf.name}", key=f"download_{csvf.name}")
        for special in ["results_export.xlsx", "detailed_outputs.xlsx", "results_report.pdf", "surrogate_export.xlsx", "surrogate_report.pdf"]:
            download_file_button(folder / special, f"Download {special}", key=f"download_{special}")
        if st.button("Create ZIP bundle for this run"):
            zip_path = create_zip_from_folder(folder)
            st.success(f"ZIP created: {zip_path}")
            download_file_button(zip_path, "Download ZIP bundle")
    else:
        st.info("No folder found yet.")

with tabs[5]:
    st.subheader("Deployment guide")
    st.markdown(
        """
        **Run locally**

        ```bash
        pip install -r requirements.txt
        streamlit run streamlit_app.py
        ```

        **Recommended publication workflow**

        1. Define building/HVAC/degradation inputs from the setup icon.  
        2. Upload EPW or CSV weather, or use the synthetic daily weather generator.  
        3. Select S0, S1, S2, or S3 through the strategy selector.  
        4. Run one-axis, two-axis, or three-axis scenario modelling.  
        5. Use Extra UI Tools for validation upload, benchmark comparison, and zone analysis.  
        6. Export CSV, Excel, PDF, figures, and ZIP bundle.  
        7. Train the CatBoost surrogate using `matrix_ml_dataset.csv` or `three_axis_ml_dataset.csv`.
        """
    )
