"""Post-processing and UI helper tools for HVAC_ROM_Degradation_Suite.

These functions deliberately do not duplicate the HVAC model. They parse files, call
or consume outputs from hvac_v3_engine.py, and generate validation/benchmark/zone
reports from the engine results.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from hvac_v3_engine import BuildingSpec, HVACConfig, ensure_365_daily_weather, read_epw_daily


def infer_col(cols, candidates):
    mapping = {str(c).strip().lower().replace("_", " ").replace("-", " "): c for c in cols}
    for cand in candidates:
        key = str(cand).strip().lower().replace("_", " ").replace("-", " ")
        if key in mapping:
            return mapping[key]
    for c in cols:
        low = str(c).strip().lower().replace("_", " ").replace("-", " ")
        for cand in candidates:
            key = str(cand).strip().lower().replace("_", " ").replace("-", " ")
            if key in low or low in key:
                return c
    return None


def read_csv_fallback(file_obj) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception:
            pass
    raise ValueError("Could not read CSV weather/validation file with common encodings.")


def read_epw_upload_to_daily(uploaded_file) -> pd.DataFrame:
    """Parse a Streamlit UploadedFile EPW into engine daily weather."""
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    rows = []
    for line in content[8:]:
        parts = line.split(",")
        if len(parts) < 14:
            continue
        try:
            year = int(float(parts[0])); month = int(float(parts[1])); day = int(float(parts[2])); hour = int(float(parts[3]))
            dry = float(parts[6]); rh = float(parts[8]); ghi = float(parts[13])
        except Exception:
            continue
        ts = pd.Timestamp(year=year, month=month, day=day, hour=max(min(hour - 1, 23), 0))
        rows.append({"Date/Time": ts, "T_amb_C": dry, "RH_pct": rh, "GHI_Wm2": ghi})
    if not rows:
        raise ValueError("No valid weather rows parsed from EPW upload.")
    return aggregate_weather_upload_to_daily(pd.DataFrame(rows))


def aggregate_weather_upload_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_col = infer_col(out.columns, ["Date/Time", "date", "datetime", "timestamp", "time"])
    temp_col = infer_col(out.columns, ["T_amb_C", "Outdoor Dry-Bulb Temperature", "Outside Dry-Bulb Temperature", "temperature", "temp", "DryBulb"])
    rh_col = infer_col(out.columns, ["RH_pct", "RH", "Relative Humidity", "humidity"])
    ghi_col = infer_col(out.columns, ["GHI_Wm2", "GHI", "Global Solar Radiation", "Global Horizontal Solar", "solar"])
    dni_col = infer_col(out.columns, ["DNI", "Direct Normal Solar", "Direct Normal Radiation"])
    dhi_col = infer_col(out.columns, ["DHI", "Diffuse Horizontal Solar", "Diffuse Radiation"])

    if {"day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"}.issubset(out.columns):
        return ensure_365_daily_weather(out)

    if date_col is None or temp_col is None:
        raise ValueError("Weather CSV must contain date/time and dry-bulb temperature columns, or the five engine daily weather columns.")

    temp = pd.to_numeric(out[temp_col], errors="coerce")
    rh = pd.to_numeric(out[rh_col], errors="coerce") if rh_col else pd.Series([60.0] * len(out))
    if ghi_col:
        ghi = pd.to_numeric(out[ghi_col], errors="coerce")
    else:
        ghi = pd.Series([0.0] * len(out), dtype=float)
        if dni_col:
            ghi = ghi.add(pd.to_numeric(out[dni_col], errors="coerce").fillna(0), fill_value=0)
        if dhi_col:
            ghi = ghi.add(pd.to_numeric(out[dhi_col], errors="coerce").fillna(0), fill_value=0)

    work = pd.DataFrame({
        "Date/Time": pd.to_datetime(out[date_col], errors="coerce"),
        "temp": temp,
        "rh": rh,
        "ghi": ghi,
    }).dropna(subset=["Date/Time", "temp"])
    if len(work) == 0:
        raise ValueError("No valid weather rows remained after parsing.")
    work["date_only"] = work["Date/Time"].dt.floor("D")
    daily = work.groupby("date_only", as_index=False).agg(
        T_mean_C=("temp", "mean"),
        T_max_C=("temp", "max"),
        RH_mean_pct=("rh", "mean"),
        GHI_mean_Wm2=("ghi", "mean"),
    )
    daily["day_of_year"] = daily["date_only"].dt.dayofyear
    daily = daily[~((daily["date_only"].dt.month == 2) & (daily["date_only"].dt.day == 29))].copy()
    return ensure_365_daily_weather(daily[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]])


def read_weather_upload(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".epw":
        return read_epw_upload_to_daily(uploaded_file)
    if suffix in [".csv", ".txt"]:
        df = read_csv_fallback(uploaded_file)
        return aggregate_weather_upload_to_daily(df)
    raise ValueError("Unsupported weather file type. Upload .epw or .csv.")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def find_result_paths(result_dir: str | Path) -> Dict[str, Path]:
    p = Path(result_dir)
    summary_candidates = [
        p / "three_axis_summary.csv", p / "matrix_summary.csv",
        p / "one_axis_severity_summary.csv", p / "one_axis_strategy_summary.csv",
    ]
    annual_candidates = [
        p / "annual_three_axis.csv", p / "annual_matrix.csv",
        p / "annual_one_axis_severity.csv", p / "annual_one_axis_strategy.csv",
    ]
    dataset_candidates = [
        p / "three_axis_ml_dataset.csv", p / "matrix_ml_dataset.csv",
        p / "one_axis_severity_ml_dataset.csv", p / "one_axis_strategy_ml_dataset.csv",
    ]
    return {
        "summary": next((x for x in summary_candidates if x.exists()), summary_candidates[0]),
        "annual": next((x for x in annual_candidates if x.exists()), annual_candidates[0]),
        "daily": next((x for x in dataset_candidates if x.exists()), dataset_candidates[0]),
        "baseline_summary": p / "baseline_no_degradation_summary.csv",
        "baseline_daily": p / "baseline_no_degradation_daily.csv",
        "weather": p / "baseline_daily_weather.csv",
        "metadata": p / "run_metadata.json",
    }


def build_detailed_tables(
    result_dir: str | Path,
    bldg: Optional[BuildingSpec] = None,
    cfg: Optional[HVACConfig] = None,
    zone_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    paths = find_result_paths(result_dir)
    daily = safe_read_csv(paths["daily"])
    annual = safe_read_csv(paths["annual"])
    summary = safe_read_csv(paths["summary"])
    weather = safe_read_csv(paths["weather"])
    baseline_summary = safe_read_csv(paths["baseline_summary"])

    if len(daily) == 0:
        raise FileNotFoundError("No engine daily dataset found in the result folder.")

    total = pd.to_numeric(daily.get("energy_kwh_day", 0.0), errors="coerce").fillna(0.0)
    cop = pd.to_numeric(daily.get("COP_eff", 1.0), errors="coerce").replace(0, np.nan).fillna(1.0)
    q_hvac = pd.to_numeric(daily.get("Q_HVAC_kw", 0.0), errors="coerce").fillna(0.0)
    hvac_elec = (q_hvac * 24.0 / cop).clip(lower=0.0)
    fan_elec = (total - hvac_elec).clip(lower=0.0)
    mode = daily.get("mode", pd.Series([""] * len(daily))).astype(str)

    fuel_breakdown = pd.DataFrame({
        "scenario_combo_3axis": daily.get("scenario_combo_3axis", ""),
        "strategy": daily.get("strategy", ""),
        "severity": daily.get("severity", ""),
        "climate": daily.get("climate", ""),
        "year": daily.get("year", ""),
        "day": daily.get("day", ""),
        "mode": mode,
        "cooling_electricity_kwh_day": hvac_elec.where(mode == "cooling", 0.0),
        "heating_electricity_kwh_day": hvac_elec.where(mode == "heating", 0.0),
        "fan_electricity_kwh_day": fan_elec,
        "gas_kwh_day": 0.0,
        "total_electricity_kwh_day": total,
        "total_energy_kwh_day": total,
        "co2_kg_day": daily.get("co2_kg_day", 0.0),
    })

    comfort_cols = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year", "day", "day_of_year", "occ", "T_sp_C", "alpha_flow", "comfort_dev_C", "occupied_discomfort_flag", "delta", "COP_eff"] if c in daily.columns]
    comfort = daily[comfort_cols].copy()
    if "comfort_dev_C" in comfort.columns:
        comfort["comfort_class"] = np.select(
            [comfort["comfort_dev_C"] <= 0.3, comfort["comfort_dev_C"] <= 1.0, comfort["comfort_dev_C"] <= 2.0],
            ["within_target", "minor_deviation", "moderate_deviation"], default="high_deviation")

    site_data = weather.copy()
    if bldg is not None:
        site_data["building_type"] = bldg.building_type
        site_data["location"] = bldg.location
        site_data["conditioned_area_m2"] = bldg.conditioned_area_m2
        site_data["floors"] = bldg.floors
        site_data["n_spaces"] = bldg.n_spaces
    if cfg is not None:
        site_data["hvac_system_type"] = cfg.hvac_system_type

    internal_cols = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year", "day", "day_of_year", "occ", "people_count", "sensible_people_kw", "internal_gains_kw", "Q_cool_kw", "Q_heat_kw", "Q_HVAC_kw"] if c in daily.columns]
    internal_gains = daily[internal_cols].copy()

    validation_template = summary.copy()
    for col in ["Reference Source", "Reference Energy MWh", "Reference CO2 tonne", "Reference Comfort C", "Energy Error %", "CO2 Error %", "Comfort Error %"]:
        if col not in validation_template.columns:
            validation_template[col] = np.nan

    benchmark_rows = []
    if len(baseline_summary) and "Total Energy MWh" in baseline_summary.columns:
        ref = baseline_summary.iloc[0]
        ref_energy = float(ref.get("Total Energy MWh", np.nan))
        ref_co2 = float(ref.get("Total CO2 tonne", np.nan))
        ref_comfort = float(ref.get("Mean Comfort Deviation C", np.nan))
        for _, row in summary.iterrows():
            energy = float(row.get("Total Energy MWh", np.nan))
            co2 = float(row.get("Total CO2 tonne", np.nan))
            comfort_val = float(row.get("Mean Comfort Deviation C", np.nan))
            benchmark_rows.append({
                "scenario_combo_3axis": row.get("scenario_combo_3axis", ""),
                "benchmark_reference": "baseline_no_degradation",
                "energy_delta_MWh": energy - ref_energy,
                "energy_delta_pct": ((energy - ref_energy) / ref_energy * 100.0) if ref_energy else np.nan,
                "co2_delta_tonne": co2 - ref_co2,
                "co2_delta_pct": ((co2 - ref_co2) / ref_co2 * 100.0) if ref_co2 else np.nan,
                "comfort_delta_C": comfort_val - ref_comfort,
            })
    benchmark_summary = pd.DataFrame(benchmark_rows)

    zone_analysis = build_zone_analysis_from_daily(daily, zone_df) if zone_df is not None and len(zone_df) else pd.DataFrame({"note": ["No zone table supplied."]})

    kpi_summary = summary[[c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "Total Energy MWh", "Total Cost USD", "Total CO2 tonne", "Mean COP", "Mean Degradation Index", "Mean Comfort Deviation C", "Occupied Discomfort Days", "Filter Replacements count", "HX Cleanings count"] if c in summary.columns]].copy()

    return {
        "fuel_breakdown": fuel_breakdown,
        "comfort": comfort,
        "site_data": site_data,
        "internal_gains": internal_gains,
        "validation_template": validation_template,
        "benchmark_summary": benchmark_summary,
        "zone_analysis": zone_analysis,
        "kpi_summary": kpi_summary,
    }


def build_zone_analysis_from_daily(daily: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    z = zone_df.copy()
    for col in ["zone_name", "zone_type", "area_m2", "occ_density"]:
        if col not in z.columns:
            if col == "zone_name": z[col] = [f"Zone_{i+1}" for i in range(len(z))]
            elif col == "zone_type": z[col] = "Custom"
            else: z[col] = 0.0
    z["area_m2"] = pd.to_numeric(z["area_m2"], errors="coerce").fillna(0.0)
    z["occ_density"] = pd.to_numeric(z["occ_density"], errors="coerce").fillna(0.0)
    total_area = max(float(z["area_m2"].sum()), 1e-9)
    area_weights = z["area_m2"] / total_area
    rows = []
    base_cols = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year", "day", "day_of_year", "energy_kwh_day", "co2_kg_day", "comfort_dev_C", "delta"] if c in daily.columns]
    for i, zone in z.reset_index(drop=True).iterrows():
        part = daily[base_cols].copy()
        w = float(area_weights.iloc[i])
        part["zone_name"] = zone["zone_name"]
        part["zone_type"] = zone["zone_type"]
        part["zone_area_m2"] = float(zone["area_m2"])
        part["zone_occ_density"] = float(zone["occ_density"])
        part["zone_energy_kwh_day"] = pd.to_numeric(daily["energy_kwh_day"], errors="coerce").fillna(0.0) * w
        part["zone_co2_kg_day"] = pd.to_numeric(daily["co2_kg_day"], errors="coerce").fillna(0.0) * w
        part["zone_comfort_dev_C"] = pd.to_numeric(daily["comfort_dev_C"], errors="coerce").fillna(0.0) * (0.95 + 0.10 * w)
        part["zone_degradation_index"] = pd.to_numeric(daily["delta"], errors="coerce").fillna(0.0)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_detailed_outputs(result_dir: str | Path, tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = str(path)

    with pd.ExcelWriter(out / "detailed_outputs.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.head(100000).to_excel(writer, sheet_name=name[:31], index=False)
    paths["detailed_excel"] = str(out / "detailed_outputs.xlsx")
    return paths


def load_validation_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return read_csv_fallback(uploaded_file)


def build_validation_comparison(summary_df: pd.DataFrame, validation_df: pd.DataFrame, source_name: str = "External") -> pd.DataFrame:
    """Compare model summary with external values. Flexible column matching is used."""
    if validation_df is None or len(validation_df) == 0:
        return pd.DataFrame()
    val = validation_df.copy()
    scen_col = infer_col(val.columns, ["scenario_combo_3axis", "Scenario Key", "scenario", "case"])
    energy_col = infer_col(val.columns, ["Total Energy MWh", "Energy MWh", "Total HVAC Energy (kWh)", "Energy Consumption (kWh)", "energy_kwh_day"])
    co2_col = infer_col(val.columns, ["Total CO2 tonne", "CO2 tonne", "Total CO2 Production (kg)", "Carbon Footprint (kgCO2)"])
    comfort_col = infer_col(val.columns, ["Mean Comfort Deviation C", "Comfort Deviation", "Comfort Deviation Mean (C)"])

    rows = []
    for _, m in summary_df.iterrows():
        scenario = m.get("scenario_combo_3axis", "")
        ref_rows = val
        if scen_col is not None:
            matching = val[val[scen_col].astype(str) == str(scenario)]
            if len(matching):
                ref_rows = matching
        ref = ref_rows.iloc[0]
        ref_energy = np.nan
        if energy_col is not None:
            ref_energy = float(ref[energy_col])
            if "kWh" in str(energy_col):
                ref_energy = ref_energy / 1000.0
        ref_co2 = np.nan
        if co2_col is not None:
            ref_co2 = float(ref[co2_col])
            if "kg" in str(co2_col).lower():
                ref_co2 = ref_co2 / 1000.0
        ref_comfort = float(ref[comfort_col]) if comfort_col is not None else np.nan
        model_energy = float(m.get("Total Energy MWh", np.nan))
        model_co2 = float(m.get("Total CO2 tonne", np.nan))
        model_comfort = float(m.get("Mean Comfort Deviation C", np.nan))
        rows.append({
            "scenario_combo_3axis": scenario,
            "Reference Source": source_name,
            "Model Energy MWh": model_energy,
            "Reference Energy MWh": ref_energy,
            "Energy Error %": ((model_energy - ref_energy) / ref_energy * 100.0) if ref_energy else np.nan,
            "Model CO2 tonne": model_co2,
            "Reference CO2 tonne": ref_co2,
            "CO2 Error %": ((model_co2 - ref_co2) / ref_co2 * 100.0) if ref_co2 else np.nan,
            "Model Comfort C": model_comfort,
            "Reference Comfort C": ref_comfort,
            "Comfort Error %": ((model_comfort - ref_comfort) / ref_comfort * 100.0) if ref_comfort else np.nan,
        })
    return pd.DataFrame(rows)


def create_zip_from_folder(folder: str | Path, zip_name: Optional[str] = None) -> str:
    folder = Path(folder)
    zip_path = folder.parent / (zip_name or f"{folder.name}_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            if path.is_file() and path != zip_path:
                zf.write(path, arcname=path.relative_to(folder))
    return str(zip_path)
