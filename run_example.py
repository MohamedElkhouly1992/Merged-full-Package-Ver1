from pathlib import Path
import pandas as pd

from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model
from report_addons import build_detailed_tables, save_detailed_outputs, create_zip_from_folder

out = Path("example_run")
bldg = BuildingSpec(
    building_type="Educational / University building",
    location="Example CSV weather",
    conditioned_area_m2=1000.0,
    floors=2,
    n_spaces=12,
    occupancy_density_p_m2=0.08,
)
cfg = HVACConfig(years=1, hvac_system_type="Chiller_AHU", degradation_model="physics")
zone_df = pd.DataFrame([
    {"zone_name": "Lecture", "zone_type": "Lecture", "area_m2": 500.0, "occ_density": 0.12, "term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
    {"zone_name": "Office", "zone_type": "Office", "area_m2": 300.0, "occ_density": 0.05, "term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
    {"zone_name": "Service", "zone_type": "Service", "area_m2": 200.0, "occ_density": 0.02, "term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
])

result = run_scenario_model(
    output_dir=out,
    axis_mode="one_strategy",
    bldg=bldg,
    cfg=cfg,
    weather_mode="csv",
    csv_path="examples/sample_daily_weather.csv",
    fixed_severity="Moderate",
    fixed_climate="C0_Baseline",
    zone_df=zone_df,
    include_baseline_layer=True,
    degradation_model="physics",
)
tables = build_detailed_tables(out, bldg=bldg, cfg=cfg, zone_df=zone_df)
save_detailed_outputs(out, tables)
zip_path = create_zip_from_folder(out, "example_run_bundle.zip")
print("Run complete")
print(result)
print("ZIP:", zip_path)
