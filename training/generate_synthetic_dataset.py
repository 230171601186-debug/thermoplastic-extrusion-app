from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 5000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_PATH = DATASET_DIR / "Thermoplastic_Extrusion_5000Rows_ML_Final_Dataset.csv"

MATERIALS = ["PA12", "PE", "TPU", "PVC", "PP"]
GRADE_LEVELS = ["G1", "G2", "G3"]
TUBE_TYPES = ["Composite", "Industrial", "Pneumatic", "Hydraulic", "PTFE"]
BRAID_TYPES = ["None", "1-Wire", "2-Wire", "Textile", "Helical"]
MACHINE_IDS = ["EXT-01", "EXT-02", "EXT-03", "EXT-04", "EXT-05"]
SHIFTS = ["A", "B", "C"]
FAULTS = [
    "No Fault",
    "Machine Cleaning",
    "Start Up Delay",
    "Carrier Problem",
    "Power Cut",
    "Material Scorch",
    "Pressure Instability",
    "Cooling Failure",
    "Machine Breakdown",
    "High Throughput Stress",
    "Dimension Variation",
    "Minor Dimension Variation",
]

MATERIAL_PROFILE = {
    "PA12": {"temp": 214.0, "pressure": 140.0, "flow": 1.15, "rpm": 8.0, "speed": 1.35, "cool": 18.0},
    "PE": {"temp": 205.0, "pressure": 118.0, "flow": 0.98, "rpm": 6.5, "speed": 1.15, "cool": 17.2},
    "TPU": {"temp": 210.0, "pressure": 131.0, "flow": 1.08, "rpm": 7.4, "speed": 1.27, "cool": 17.6},
    "PVC": {"temp": 220.0, "pressure": 150.0, "flow": 1.20, "rpm": 8.8, "speed": 1.45, "cool": 19.2},
    "PP": {"temp": 200.0, "pressure": 116.0, "flow": 0.94, "rpm": 6.0, "speed": 1.10, "cool": 16.9},
}
GRADE_PROFILE = {"G1": 0.0, "G2": 5.0, "G3": 10.0}
TUBE_PROFILE = {
    "Composite": {"difficulty": 1.0, "thermal": 2.0},
    "Industrial": {"difficulty": 0.9, "thermal": 1.8},
    "Pneumatic": {"difficulty": 1.2, "thermal": 1.9},
    "Hydraulic": {"difficulty": 1.1, "thermal": 2.1},
    "PTFE": {"difficulty": 1.5, "thermal": 2.8},
}
BRAID_PROFILE = {"None": 0.0, "1-Wire": 0.15, "2-Wire": 0.30, "Textile": 0.55, "Helical": 0.75}
MACHINE_PROFILE = {
    "EXT-01": {"speed_bias": 0.92, "temp_bias": 2.5, "pressure_bias": 1.00, "rpm_bias": 1.2},
    "EXT-02": {"speed_bias": 0.96, "temp_bias": 0.0, "pressure_bias": 0.98, "rpm_bias": 0.0},
    "EXT-03": {"speed_bias": 1.02, "temp_bias": -1.0, "pressure_bias": 1.00, "rpm_bias": 1.0},
    "EXT-04": {"speed_bias": 1.08, "temp_bias": -2.0, "pressure_bias": 1.04, "rpm_bias": 2.5},
    "EXT-05": {"speed_bias": 1.12, "temp_bias": 1.5, "pressure_bias": 1.06, "rpm_bias": 3.0},
}
FAULT_DOWNTIME = {
    "No Fault": (0.0, 0.0),
    "Machine Cleaning": (4.0, 18.0),
    "Start Up Delay": (5.0, 22.0),
    "Carrier Problem": (8.0, 30.0),
    "Power Cut": (10.0, 45.0),
    "Material Scorch": (11.0, 42.0),
    "Pressure Instability": (9.0, 46.0),
    "Cooling Failure": (16.0, 60.0),
    "Machine Breakdown": (40.0, 150.0),
    "High Throughput Stress": (8.0, 24.0),
    "Dimension Variation": (12.0, 50.0),
    "Minor Dimension Variation": (5.0, 16.0),
}


def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def choose_fault(rng, risk_score, risk_components, material, tube, shift, machine, difficulty):
    machine_risk = 0.12 if machine in ["EXT-01", "EXT-05"] else 0.0
    scores = {
        "Material Scorch": 0.08 + 2.4 * risk_components["thermal"] + 1.0 * risk_components["pressure"],
        "Pressure Instability": 0.08 + 2.6 * risk_components["pressure"],
        "Cooling Failure": 0.08 + 2.6 * risk_components["cooling"],
        "Carrier Problem": 0.08 + 2.8 * risk_components["mismatch"],
        "Machine Breakdown": 0.08 + 0.7 * risk_score + machine_risk,
        "High Throughput Stress": 0.08 + 2.8 * risk_components["throughput"],
        "Dimension Variation": 0.08 + 2.1 * risk_components["geometry"] + risk_components["mismatch"],
        "Minor Dimension Variation": 0.08 + 1.8 * risk_components["mismatch"] + 0.4 * risk_components["geometry"],
        "Machine Cleaning": 0.08 + 0.25 * (shift == "A") + 0.20 * (material == "PVC"),
        "Start Up Delay": 0.08 + 0.30 * (shift == "A") + 0.10 * (shift == "B"),
        "Power Cut": 0.08 + machine_risk + 0.6 * risk_components["cooling"],
    }
    # A stronger but still stochastic dependence on pre-production risk keeps the
    # process realistic while making efficiency meaningfully explainable from the
    # customer/product specification and intended process conditions.
    fault_probability = clip(0.03 + 1.15 * risk_score + rng.normal(0.0, 0.04), 0.01, 0.82)
    if rng.random() >= fault_probability:
        return "No Fault"
    fault_weights = np.array([scores[fault] for fault in FAULTS[1:]], dtype=float)
    fault_weights /= fault_weights.sum()
    return rng.choice(FAULTS[1:], p=fault_weights)


def make_row(i, rng):
    material = rng.choice(MATERIALS, p=[0.38, 0.22, 0.18, 0.12, 0.10])
    grade = rng.choice(GRADE_LEVELS, p=[0.40, 0.38, 0.22])
    tube = rng.choice(TUBE_TYPES, p=[0.18, 0.18, 0.24, 0.20, 0.20])
    braid = rng.choice(BRAID_TYPES, p=[0.16, 0.20, 0.16, 0.30, 0.18])
    machine = rng.choice(MACHINE_IDS, p=[0.22, 0.22, 0.18, 0.20, 0.18])
    shift = rng.choice(SHIFTS, p=[0.33, 0.33, 0.34])

    material_p = MATERIAL_PROFILE[material]
    machine_p = MACHINE_PROFILE[machine]
    tube_p = TUBE_PROFILE[tube]
    braid_difficulty = BRAID_PROFILE[braid]
    difficulty = tube_p["difficulty"] + 0.4 * braid_difficulty

    target_od = rng.uniform(7.0, 20.0)
    target_wall = rng.uniform(1.0, 4.8)
    target_id = target_od - 2.0 * target_wall
    if target_id <= 0.5:
        target_wall = (target_od - 0.8) / 2.0
        target_id = target_od - 2.0 * target_wall

    customer_length = np.exp(rng.normal(loc=math.log(5000.0), scale=0.55))
    customer_length = clip(customer_length, 900.0, 17000.0)
    planned_length = customer_length * rng.uniform(1.03, 1.16)

    ambient_temperature = rng.uniform(22.0, 35.0)
    humidity = rng.uniform(35.0, 82.0)

    # Process setpoints: product and geometry drive the center; noise keeps them imperfect.
    thermal_target = (
        material_p["temp"]
        + 0.55 * GRADE_PROFILE[grade]
        + 3.2 * tube_p["thermal"]
        + 2.2 * difficulty
        + machine_p["temp_bias"]
    )
    extruder_temp = thermal_target + rng.normal(0.0, 4.0)
    extruder_temp = clip(extruder_temp, 180.0, 245.0)

    die_temp = extruder_temp - 7.0 + 0.18 * (target_od - 13.5) + 0.8 * tube_p["thermal"] + rng.normal(0.0, 3.8)
    die_temp = clip(die_temp, 180.0, 235.0)

    pressure_target = (
        128.0
        + 0.38 * (material_p["pressure"] - 130.0)
        + 2.7 * (target_wall - 2.8)
        + 0.72 * (target_od - 13.5)
        + 4.0 * (grade == "G3")
        + 5.0 * (difficulty - 1.0)
        + 2.0 * (machine_p["pressure_bias"] - 1.0) * 10.0
    )
    melt_pressure = pressure_target + rng.normal(0.0, 5.5)
    melt_pressure = clip(melt_pressure, 90.0, 200.0)

    air_pressure = 5.2 + 0.62 * target_wall + 0.65 * difficulty + 0.48 * braid_difficulty + rng.normal(0.0, 0.65)
    air_pressure = clip(air_pressure, 3.0, 12.0)

    screw_rpm = (
        66.0
        + 4.0 * (grade == "G2")
        + 7.0 * (grade == "G3")
        + 7.0 * (difficulty - 1.0)
        + 3.0 * (target_wall - 2.8)
        + 0.50 * (target_od - 13.5)
        + 1.8 * (material_p["rpm"] - 7.0)
        + 4.5 * braid_difficulty
        + 1.4 * machine_p["rpm_bias"]
        + rng.normal(0.0, 3.8)
    )
    screw_rpm = clip(screw_rpm, 48.0, 100.0)

    line_speed = (
        14.0
        + 1.5 * (material_p["speed"] - 1.2)
        + 0.55 * (grade == "G2")
        + 1.0 * (grade == "G3")
        - 0.45 * (target_od - 13.5)
        - 0.85 * (target_wall - 2.8)
        - 1.0 * (difficulty - 1.0)
        + 0.00012 * (customer_length - 5000.0)
        + 2.0 * (machine_p["speed_bias"] - 1.0)
        + rng.normal(0.0, 0.9)
    )
    line_speed = clip(line_speed, 8.0, 24.0)

    cooling_target = (
        16.0
        + 0.75 * (material_p["cool"] - 17.5)
        + 1.35 * tube_p["thermal"]
        + 0.19 * (target_od - 13.5)
        + 0.18 * (line_speed - 13.5)
        + 0.06 * (ambient_temperature - 28.5)
        + 0.70 * braid_difficulty
    )
    cooling_temp = cooling_target + rng.normal(0.0, 0.90)
    cooling_temp = clip(cooling_temp, 12.0, 28.0)

    puller_speed = line_speed + 0.10 * braid_difficulty + rng.normal(0.0, 0.42)
    puller_speed = clip(puller_speed, 7.0, 25.0)

    extrusion_rate = (
        42.0
        + 0.78 * screw_rpm
        + 4.5 * (material_p["flow"] - 1.0)
        + 1.25 * (target_od - 13.5)
        + 3.5 * (target_wall - 2.8)
        + 5.0 * (tube_p["difficulty"] - 1.0)
        + 1.5 * machine_p["pressure_bias"]
        + rng.normal(0.0, 6.0)
    )
    extrusion_rate = clip(extrusion_rate, 25.0, 170.0)

    planned_time = clip(planned_length / max(line_speed, 8.0) * 1.10 + 35.0 + rng.normal(0.0, 12.0), 90.0, 900.0)

    line_speed = round(float(line_speed), 2)
    puller_speed = round(float(puller_speed), 2)
    extrusion_rate = round(float(extrusion_rate), 2)
    planned_time = round(float(planned_time), 2)

    geometry_risk = clip(
        0.35 * (target_wall - 1.0) / 3.8
        + 0.30 * (target_od - 7.0) / 13.0
        + 0.35 * (difficulty - 0.9) / 1.3,
        0.0,
        1.0,
    )
    risk_components = {
        "thermal": clip(
            0.75 * max(0.0, abs(thermal_target - (material_p["temp"] + 3.0)) - 7.0) / 20.0
            + 0.25 * abs(extruder_temp - thermal_target) / 12.0,
            0.0,
            1.0,
        ),
        "pressure": clip(
            0.75 * max(0.0, abs(pressure_target - 130.0) - 8.0) / 35.0
            + 0.25 * abs(melt_pressure - pressure_target) / 18.0,
            0.0,
            1.0,
        ),
        "cooling": clip(
            0.75 * abs(cooling_target - 18.5) / 6.0
            + 0.25 * abs(cooling_temp - cooling_target) / 3.0,
            0.0,
            1.0,
        ),
        "mismatch": clip(
            0.65 * geometry_risk + 0.35 * abs(line_speed - puller_speed) / 1.6,
            0.0,
            1.0,
        ),
        "throughput": clip(
            0.55 * max(0.0, screw_rpm - 72.0) / 22.0
            + 0.45 * max(0.0, line_speed - 14.0) / 8.0,
            0.0,
            1.0,
        ),
        "geometry": geometry_risk,
        "length": clip(math.log(customer_length / 900.0) / math.log(17000.0 / 900.0), 0.0, 1.0),
    }
    risk_score = clip(
        0.20 * risk_components["thermal"]
        + 0.18 * risk_components["pressure"]
        + 0.15 * risk_components["cooling"]
        + 0.15 * risk_components["mismatch"]
        + 0.13 * risk_components["throughput"]
        + 0.13 * risk_components["geometry"]
        + 0.06 * risk_components["length"],
        0.0,
        1.0,
    )

    # Fault occurrence and severity respond to pre-production risk, but remain stochastic.
    fault = choose_fault(rng, risk_score, risk_components, material, tube, shift, machine, difficulty)
    if fault == "Material Scorch":
        extruder_temp = clip(extruder_temp + rng.uniform(5.0, 13.0), 180.0, 245.0)
        melt_pressure = clip(melt_pressure + rng.uniform(3.0, 9.0), 90.0, 200.0)
    elif fault == "Pressure Instability":
        melt_pressure = clip(melt_pressure + rng.normal(0.0, 9.0), 90.0, 200.0)
    elif fault == "Cooling Failure":
        cooling_temp = clip(cooling_temp + rng.uniform(3.0, 7.0), 12.0, 28.0)
    elif fault in ["Carrier Problem", "Dimension Variation", "Minor Dimension Variation"]:
        puller_speed = clip(puller_speed + rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 2.0), 7.0, 25.0)
    elif fault == "High Throughput Stress":
        screw_rpm = clip(screw_rpm + rng.uniform(3.0, 8.0), 48.0, 100.0)
        line_speed = clip(line_speed + rng.uniform(0.5, 1.5), 8.0, 24.0)
    line_speed = round(float(line_speed), 2)
    puller_speed = round(float(puller_speed), 2)
    if fault == "No Fault":
        downtime_min = 0.0
    else:
        lo, hi = FAULT_DOWNTIME[fault]
        fault_severity_risk = {
            "Machine Cleaning": 0.35 * risk_components["geometry"] + 0.30 * risk_components["length"],
            "Start Up Delay": 0.40 * risk_components["throughput"] + 0.25 * risk_components["length"],
            "Carrier Problem": risk_components["mismatch"],
            "Power Cut": 0.50 * risk_components["cooling"] + 0.20 * risk_score,
            "Material Scorch": 0.70 * risk_components["thermal"] + 0.30 * risk_components["pressure"],
            "Pressure Instability": risk_components["pressure"],
            "Cooling Failure": risk_components["cooling"],
            "Machine Breakdown": 0.65 * risk_score + 0.35 * risk_components["geometry"],
            "High Throughput Stress": risk_components["throughput"],
            "Dimension Variation": 0.65 * risk_components["geometry"] + 0.35 * risk_components["mismatch"],
            "Minor Dimension Variation": 0.55 * risk_components["mismatch"] + 0.25 * risk_components["geometry"],
        }[fault]
        severity_fraction = clip(
            0.12 + 0.72 * risk_score + 0.18 * fault_severity_risk + rng.normal(0.0, 0.12),
            0.05,
            0.98,
        )
        downtime_min = lo + severity_fraction * (hi - lo)

    run_time = min(max(0.0, planned_time - downtime_min), planned_time)
    run_time = round(float(run_time), 2)
    actual_length = round(float(line_speed * run_time), 2)

    # Keep the existing production-accounting structure, but make the baseline
    # process risk materially affect expected throughput and scrap before the
    # stochastic fault/no-fault realization is applied.
    performance_risk_multiplier = clip(1.0 - 0.42 * risk_score + rng.normal(0.0, 0.06), 0.55, 1.10)

    # Quality / efficiency via realistic engineering logic.
    ideal_capacity = line_speed * planned_time * (1.02 + 0.06 * tube_p["difficulty"])
    performance = clip((actual_length / ideal_capacity) * performance_risk_multiplier, 0.50, 1.0)

    # Scrap risk from process abnormalities and fault conditions.
    thermal_deviation = max(0.0, abs(extruder_temp - thermal_target) - 3.0) * 0.0022
    pressure_deviation = max(0.0, abs(melt_pressure - 130.0) - 8.0) * 0.0018
    cooling_deviation = max(0.0, abs(cooling_temp - 18.5) - 1.5) * 0.007
    speed_mismatch = max(0.0, abs(line_speed - puller_speed) - 0.2) * 0.025
    rpm_deviation = max(0.0, abs(screw_rpm - 72.0) - 9.0) * 0.002
    geometry_effect = 0.006 + max(0.0, target_wall - 2.8) * 0.008 + max(0.0, target_od - 13.5) * 0.003 + max(0.0, difficulty - 1.0) * 0.012
    material_effect = 0.006 + (material in ["PVC", "TPU"]) * 0.012 + (material == "PA12") * 0.006 + (grade == "G3") * 0.005
    braid_effect = 0.011 * braid_difficulty
    fault_effect = {
        "No Fault": 0.0,
        "Machine Cleaning": 0.010,
        "Start Up Delay": 0.012,
        "Carrier Problem": 0.025,
        "Power Cut": 0.015,
        "Material Scorch": 0.045,
        "Pressure Instability": 0.038,
        "Cooling Failure": 0.040,
        "Machine Breakdown": 0.030,
        "High Throughput Stress": 0.032,
        "Dimension Variation": 0.030,
        "Minor Dimension Variation": 0.018,
    }[fault]

    scrap_rate = round(
        clip(
            0.010
            + 0.045 * risk_score
            + thermal_deviation
            + pressure_deviation
            + cooling_deviation
            + speed_mismatch
            + rpm_deviation
            + geometry_effect
            + material_effect
            + braid_effect
            + fault_effect
            + rng.normal(0.0, 0.006),
            0.002,
            0.16,
        ),
        5,
    )
    scrap_length = actual_length * scrap_rate
    good_length = actual_length - scrap_length
    if good_length < 0:
        scrap_length = actual_length * 0.18
        good_length = actual_length - scrap_length
    scrap_length = round(float(scrap_length), 2)
    good_length = round(float(good_length), 2)
    actual_length = round(float(actual_length), 2)

    quality = clip(float(good_length / actual_length), 0.0, 1.0) if actual_length > 0 else 1.0
    availability = clip(float(run_time / planned_time), 0.0, 1.0) if planned_time > 0 else 1.0
    oee = clip(float(availability * performance * quality), 0.0, 1.0)
    operational_efficiency_pct = clip(float(oee * 100.0), 0.0, 100.0)

    shortfall_m = max(0.0, planned_length - actual_length)
    order_status = "Completed" if shortfall_m <= 0.12 * planned_length else "Delayed"
    if order_status == "Delayed":
        shortfall_m = max(shortfall_m, 0.04 * planned_length)

    power_consumption = 150.0 + 1.5 * extrusion_rate + 0.9 * melt_pressure + 0.12 * extruder_temp + rng.normal(0.0, 18.0)
    power_consumption = clip(power_consumption, 80.0, 1000.0)

    return {
        "Record_ID": i,
        "Material_Type": material,
        "Material_Grade": grade,
        "Tube_Type": tube,
        "Braid_Type": braid,
        "Customer_Required_Length_m": round(float(customer_length), 2),
        "Target_Tube_OD_mm": round(float(target_od), 2),
        "Target_Tube_Wall_mm": round(float(target_wall), 2),
        "Target_Tube_ID_mm": round(float(target_id), 2),
        "Machine_ID": machine,
        "Shift": shift,
        "Ambient_Temperature_C": round(float(ambient_temperature), 2),
        "Humidity_percent": round(float(humidity), 2),
        "Planned_Scrap_Allowance_pct": round(float(rng.uniform(1.0, 4.0)), 2),
        "Planned_Processing_Length_m": round(float(planned_length), 2),
        "Planned_Production_Time_min": round(float(planned_time), 2),
        "Downtime_min": round(float(downtime_min), 2),
        "Run_Time_min": round(float(run_time), 2),
        "Extruder_Temperature_C": round(float(extruder_temp), 2),
        "Die_Temperature_C": round(float(die_temp), 2),
        "Cooling_Tank_Temperature_C": round(float(cooling_temp), 2),
        "Melt_Pressure_bar": round(float(melt_pressure), 2),
        "Air_Pressure_bar": round(float(air_pressure), 2),
        "Screw_RPM": round(float(screw_rpm), 2),
        "Line_Speed_m_min": round(float(line_speed), 2),
        "Puller_Speed_m_min": round(float(puller_speed), 2),
        "Extrusion_Speed_kg_hr": round(float(extrusion_rate), 2),
        "Power_Consumption_kWh": round(float(power_consumption), 2),
        "Actual_Processing_Length_m": round(float(actual_length), 2),
        "Scrap_Length_m": round(float(scrap_length), 2),
        "Good_Length_m": round(float(good_length), 2),
        "Length_Shortfall_m": round(float(shortfall_m), 2),
        "Order_Status": order_status,
        "Scrap_Rate": round(float(scrap_rate), 5),
        "Availability": round(float(availability), 6),
        "Performance": round(float(performance), 6),
        "Quality": round(float(quality), 6),
        "OEE": round(float(oee), 6),
        "Operational_Efficiency_pct": round(float(operational_efficiency_pct), 4),
        "Fault_Code": fault,
    }


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = [make_row(i + 1, rng) for i in range(N_ROWS)]
    df = pd.DataFrame(rows)

    columns = [
        "Record_ID",
        "Material_Type",
        "Material_Grade",
        "Tube_Type",
        "Braid_Type",
        "Customer_Required_Length_m",
        "Target_Tube_OD_mm",
        "Target_Tube_Wall_mm",
        "Target_Tube_ID_mm",
        "Machine_ID",
        "Shift",
        "Ambient_Temperature_C",
        "Humidity_percent",
        "Planned_Scrap_Allowance_pct",
        "Planned_Processing_Length_m",
        "Planned_Production_Time_min",
        "Downtime_min",
        "Run_Time_min",
        "Extruder_Temperature_C",
        "Die_Temperature_C",
        "Cooling_Tank_Temperature_C",
        "Melt_Pressure_bar",
        "Air_Pressure_bar",
        "Screw_RPM",
        "Line_Speed_m_min",
        "Puller_Speed_m_min",
        "Extrusion_Speed_kg_hr",
        "Power_Consumption_kWh",
        "Actual_Processing_Length_m",
        "Scrap_Length_m",
        "Good_Length_m",
        "Length_Shortfall_m",
        "Order_Status",
        "Scrap_Rate",
        "Availability",
        "Performance",
        "Quality",
        "OEE",
        "Operational_Efficiency_pct",
        "Fault_Code",
    ]
    df = df[columns]

    assert len(df) == N_ROWS, f"Expected {N_ROWS} rows, got {len(df)}"
    assert len(df.columns) == 40, f"Expected 40 columns, got {len(df.columns)}"
    assert df.isna().sum().sum() == 0, "Missing values present"
    assert not df.duplicated(subset=["Record_ID"]).any(), "Duplicate Record_ID values found"
    expected_categories = {
        "Material_Type": MATERIALS,
        "Material_Grade": GRADE_LEVELS,
        "Tube_Type": TUBE_TYPES,
        "Braid_Type": BRAID_TYPES,
        "Machine_ID": MACHINE_IDS,
        "Shift": SHIFTS,
        "Fault_Code": FAULTS,
    }
    for column, categories in expected_categories.items():
        assert set(df[column]).issubset(categories), f"Unexpected category in {column}"
    assert (df["Target_Tube_ID_mm"] - (df["Target_Tube_OD_mm"] - 2.0 * df["Target_Tube_Wall_mm"])).abs().max() < 0.25, "Tube geometry not consistent"
    assert (df["Customer_Required_Length_m"] > 0).all(), "Customer length must be positive"
    assert (df["Planned_Processing_Length_m"] > 0).all(), "Planned processing length must be positive"
    assert (df["Downtime_min"] >= 0).all(), "Negative downtime found"
    assert (df["Run_Time_min"] >= 0).all(), "Negative runtime found"
    assert (df["Run_Time_min"] <= df["Planned_Production_Time_min"]).all(), "Runtime exceeds planned time"
    assert (df["Actual_Processing_Length_m"] >= 0).all(), "Actual length is negative"
    assert (df["Scrap_Length_m"] >= 0).all(), "Scrap length negative"
    assert (df["Good_Length_m"] >= 0).all(), "Good length negative"
    assert ((df["Scrap_Rate"] >= 0) & (df["Scrap_Rate"] <= 1.0)).all(), "Scrap rate out of range"
    assert df["Scrap_Rate"].std(ddof=0) > 0.002, "Scrap rate variation is too small"
    assert np.allclose(df["Good_Length_m"] + df["Scrap_Length_m"], df["Actual_Processing_Length_m"], atol=1e-2), "Good + scrap does not equal actual length"
    assert np.allclose(df["Scrap_Length_m"], df["Actual_Processing_Length_m"] * df["Scrap_Rate"], atol=1e-2), "Scrap length formula mismatch"
    assert np.allclose(df["Good_Length_m"], df["Actual_Processing_Length_m"] - df["Scrap_Length_m"], atol=1e-2), "Good length formula mismatch"
    assert ((df["Quality"] >= 0) & (df["Quality"] <= 1.0)).all(), "Quality out of range"
    assert ((df["Availability"] >= 0) & (df["Availability"] <= 1.0)).all(), "Availability out of range"
    assert ((df["Performance"] >= 0) & (df["Performance"] <= 1.0)).all(), "Performance out of range"
    assert ((df["OEE"] >= 0) & (df["OEE"] <= 1.0)).all(), "OEE out of range"
    assert ((df["Operational_Efficiency_pct"] >= 0) & (df["Operational_Efficiency_pct"] <= 100)).all(), "Operational efficiency out of range"
    assert np.allclose(
        df["Run_Time_min"],
        np.maximum(0.0, df["Planned_Production_Time_min"] - df["Downtime_min"]),
        atol=1e-2,
    ), "Runtime is not planned time minus downtime"
    assert np.allclose(
        df["Availability"],
        df["Run_Time_min"] / df["Planned_Production_Time_min"],
        atol=1e-3,
    ), "Availability formula mismatch"
    assert np.allclose(df["OEE"], df["Availability"] * df["Performance"] * df["Quality"], atol=1e-3), "OEE is not Availability × Performance × Quality"
    assert np.allclose(df["Operational_Efficiency_pct"], df["OEE"] * 100.0, atol=1e-2), "Operational Efficiency not OEE × 100"
    assert np.allclose(df["Actual_Processing_Length_m"], df["Line_Speed_m_min"] * df["Run_Time_min"], atol=1e-1), "Actual length formula mismatch"
    positive_length = df["Actual_Processing_Length_m"] > 0
    assert np.allclose(
        df.loc[positive_length, "Quality"],
        df.loc[positive_length, "Good_Length_m"] / df.loc[positive_length, "Actual_Processing_Length_m"],
        atol=1e-3,
    ), "Quality formula mismatch"
    process_bounds = {
        "Extruder_Temperature_C": (180.0, 245.0),
        "Die_Temperature_C": (180.0, 235.0),
        "Cooling_Tank_Temperature_C": (12.0, 28.0),
        "Melt_Pressure_bar": (90.0, 200.0),
        "Air_Pressure_bar": (3.0, 12.0),
        "Screw_RPM": (48.0, 100.0),
        "Line_Speed_m_min": (8.0, 24.0),
        "Puller_Speed_m_min": (7.0, 25.0),
        "Extrusion_Speed_kg_hr": (25.0, 170.0),
    }
    for target, (lower, upper) in process_bounds.items():
        assert df[target].nunique() > 1 and df[target].std() > 0.01, f"{target} has no meaningful variation"
        clipped_fraction = ((df[target] <= lower + 0.01) | (df[target] >= upper - 0.01)).mean()
        assert clipped_fraction < 0.05, f"{target} is almost completely clipped ({clipped_fraction:.1%})"

        max_numeric_r2 = max(
            df[[target, feature]].corr().iloc[0, 1] ** 2
            for feature in ["Target_Tube_OD_mm", "Target_Tube_Wall_mm", "Target_Tube_ID_mm", "Customer_Required_Length_m"]
        )
        max_category_r2 = max(
            df.groupby(feature, observed=False)[target].mean().var() / df[target].var()
            for feature in ["Material_Type", "Material_Grade", "Tube_Type", "Braid_Type"]
        )
        assert max(max_numeric_r2, max_category_r2) < 0.999, f"{target} is nearly deterministic from one input"

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote synthetic dataset to {OUTPUT_PATH}")
    print(df.shape)
    scrap_summary = df["Scrap_Rate"].agg(["min", "max", "mean", "median", "std"])
    print("\nScrap_Rate:")
    print(scrap_summary.to_string(float_format=lambda value: f"{value:.5f}"))
    print(f"unique values: {df['Scrap_Rate'].nunique()}")
    efficiency_bins = [0, 50, 60, 70, 80, 90, 100]
    efficiency_counts = pd.cut(
        df["Operational_Efficiency_pct"],
        bins=efficiency_bins,
        right=False,
        include_lowest=True,
    ).value_counts(sort=False)
    print("\nOperational_Efficiency_pct histogram counts:")
    print(efficiency_counts.to_string())
    example_indices = [
        int((df["Scrap_Rate"] - df["Scrap_Rate"].quantile(q)).abs().idxmin())
        for q in (0.05, 0.50, 0.95)
    ]
    example_columns = [
        "Scrap_Rate",
        "Material_Type",
        "Tube_Type",
        "Target_Tube_OD_mm",
        "Target_Tube_Wall_mm",
        "Extruder_Temperature_C",
        "Cooling_Tank_Temperature_C",
        "Melt_Pressure_bar",
        "Screw_RPM",
        "Line_Speed_m_min",
        "Puller_Speed_m_min",
        "Fault_Code",
    ]
    print("\nExample process conditions at low, median, and high scrap:")
    print(df.loc[example_indices, example_columns].to_string(index=False))
    print(df[["Material_Type", "Material_Grade", "Screw_RPM", "Cooling_Tank_Temperature_C", "Scrap_Rate", "Fault_Code"]].head().to_string(index=False))
    print(f"Fault distribution:\n{df['Fault_Code'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
