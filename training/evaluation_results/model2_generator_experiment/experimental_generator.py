"""Isolated, explainable synthetic extrusion-data experiment.

This module writes only experimental_dataset.csv beside itself. All numerical
profiles and bounds are synthetic assumptions, not Polyhose specifications.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROW_COUNT = 5000
OUTPUT_PATH = Path(__file__).resolve().parent / "experimental_dataset.csv"

MATERIALS = ["PA12", "PE", "TPU", "PVC", "PP"]
GRADES = ["G1", "G2", "G3"]
TUBE_TYPES = ["Composite", "Industrial", "Pneumatic", "Hydraulic", "PTFE"]
BRAID_TYPES = ["None", "1-Wire", "2-Wire", "Textile", "Helical"]
MACHINE_IDS = ["EXT-01", "EXT-02", "EXT-03", "EXT-04", "EXT-05"]
SHIFTS = ["A", "B", "C"]
FAULT_CODES = [
    "No Fault", "Machine Cleaning", "Start Up Delay", "Carrier Problem", "Power Cut",
    "Material Scorch", "Pressure Instability", "Cooling Failure", "Machine Breakdown",
    "High Throughput Stress", "Dimension Variation", "Minor Dimension Variation",
]

# Reused synthetic profiles from the existing generator; they are not validated machine data.
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
    "No Fault": (0.0, 0.0), "Machine Cleaning": (4.0, 18.0), "Start Up Delay": (5.0, 22.0),
    "Carrier Problem": (8.0, 30.0), "Power Cut": (10.0, 45.0), "Material Scorch": (11.0, 42.0),
    "Pressure Instability": (9.0, 46.0), "Cooling Failure": (16.0, 60.0),
    "Machine Breakdown": (40.0, 150.0), "High Throughput Stress": (8.0, 24.0),
    "Dimension Variation": (12.0, 50.0), "Minor Dimension Variation": (5.0, 16.0),
}
PROCESS_BOUNDS = {
    "Extruder_Temperature_C": (180.0, 245.0), "Die_Temperature_C": (180.0, 235.0),
    "Cooling_Tank_Temperature_C": (12.0, 28.0), "Melt_Pressure_bar": (90.0, 200.0),
    "Air_Pressure_bar": (3.0, 12.0), "Screw_RPM": (48.0, 100.0),
    "Line_Speed_m_min": (8.0, 24.0), "Puller_Speed_m_min": (7.0, 25.0),
    "Extrusion_Speed_kg_hr": (25.0, 170.0),
}

COLUMNS = [
    "Record_ID", "Material_Type", "Material_Grade", "Tube_Type", "Braid_Type",
    "Customer_Required_Length_m", "Target_Tube_OD_mm", "Target_Tube_Wall_mm", "Target_Tube_ID_mm",
    "Machine_ID", "Shift", "Ambient_Temperature_C", "Humidity_percent", "Planned_Scrap_Allowance_pct",
    "Planned_Processing_Length_m", "Planned_Production_Time_min", "Downtime_min", "Run_Time_min",
    "Extruder_Temperature_C", "Die_Temperature_C", "Cooling_Tank_Temperature_C", "Melt_Pressure_bar",
    "Air_Pressure_bar", "Screw_RPM", "Line_Speed_m_min", "Puller_Speed_m_min", "Extrusion_Speed_kg_hr",
    "Power_Consumption_kWh", "Actual_Processing_Length_m", "Scrap_Length_m", "Good_Length_m",
    "Length_Shortfall_m", "Order_Status", "Scrap_Rate", "Availability", "Performance", "Quality", "OEE",
    "Operational_Efficiency_pct", "Fault_Code",
]


def bounded(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def sample_customer_and_context(rng: np.random.Generator) -> dict:
    material = str(rng.choice(MATERIALS, p=[0.38, 0.22, 0.18, 0.12, 0.10]))
    grade = str(rng.choice(GRADES, p=[0.40, 0.38, 0.22]))
    tube = str(rng.choice(TUBE_TYPES, p=[0.18, 0.18, 0.24, 0.20, 0.20]))
    braid = str(rng.choice(BRAID_TYPES, p=[0.16, 0.20, 0.16, 0.30, 0.18]))
    machine = str(rng.choice(MACHINE_IDS, p=[0.22, 0.22, 0.18, 0.20, 0.18]))
    shift = str(rng.choice(SHIFTS, p=[0.33, 0.33, 0.34]))
    od = float(rng.uniform(7.0, 20.0))
    wall = float(rng.uniform(1.0, 4.8))
    inner_diameter = od - 2.0 * wall
    if inner_diameter <= 0.5:
        wall = (od - 0.8) / 2.0
        inner_diameter = od - 2.0 * wall
    required_length = bounded(float(np.exp(rng.normal(math.log(5000.0), 0.55))), 900.0, 17000.0)
    planned_length = required_length * float(rng.uniform(1.03, 1.16))
    return {
        "material": material, "grade": grade, "tube": tube, "braid": braid,
        "machine": machine, "shift": shift, "od": od, "wall": wall, "id": inner_diameter,
        "required_length": required_length, "planned_length": planned_length,
        "ambient": float(rng.uniform(22.0, 35.0)), "humidity": float(rng.uniform(35.0, 82.0)),
        "difficulty": TUBE_PROFILE[tube]["difficulty"] + 0.4 * BRAID_PROFILE[braid],
    }


def calculate_process_settings(spec: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    """Compute recipe centers and realized settings; return nominal line speed separately."""
    material = MATERIAL_PROFILE[spec["material"]]
    tube = TUBE_PROFILE[spec["tube"]]
    machine = MACHINE_PROFILE[spec["machine"]]
    difficulty = spec["difficulty"]
    grade_offset = GRADE_PROFILE[spec["grade"]]
    braid_factor = BRAID_PROFILE[spec["braid"]]

    extruder_center = material["temp"] + 0.55 * grade_offset + 3.2 * tube["thermal"] + 2.2 * difficulty + machine["temp_bias"]
    extruder = bounded(extruder_center + rng.normal(0, 4.0), *PROCESS_BOUNDS["Extruder_Temperature_C"])
    die_center = extruder - 7.0 + 0.18 * (spec["od"] - 13.5) + 0.8 * tube["thermal"]
    die = bounded(die_center + rng.normal(0, 3.8), *PROCESS_BOUNDS["Die_Temperature_C"])

    pressure_center = (
        128.0 + 0.38 * (material["pressure"] - 130.0) + 2.7 * (spec["wall"] - 2.8)
        + 0.72 * (spec["od"] - 13.5) + 4.0 * (spec["grade"] == "G3")
        + 5.0 * (difficulty - 1.0) + 2.0 * (machine["pressure_bias"] - 1.0) * 10.0
    )
    melt_pressure = bounded(pressure_center + rng.normal(0, 5.5), *PROCESS_BOUNDS["Melt_Pressure_bar"])
    air_pressure = bounded(
        5.2 + 0.62 * spec["wall"] + 0.65 * difficulty + 0.48 * braid_factor + rng.normal(0, 0.65),
        *PROCESS_BOUNDS["Air_Pressure_bar"],
    )
    rpm = (
        66.0 + 4.0 * (spec["grade"] == "G2") + 7.0 * (spec["grade"] == "G3")
        + 7.0 * (difficulty - 1.0) + 3.0 * (spec["wall"] - 2.8)
        + 0.50 * (spec["od"] - 13.5) + 1.8 * (material["rpm"] - 7.0)
        + 4.5 * braid_factor + 1.4 * machine["rpm_bias"] + rng.normal(0, 3.8)
    )
    rpm = bounded(rpm, *PROCESS_BOUNDS["Screw_RPM"])

    # Synthetic nominal speed is a recipe reference based on specification. Actual speed
    # adds machine calibration and stochastic operating variation; neither is factory data.
    nominal_line_speed = (
        14.0 + 1.5 * (material["speed"] - 1.2) + 0.55 * (spec["grade"] == "G2")
        + 1.0 * (spec["grade"] == "G3") - 0.45 * (spec["od"] - 13.5)
        - 0.85 * (spec["wall"] - 2.8) - 1.0 * (difficulty - 1.0)
        + 0.00012 * (spec["required_length"] - 5000.0)
    )
    line_speed = bounded(
        nominal_line_speed + 2.0 * (machine["speed_bias"] - 1.0) + rng.normal(0, 0.9),
        *PROCESS_BOUNDS["Line_Speed_m_min"],
    )
    cool_center = (
        16.0 + 0.75 * (material["cool"] - 17.5) + 1.35 * tube["thermal"]
        + 0.19 * (spec["od"] - 13.5) + 0.18 * (line_speed - 13.5)
        + 0.06 * (spec["ambient"] - 28.5) + 0.70 * braid_factor
    )
    cooling = bounded(cool_center + rng.normal(0, 0.90), *PROCESS_BOUNDS["Cooling_Tank_Temperature_C"])
    puller = bounded(line_speed + 0.10 * braid_factor + rng.normal(0, 0.42), *PROCESS_BOUNDS["Puller_Speed_m_min"])
    extrusion = (
        42.0 + 0.78 * rpm + 4.5 * (material["flow"] - 1.0) + 1.25 * (spec["od"] - 13.5)
        + 3.5 * (spec["wall"] - 2.8) + 5.0 * (TUBE_PROFILE[spec["tube"]]["difficulty"] - 1.0)
        + 1.5 * machine["pressure_bias"] + rng.normal(0, 6.0)
    )
    extrusion = bounded(extrusion, *PROCESS_BOUNDS["Extrusion_Speed_kg_hr"])

    process = {
        "Extruder_Temperature_C": extruder,
        "Die_Temperature_C": die,
        "Cooling_Tank_Temperature_C": cooling,
        "Melt_Pressure_bar": melt_pressure,
        "Air_Pressure_bar": air_pressure,
        "Screw_RPM": rpm,
        "Line_Speed_m_min": line_speed,
        "Puller_Speed_m_min": puller,
        "Extrusion_Speed_kg_hr": extrusion,
    }
    references = {"nominal_line_speed_m_min": nominal_line_speed, "extruder_center_C": extruder_center, "pressure_center_bar": pressure_center, "cooling_center_C": cool_center}
    return process, references


def calculate_process_risk(spec: dict, process: dict, references: dict) -> tuple[dict, float]:
    geometry_risk = bounded(
        0.35 * (spec["wall"] - 1.0) / 3.8
        + 0.30 * (spec["od"] - 7.0) / 13.0
        + 0.35 * (spec["difficulty"] - 0.9) / 1.3,
        0.0, 1.0,
    )
    components = {
        "thermal": bounded(
            0.75 * max(0.0, abs(references["extruder_center_C"] - (MATERIAL_PROFILE[spec["material"]]["temp"] + 3.0)) - 7.0) / 20.0
            + 0.25 * abs(process["Extruder_Temperature_C"] - references["extruder_center_C"]) / 12.0,
            0.0, 1.0,
        ),
        "pressure": bounded(
            0.75 * max(0.0, abs(references["pressure_center_bar"] - 130.0) - 8.0) / 35.0
            + 0.25 * abs(process["Melt_Pressure_bar"] - references["pressure_center_bar"]) / 18.0,
            0.0, 1.0,
        ),
        "cooling": bounded(
            0.75 * abs(references["cooling_center_C"] - 18.5) / 6.0
            + 0.25 * abs(process["Cooling_Tank_Temperature_C"] - references["cooling_center_C"]) / 3.0,
            0.0, 1.0,
        ),
        "mismatch": bounded(
            0.65 * geometry_risk + 0.35 * abs(process["Line_Speed_m_min"] - process["Puller_Speed_m_min"]) / 1.6,
            0.0, 1.0,
        ),
        "throughput": bounded(
            0.55 * max(0.0, process["Screw_RPM"] - 72.0) / 22.0
            + 0.45 * max(0.0, process["Line_Speed_m_min"] - 14.0) / 8.0,
            0.0, 1.0,
        ),
        "geometry": geometry_risk,
        "length": bounded(math.log(spec["required_length"] / 900.0) / math.log(17000.0 / 900.0), 0.0, 1.0),
    }
    score = bounded(
        0.20 * components["thermal"] + 0.18 * components["pressure"] + 0.15 * components["cooling"]
        + 0.15 * components["mismatch"] + 0.13 * components["throughput"]
        + 0.13 * components["geometry"] + 0.06 * components["length"],
        0.0, 1.0,
    )
    return components, score


def select_fault(rng: np.random.Generator, score: float, risk: dict, spec: dict) -> str:
    machine_risk = 0.12 if spec["machine"] in ("EXT-01", "EXT-05") else 0.0
    weights = {
        "Material Scorch": 0.08 + 2.4 * risk["thermal"] + risk["pressure"],
        "Pressure Instability": 0.08 + 2.6 * risk["pressure"],
        "Cooling Failure": 0.08 + 2.6 * risk["cooling"],
        "Carrier Problem": 0.08 + 2.8 * risk["mismatch"],
        "Machine Breakdown": 0.08 + 0.7 * score + machine_risk,
        "High Throughput Stress": 0.08 + 2.8 * risk["throughput"],
        "Dimension Variation": 0.08 + 2.1 * risk["geometry"] + risk["mismatch"],
        "Minor Dimension Variation": 0.08 + 1.8 * risk["mismatch"] + 0.4 * risk["geometry"],
        "Machine Cleaning": 0.08 + 0.25 * (spec["shift"] == "A") + 0.20 * (spec["material"] == "PVC"),
        "Start Up Delay": 0.08 + 0.30 * (spec["shift"] == "A") + 0.10 * (spec["shift"] == "B"),
        "Power Cut": 0.08 + machine_risk + 0.6 * risk["cooling"],
    }
    fault_probability = bounded(0.03 + 1.15 * score + rng.normal(0.0, 0.04), 0.01, 0.82)
    if rng.random() >= fault_probability:
        return "No Fault"
    labels = FAULT_CODES[1:]
    probabilities = np.array([weights[label] for label in labels], dtype=float)
    probabilities /= probabilities.sum()
    return str(rng.choice(labels, p=probabilities))


def apply_fault_to_process(fault: str, process: dict, rng: np.random.Generator) -> None:
    if fault == "Material Scorch":
        process["Extruder_Temperature_C"] = bounded(process["Extruder_Temperature_C"] + rng.uniform(5, 13), *PROCESS_BOUNDS["Extruder_Temperature_C"])
        process["Melt_Pressure_bar"] = bounded(process["Melt_Pressure_bar"] + rng.uniform(3, 9), *PROCESS_BOUNDS["Melt_Pressure_bar"])
    elif fault == "Pressure Instability":
        process["Melt_Pressure_bar"] = bounded(process["Melt_Pressure_bar"] + rng.normal(0, 9), *PROCESS_BOUNDS["Melt_Pressure_bar"])
    elif fault == "Cooling Failure":
        process["Cooling_Tank_Temperature_C"] = bounded(process["Cooling_Tank_Temperature_C"] + rng.uniform(3, 7), *PROCESS_BOUNDS["Cooling_Tank_Temperature_C"])
    elif fault in ("Carrier Problem", "Dimension Variation", "Minor Dimension Variation"):
        process["Puller_Speed_m_min"] = bounded(process["Puller_Speed_m_min"] + rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 2.0), *PROCESS_BOUNDS["Puller_Speed_m_min"])
    elif fault == "High Throughput Stress":
        process["Screw_RPM"] = bounded(process["Screw_RPM"] + rng.uniform(3, 8), *PROCESS_BOUNDS["Screw_RPM"])
        process["Line_Speed_m_min"] = bounded(process["Line_Speed_m_min"] + rng.uniform(0.5, 1.5), *PROCESS_BOUNDS["Line_Speed_m_min"])


def fault_severity_risk(fault: str, risk: dict, score: float) -> float:
    formulas = {
        "Machine Cleaning": 0.35 * risk["geometry"] + 0.30 * risk["length"],
        "Start Up Delay": 0.40 * risk["throughput"] + 0.25 * risk["length"],
        "Carrier Problem": risk["mismatch"],
        "Power Cut": 0.50 * risk["cooling"] + 0.20 * score,
        "Material Scorch": 0.70 * risk["thermal"] + 0.30 * risk["pressure"],
        "Pressure Instability": risk["pressure"],
        "Cooling Failure": risk["cooling"],
        "Machine Breakdown": 0.65 * score + 0.35 * risk["geometry"],
        "High Throughput Stress": risk["throughput"],
        "Dimension Variation": 0.65 * risk["geometry"] + 0.35 * risk["mismatch"],
        "Minor Dimension Variation": 0.55 * risk["mismatch"] + 0.25 * risk["geometry"],
    }
    return float(formulas[fault])


def calculate_operating_conditions(spec: dict, process: dict, risk: dict, score: float, rng: np.random.Generator) -> dict:
    planned_time = bounded(spec["planned_length"] / max(8.0, process["Line_Speed_m_min"]) * 1.10 + 35.0 + rng.normal(0, 12.0), 90.0, 900.0)
    fault = select_fault(rng, score, risk, spec)
    apply_fault_to_process(fault, process, rng)
    downtime = 0.0
    if fault != "No Fault":
        lower, upper = FAULT_DOWNTIME[fault]
        severity = bounded(0.12 + 0.72 * score + 0.18 * fault_severity_risk(fault, risk, score) + rng.normal(0, 0.12), 0.05, 0.98)
        downtime = lower + severity * (upper - lower)
    runtime = bounded(planned_time - downtime, 0.0, planned_time)
    return {
        "planned_time": round(planned_time, 2),
        "downtime": round(downtime, 2),
        "runtime": round(runtime, 2),
        "fault": fault,
    }


def calculate_scrap_rate(spec: dict, process: dict, score: float, fault: str, rng: np.random.Generator) -> float:
    material = spec["material"]
    grade = spec["grade"]
    tube_profile = TUBE_PROFILE[spec["tube"]]
    material_profile = MATERIAL_PROFILE[material]
    braid = BRAID_PROFILE[spec["braid"]]
    thermal_deviation = max(0.0, abs(process["Extruder_Temperature_C"] - material_profile["temp"]) - 3.0) * 0.0022
    pressure_deviation = max(0.0, abs(process["Melt_Pressure_bar"] - 130.0) - 8.0) * 0.0018
    cooling_deviation = max(0.0, abs(process["Cooling_Tank_Temperature_C"] - 18.5) - 1.5) * 0.007
    mismatch = max(0.0, abs(process["Line_Speed_m_min"] - process["Puller_Speed_m_min"]) - 0.2) * 0.025
    rpm_deviation = max(0.0, abs(process["Screw_RPM"] - 72.0) - 9.0) * 0.002
    geometry_effect = 0.006 + max(0.0, spec["wall"] - 2.8) * 0.008 + max(0.0, spec["od"] - 13.5) * 0.003 + max(0.0, spec["difficulty"] - 1.0) * 0.012
    material_effect = 0.006 + 0.012 * (material in ("PVC", "TPU")) + 0.006 * (material == "PA12") + 0.005 * (grade == "G3")
    fault_effect = {
        "No Fault": 0.0, "Machine Cleaning": 0.010, "Start Up Delay": 0.012,
        "Carrier Problem": 0.025, "Power Cut": 0.015, "Material Scorch": 0.045,
        "Pressure Instability": 0.038, "Cooling Failure": 0.040, "Machine Breakdown": 0.030,
        "High Throughput Stress": 0.032, "Dimension Variation": 0.030, "Minor Dimension Variation": 0.018,
    }[fault]
    return round(bounded(
        0.010 + 0.045 * score + thermal_deviation + pressure_deviation + cooling_deviation
        + mismatch + rpm_deviation + geometry_effect + material_effect + 0.011 * braid
        + fault_effect + rng.normal(0, 0.006),
        0.002, 0.16,
    ), 5)


def calculate_production_accounting(
    spec: dict, process: dict, references: dict, risk: dict, score: float, operating: dict, rng: np.random.Generator
) -> dict:
    """Derive OEE only from availability, measured rate performance, and good-output quality."""
    line_speed = round(float(process["Line_Speed_m_min"]), 2)
    puller_speed = round(float(process["Puller_Speed_m_min"]), 2)
    runtime = operating["runtime"]
    planned_time = operating["planned_time"]
    actual_length = round(line_speed * runtime, 2)
    # PERFORMANCE DEFINITION: compare actual output rate with the synthetic nominal
    # recipe speed over the time actually running. Runtime is not embedded in its denominator.
    if runtime > 0 and references["nominal_line_speed_m_min"] > 0:
        performance = bounded(actual_length / (references["nominal_line_speed_m_min"] * runtime), 0.0, 1.0)
    else:
        performance = 0.0
    scrap_rate = calculate_scrap_rate(spec, process, score, operating["fault"], rng)
    scrap_length = round(actual_length * scrap_rate, 2)
    good_length = round(actual_length - scrap_length, 2)
    if good_length < 0:
        raise ValueError("Synthetic scrap accounting produced negative good output")
    quality = bounded(good_length / actual_length, 0.0, 1.0) if actual_length > 0 else 1.0
    availability = bounded(runtime / planned_time, 0.0, 1.0) if planned_time > 0 else 1.0
    oee = bounded(availability * performance * quality, 0.0, 1.0)
    efficiency = oee * 100.0
    shortfall = max(0.0, spec["planned_length"] - actual_length)
    order_status = "Completed" if shortfall <= 0.12 * spec["planned_length"] else "Delayed"

    power = bounded(
        150.0 + 1.5 * process["Extrusion_Speed_kg_hr"] + 0.9 * process["Melt_Pressure_bar"]
        + 0.12 * process["Extruder_Temperature_C"] + rng.normal(0, 18.0),
        80.0, 1000.0,
    )
    return {
        "Planned_Production_Time_min": round(planned_time, 2),
        "Downtime_min": round(operating["downtime"], 2),
        "Run_Time_min": round(runtime, 2),
        "Power_Consumption_kWh": round(power, 2),
        "Actual_Processing_Length_m": round(actual_length, 2),
        "Scrap_Length_m": scrap_length,
        "Good_Length_m": good_length,
        "Length_Shortfall_m": round(shortfall, 2),
        "Order_Status": order_status,
        "Scrap_Rate": scrap_rate,
        "Availability": round(availability, 6),
        "Performance": round(performance, 6),
        "Quality": round(quality, 6),
        "OEE": round(oee, 6),
        "Operational_Efficiency_pct": round(efficiency, 4),
    }


def make_row(record_id: int, rng: np.random.Generator) -> dict:
    spec = sample_customer_and_context(rng)
    process, references = calculate_process_settings(spec, rng)
    risk, score = calculate_process_risk(spec, process, references)
    operating = calculate_operating_conditions(spec, process, risk, score, rng)
    accounting = calculate_production_accounting(spec, process, references, risk, score, operating, rng)
    return {
        "Record_ID": record_id,
        "Material_Type": spec["material"],
        "Material_Grade": spec["grade"],
        "Tube_Type": spec["tube"],
        "Braid_Type": spec["braid"],
        "Customer_Required_Length_m": round(spec["required_length"], 2),
        "Target_Tube_OD_mm": round(spec["od"], 2),
        "Target_Tube_Wall_mm": round(spec["wall"], 2),
        "Target_Tube_ID_mm": round(spec["id"], 2),
        "Machine_ID": spec["machine"],
        "Shift": spec["shift"],
        "Ambient_Temperature_C": round(spec["ambient"], 2),
        "Humidity_percent": round(spec["humidity"], 2),
        "Planned_Scrap_Allowance_pct": round(float(rng.uniform(1.0, 4.0)), 2),
        "Planned_Processing_Length_m": round(spec["planned_length"], 2),
        **accounting,
        **{name: round(float(value), 2) for name, value in process.items()},
        "Fault_Code": operating["fault"],
    }


def generate_dataset(seed: int = SEED, row_count: int = ROW_COUNT) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [make_row(index + 1, rng) for index in range(row_count)]
    frame = pd.DataFrame(rows)[COLUMNS]
    validate_dataset(frame, expected_rows=row_count)
    return frame


def validate_dataset(frame: pd.DataFrame, expected_rows: int = ROW_COUNT) -> dict:
    if frame.shape != (expected_rows, 40):
        raise AssertionError(f"Expected ({expected_rows}, 40), got {frame.shape}")
    if list(frame.columns) != COLUMNS:
        raise AssertionError("Column names or column order changed")
    if frame.isna().any().any():
        raise AssertionError("Missing values present")
    if frame["Record_ID"].nunique() != expected_rows:
        raise AssertionError("Record_ID values must be unique")
    allowed = {"Material_Type": MATERIALS, "Material_Grade": GRADES, "Tube_Type": TUBE_TYPES, "Braid_Type": BRAID_TYPES, "Machine_ID": MACHINE_IDS, "Shift": SHIFTS, "Fault_Code": FAULT_CODES}
    for column, categories in allowed.items():
        if not set(frame[column]).issubset(categories):
            raise AssertionError(f"Unexpected {column} category")
    geometry_error = (frame.Target_Tube_ID_mm - (frame.Target_Tube_OD_mm - 2 * frame.Target_Tube_Wall_mm)).abs()
    if geometry_error.max() >= 0.25 or (frame.Target_Tube_ID_mm <= 0.5).any():
        raise AssertionError("Tube geometry rule violated")
    if (frame.Customer_Required_Length_m <= 0).any() or (frame.Planned_Production_Time_min <= 0).any():
        raise AssertionError("Lengths and planned time must be positive")
    if (frame.Downtime_min < 0).any() or (frame.Run_Time_min < 0).any() or (frame.Run_Time_min > frame.Planned_Production_Time_min).any():
        raise AssertionError("Downtime/runtime constraints violated")
    if not np.allclose(frame.Actual_Processing_Length_m, frame.Line_Speed_m_min * frame.Run_Time_min, atol=0.011):
        raise AssertionError("Actual processing length identity failed")
    if not np.allclose(frame.Good_Length_m + frame.Scrap_Length_m, frame.Actual_Processing_Length_m, atol=0.011):
        raise AssertionError("Good + scrap = actual length identity failed")
    if not np.allclose(frame.Scrap_Length_m, frame.Actual_Processing_Length_m * frame.Scrap_Rate, atol=0.011):
        raise AssertionError("Scrap length identity failed")
    expected_shortfall = np.maximum(
        0.0, frame.Planned_Processing_Length_m - frame.Actual_Processing_Length_m
    )
    if not np.allclose(frame.Length_Shortfall_m, expected_shortfall, atol=0.011):
        raise AssertionError("Length shortfall identity failed")
    expected_runtime = np.maximum(
        0.0, frame.Planned_Production_Time_min - frame.Downtime_min
    )
    if not np.allclose(frame.Run_Time_min, expected_runtime, atol=0.011):
        raise AssertionError("Runtime identity failed")
    if not np.allclose(
        frame.Operational_Efficiency_pct, frame.OEE * 100.0, atol=0.01
    ):
        raise AssertionError("Efficiency/OEE identity failed")
    expected_status = np.where(
        expected_shortfall <= 0.12 * frame.Planned_Processing_Length_m,
        "Completed",
        "Delayed",
    )
    if not np.array_equal(frame.Order_Status.to_numpy(), expected_status):
        raise AssertionError("Order status is inconsistent with the shortfall rule")
    if not np.allclose(frame.Availability, frame.Run_Time_min / frame.Planned_Production_Time_min, atol=1e-3):
        raise AssertionError("Availability identity failed")
    positive_output = frame.Actual_Processing_Length_m > 0
    if not np.allclose(
        frame.loc[positive_output, "Quality"],
        frame.loc[positive_output, "Good_Length_m"] / frame.loc[positive_output, "Actual_Processing_Length_m"],
        atol=1e-3,
    ):
        raise AssertionError("Quality identity failed")
    if not np.allclose(frame.OEE, frame.Availability * frame.Performance * frame.Quality, atol=1e-3):
        raise AssertionError("OEE identity failed")
    if not np.allclose(frame.Operational_Efficiency_pct, frame.OEE * 100.0, atol=1e-2):
        raise AssertionError("Operational efficiency identity failed")
    for target, (low, high) in PROCESS_BOUNDS.items():
        if not frame[target].between(low, high).all():
            raise AssertionError(f"{target} outside this experiment's inherited synthetic range [{low}, {high}]")
    if not frame.Scrap_Rate.between(0.0, 1.0).all() or not frame.Operational_Efficiency_pct.between(0.0, 100.0).all():
        raise AssertionError("Outcome fraction/percentage bounds violated")
    return {
        "rows": int(len(frame)), "columns": int(frame.shape[1]), "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_record_ids": int(frame.Record_ID.duplicated().sum()),
        "geometry_max_abs_residual_mm": float(geometry_error.max()),
        "actual_length_max_abs_residual_m": float((frame.Actual_Processing_Length_m - frame.Line_Speed_m_min * frame.Run_Time_min).abs().max()),
        "efficiency_min": float(frame.Operational_Efficiency_pct.min()), "efficiency_max": float(frame.Operational_Efficiency_pct.max()),
        "scrap_min": float(frame.Scrap_Rate.min()), "scrap_max": float(frame.Scrap_Rate.max()),
        "fault_count": int((frame.Fault_Code != "No Fault").sum()),
        "braid_none_count": int((frame.Braid_Type == "None").sum()),
    }


def main() -> None:
    frame = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Experimental dataset written only to {OUTPUT_PATH}")
    print(frame.shape)
    print(validate_dataset(frame))


if __name__ == "__main__":
    main()
