from typing import Dict, Tuple

PIPE_ROUGHNESS: Dict[str, Dict[str, float]] = {
    "steel": {
        "new": 0.045e-3,
        "10years": 0.15e-3,
        "20years": 0.3e-3,
        "30years": 0.5e-3,
        "old": 1.0e-3,
    },
    "cast_iron": {
        "new": 0.26e-3,
        "10years": 0.5e-3,
        "20years": 1.0e-3,
        "30years": 1.5e-3,
        "old": 2.5e-3,
    },
    "pvc": {
        "new": 0.0015e-3,
        "10years": 0.003e-3,
        "20years": 0.005e-3,
        "30years": 0.008e-3,
        "old": 0.015e-3,
    },
}

INSULATION_CONDUCTIVITY: Dict[str, float] = {
    "none": 2.0,
    "polyurethane": 0.026,
    "rock_wool": 0.040,
    "glass_wool": 0.035,
}

PIPE_WALL_CONDUCTIVITY: Dict[str, float] = {
    "steel": 45.0,
    "cast_iron": 52.0,
    "pvc": 0.17,
}

WALL_THICKNESS: Dict[str, Dict[float, float]] = {
    "steel": {
        0.05: 0.003, 0.08: 0.004, 0.1: 0.004, 0.15: 0.006, 0.2: 0.008,
        0.25: 0.010, 0.3: 0.012, 0.4: 0.014, 0.5: 0.016, 0.6: 0.018,
        0.8: 0.022, 1.0: 0.026,
    },
    "cast_iron": {
        0.05: 0.007, 0.08: 0.008, 0.1: 0.009, 0.15: 0.011, 0.2: 0.013,
        0.25: 0.015, 0.3: 0.017, 0.4: 0.021, 0.5: 0.025, 0.6: 0.028,
        0.8: 0.034, 1.0: 0.040,
    },
    "pvc": {
        0.05: 0.002, 0.08: 0.003, 0.1: 0.003, 0.15: 0.004, 0.2: 0.005,
        0.25: 0.006, 0.3: 0.007, 0.4: 0.009, 0.5: 0.011, 0.6: 0.013,
        0.8: 0.017, 1.0: 0.021,
    },
}

INNER_CONVECTION_COEFFICIENT = 1500.0
OUTER_SOIL_CONDUCTIVITY = 1.5


def get_pipe_roughness(material: str, age_years: float) -> float:
    age_key = "new"
    if age_years < 5:
        age_key = "new"
    elif age_years < 15:
        age_key = "10years"
    elif age_years < 25:
        age_key = "20years"
    elif age_years < 35:
        age_key = "30years"
    else:
        age_key = "old"
    material_roughness = PIPE_ROUGHNESS.get(material, PIPE_ROUGHNESS["steel"])
    return material_roughness.get(age_key, material_roughness["new"])


def get_wall_thickness(material: str, diameter: float) -> float:
    thickness_dict = WALL_THICKNESS.get(material, WALL_THICKNESS["steel"])
    closest_d = min(thickness_dict.keys(), key=lambda d: abs(d - diameter))
    return thickness_dict[closest_d]


def get_water_properties(temperature_c: float) -> Tuple[float, float]:
    t = max(0.0, min(200.0, temperature_c))
    a1 = 999.83952
    a2 = 16.945176e-3
    a3 = -7.9870401e-6
    a4 = -46.170461e-9
    a5 = 105.56302e-12
    a6 = -280.54253e-15
    density = a1 + a2 * t + a3 * t ** 2 + a4 * t ** 3 + a5 * t ** 4 + a6 * t ** 5
    density = density / (1.0 + 16.879850e-6 * t)
    a = 2.414e-5
    b = 247.8
    c = 140.0
    viscosity = a * 10 ** (b / (t + c))
    return density, viscosity
