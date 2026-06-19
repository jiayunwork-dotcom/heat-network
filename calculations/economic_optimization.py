from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from models import (
    PipeNetwork, Pipe, NetworkResults, PipeSectionResult,
    INSULATION_MATERIAL_NONE, INSULATION_MATERIAL_POLYURETHANE,
    INSULATION_MATERIAL_ROCK_WOOL, INSULATION_MATERIAL_GLASS_WOOL,
    INSULATION_CONDUCTIVITY, PIPE_WALL_CONDUCTIVITY,
    INNER_CONVECTION_COEFFICIENT, OUTER_SOIL_CONDUCTIVITY,
    get_wall_thickness,
)
from calculations.thermal import compute_overall_heat_transfer_coefficient


DEFAULT_ELECTRICITY_PRICE = 0.7
DEFAULT_GAS_PRICE = 3.5
DEFAULT_GAS_CALORIFIC_VALUE = 36.0
DEFAULT_BOILER_EFFICIENCY = 0.9
DEFAULT_DAILY_HOURS = 24
DEFAULT_ANNUAL_HOURS = 8000
DEFAULT_PAYBACK_THRESHOLD = 5.0

INSULATION_UNIT_COST = {
    INSULATION_MATERIAL_POLYURETHANE: 120.0,
    INSULATION_MATERIAL_ROCK_WOOL: 80.0,
    INSULATION_MATERIAL_GLASS_WOOL: 60.0,
    INSULATION_MATERIAL_NONE: 0.0,
}

MAINTENANCE_COST_TIERS = [
    (20.0, 15.0),
    (10.0, 8.0),
    (0.0, 3.0),
]


@dataclass
class OperatingCostBreakdown:
    electricity_cost_daily: float = 0.0
    heat_loss_cost_daily: float = 0.0
    maintenance_cost_annual: float = 0.0
    total_daily_cost: float = 0.0
    total_annual_cost: float = 0.0
    electricity_cost_annual: float = 0.0
    heat_loss_cost_annual: float = 0.0


@dataclass
class PipeInsulationRetrofit:
    pipe_id: int
    pipe_name: str
    original_thickness: float
    new_thickness: float
    original_heat_loss: float
    new_heat_loss: float
    heat_loss_saving: float
    annual_gas_saving: float
    investment: float
    payback_period: float
    insulation_material: str


@dataclass
class PumpEfficiencyRetrofit:
    pipe_id: int
    pipe_name: str
    rated_head: float
    actual_head: float
    current_power: float
    status: str
    has_vfd_recommendation: bool
    vfd_annual_saving: float
    vfd_investment: float
    vfd_payback_period: float


@dataclass
class ComprehensiveRetrofitItem:
    item_id: str
    item_name: str
    retrofit_type: str
    investment: float
    annual_saving: float
    payback_period: float
    detail: str = ""


@dataclass
class EconomicOptimizationResult:
    operating_cost: OperatingCostBreakdown = field(default_factory=OperatingCostBreakdown)
    insulation_retrofits: List[PipeInsulationRetrofit] = field(default_factory=list)
    pump_retrofits: List[PumpEfficiencyRetrofit] = field(default_factory=list)
    comprehensive_list: List[ComprehensiveRetrofitItem] = field(default_factory=list)
    total_investment: float = 0.0
    total_annual_saving: float = 0.0
    overall_payback_period: float = 0.0


def calculate_operating_cost(
    network: PipeNetwork,
    results: NetworkResults,
    electricity_price: float = DEFAULT_ELECTRICITY_PRICE,
    gas_price: float = DEFAULT_GAS_PRICE,
    gas_calorific_value: float = DEFAULT_GAS_CALORIFIC_VALUE,
    boiler_efficiency: float = DEFAULT_BOILER_EFFICIENCY,
    daily_hours: float = DEFAULT_DAILY_HOURS,
) -> OperatingCostBreakdown:
    total_pump_power_kw = results.total_pump_power
    electricity_cost_daily = total_pump_power_kw * daily_hours * electricity_price

    total_heat_loss_w = results.total_heat_loss
    total_heat_loss_mj_daily = total_heat_loss_w * 3600 * daily_hours / 1e6
    gas_consumption_daily = total_heat_loss_mj_daily / (gas_calorific_value * boiler_efficiency)
    heat_loss_cost_daily = gas_consumption_daily * gas_price

    maintenance_cost_annual = 0.0
    for pipe in network.pipes.values():
        age = pipe.pipe_age
        unit_cost = 0.0
        for threshold, cost in MAINTENANCE_COST_TIERS:
            if age >= threshold:
                unit_cost = cost
                break
        maintenance_cost_annual += pipe.length * unit_cost

    maintenance_cost_daily = maintenance_cost_annual / 365.0

    total_daily_cost = electricity_cost_daily + heat_loss_cost_daily + maintenance_cost_daily
    total_annual_cost = total_daily_cost * 365.0
    electricity_cost_annual = electricity_cost_daily * 365.0
    heat_loss_cost_annual = heat_loss_cost_daily * 365.0

    return OperatingCostBreakdown(
        electricity_cost_daily=electricity_cost_daily,
        heat_loss_cost_daily=heat_loss_cost_daily,
        maintenance_cost_annual=maintenance_cost_annual,
        total_daily_cost=total_daily_cost,
        total_annual_cost=total_annual_cost,
        electricity_cost_annual=electricity_cost_annual,
        heat_loss_cost_annual=heat_loss_cost_annual,
    )


def _compute_insulation_heat_loss(
    pipe: Pipe,
    new_insulation_thickness: float,
    inlet_temp: float,
    flow_rate: float,
    env_temp: float,
    cp_water: float,
) -> Tuple[float, float]:
    import numpy as np

    d_inner = pipe.diameter
    wall_thickness = get_wall_thickness(pipe.material, d_inner)
    d_wall_outer = d_inner + 2 * wall_thickness
    d_insul_outer = d_wall_outer + 2 * new_insulation_thickness

    r_inner_conv = 1.0 / (INNER_CONVECTION_COEFFICIENT * np.pi * d_inner)
    cond_wall = PIPE_WALL_CONDUCTIVITY.get(pipe.material, 45.0)
    r_wall = np.log(d_wall_outer / d_inner) / (2.0 * np.pi * cond_wall)

    if new_insulation_thickness > 0 and pipe.insulation_material != INSULATION_MATERIAL_NONE:
        cond_insul = INSULATION_CONDUCTIVITY.get(pipe.insulation_material, 0.03)
        r_insul = np.log(d_insul_outer / d_wall_outer) / (2.0 * np.pi * cond_insul)
    else:
        r_insul = 0.0

    burial_depth = max(pipe.burial_depth, d_insul_outer / 2.0 + 0.1)
    z_ratio = 2.0 * burial_depth / d_insul_outer
    if z_ratio > 2.0:
        soil_factor = np.log(z_ratio + np.sqrt(z_ratio ** 2 - 1.0))
    else:
        soil_factor = np.arccosh(z_ratio)
    r_soil = soil_factor / (2.0 * np.pi * OUTER_SOIL_CONDUCTIVITY)

    total_r_per_m = r_inner_conv + r_wall + r_insul + r_soil
    U_per_m = 1.0 / total_r_per_m

    from models import get_water_properties
    density, _ = get_water_properties(inlet_temp)
    mass_flow = abs(flow_rate) * density
    UA = U_per_m * pipe.length
    exponent = -UA / max(mass_flow * cp_water, 1e-6)
    outlet_temp = env_temp + (inlet_temp - env_temp) * np.exp(exponent)
    temp_drop = inlet_temp - outlet_temp

    avg_temp = 0.5 * (inlet_temp + outlet_temp)
    density_avg, _ = get_water_properties(avg_temp)
    mass_flow_avg = abs(flow_rate) * density_avg
    heat_loss = mass_flow_avg * cp_water * temp_drop

    return heat_loss, U_per_m


def analyze_insulation_retrofit(
    network: PipeNetwork,
    results: NetworkResults,
    gas_price: float = DEFAULT_GAS_PRICE,
    gas_calorific_value: float = DEFAULT_GAS_CALORIFIC_VALUE,
    boiler_efficiency: float = DEFAULT_BOILER_EFFICIENCY,
    payback_threshold: float = DEFAULT_PAYBACK_THRESHOLD,
    cp_water: float = 4186.0,
    annual_hours: float = 8760,
) -> List[PipeInsulationRetrofit]:
    retrofits = []

    for pid, pipe in network.pipes.items():
        if pipe.insulation_material == INSULATION_MATERIAL_NONE:
            continue
        if pipe.insulation_thickness <= 0:
            continue

        pr = results.pipe_results.get(pid)
        if pr is None:
            continue
        if pr.flow_rate <= 1e-6:
            continue

        original_heat_loss = pr.heat_loss
        new_thickness = pipe.insulation_thickness * 2.0

        inlet_temp = pr.inlet_temperature
        env_temp = network.environment_temperature

        new_heat_loss, _ = _compute_insulation_heat_loss(
            pipe, new_thickness, inlet_temp, pr.flow_rate, env_temp, cp_water
        )

        heat_loss_saving = original_heat_loss - new_heat_loss
        if heat_loss_saving <= 0:
            continue

        heat_loss_saving_mj_per_hour = heat_loss_saving * 3600 / 1e6
        annual_gas_saving_mj = heat_loss_saving_mj_per_hour * annual_hours / boiler_efficiency
        annual_gas_saving_volume = annual_gas_saving_mj / gas_calorific_value
        annual_gas_saving_yuan = annual_gas_saving_volume * gas_price

        unit_cost = INSULATION_UNIT_COST.get(pipe.insulation_material, 0.0)
        investment = pipe.length * unit_cost

        if annual_gas_saving_yuan <= 0:
            continue

        payback_period = investment / annual_gas_saving_yuan

        retrofits.append(PipeInsulationRetrofit(
            pipe_id=pid,
            pipe_name=pipe.name,
            original_thickness=pipe.insulation_thickness,
            new_thickness=new_thickness,
            original_heat_loss=original_heat_loss,
            new_heat_loss=new_heat_loss,
            heat_loss_saving=heat_loss_saving,
            annual_gas_saving=annual_gas_saving_yuan,
            investment=investment,
            payback_period=payback_period,
            insulation_material=pipe.insulation_material,
        ))

    retrofits.sort(key=lambda x: x.payback_period)
    return retrofits


def analyze_pump_efficiency(
    network: PipeNetwork,
    results: NetworkResults,
    electricity_price: float = DEFAULT_ELECTRICITY_PRICE,
    annual_hours: float = DEFAULT_ANNUAL_HOURS,
    vfd_efficiency_improvement: float = 0.15,
    vfd_unit_cost_per_kw: float = 800.0,
) -> List[PumpEfficiencyRetrofit]:
    retrofits = []

    for pid, pipe in network.pipes.items():
        if not pipe.has_pump or pipe.rated_head is None:
            continue

        pr = results.pipe_results.get(pid)
        if pr is None:
            continue

        actual_head = abs(pr.pump_head)
        rated_head = pipe.rated_head
        current_power = pr.power_consumption

        head_ratio = actual_head / rated_head if rated_head > 0 else 0

        if head_ratio < 0.6 or head_ratio > 1.1:
            status = "工况偏离严重"
            has_vfd_recommendation = True
        else:
            status = "工况正常"
            has_vfd_recommendation = False

        if current_power > 0 and has_vfd_recommendation:
            vfd_power_saving = current_power * vfd_efficiency_improvement
            vfd_annual_saving = vfd_power_saving * annual_hours * electricity_price
            vfd_investment = current_power * vfd_unit_cost_per_kw
            vfd_payback_period = vfd_investment / vfd_annual_saving if vfd_annual_saving > 0 else float('inf')
        else:
            vfd_annual_saving = 0.0
            vfd_investment = 0.0
            vfd_payback_period = float('inf')

        retrofits.append(PumpEfficiencyRetrofit(
            pipe_id=pid,
            pipe_name=pipe.name,
            rated_head=rated_head,
            actual_head=actual_head,
            current_power=current_power,
            status=status,
            has_vfd_recommendation=has_vfd_recommendation,
            vfd_annual_saving=vfd_annual_saving,
            vfd_investment=vfd_investment,
            vfd_payback_period=vfd_payback_period,
        ))

    return retrofits


def generate_comprehensive_retrofit_list(
    insulation_retrofits: List[PipeInsulationRetrofit],
    pump_retrofits: List[PumpEfficiencyRetrofit],
    payback_threshold: float = DEFAULT_PAYBACK_THRESHOLD,
) -> Tuple[List[ComprehensiveRetrofitItem], float, float, float]:
    items = []

    for ir in insulation_retrofits:
        if ir.payback_period > payback_threshold:
            continue
        items.append(ComprehensiveRetrofitItem(
            item_id=f"insulation_{ir.pipe_id}",
            item_name=ir.pipe_name,
            retrofit_type="保温改造",
            investment=ir.investment,
            annual_saving=ir.annual_gas_saving,
            payback_period=ir.payback_period,
            detail=f"保温层厚度 {ir.original_thickness*1000:.0f}mm → {ir.new_thickness*1000:.0f}mm，年节气 {ir.heat_loss_saving/1000:.2f} kW",
        ))

    for pr in pump_retrofits:
        if not pr.has_vfd_recommendation:
            continue
        if pr.vfd_payback_period > payback_threshold:
            continue
        items.append(ComprehensiveRetrofitItem(
            item_id=f"pump_{pr.pipe_id}",
            item_name=pr.pipe_name + " 变频改造",
            retrofit_type="泵站变频",
            investment=pr.vfd_investment,
            annual_saving=pr.vfd_annual_saving,
            payback_period=pr.vfd_payback_period,
            detail=f"额定扬程 {pr.rated_head:.1f}m，实际 {pr.actual_head:.1f}m，{pr.status}",
        ))

    items.sort(key=lambda x: x.payback_period)

    total_investment = sum(item.investment for item in items)
    total_annual_saving = sum(item.annual_saving for item in items)
    overall_payback = total_investment / total_annual_saving if total_annual_saving > 0 else float('inf')

    return items, total_investment, total_annual_saving, overall_payback


def run_economic_optimization(
    network: PipeNetwork,
    results: NetworkResults,
    electricity_price: float = DEFAULT_ELECTRICITY_PRICE,
    gas_price: float = DEFAULT_GAS_PRICE,
    gas_calorific_value: float = DEFAULT_GAS_CALORIFIC_VALUE,
    boiler_efficiency: float = DEFAULT_BOILER_EFFICIENCY,
    payback_threshold: float = DEFAULT_PAYBACK_THRESHOLD,
    daily_hours: float = DEFAULT_DAILY_HOURS,
    annual_hours_pump: float = DEFAULT_ANNUAL_HOURS,
) -> EconomicOptimizationResult:
    op_cost = calculate_operating_cost(
        network, results,
        electricity_price=electricity_price,
        gas_price=gas_price,
        gas_calorific_value=gas_calorific_value,
        boiler_efficiency=boiler_efficiency,
        daily_hours=daily_hours,
    )

    insulation_retrofits = analyze_insulation_retrofit(
        network, results,
        gas_price=gas_price,
        gas_calorific_value=gas_calorific_value,
        boiler_efficiency=boiler_efficiency,
        payback_threshold=payback_threshold,
        cp_water=network.water_specific_heat,
    )

    pump_retrofits = analyze_pump_efficiency(
        network, results,
        electricity_price=electricity_price,
        annual_hours=annual_hours_pump,
    )

    comprehensive_list, total_investment, total_annual_saving, overall_payback = \
        generate_comprehensive_retrofit_list(
            insulation_retrofits, pump_retrofits, payback_threshold
        )

    return EconomicOptimizationResult(
        operating_cost=op_cost,
        insulation_retrofits=insulation_retrofits,
        pump_retrofits=pump_retrofits,
        comprehensive_list=comprehensive_list,
        total_investment=total_investment,
        total_annual_saving=total_annual_saving,
        overall_payback_period=overall_payback,
    )
