from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from models import (
    PipeNetwork, Pipe, NetworkResults, PipeSectionResult,
    INSULATION_MATERIAL_NONE, INSULATION_MATERIAL_POLYURETHANE,
    INSULATION_MATERIAL_ROCK_WOOL, INSULATION_MATERIAL_GLASS_WOOL,
    INSULATION_CONDUCTIVITY, PIPE_WALL_CONDUCTIVITY,
    INNER_CONVECTION_COEFFICIENT, OUTER_SOIL_CONDUCTIVITY,
    get_wall_thickness, NODE_TYPE_END_USER, NODE_TYPE_SOURCE,
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
    criticality: float = 0.0
    composite_score: float = 0.0
    pipe_id: Optional[int] = None


@dataclass
class PhasePlanItem:
    year: int
    items: List[ComprehensiveRetrofitItem]
    annual_investment: float
    annual_saving: float
    cumulative_saving: float


@dataclass
class SensitivityPoint:
    parameter_multiplier: float
    parameter_value: float
    total_annual_saving: float
    overall_payback_period: float


@dataclass
class SensitivityAnalysisResult:
    parameter_name: str
    base_value: float
    points: List[SensitivityPoint]


@dataclass
class CashFlowPoint:
    year: int
    investment: float
    saving: float
    cumulative_cash_flow: float


@dataclass
class EconomicOptimizationResult:
    operating_cost: OperatingCostBreakdown = field(default_factory=OperatingCostBreakdown)
    insulation_retrofits: List[PipeInsulationRetrofit] = field(default_factory=list)
    pump_retrofits: List[PumpEfficiencyRetrofit] = field(default_factory=list)
    comprehensive_list: List[ComprehensiveRetrofitItem] = field(default_factory=list)
    total_investment: float = 0.0
    total_annual_saving: float = 0.0
    overall_payback_period: float = 0.0
    pipe_criticality: Dict[int, float] = field(default_factory=dict)
    phase_plan: List[PhasePlanItem] = field(default_factory=list)
    electricity_sensitivity: Optional[SensitivityAnalysisResult] = None
    gas_sensitivity: Optional[SensitivityAnalysisResult] = None
    cash_flow: List[CashFlowPoint] = field(default_factory=list)
    payback_year: Optional[int] = None


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


def calculate_pipe_criticality(
    network: PipeNetwork,
    results: NetworkResults,
) -> Dict[int, float]:
    total_supply_flow = 0.0
    for pr in results.pipe_results.values():
        total_supply_flow += abs(pr.flow_rate)

    pipe_criticality = {}

    for pid, pipe in network.pipes.items():
        pr = results.pipe_results.get(pid)
        if pr is None or abs(pr.flow_rate) < 1e-6:
            pipe_criticality[pid] = 0.0
            continue

        downstream_user_count = 0
        visited = set()
        queue = deque()
        queue.append(pipe.end_node_id)
        visited.add(pipe.end_node_id)

        while queue:
            node_id = queue.popleft()
            node = network.nodes.get(node_id)
            if node is None:
                continue
            if node.type == NODE_TYPE_END_USER:
                downstream_user_count += 1
            for dp in network.get_downstream_pipes(node_id):
                if dp.end_node_id not in visited:
                    visited.add(dp.end_node_id)
                    queue.append(dp.end_node_id)

        flow_ratio = abs(pr.flow_rate) / max(total_supply_flow, 1e-6)
        criticality = downstream_user_count * flow_ratio
        pipe_criticality[pid] = criticality

    return pipe_criticality


def generate_comprehensive_retrofit_list(
    insulation_retrofits: List[PipeInsulationRetrofit],
    pump_retrofits: List[PumpEfficiencyRetrofit],
    pipe_criticality: Dict[int, float],
    payback_threshold: float = DEFAULT_PAYBACK_THRESHOLD,
) -> Tuple[List[ComprehensiveRetrofitItem], float, float, float]:
    items = []

    for ir in insulation_retrofits:
        if ir.payback_period > payback_threshold:
            continue
        criticality = pipe_criticality.get(ir.pipe_id, 0.0)
        composite_score = (1.0 / max(ir.payback_period, 0.01)) * criticality if criticality > 0 else (1.0 / max(ir.payback_period, 0.01))
        items.append(ComprehensiveRetrofitItem(
            item_id=f"insulation_{ir.pipe_id}",
            item_name=ir.pipe_name,
            retrofit_type="保温改造",
            investment=ir.investment,
            annual_saving=ir.annual_gas_saving,
            payback_period=ir.payback_period,
            detail=f"保温层厚度 {ir.original_thickness*1000:.0f}mm → {ir.new_thickness*1000:.0f}mm，年节气 {ir.heat_loss_saving/1000:.2f} kW",
            criticality=criticality,
            composite_score=composite_score,
            pipe_id=ir.pipe_id,
        ))

    for pr in pump_retrofits:
        if not pr.has_vfd_recommendation:
            continue
        if pr.vfd_payback_period > payback_threshold:
            continue
        criticality = pipe_criticality.get(pr.pipe_id, 0.0)
        composite_score = (1.0 / max(pr.vfd_payback_period, 0.01)) * criticality if criticality > 0 else (1.0 / max(pr.vfd_payback_period, 0.01))
        items.append(ComprehensiveRetrofitItem(
            item_id=f"pump_{pr.pipe_id}",
            item_name=pr.pipe_name + " 变频改造",
            retrofit_type="泵站变频",
            investment=pr.vfd_investment,
            annual_saving=pr.vfd_annual_saving,
            payback_period=pr.vfd_payback_period,
            detail=f"额定扬程 {pr.rated_head:.1f}m，实际 {pr.actual_head:.1f}m，{pr.status}",
            criticality=criticality,
            composite_score=composite_score,
            pipe_id=pr.pipe_id,
        ))

    items.sort(key=lambda x: -x.composite_score)

    total_investment = sum(item.investment for item in items)
    total_annual_saving = sum(item.annual_saving for item in items)
    overall_payback = total_investment / total_annual_saving if total_annual_saving > 0 else float('inf')

    return items, total_investment, total_annual_saving, overall_payback


def generate_phase_plan(
    retrofit_items: List[ComprehensiveRetrofitItem],
    annual_budget: float = 500000.0,
    max_years: int = 5,
) -> List[PhasePlanItem]:
    if not retrofit_items:
        return []

    sorted_items = sorted(retrofit_items, key=lambda x: x.payback_period)

    phase_plan = []
    remaining_items = sorted_items[:]
    cumulative_saving = 0.0
    year = 1

    while remaining_items and year <= max_years:
        year_items = []
        year_investment = 0.0
        year_saving = 0.0

        i = 0
        while i < len(remaining_items):
            item = remaining_items[i]
            if item.investment > annual_budget:
                if not year_items and year_investment == 0.0:
                    year_items.append(item)
                    year_investment += item.investment
                    year_saving += item.annual_saving
                    remaining_items.pop(i)
                    continue
                else:
                    i += 1
                    continue

            if year_investment + item.investment <= annual_budget:
                year_items.append(item)
                year_investment += item.investment
                year_saving += item.annual_saving
                remaining_items.pop(i)
            else:
                i += 1

        if year_items:
            cumulative_saving += year_saving
            phase_plan.append(PhasePlanItem(
                year=year,
                items=year_items,
                annual_investment=year_investment,
                annual_saving=year_saving,
                cumulative_saving=cumulative_saving,
            ))
            year += 1
        else:
            break

    if remaining_items and year <= max_years:
        for item in remaining_items:
            if year > max_years:
                break
            cumulative_saving += item.annual_saving
            phase_plan.append(PhasePlanItem(
                year=year,
                items=[item],
                annual_investment=item.investment,
                annual_saving=item.annual_saving,
                cumulative_saving=cumulative_saving,
            ))
            year += 1

    return phase_plan


def perform_sensitivity_analysis(
    network: PipeNetwork,
    results: NetworkResults,
    base_electricity_price: float,
    base_gas_price: float,
    gas_calorific_value: float,
    boiler_efficiency: float,
    payback_threshold: float,
    pipe_criticality: Dict[int, float],
    parameter_name: str,
    num_points: int = 5,
) -> SensitivityAnalysisResult:
    if parameter_name == "electricity":
        base_value = base_electricity_price
    elif parameter_name == "gas":
        base_value = base_gas_price
    else:
        raise ValueError(f"Unknown parameter: {parameter_name}")

    multipliers = np.linspace(0.5, 2.0, num_points)
    points = []

    for mult in multipliers:
        if parameter_name == "electricity":
            elec_price = base_electricity_price * mult
            gas_price_val = base_gas_price
        else:
            elec_price = base_electricity_price
            gas_price_val = base_gas_price * mult

        ins_retrofits = analyze_insulation_retrofit(
            network, results,
            gas_price=gas_price_val,
            gas_calorific_value=gas_calorific_value,
            boiler_efficiency=boiler_efficiency,
            payback_threshold=payback_threshold,
            cp_water=network.water_specific_heat,
        )

        pump_retrofits = analyze_pump_efficiency(
            network, results,
            electricity_price=elec_price,
        )

        items, total_inv, total_saving, overall_payback = generate_comprehensive_retrofit_list(
            ins_retrofits, pump_retrofits, pipe_criticality, payback_threshold
        )

        points.append(SensitivityPoint(
            parameter_multiplier=mult,
            parameter_value=base_value * mult,
            total_annual_saving=total_saving,
            overall_payback_period=overall_payback,
        ))

    return SensitivityAnalysisResult(
        parameter_name=parameter_name,
        base_value=base_value,
        points=points,
    )


def calculate_cash_flow(
    phase_plan: List[PhasePlanItem],
) -> Tuple[List[CashFlowPoint], Optional[int]]:
    if not phase_plan:
        return [], None

    max_year = max(item.year for item in phase_plan)
    cash_flow = []

    cumulative = 0.0
    payback_year = None

    for year in range(0, max_year + 1):
        year_investment = 0.0
        year_saving = 0.0

        if year == 0:
            for phase in phase_plan:
                if phase.year == 1:
                    year_investment = phase.annual_investment
                    break
            cumulative = -year_investment
        else:
            for phase in phase_plan:
                if phase.year <= year:
                    year_saving += phase.annual_saving
            if year >= 2:
                for phase in phase_plan:
                    if phase.year == year:
                        year_investment = phase.annual_investment
                        break
            cumulative = cumulative + year_saving - year_investment

        if payback_year is None and cumulative >= 0 and year > 0:
            payback_year = year

        cash_flow.append(CashFlowPoint(
            year=year,
            investment=year_investment,
            saving=year_saving,
            cumulative_cash_flow=cumulative,
        ))

    return cash_flow, payback_year


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
    annual_budget: float = 500000.0,
    max_phase_years: int = 5,
) -> EconomicOptimizationResult:
    op_cost = calculate_operating_cost(
        network, results,
        electricity_price=electricity_price,
        gas_price=gas_price,
        gas_calorific_value=gas_calorific_value,
        boiler_efficiency=boiler_efficiency,
        daily_hours=daily_hours,
    )

    pipe_criticality = calculate_pipe_criticality(network, results)

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
            insulation_retrofits, pump_retrofits, pipe_criticality, payback_threshold
        )

    phase_plan = generate_phase_plan(
        comprehensive_list,
        annual_budget=annual_budget,
        max_years=max_phase_years,
    )

    electricity_sensitivity = perform_sensitivity_analysis(
        network, results,
        base_electricity_price=electricity_price,
        base_gas_price=gas_price,
        gas_calorific_value=gas_calorific_value,
        boiler_efficiency=boiler_efficiency,
        payback_threshold=payback_threshold,
        pipe_criticality=pipe_criticality,
        parameter_name="electricity",
        num_points=5,
    )

    gas_sensitivity = perform_sensitivity_analysis(
        network, results,
        base_electricity_price=electricity_price,
        base_gas_price=gas_price,
        gas_calorific_value=gas_calorific_value,
        boiler_efficiency=boiler_efficiency,
        payback_threshold=payback_threshold,
        pipe_criticality=pipe_criticality,
        parameter_name="gas",
        num_points=5,
    )

    cash_flow, payback_year = calculate_cash_flow(phase_plan)

    return EconomicOptimizationResult(
        operating_cost=op_cost,
        insulation_retrofits=insulation_retrofits,
        pump_retrofits=pump_retrofits,
        comprehensive_list=comprehensive_list,
        total_investment=total_investment,
        total_annual_saving=total_annual_saving,
        overall_payback_period=overall_payback,
        pipe_criticality=pipe_criticality,
        phase_plan=phase_plan,
        electricity_sensitivity=electricity_sensitivity,
        gas_sensitivity=gas_sensitivity,
        cash_flow=cash_flow,
        payback_year=payback_year,
    )
