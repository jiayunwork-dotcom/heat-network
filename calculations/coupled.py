import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize

from models import (
    PipeNetwork, Node, Pipe, PipeSectionResult, NodeResult, NetworkResults,
    NODE_TYPE_SOURCE, NODE_TYPE_END_USER,
    get_water_properties,
)
from calculations.hydraulic import solve_hydraulics, compute_pump_head
from calculations.thermal import solve_thermal


def solve_coupled(
    network: PipeNetwork,
    source_supply_temps: Optional[Dict[int, float]] = None,
    temp_tolerance: float = 0.01,
    max_coupled_iter: int = 30,
) -> NetworkResults:
    source_nodes = network.get_nodes_by_type(NODE_TYPE_SOURCE)
    if source_supply_temps is None:
        source_supply_temps = {}
        for i, sn in enumerate(source_nodes):
            source_supply_temps[sn.id] = 110.0 if i == 0 else 108.0
    node_temps = {}
    avg_init_t = np.mean(list(source_supply_temps.values()))
    for nid in network.nodes:
        node_temps[nid] = avg_init_t if nid not in source_supply_temps else source_supply_temps[nid]
    total_iter = 0
    converged = False
    pressures = {}
    flows = {}
    pipe_hyd_res = {}
    pipe_therm_res = {}
    for coupled_iter in range(max_coupled_iter):
        pressures, flows, pipe_hyd_res, hyd_iter, hyd_conv = solve_hydraulics(
            network, node_temperatures=node_temps
        )
        total_iter += hyd_iter
        new_temps, pipe_therm_res = solve_thermal(
            network, flows, source_supply_temps=source_supply_temps,
            env_temp=network.environment_temperature,
            cp_water=network.water_specific_heat,
        )
        max_temp_diff = 0.0
        for nid in network.nodes:
            if nid in new_temps and nid in node_temps:
                if nid not in source_supply_temps:
                    diff = abs(new_temps[nid] - node_temps[nid])
                    max_temp_diff = max(max_temp_diff, diff)
        node_temps = new_temps
        if max_temp_diff < temp_tolerance and hyd_conv:
            converged = True
            total_iter = coupled_iter + 1
            break
        total_iter = coupled_iter + 1
    results = NetworkResults()
    results.iterations = total_iter
    results.converged = converged
    g = 9.81
    cp_water = network.water_specific_heat
    for pid in network.pipes:
        pipe = network.pipes[pid]
        pr = PipeSectionResult(pipe_id=pid)
        if pid in pipe_hyd_res:
            pr.flow_rate = pipe_hyd_res[pid].flow_rate
            pr.velocity = pipe_hyd_res[pid].velocity
            pr.reynolds = pipe_hyd_res[pid].reynolds
            pr.friction_factor = pipe_hyd_res[pid].friction_factor
            pr.pressure_loss = pipe_hyd_res[pid].pressure_loss
            pr.local_pressure_loss = pipe_hyd_res[pid].local_pressure_loss
            pr.total_pressure_loss = pipe_hyd_res[pid].total_pressure_loss
            pr.pump_head = pipe_hyd_res[pid].pump_head
        if pid in pipe_therm_res:
            pr.inlet_temperature = pipe_therm_res[pid].inlet_temperature
            pr.outlet_temperature = pipe_therm_res[pid].outlet_temperature
            pr.temperature_drop = pipe_therm_res[pid].temperature_drop
            pr.heat_loss = pipe_therm_res[pid].heat_loss
            pr.heat_transfer_coefficient = pipe_therm_res[pid].heat_transfer_coefficient
        if pipe.has_pump and pr.flow_rate > 0:
            density, _ = get_water_properties((pr.inlet_temperature + pr.outlet_temperature) / 2.0)
            flow_vol = pr.flow_rate
            head = abs(pr.pump_head)
            pump_eff = 0.75
            if pipe.pump_efficiency_curve and len(pipe.pump_efficiency_curve) >= 2:
                qp = [p[0] for p in pipe.pump_efficiency_curve]
                hp = [p[1] for p in pipe.pump_efficiency_curve]
                q_abs = abs(pr.flow_rate)
                for k in range(len(qp) - 1):
                    if qp[k] <= q_abs <= qp[k + 1]:
                        t = (q_abs - qp[k]) / max(qp[k + 1] - qp[k], 1e-6)
                        h_ratio = (hp[k] + t * (hp[k + 1] - hp[k])) / max(pipe.rated_head, 1e-6)
                        pump_eff = 0.6 + 0.2 * h_ratio * (2 - h_ratio)
                        break
            shaft_power = density * g * flow_vol * head / max(pump_eff, 1e-3)
            pr.power_consumption = shaft_power / 1000.0
        results.pipe_results[pid] = pr
    for nid in network.nodes:
        nr = NodeResult(node_id=nid)
        nr.pressure = pressures.get(nid, 0.0)
        nr.temperature = node_temps.get(nid, 0.0)
        conn_pipes = network.get_connected_pipes(nid)
        for conn_pipe in conn_pipes:
            pr = results.pipe_results.get(conn_pipe.id)
            if pr and pr.flow_rate > 0:
                if conn_pipe.start_node_id == nid:
                    nr.flow_out += pr.flow_rate
                elif conn_pipe.end_node_id == nid:
                    nr.flow_in += pr.flow_rate
        results.node_results[nid] = nr
    total_heat_supplied = 0.0
    for sn in source_nodes:
        for dp in network.get_downstream_pipes(sn.id):
            pr = results.pipe_results.get(dp.id)
            if pr and pr.flow_rate > 0:
                density, _ = get_water_properties(pr.inlet_temperature)
                mass_flow = pr.flow_rate * density
                t_diff = pr.inlet_temperature - network.environment_temperature
                total_heat_supplied += mass_flow * cp_water * t_diff
    total_heat_loss = sum(pr.heat_loss for pr in results.pipe_results.values())
    results.total_heat_supplied = total_heat_supplied
    results.total_heat_loss = total_heat_loss
    results.heat_loss_rate = total_heat_loss / max(total_heat_supplied, 1e-6) * 100.0
    results.total_pump_power = sum(pr.power_consumption for pr in results.pipe_results.values())
    heat_delivered_gj = max(total_heat_supplied - total_heat_loss, 0) / 1.0e9
    results.specific_energy_consumption = (results.total_pump_power * 3600.0 / 3.6e6) / max(heat_delivered_gj, 1e-6) if heat_delivered_gj > 0 else 0.0
    return results


def optimize_source_allocation_equal(network: PipeNetwork) -> Dict[int, float]:
    sources = network.get_nodes_by_type(NODE_TYPE_SOURCE)
    total_capacity = sum((s.rated_capacity or 0) for s in sources)
    if total_capacity <= 0:
        return {s.id: 1.0 / max(len(sources), 1) for s in sources}
    return {s.id: (s.rated_capacity or 0) / total_capacity for s in sources}


def optimize_source_allocation_min_energy(
    network: PipeNetwork,
    source_supply_temps: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, float], NetworkResults]:
    sources = network.get_nodes_by_type(NODE_TYPE_SOURCE)
    n_sources = len(sources)
    if n_sources <= 1:
        ratio = {s.id: 1.0 for s in sources}
        result = solve_coupled(network, source_supply_temps)
        return ratio, result
    end_users = network.get_nodes_by_type(NODE_TYPE_END_USER)
    total_demand = sum((eu.design_flow or 0) for eu in end_users)
    source_ids = [s.id for s in sources]
    source_map = {i: sid for i, sid in enumerate(source_ids)}
    capacities = np.array([(s.rated_capacity or total_demand) for s in sources], dtype=float)
    capacities = capacities / capacities.sum() * total_demand
    initial_x = capacities / capacities.sum()
    def objective(x):
        x_norm = np.abs(x) / np.sum(np.abs(x))
        for i, sid in enumerate(source_ids):
            flow_fraction = x_norm[i] * total_demand
            assigned = 0.0
            down_pipes = network.get_downstream_pipes(sid)
            if down_pipes:
                per_pipe = flow_fraction / len(down_pipes)
                for dp in down_pipes:
                    original = network.nodes[dp.end_node_id]
        result = solve_coupled(network, source_supply_temps)
        return result.total_pump_power + 0.01 * np.sum((x_norm - initial_x) ** 2)
    bounds = [(0.0, 1.0)] * n_sources
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1.0}
    try:
        opt_result = minimize(
            objective,
            initial_x,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraint],
            options={'maxiter': 50, 'ftol': 1e-4},
        )
        optimal_x = np.abs(opt_result.x) / np.sum(np.abs(opt_result.x))
    except Exception:
        optimal_x = initial_x
    ratios = {source_map[i]: float(optimal_x[i]) for i in range(n_sources)}
    final_result = solve_coupled(network, source_supply_temps)
    return ratios, final_result


def analyze_hydraulic_balance(
    network: PipeNetwork,
    results: NetworkResults,
    tolerance: float = 0.15,
) -> Tuple[Dict[int, float], List[Dict]]:
    end_users = network.get_nodes_by_type(NODE_TYPE_END_USER)
    flow_ratios: Dict[int, float] = {}
    suggestions: List[Dict] = []
    for eu in end_users:
        if eu.design_flow and eu.design_flow > 0:
            nr = results.node_results.get(eu.id)
            actual_flow = nr.flow_in if nr else 0.0
            ratio = actual_flow / eu.design_flow
            flow_ratios[eu.id] = ratio
            if abs(ratio - 1.0) > tolerance:
                incoming = network.get_upstream_pipes(eu.id)
                valve_pipes = [p for p in incoming if p.has_valve]
                target_pipe = valve_pipes[0] if valve_pipes else (incoming[0] if incoming else None)
                if target_pipe:
                    if ratio > 1.0:
                        closing_pct = min(90, (ratio - 1.0) / ratio * 100 * 0.8)
                        suggestion = {
                            "user_id": eu.id,
                            "user_name": eu.name,
                            "design_flow": eu.design_flow,
                            "actual_flow": actual_flow,
                            "flow_ratio": ratio,
                            "pipe_id": target_pipe.id,
                            "pipe_name": target_pipe.name,
                            "action": "关小阀门",
                            "adjustment": closing_pct,
                            "reason": f"流量偏大{(ratio-1)*100:.1f}%",
                        }
                    else:
                        opening_pct = min(100, (1.0 - ratio) * 100 * 1.2)
                        suggestion = {
                            "user_id": eu.id,
                            "user_name": eu.name,
                            "design_flow": eu.design_flow,
                            "actual_flow": actual_flow,
                            "flow_ratio": ratio,
                            "pipe_id": target_pipe.id,
                            "pipe_name": target_pipe.name,
                            "action": "开大阀门",
                            "adjustment": opening_pct,
                            "reason": f"流量偏小{(1-ratio)*100:.1f}%",
                        }
                    suggestions.append(suggestion)
    return flow_ratios, suggestions
