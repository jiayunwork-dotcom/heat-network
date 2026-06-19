import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from models import (
    PipeNetwork, Pipe, Node, PipeSectionResult,
    NODE_TYPE_SOURCE, NODE_TYPE_END_USER, NODE_TYPE_HEAT_EXCHANGER, NODE_TYPE_BRANCH,
    INSULATION_CONDUCTIVITY, PIPE_WALL_CONDUCTIVITY,
    INNER_CONVECTION_COEFFICIENT, OUTER_SOIL_CONDUCTIVITY,
    get_wall_thickness, get_water_properties,
)


def compute_overall_heat_transfer_coefficient(pipe: Pipe) -> float:
    d_inner = pipe.diameter
    wall_thickness = get_wall_thickness(pipe.material, d_inner)
    d_wall_outer = d_inner + 2 * wall_thickness
    insul_thick = pipe.insulation_thickness if pipe.insulation_material != "none" else 0.0
    d_insul_outer = d_wall_outer + 2 * insul_thick
    r_inner_conv = 1.0 / (INNER_CONVECTION_COEFFICIENT * np.pi * d_inner)
    cond_wall = PIPE_WALL_CONDUCTIVITY.get(pipe.material, 45.0)
    r_wall = np.log(d_wall_outer / d_inner) / (2.0 * np.pi * cond_wall)
    if insul_thick > 0:
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
    return U_per_m


def compute_pipe_temperature_drop(
    pipe: Pipe, flow_rate: float, inlet_temp: float, env_temp: float,
    cp_water: float = 4186.0,
) -> Tuple[float, float, float]:
    if abs(flow_rate) < 1e-10:
        return inlet_temp, 0.0, 0.0
    density, _ = get_water_properties(inlet_temp)
    mass_flow = abs(flow_rate) * density
    U_per_m = compute_overall_heat_transfer_coefficient(pipe)
    wall_thickness = get_wall_thickness(pipe.material, pipe.diameter)
    d_outer = pipe.diameter + 2 * wall_thickness + 2 * pipe.insulation_thickness
    A_total = np.pi * d_outer * pipe.length
    UA = U_per_m * pipe.length
    exponent = -UA / max(mass_flow * cp_water, 1e-6)
    outlet_temp = env_temp + (inlet_temp - env_temp) * np.exp(exponent)
    temp_drop = inlet_temp - outlet_temp
    avg_temp = 0.5 * (inlet_temp + outlet_temp)
    density_avg, _ = get_water_properties(avg_temp)
    mass_flow_avg = abs(flow_rate) * density_avg
    heat_loss = mass_flow_avg * cp_water * temp_drop
    return outlet_temp, temp_drop, heat_loss


def _topological_sort(network: PipeNetwork) -> List[int]:
    in_degree = {nid: 0 for nid in network.nodes}
    for pipe in network.pipes.values():
        if "回水" in pipe.name or "return" in pipe.name.lower():
            continue
        if pipe.start_node_id in in_degree and pipe.end_node_id in in_degree:
            in_degree[pipe.end_node_id] += 1
    source_ids = [n.id for n in network.get_nodes_by_type(NODE_TYPE_SOURCE)]
    for sid in source_ids:
        in_degree[sid] = 0
    queue = deque()
    visited = set()
    for nid, deg in in_degree.items():
        if deg == 0:
            queue.append(nid)
            visited.add(nid)
    result = []
    while queue:
        nid = queue.popleft()
        result.append(nid)
        for pipe in network.get_downstream_pipes(nid):
            if "回水" in pipe.name or "return" in pipe.name.lower():
                continue
            child = pipe.end_node_id
            if child in in_degree:
                in_degree[child] -= 1
                if in_degree[child] <= 0 and child not in visited:
                    queue.append(child)
                    visited.add(child)
    for nid in network.nodes:
        if nid not in visited:
            result.append(nid)
    return result


def solve_thermal(
    network: PipeNetwork,
    flow_rates: Dict[int, float],
    source_supply_temps: Optional[Dict[int, float]] = None,
    env_temp: Optional[float] = None,
    cp_water: float = 4186.0,
) -> Tuple[Dict[int, float], Dict[int, PipeSectionResult]]:
    env_t = env_temp if env_temp is not None else network.environment_temperature
    node_temps: Dict[int, float] = {}
    source_nodes = network.get_nodes_by_type(NODE_TYPE_SOURCE)
    for sn in source_nodes:
        if source_supply_temps and sn.id in source_supply_temps:
            node_temps[sn.id] = source_supply_temps[sn.id]
        elif sn.supply_pressure:
            node_temps[sn.id] = 110.0 if sn.id == 0 else 108.0
        else:
            node_temps[sn.id] = 105.0
    pipe_results: Dict[int, PipeSectionResult] = {}
    for pid in network.pipes:
        pipe_results[pid] = PipeSectionResult(pipe_id=pid)
        pipe_results[pid].heat_transfer_coefficient = compute_overall_heat_transfer_coefficient(network.pipes[pid])
    sorted_nodes = _topological_sort(network)
    for nid in sorted_nodes:
        node = network.nodes[nid]
        if nid not in node_temps:
            incoming = network.get_upstream_pipes(nid)
            total_mass_flow = 0.0
            weighted_temp = 0.0
            for ip in incoming:
                if "回水" in ip.name or "return" in ip.name.lower():
                    continue
                flow = flow_rates.get(ip.id, 0.0)
                if flow > 0 and ip.id in pipe_results:
                    density, _ = get_water_properties(pipe_results[ip.id].outlet_temperature if pipe_results[ip.id].outlet_temperature > 0 else 80.0)
                    mass = flow * density
                    temp = pipe_results[ip.id].outlet_temperature
                    total_mass_flow += mass
                    weighted_temp += mass * temp
            if total_mass_flow > 0:
                node_temps[nid] = weighted_temp / total_mass_flow
            else:
                node_temps[nid] = env_t + 20.0
        downstream = network.get_downstream_pipes(nid)
        for dp in downstream:
            if "回水" in dp.name or "return" in dp.name.lower():
                continue
            flow = flow_rates.get(dp.id, 0.0)
            if flow > 1e-10:
                inlet_t = node_temps[nid]
                out_t, drop, hloss = compute_pipe_temperature_drop(dp, flow, inlet_t, env_t, cp_water)
                pipe_results[dp.id].flow_rate = flow
                pipe_results[dp.id].inlet_temperature = inlet_t
                pipe_results[dp.id].outlet_temperature = out_t
                pipe_results[dp.id].temperature_drop = drop
                pipe_results[dp.id].heat_loss = hloss
                pipe_results[dp.id].heat_transfer_coefficient = compute_overall_heat_transfer_coefficient(dp)
            elif flow < -1e-10:
                pass
    remaining = [nid for nid in network.nodes if nid not in node_temps]
    for nid in remaining:
        node_temps[nid] = env_t + 15.0
    for pid, pres in pipe_results.items():
        if pres.inlet_temperature == 0 and pres.outlet_temperature == 0:
            pipe = network.pipes[pid]
            t_s = node_temps.get(pipe.start_node_id, 80.0)
            t_e = node_temps.get(pipe.end_node_id, 80.0)
            flow = abs(flow_rates.get(pid, 0.0))
            if flow > 1e-10:
                out_t, drop, hloss = compute_pipe_temperature_drop(pipe, flow, t_s, env_t, cp_water)
                pres.inlet_temperature = t_s
                pres.outlet_temperature = out_t
                pres.temperature_drop = drop
                pres.heat_loss = hloss
    return node_temps, pipe_results
