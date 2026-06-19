import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from models import (
    PipeNetwork, Pipe, Node, PipeSectionResult, NodeResult, NetworkResults,
    NODE_TYPE_SOURCE, NODE_TYPE_END_USER,
    get_pipe_roughness, get_water_properties,
)


def solve_colebrook_white(reynolds: float, roughness: float, diameter: float) -> float:
    if reynolds < 2300:
        return 64.0 / reynolds if reynolds > 0 else 0.02
    r_d = roughness / diameter
    f = 0.02
    for _ in range(30):
        lhs = 1.0 / np.sqrt(max(f, 1e-10))
        rhs = -2.0 * np.log10(r_d / 3.7 + 2.51 / (reynolds * np.sqrt(max(f, 1e-10))))
        error = lhs - rhs
        if abs(error) < 1e-7:
            break
        df = -0.5 * (f ** 1.5) * error
        f_new = f + 0.5 * df
        if f_new < 1e-4:
            f_new = 1e-4
        if f_new > 0.1:
            f_new = 0.1
        f = f_new
    return max(f, 1e-5)


def compute_pipe_hydraulics(
    pipe: Pipe, flow_rate: float, temperature: float,
) -> Tuple[float, float, float, float, float]:
    if abs(flow_rate) < 1e-15:
        return 0.0, 0.0, 0.02, 0.0, 0.0
    density, viscosity = get_water_properties(temperature)
    area = np.pi * (pipe.diameter / 2.0) ** 2
    velocity = flow_rate / area / density
    reynolds = abs(velocity) * pipe.diameter / max(viscosity, 1e-7)
    roughness = get_pipe_roughness(pipe.material, pipe.pipe_age)
    f = solve_colebrook_white(reynolds, roughness, pipe.diameter)
    g = 9.81
    v_sq = velocity ** 2
    eq_len = pipe.length * (1.0 + pipe.equivalent_local_length_ratio)
    head_loss = f * eq_len / pipe.diameter * v_sq / (2.0 * g)
    local_ratio = pipe.equivalent_local_length_ratio / (1.0 + pipe.equivalent_local_length_ratio)
    head_loss_local = head_loss * local_ratio
    head_loss_friction = head_loss * (1.0 - local_ratio)
    sign = 1.0 if flow_rate >= 0 else -1.0
    return (
        velocity,
        max(reynolds, 10),
        f,
        head_loss_friction * sign,
        head_loss_local * sign,
    )


def compute_pump_head(pipe: Pipe, flow_rate: float) -> float:
    if not pipe.has_pump or pipe.rated_head is None:
        return 0.0
    q_abs = abs(flow_rate)
    if pipe.pump_efficiency_curve and len(pipe.pump_efficiency_curve) >= 2:
        q_points = np.array([p[0] for p in pipe.pump_efficiency_curve])
        h_points = np.array([p[1] for p in pipe.pump_efficiency_curve])
        if q_abs >= q_points[-1]:
            return 0.0
        head = np.interp(q_abs, q_points, h_points)
    else:
        q_max = max(pipe.rated_head / 25.0, 0.1)
        q_ratio = min(q_abs / q_max, 1.0)
        head = pipe.rated_head * max(0, 1.0 - q_ratio ** 2)
    return head if flow_rate >= 0 else -head


def _estimate_supply_network_flows(network: PipeNetwork) -> Tuple[Dict[int, float], Dict[int, int]]:
    flows = {}
    pipe_depth = {}
    end_users = network.get_nodes_by_type(NODE_TYPE_END_USER)
    user_demand = {eu.id: eu.design_flow or 0.0 for eu in end_users}
    node_out_flow: Dict[int, float] = {}
    for uid, dem in user_demand.items():
        node_out_flow[uid] = node_out_flow.get(uid, 0) + dem
    visited = set()
    queue = deque([eu.id for eu in end_users])
    upstream_map: Dict[int, List[int]] = {}
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        incoming = network.get_upstream_pipes(nid)
        if not incoming:
            continue
        total_out = node_out_flow.get(nid, 0)
        n_incoming = len(incoming)
        per_pipe = total_out / max(n_incoming, 1)
        for ip in incoming:
            if ip.start_node_id == ip.end_node_id:
                continue
            flows[ip.id] = per_pipe
            pipe_depth[ip.id] = 0
            start_node = ip.start_node_id
            node_out_flow[start_node] = node_out_flow.get(start_node, 0) + per_pipe
            if start_node not in visited:
                queue.append(start_node)
    return flows, pipe_depth


def solve_hydraulics_simplified(
    network: PipeNetwork,
    node_temperatures: Optional[Dict[int, float]] = None,
    max_iterations: int = 80,
    tolerance: float = 1e-4,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, PipeSectionResult], int, bool]:
    if node_temperatures is None:
        avg_temp = 80.0
        node_temperatures = {nid: avg_temp for nid in network.nodes}
    g = 9.81
    source_ids = [n.id for n in network.get_nodes_by_type(NODE_TYPE_SOURCE)]
    end_users = network.get_nodes_by_type(NODE_TYPE_END_USER)
    init_flows, _ = _estimate_supply_network_flows(network)
    pipe_flows: Dict[int, float] = {}
    for pid in network.pipes:
        if pid in init_flows and init_flows[pid] > 0:
            pipe_flows[pid] = init_flows[pid]
        else:
            pipe_flows[pid] = 0.001
    source_pressures: Dict[int, float] = {}
    for sid in source_ids:
        sn = network.nodes[sid]
        source_pressures[sid] = sn.supply_pressure if sn.supply_pressure else 50.0
    pipe_results: Dict[int, PipeSectionResult] = {}
    for pid in network.pipes:
        pipe_results[pid] = PipeSectionResult(pipe_id=pid)
    converged = False
    iteration = 0
    supply_pipes = {}
    return_pipes = {}
    for pid, pipe in network.pipes.items():
        sn_type = network.nodes[pipe.start_node_id].type
        en_type = network.nodes[pipe.end_node_id].type
        if "回水" in pipe.name or "return" in pipe.name.lower():
            return_pipes[pid] = pipe
        else:
            supply_pipes[pid] = pipe
    for iteration in range(max_iterations):
        node_pressures: Dict[int, float] = {sid: source_pressures[sid] for sid in source_ids}
        max_correction = 0.0
        sorted_nodes = _topo_sort_supply(network, source_ids)
        for nid in sorted_nodes:
            if nid in node_pressures:
                continue
            incoming = network.get_upstream_pipes(nid)
            best_p = None
            for ip in incoming:
                if ip.id not in supply_pipes:
                    continue
                start_p = node_pressures.get(ip.start_node_id)
                if start_p is None or ip.id not in pipe_flows:
                    continue
                flow = pipe_flows[ip.id]
                avg_t = 0.5 * (
                    node_temperatures.get(ip.start_node_id, 80.0) +
                    node_temperatures.get(ip.end_node_id, 80.0)
                )
                density, _ = get_water_properties(avg_t)
                vel, re, f, hf, hl = compute_pipe_hydraulics(ip, flow, avg_t)
                pump_h = compute_pump_head(ip, flow)
                head_loss_m = hf + hl
                p_downstream = start_p - head_loss_m + pump_h
                if best_p is None or p_downstream > best_p:
                    best_p = p_downstream
            if best_p is not None:
                node_pressures[nid] = best_p
        demand_errors: Dict[int, float] = {}
        for eu in end_users:
            if eu.design_flow is None or eu.design_flow <= 0:
                continue
            actual_in = 0.0
            for ip in network.get_upstream_pipes(eu.id):
                if ip.id in pipe_flows and ip.id in supply_pipes:
                    actual_in += pipe_flows[ip.id]
            demand_errors[eu.id] = (eu.design_flow - actual_in) / max(eu.design_flow, 1e-6)
        for nid in sorted_nodes:
            if nid in source_ids:
                continue
            incoming = [p for p in network.get_upstream_pipes(nid) if p.id in supply_pipes]
            outgoing = [p for p in network.get_downstream_pipes(nid) if p.id in supply_pipes]
            if not incoming:
                continue
            if nid in demand_errors:
                correction_ratio = 1.0 + 0.5 * demand_errors[nid]
            else:
                total_out = sum(pipe_flows.get(op.id, 0) for op in outgoing)
                total_in = sum(pipe_flows.get(ip.id, 0) for ip in incoming)
                if total_in > 1e-10 and total_out > 1e-10:
                    correction_ratio = total_out / total_in
                else:
                    correction_ratio = 1.0
            for ip in incoming:
                old_flow = pipe_flows.get(ip.id, 0.001)
                new_flow = max(old_flow * correction_ratio, 1e-6)
                max_correction = max(max_correction, abs(new_flow - old_flow) / max(old_flow, 1e-6))
                pipe_flows[ip.id] = new_flow
        if max_correction < tolerance and iteration > 5:
            converged = True
            break
    for pid, pipe in network.pipes.items():
        flow = pipe_flows.get(pid, 0.0)
        avg_t = 0.5 * (
            node_temperatures.get(pipe.start_node_id, 80.0) +
            node_temperatures.get(pipe.end_node_id, 80.0)
        )
        density, _ = get_water_properties(avg_t)
        vel, re, f, hf, hl = compute_pipe_hydraulics(pipe, flow, avg_t)
        pump_h = compute_pump_head(pipe, flow)
        pr = pipe_results[pid]
        pr.flow_rate = flow
        pr.velocity = vel
        pr.reynolds = re
        pr.friction_factor = f
        pr.pressure_loss = density * g * hf
        pr.local_pressure_loss = density * g * hl
        pr.total_pressure_loss = density * g * (hf + hl)
        pr.pump_head = pump_h
    min_pressure = 10.0
    all_nodes = list(network.nodes.keys())
    node_pressures_final: Dict[int, float] = {}
    for sid in source_ids:
        node_pressures_final[sid] = source_pressures[sid]
    for _ in range(3):
        for nid in all_nodes:
            if nid in node_pressures_final:
                continue
            incoming = network.get_upstream_pipes(nid)
            best_p = None
            for ip in incoming:
                p_start = node_pressures_final.get(ip.start_node_id)
                if p_start is None:
                    continue
                pr = pipe_results[ip.id]
                density, _ = get_water_properties(80.0)
                head_loss_m = (pr.pressure_loss + pr.local_pressure_loss) / (density * g)
                pump_term = pr.pump_head if ip.has_pump else 0
                p_end = p_start - head_loss_m + pump_term
                if best_p is None or p_end > best_p:
                    best_p = p_end
            if best_p is not None:
                node_pressures_final[nid] = best_p
    for nid in all_nodes:
        if nid not in node_pressures_final:
            node_pressures_final[nid] = min_pressure
        node_pressures_final[nid] = max(node_pressures_final[nid], min_pressure)
    return node_pressures_final, pipe_flows, pipe_results, iteration + 1, converged


def _topo_sort_supply(network: PipeNetwork, source_ids: List[int]) -> List[int]:
    in_count = {nid: 0 for nid in network.nodes}
    for pipe in network.pipes.values():
        if "回水" in pipe.name or "return" in pipe.name.lower():
            continue
        if pipe.end_node_id in in_count and pipe.start_node_id != pipe.end_node_id:
            in_count[pipe.end_node_id] += 1
    q = deque(source_ids)
    result = []
    visited = set(source_ids)
    while q:
        nid = q.popleft()
        result.append(nid)
        for dp in network.get_downstream_pipes(nid):
            if "回水" in dp.name or "return" in dp.name.lower():
                continue
            child = dp.end_node_id
            if child in in_count:
                in_count[child] -= 1
                if in_count[child] <= 0 and child not in visited:
                    visited.add(child)
                    q.append(child)
    for nid in network.nodes:
        if nid not in visited:
            result.append(nid)
    return result


def solve_hydraulics(
    network: PipeNetwork,
    node_temperatures: Optional[Dict[int, float]] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, PipeSectionResult], int, bool]:
    return solve_hydraulics_simplified(network, node_temperatures, max_iterations, tolerance)
