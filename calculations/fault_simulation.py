import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from models import (
    PipeNetwork, Pipe, Node, NetworkResults, PipeSectionResult, NodeResult,
    NODE_TYPE_SOURCE, NODE_TYPE_END_USER, NODE_TYPE_BRANCH,
    get_water_properties,
)
from calculations.coupled import solve_coupled
from calculations.hydraulic import compute_pump_head
from calculations.thermal import compute_pipe_temperature_drop


FAULT_TYPE_PIPE_BURST = "pipe_burst"
FAULT_TYPE_PUMP_FAILURE = "pump_failure"
FAULT_TYPE_SOURCE_SHUTDOWN = "source_shutdown"

FAULT_TYPE_CN = {
    FAULT_TYPE_PIPE_BURST: "管段爆管",
    FAULT_TYPE_PUMP_FAILURE: "泵站故障",
    FAULT_TYPE_SOURCE_SHUTDOWN: "热源停机",
}

MIN_OPERATING_PRESSURE = 20.0


@dataclass
class FaultConfig:
    fault_type: str
    target_id: int
    params: Dict = field(default_factory=dict)

    def describe(self, network: PipeNetwork) -> str:
        if self.fault_type == FAULT_TYPE_PIPE_BURST:
            pipe = network.pipes.get(self.target_id)
            name = pipe.name if pipe else f"管段{self.target_id}"
            return f"管段爆管: {name} (ID:{self.target_id})"
        elif self.fault_type == FAULT_TYPE_PUMP_FAILURE:
            pipe = network.pipes.get(self.target_id)
            name = pipe.name if pipe else f"管段{self.target_id}"
            return f"泵站故障: {name} (ID:{self.target_id})"
        elif self.fault_type == FAULT_TYPE_SOURCE_SHUTDOWN:
            node = network.nodes.get(self.target_id)
            name = node.name if node else f"热源{self.target_id}"
            return f"热源停机: {name} (ID:{self.target_id})"
        return f"未知故障 (类型:{self.fault_type}, 目标:{self.target_id})"


@dataclass
class ImpactAssessment:
    affected_user_ids: List[int] = field(default_factory=list)
    affected_user_names: List[str] = field(default_factory=list)
    disconnected_user_ids: List[int] = field(default_factory=list)
    disconnected_user_names: List[str] = field(default_factory=list)
    user_temp_changes: Dict[int, float] = field(default_factory=dict)
    user_pressure_changes: Dict[int, float] = field(default_factory=dict)
    max_pressure_drop_change: float = 0.0
    total_heat_capacity_drop_pct: float = 0.0
    original_total_heat: float = 0.0
    fault_total_heat: float = 0.0
    low_pressure_users: List[int] = field(default_factory=list)
    summary_text: str = ""


def apply_faults_to_network(
    original_network: PipeNetwork,
    faults: List[FaultConfig],
    source_temps: Dict[int, float],
) -> Tuple[PipeNetwork, Dict[int, float], List[int]]:
    network = copy.deepcopy(original_network)
    new_source_temps = dict(source_temps)

    burst_pipe_ids = set()
    failed_pump_ids = set()
    shutdown_source_ids = set()

    for fault in faults:
        if fault.fault_type == FAULT_TYPE_PIPE_BURST:
            burst_pipe_ids.add(fault.target_id)
        elif fault.fault_type == FAULT_TYPE_PUMP_FAILURE:
            failed_pump_ids.add(fault.target_id)
        elif fault.fault_type == FAULT_TYPE_SOURCE_SHUTDOWN:
            shutdown_source_ids.add(fault.target_id)

    for pid in list(network.pipes.keys()):
        if pid in burst_pipe_ids:
            del network.pipes[pid]

    for pid, pipe in network.pipes.items():
        if pid in failed_pump_ids and pipe.has_pump:
            pipe.has_pump = False
            pipe.rated_head = None
            pipe.pump_efficiency_curve = None

    active_source_ids = []
    for sid in shutdown_source_ids:
        if sid in network.nodes:
            node = network.nodes[sid]
            node.type = NODE_TYPE_BRANCH
            node.supply_pressure = None
            node.rated_capacity = 0.0
        if sid in new_source_temps:
            del new_source_temps[sid]

    for nid, node in network.nodes.items():
        if nid not in shutdown_source_ids and node.type == NODE_TYPE_SOURCE:
            active_source_ids.append(nid)

    return network, new_source_temps, active_source_ids


def _adjust_flow_and_temperature_for_pump_failure(
    original_network: PipeNetwork,
    modified_network: PipeNetwork,
    original_results: NetworkResults,
    fault_results: NetworkResults,
    source_temps: Dict[int, float],
) -> NetworkResults:
    adjusted = copy.deepcopy(fault_results)
    g = 9.81

    pressure_ratios: Dict[int, float] = {}
    for uid, user in enumerate(original_network.get_nodes_by_type(NODE_TYPE_END_USER)):
        orig_nr = original_results.node_results.get(user.id)
        fault_nr = adjusted.node_results.get(user.id)
        if orig_nr and fault_nr:
            orig_p = orig_nr.pressure - user.elevation
            fault_p = fault_nr.pressure - user.elevation
            if orig_p > 1.0:
                ratio = max(0.1, min(1.0, np.sqrt(max(0.0, fault_p) / max(1.0, orig_p))))
                pressure_ratios[user.id] = ratio
            else:
                pressure_ratios[user.id] = 1.0

    pipe_flow_ratios: Dict[int, float] = {}
    for pid, pipe in modified_network.pipes.items():
        orig_pr = original_results.pipe_results.get(pid)
        if orig_pr and abs(orig_pr.flow_rate) > 1e-6:
            affected_users = []
            for uid, ratio in pressure_ratios.items():
                user_node = original_network.nodes[uid]
                if _is_pipe_upstream_of_node(modified_network, pipe, user_node):
                    affected_users.append(ratio)
            if affected_users:
                avg_ratio = sum(affected_users) / len(affected_users)
                pipe_flow_ratios[pid] = avg_ratio
            else:
                pipe_flow_ratios[pid] = 1.0
        else:
            pipe_flow_ratios[pid] = 1.0

    for pid, ratio in pipe_flow_ratios.items():
        if pid in adjusted.pipe_results:
            pr = adjusted.pipe_results[pid]
            orig_flow = pr.flow_rate
            new_flow = orig_flow * ratio
            pr.flow_rate = new_flow

            orig_pipe = original_network.pipes.get(pid)
            if orig_pipe and abs(new_flow) > 1e-10 and pr.inlet_temperature > 0:
                env_t = modified_network.environment_temperature
                cp_w = modified_network.water_specific_heat
                out_t, drop, hloss = compute_pipe_temperature_drop(
                    orig_pipe, abs(new_flow), pr.inlet_temperature, env_t, cp_w
                )
                pr.outlet_temperature = out_t
                pr.temperature_drop = drop
                pr.heat_loss = hloss
                if pr.pump_head != 0:
                    sign = 1.0 if new_flow >= 0 else -1.0
                    pr.pump_head = sign * compute_pump_head(orig_pipe, abs(new_flow))

    node_temps: Dict[int, float] = {}
    for sid in source_temps:
        if sid in modified_network.nodes:
            node_temps[sid] = source_temps[sid]

    sorted_nodes = []
    visited = set()
    source_ids = [n.id for n in modified_network.get_nodes_by_type(NODE_TYPE_SOURCE)]
    queue = deque(source_ids)
    for sid in source_ids:
        visited.add(sid)
        sorted_nodes.append(sid)
    while queue:
        nid = queue.popleft()
        for pipe in modified_network.get_downstream_pipes(nid):
            neighbor = pipe.end_node_id if pipe.start_node_id == nid else pipe.start_node_id
            if neighbor not in visited:
                visited.add(neighbor)
                sorted_nodes.append(neighbor)
                queue.append(neighbor)

    for nid in sorted_nodes:
        if nid in node_temps:
            continue
        incoming = modified_network.get_upstream_pipes(nid)
        total_mass = 0.0
        weighted_temp = 0.0
        for ip in incoming:
            if "回水" in ip.name or "return" in ip.name.lower():
                continue
            pr = adjusted.pipe_results.get(ip.id)
            if pr and abs(pr.flow_rate) > 1e-10:
                density, _ = get_water_properties(pr.outlet_temperature if pr.outlet_temperature > 0 else 80.0)
                mass = abs(pr.flow_rate) * density
                total_mass += mass
                weighted_temp += mass * pr.outlet_temperature
        if total_mass > 0:
            node_temps[nid] = weighted_temp / total_mass
        else:
            node_temps[nid] = modified_network.environment_temperature + 20.0

        nr = adjusted.node_results.get(nid)
        if nr:
            nr.temperature = node_temps[nid]

        downstream = modified_network.get_downstream_pipes(nid)
        for dp in downstream:
            if "回水" in dp.name or "return" in dp.name.lower():
                continue
            pr = adjusted.pipe_results.get(dp.id)
            if pr and abs(pr.flow_rate) > 1e-10:
                orig_pipe = original_network.pipes.get(dp.id)
                if orig_pipe:
                    env_t = modified_network.environment_temperature
                    cp_w = modified_network.water_specific_heat
                    out_t, drop, hloss = compute_pipe_temperature_drop(
                        orig_pipe, abs(pr.flow_rate), node_temps[nid], env_t, cp_w
                    )
                    pr.inlet_temperature = node_temps[nid]
                    pr.outlet_temperature = out_t
                    pr.temperature_drop = drop
                    pr.heat_loss = hloss

    total_heat = 0.0
    for nr in adjusted.node_results.values():
        node = modified_network.nodes.get(nr.node_id)
        if node and node.type == NODE_TYPE_END_USER and node.design_flow:
            orig_nr = original_results.node_results.get(nr.node_id)
            if orig_nr:
                ratio = pressure_ratios.get(nr.node_id, 1.0)
                actual_flow = node.design_flow * ratio
                density, _ = get_water_properties(nr.temperature)
                cp_w = modified_network.water_specific_heat
                heat_load = density * actual_flow * cp_w * max(0.0, nr.temperature - 40.0)
                nr.flow_in = actual_flow
                nr.heat_load = heat_load
                total_heat += heat_load

    adjusted.total_heat_supplied = total_heat
    total_loss = sum(pr.heat_loss for pr in adjusted.pipe_results.values())
    if total_heat > 0:
        adjusted.heat_loss_rate = total_loss / total_heat * 100.0

    return adjusted


def _is_pipe_upstream_of_node(network: PipeNetwork, pipe: Pipe, target_node: Node) -> bool:
    visited = set()
    queue = deque([pipe.end_node_id, pipe.start_node_id])
    while queue:
        nid = queue.popleft()
        if nid == target_node.id:
            return True
        if nid in visited:
            continue
        visited.add(nid)
        for down_pipe in network.get_downstream_pipes(nid):
            neighbor = down_pipe.end_node_id if down_pipe.start_node_id == nid else down_pipe.start_node_id
            if neighbor not in visited:
                queue.append(neighbor)
    return False


def _find_disconnected_nodes(
    network: PipeNetwork,
    source_ids: List[int],
) -> set:
    visited = set()
    queue = deque(source_ids)
    for sid in source_ids:
        visited.add(sid)

    while queue:
        nid = queue.popleft()
        for pipe in network.get_connected_pipes(nid):
            neighbor = pipe.end_node_id if pipe.start_node_id == nid else pipe.start_node_id
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    all_node_ids = set(network.nodes.keys())
    return all_node_ids - visited


def simulate_faults(
    original_network: PipeNetwork,
    original_results: NetworkResults,
    faults: List[FaultConfig],
    source_temps: Dict[int, float],
) -> Tuple[Optional[NetworkResults], ImpactAssessment, PipeNetwork]:
    if not faults:
        return original_results, ImpactAssessment(), original_network

    modified_network, modified_source_temps, active_source_ids = apply_faults_to_network(
        original_network, faults, source_temps
    )

    if not active_source_ids:
        impact = ImpactAssessment()
        impact.summary_text = "所有热源均已停运，系统完全瘫痪！"
        all_users = original_network.get_nodes_by_type(NODE_TYPE_END_USER)
        impact.disconnected_user_ids = [u.id for u in all_users]
        impact.disconnected_user_names = [u.name for u in all_users]
        impact.affected_user_ids = impact.disconnected_user_ids
        impact.affected_user_names = impact.disconnected_user_names
        impact.total_heat_capacity_drop_pct = 100.0
        impact.original_total_heat = original_results.total_heat_supplied
        impact.fault_total_heat = 0.0
        return None, impact, modified_network

    valid, errors = modified_network.validate()
    fault_results = None
    if valid:
        try:
            fault_results = solve_coupled(modified_network, modified_source_temps)
        except Exception:
            fault_results = None

    has_pump_failure = any(f.fault_type == FAULT_TYPE_PUMP_FAILURE for f in faults)
    if has_pump_failure and fault_results is not None and original_results is not None:
        fault_results = _adjust_flow_and_temperature_for_pump_failure(
            original_network, modified_network,
            original_results, fault_results,
            modified_source_temps,
        )

    impact = _assess_impact(
        original_network, original_results,
        modified_network, fault_results,
        faults, active_source_ids,
    )

    return fault_results, impact, modified_network


def _assess_impact(
    original_network: PipeNetwork,
    original_results: NetworkResults,
    modified_network: PipeNetwork,
    fault_results: Optional[NetworkResults],
    faults: List[FaultConfig],
    active_source_ids: List[int],
) -> ImpactAssessment:
    impact = ImpactAssessment()

    disconnected_ids = _find_disconnected_nodes(
        modified_network, active_source_ids
    )

    original_users = original_network.get_nodes_by_type(NODE_TYPE_END_USER)
    total_users = len(original_users)

    for user in original_users:
        if user.id in disconnected_ids:
            impact.disconnected_user_ids.append(user.id)
            impact.disconnected_user_names.append(user.name)
            impact.affected_user_ids.append(user.id)
            impact.affected_user_names.append(user.name)
            impact.user_temp_changes[user.id] = -999.0
            impact.user_pressure_changes[user.id] = -999.0
            continue

        orig_nr = original_results.node_results.get(user.id)
        fault_nr = fault_results.node_results.get(user.id) if fault_results else None

        if orig_nr and fault_nr:
            temp_change = fault_nr.temperature - orig_nr.temperature
            press_change = fault_nr.pressure - orig_nr.pressure

            impact.user_temp_changes[user.id] = temp_change
            impact.user_pressure_changes[user.id] = press_change

            if abs(temp_change) > 0.1 or abs(press_change) > 0.5:
                if user.id not in impact.affected_user_ids:
                    impact.affected_user_ids.append(user.id)
                    impact.affected_user_names.append(user.name)

            if fault_nr.pressure < MIN_OPERATING_PRESSURE:
                impact.low_pressure_users.append(user.id)
                if user.id not in impact.affected_user_ids:
                    impact.affected_user_ids.append(user.id)
                    impact.affected_user_names.append(user.name)

    max_p_drop_orig = 0.0
    max_p_drop_fault = 0.0

    if original_results:
        for pid, pr in original_results.pipe_results.items():
            if pr.total_pressure_loss > max_p_drop_orig:
                max_p_drop_orig = pr.total_pressure_loss

    if fault_results:
        for pid, pr in fault_results.pipe_results.items():
            if pr.total_pressure_loss > max_p_drop_fault:
                max_p_drop_fault = pr.total_pressure_loss

    impact.max_pressure_drop_change = max_p_drop_fault - max_p_drop_orig

    impact.original_total_heat = original_results.total_heat_supplied if original_results else 0.0
    impact.fault_total_heat = fault_results.total_heat_supplied if fault_results else 0.0
    if impact.original_total_heat > 0:
        impact.total_heat_capacity_drop_pct = (
            (impact.original_total_heat - impact.fault_total_heat)
            / impact.original_total_heat * 100.0
        )
    else:
        impact.total_heat_capacity_drop_pct = 0.0

    parts = []
    parts.append(f"总用户数: {total_users}")
    parts.append(f"受影响用户: {len(impact.affected_user_ids)} 人")
    if impact.disconnected_user_ids:
        parts.append(f"断供用户: {len(impact.disconnected_user_ids)} 人")
    if impact.low_pressure_users:
        parts.append(f"压力不达标用户: {len(impact.low_pressure_users)} 人")
    parts.append(f"供热能力下降: {impact.total_heat_capacity_drop_pct:.1f}%")
    impact.summary_text = " | ".join(parts)

    return impact


def get_available_pump_pipes(network: PipeNetwork) -> List[Pipe]:
    return [p for p in network.pipes.values() if p.has_pump]


def get_available_source_nodes(network: PipeNetwork) -> List[Node]:
    return network.get_nodes_by_type(NODE_TYPE_SOURCE)


def get_available_pipes(network: PipeNetwork) -> List[Pipe]:
    return list(network.pipes.values())


@dataclass
class RiskAssessmentItem:
    device_id: int
    device_name: str
    device_type: str
    fault_probability: float
    heat_capacity_drop_pct: float
    risk_score: float


def get_fault_probability(pipe_age: float = None, device_type: str = None) -> float:
    if device_type == "泵站":
        return 0.3
    elif device_type == "热源":
        return 0.15
    elif device_type == "管段":
        if pipe_age > 20:
            return 0.8
        elif pipe_age >= 10:
            return 0.4
        else:
            return 0.1
    return 0.0


def calculate_risk_assessment(
    original_network: PipeNetwork,
    original_results: NetworkResults,
    source_temps: Dict[int, float],
) -> List[RiskAssessmentItem]:
    risk_items: List[RiskAssessmentItem] = []
    pipes = list(original_network.pipes.values())
    pump_pipes = [p for p in pipes if p.has_pump]
    source_nodes = original_network.get_nodes_by_type(NODE_TYPE_SOURCE)

    for pipe in pipes:
        try:
            faults = [FaultConfig(fault_type=FAULT_TYPE_PIPE_BURST, target_id=pipe.id)]
            _, impact, _ = simulate_faults(
                original_network, original_results, faults, source_temps
            )
            prob = get_fault_probability(pipe_age=pipe.pipe_age, device_type="管段")
            risk = impact.total_heat_capacity_drop_pct * prob
            risk_items.append(RiskAssessmentItem(
                device_id=pipe.id,
                device_name=pipe.name,
                device_type="管段",
                fault_probability=prob,
                heat_capacity_drop_pct=impact.total_heat_capacity_drop_pct,
                risk_score=risk,
            ))
        except Exception:
            pass

    for pump_pipe in pump_pipes:
        try:
            faults = [FaultConfig(fault_type=FAULT_TYPE_PUMP_FAILURE, target_id=pump_pipe.id)]
            _, impact, _ = simulate_faults(
                original_network, original_results, faults, source_temps
            )
            prob = get_fault_probability(device_type="泵站")
            risk = impact.total_heat_capacity_drop_pct * prob
            risk_items.append(RiskAssessmentItem(
                device_id=pump_pipe.id,
                device_name=pump_pipe.name,
                device_type="泵站",
                fault_probability=prob,
                heat_capacity_drop_pct=impact.total_heat_capacity_drop_pct,
                risk_score=risk,
            ))
        except Exception:
            pass

    for src_node in source_nodes:
        try:
            faults = [FaultConfig(fault_type=FAULT_TYPE_SOURCE_SHUTDOWN, target_id=src_node.id)]
            _, impact, _ = simulate_faults(
                original_network, original_results, faults, source_temps
            )
            prob = get_fault_probability(device_type="热源")
            risk = impact.total_heat_capacity_drop_pct * prob
            risk_items.append(RiskAssessmentItem(
                device_id=src_node.id,
                device_name=src_node.name,
                device_type="热源",
                fault_probability=prob,
                heat_capacity_drop_pct=impact.total_heat_capacity_drop_pct,
                risk_score=risk,
            ))
        except Exception:
            pass

    risk_items.sort(key=lambda x: x.risk_score, reverse=True)
    return risk_items
