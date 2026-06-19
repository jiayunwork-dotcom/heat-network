import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from models import (
    PipeNetwork, Pipe, Node, NetworkResults, PipeSectionResult, NodeResult,
    NODE_TYPE_SOURCE, NODE_TYPE_END_USER,
)
from calculations.coupled import solve_coupled


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
) -> Tuple[PipeNetwork, Dict[int, float]]:
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

    for sid in shutdown_source_ids:
        if sid in network.nodes:
            node = network.nodes[sid]
            node.supply_pressure = 0.0
            node.rated_capacity = 0.0
        if sid in new_source_temps:
            new_source_temps[sid] = 0.0

    return network, new_source_temps


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

    modified_network, modified_source_temps = apply_faults_to_network(
        original_network, faults, source_temps
    )

    active_source_ids = []
    for fault in faults:
        if fault.fault_type == FAULT_TYPE_SOURCE_SHUTDOWN:
            continue
    for n in modified_network.get_nodes_by_type(NODE_TYPE_SOURCE):
        is_shutdown = any(
            f.fault_type == FAULT_TYPE_SOURCE_SHUTDOWN and f.target_id == n.id
            for f in faults
        )
        if not is_shutdown and (n.rated_capacity or 0) > 0:
            active_source_ids.append(n.id)

    if not active_source_ids:
        impact = ImpactAssessment()
        impact.summary_text = "所有热源均已停运，系统完全瘫痪！"
        all_users = modified_network.get_nodes_by_type(NODE_TYPE_END_USER)
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

            if abs(temp_change) > 0.5 or abs(press_change) > 1.0:
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
