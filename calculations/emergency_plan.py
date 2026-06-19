import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from models import (
    PipeNetwork, Pipe, Node, NetworkResults, PipeSectionResult, NodeResult,
    NODE_TYPE_SOURCE, NODE_TYPE_END_USER, NODE_TYPE_BRANCH, NODE_TYPE_HEAT_EXCHANGER,
    get_water_properties,
)
from calculations.coupled import solve_coupled
from calculations.fault_simulation import (
    FaultConfig, ImpactAssessment,
    FAULT_TYPE_PIPE_BURST, FAULT_TYPE_PUMP_FAILURE, FAULT_TYPE_SOURCE_SHUTDOWN,
    MIN_OPERATING_PRESSURE, apply_faults_to_network,
)


DEFAULT_MIN_TEMP_THRESHOLD = 60.0
DEFAULT_TEMP_WARNING_THRESHOLD = 65.0


@dataclass
class EmergencyAction:
    priority: int
    action_type: str
    target_id: int
    target_name: str
    description: str
    detail: str = ""
    estimated_effect: str = ""


@dataclass
class EmergencyPlan:
    actions: List[EmergencyAction] = field(default_factory=list)
    summary: str = ""
    warnings: List[str] = field(default_factory=list)

    def sorted_actions(self) -> List[EmergencyAction]:
        return sorted(self.actions, key=lambda a: a.priority)


def _find_adjacent_valve_pipes(
    network: PipeNetwork, burst_pipe_id: int
) -> List[Pipe]:
    burst_pipe = network.pipes.get(burst_pipe_id)
    if not burst_pipe:
        return []
    valve_pipes = []
    for nid in [burst_pipe.start_node_id, burst_pipe.end_node_id]:
        for conn_pipe in network.get_connected_pipes(nid):
            if conn_pipe.id == burst_pipe_id:
                continue
            if conn_pipe.has_valve:
                valve_pipes.append(conn_pipe)
    return valve_pipes


def _find_alternate_path(
    network: PipeNetwork,
    burst_pipe_id: int,
    original_network: PipeNetwork,
) -> Optional[List[int]]:
    burst_pipe = original_network.pipes.get(burst_pipe_id)
    if not burst_pipe:
        return None
    start = burst_pipe.start_node_id
    end = burst_pipe.end_node_id

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        nid, path = queue.popleft()
        if nid == end:
            return path
        for pipe in network.get_connected_pipes(nid):
            if pipe.id == burst_pipe_id:
                continue
            neighbor = pipe.end_node_id if pipe.start_node_id == nid else pipe.start_node_id
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def _generate_pipe_burst_actions(
    original_network: PipeNetwork,
    modified_network: PipeNetwork,
    fault: FaultConfig,
    impact: ImpactAssessment,
) -> List[EmergencyAction]:
    actions = []
    burst_pipe = original_network.pipes.get(fault.target_id)
    if not burst_pipe:
        return actions

    action1 = EmergencyAction(
        priority=1,
        action_type="隔离故障",
        target_id=fault.target_id,
        target_name=burst_pipe.name,
        description=f"立即关闭 {burst_pipe.name} 上下游阀门，隔离故障管段",
        detail=f"管段ID: {fault.target_id}，管径{burst_pipe.diameter*1000:.0f}mm，长度{burst_pipe.length:.0f}m",
    )
    adjacent_valves = _find_adjacent_valve_pipes(original_network, fault.target_id)
    if adjacent_valves:
        valve_names = "、".join([f"{v.name}(ID:{v.id})" for v in adjacent_valves])
        action1.detail += f"。需关闭阀门: {valve_names}"
    actions.append(action1)

    alt_path = _find_alternate_path(modified_network, fault.target_id, original_network)
    if alt_path and len(alt_path) > 2:
        path_names = []
        for i in range(len(alt_path) - 1):
            s, e = alt_path[i], alt_path[i + 1]
            for pid, pipe in modified_network.pipes.items():
                if (pipe.start_node_id == s and pipe.end_node_id == e) or \
                   (pipe.start_node_id == e and pipe.end_node_id == s):
                    path_names.append(pipe.name)
                    break
        if path_names:
            actions.append(EmergencyAction(
                priority=2,
                action_type="切换路径",
                target_id=fault.target_id,
                target_name=burst_pipe.name,
                description=f"启用备用供水路径: {' → '.join(path_names)}",
                detail=f"管网存在环路，可通过备用路径恢复下游用户供水",
                estimated_effect="预计可恢复大部分受影响用户的供水",
            ))

    if impact.disconnected_user_ids:
        actions.append(EmergencyAction(
            priority=3,
            action_type="用户通知",
            target_id=0,
            target_name="受影响用户",
            description=f"通知 {len(impact.disconnected_user_ids)} 户断供用户做好应急准备",
            detail=f"断供用户: {'、'.join(impact.disconnected_user_names[:5])}" +
                   ("等" if len(impact.disconnected_user_names) > 5 else ""),
        ))

    return actions


def _generate_pump_failure_actions(
    original_network: PipeNetwork,
    modified_network: PipeNetwork,
    fault_results: Optional[NetworkResults],
    original_results: NetworkResults,
    fault: FaultConfig,
    impact: ImpactAssessment,
    source_temps: Dict[int, float],
) -> Tuple[List[EmergencyAction], Optional[NetworkResults]]:
    actions = []
    failed_pump = original_network.pipes.get(fault.target_id)
    if not failed_pump:
        return actions, fault_results

    failed_pipe_id = fault.target_id
    original_head = 0.0
    original_power = 0.0
    original_flow = 0.0
    if failed_pipe_id in original_results.pipe_results:
        pr = original_results.pipe_results[failed_pipe_id]
        original_head = abs(pr.pump_head)
        original_power = pr.power_consumption
        original_flow = abs(pr.flow_rate)

    actions.append(EmergencyAction(
        priority=1,
        action_type="泵站停机",
        target_id=fault.target_id,
        target_name=failed_pump.name,
        description=f"确认 {failed_pump.name} 泵站已安全停运",
        detail=f"原工作参数: 流量{original_flow*1000:.1f}L/s, 扬程{original_head:.1f}m, 功率{original_power:.1f}kW",
    ))

    other_pumps = []
    for pid, pipe in original_network.pipes.items():
        if pipe.has_pump and pid != failed_pipe_id:
            other_pumps.append((pid, pipe))

    if other_pumps and fault_results is not None:
        total_extra_head_needed = original_head
        compensated = False

        test_network = copy.deepcopy(original_network)
        test_pipe = test_network.pipes.get(failed_pipe_id)
        if test_pipe and test_pipe.has_pump:
            test_pipe.has_pump = False
            test_pipe.rated_head = None
            test_pipe.pump_efficiency_curve = None

        for opid, opipe in other_pumps:
            orig_pr = original_results.pipe_results.get(opid)
            if orig_pr and opipe.rated_head:
                current_head = abs(orig_pr.pump_head)
                max_head = opipe.rated_head * 1.1
                if current_head < max_head:
                    actions.append(EmergencyAction(
                        priority=2,
                        action_type="提高泵站出力",
                        target_id=opid,
                        target_name=opipe.name,
                        description=f"提高 {opipe.name} 泵站出力，补偿扬程损失",
                        detail=f"当前扬程{current_head:.1f}m，可提升至约{max_head:.1f}m",
                        estimated_effect=f"预计可补偿约{(max_head-current_head)/max(total_extra_head_needed,1)*100:.0f}%的扬程缺口",
                    ))
                    compensated = True

        if not compensated:
            actions.append(EmergencyAction(
                priority=3,
                action_type="警告",
                target_id=0,
                target_name="系统警告",
                description="剩余泵站能力不足，无法完全补偿故障泵站的扬程损失",
                detail="建议降低部分非核心用户负荷，优先保障关键用户",
            ))

    if impact.low_pressure_users:
        actions.append(EmergencyAction(
            priority=4,
            action_type="压力保障",
            target_id=0,
            target_name="低压用户",
            description=f"关注 {len(impact.low_pressure_users)} 户压力不达标用户",
            detail=f"低压用户压力低于 {MIN_OPERATING_PRESSURE} mH₂O，需重点监控",
        ))

    return actions, fault_results


def _generate_source_shutdown_actions(
    original_network: PipeNetwork,
    modified_network: PipeNetwork,
    fault_results: Optional[NetworkResults],
    original_results: NetworkResults,
    fault: FaultConfig,
    impact: ImpactAssessment,
    source_temps: Dict[int, float],
    min_temp_threshold: float = DEFAULT_TEMP_WARNING_THRESHOLD,
) -> Tuple[List[EmergencyAction], Optional[NetworkResults], Dict[int, float]]:
    actions = []
    shutdown_source = original_network.nodes.get(fault.target_id)
    if not shutdown_source:
        return actions, fault_results, source_temps

    new_source_temps = dict(source_temps)

    actions.append(EmergencyAction(
        priority=1,
        action_type="热源停机",
        target_id=fault.target_id,
        target_name=shutdown_source.name,
        description=f"确认 {shutdown_source.name} 已安全停机",
        detail=f"额定容量: {(shutdown_source.rated_capacity or 0)/1e6:.1f} MW",
    ))

    remaining_sources = []
    for node in original_network.get_nodes_by_type(NODE_TYPE_SOURCE):
        if node.id != fault.target_id and (node.rated_capacity or 0) > 0:
            remaining_sources.append(node)

    if not remaining_sources:
        actions.append(EmergencyAction(
            priority=2,
            action_type="紧急警告",
            target_id=0,
            target_name="系统警告",
            description="无剩余可用热源，系统完全瘫痪！",
            detail="建议立即启动备用锅炉或跨区域调度应急热源",
        ))
        return actions, fault_results, new_source_temps

    shutdown_capacity = shutdown_source.rated_capacity or 0.0
    total_remaining_capacity = sum((s.rated_capacity or 0) for s in remaining_sources)

    original_temp = new_source_temps.get(fault.target_id, 110.0)
    if fault.target_id in new_source_temps:
        del new_source_temps[fault.target_id]

    total_extra_heat_needed = 0.0
    if fault_results:
        end_users = original_network.get_nodes_by_type(NODE_TYPE_END_USER)
        cp_water = original_network.water_specific_heat
        for user in end_users:
            nr = fault_results.node_results.get(user.id)
            if nr and nr.temperature < min_temp_threshold:
                deficit = min_temp_threshold - nr.temperature
                if user.design_flow:
                    density, _ = get_water_properties(nr.temperature)
                    extra_heat = density * user.design_flow * cp_water * deficit
                    total_extra_heat_needed += extra_heat

    if total_extra_heat_needed > 0:
        per_source_extra = total_extra_heat_needed / len(remaining_sources)
        for src in remaining_sources:
            orig_src_temp = new_source_temps.get(src.id, 108.0)
            if orig_src_temp > 0:
                src_capacity = src.rated_capacity or (total_remaining_capacity / len(remaining_sources))
                temp_increase = min(15.0, per_source_extra / max(src_capacity, 1e-6) * 30.0)
                new_temp = min(130.0, orig_src_temp + temp_increase)
                new_source_temps[src.id] = new_temp

                actions.append(EmergencyAction(
                    priority=2,
                    action_type="提升热源温度",
                    target_id=src.id,
                    target_name=src.name,
                    description=f"将 {src.name} 供水温度从 {orig_src_temp:.1f}°C 提升至 {new_temp:.1f}°C",
                    detail=f"提升幅度 {temp_increase:.1f}°C，预计可增加供热能力约 {per_source_extra/1e6:.2f} MW",
                    estimated_effect=f"预计可使末端用户温度提升约 {temp_increase*0.6:.1f}°C",
                ))
    else:
        for src in remaining_sources:
            orig_src_temp = new_source_temps.get(src.id, 108.0)
            capacity_ratio = shutdown_capacity / max(total_remaining_capacity, 1e-6)
            temp_increase = min(10.0, original_temp * capacity_ratio * 0.3)
            new_temp = min(125.0, orig_src_temp + temp_increase)
            new_source_temps[src.id] = new_temp

            if temp_increase > 0.5:
                actions.append(EmergencyAction(
                    priority=2,
                    action_type="提升热源温度",
                    target_id=src.id,
                    target_name=src.name,
                    description=f"建议将 {src.name} 供水温度提升至 {new_temp:.1f}°C（当前 {orig_src_temp:.1f}°C）",
                    detail=f"预防性提温 {temp_increase:.1f}°C，应对停机带来的负荷转移",
                ))

    actions.append(EmergencyAction(
        priority=3,
        action_type="负荷分配",
        target_id=0,
        target_name="调度方案",
        description=f"重新分配负荷至剩余 {len(remaining_sources)} 个热源",
        detail=f"已停机容量 {(shutdown_capacity or 0)/1e6:.1f} MW，剩余容量 {total_remaining_capacity/1e6:.1f} MW",
        estimated_effect=f"容量裕度约 {(total_remaining_capacity - shutdown_capacity)/max(total_remaining_capacity,1)*100:.0f}%",
    ))

    low_temp_users = []
    if fault_results:
        for uid in impact.affected_user_ids:
            nr = fault_results.node_results.get(uid)
            if nr and nr.temperature < min_temp_threshold:
                uname = original_network.nodes[uid].name if uid in original_network.nodes else f"用户{uid}"
                low_temp_users.append((uname, nr.temperature))

    if low_temp_users:
        actions.append(EmergencyAction(
            priority=4,
            action_type="温度预警",
            target_id=0,
            target_name="低温用户",
            description=f"{len(low_temp_users)} 户用户供水温度低于 {min_temp_threshold}°C 预警阈值",
            detail="用户列表: " + "、".join([f"{n}({t:.1f}°C)" for n, t in low_temp_users[:5]]) +
                   ("等" if len(low_temp_users) > 5 else ""),
        ))

    return actions, fault_results, new_source_temps


def generate_emergency_plan(
    original_network: PipeNetwork,
    modified_network: PipeNetwork,
    original_results: NetworkResults,
    fault_results: Optional[NetworkResults],
    faults: List[FaultConfig],
    impact: ImpactAssessment,
    source_temps: Dict[int, float],
    min_temp_threshold: float = DEFAULT_TEMP_WARNING_THRESHOLD,
) -> Tuple[EmergencyPlan, Optional[NetworkResults], Dict[int, float]]:
    plan = EmergencyPlan()
    current_results = fault_results
    current_source_temps = dict(source_temps)

    if not faults:
        plan.summary = "无故障设置，无需应急预案"
        return plan, current_results, current_source_temps

    for fault in faults:
        if fault.fault_type == FAULT_TYPE_PIPE_BURST:
            actions = _generate_pipe_burst_actions(
                original_network, modified_network, fault, impact
            )
            plan.actions.extend(actions)
        elif fault.fault_type == FAULT_TYPE_PUMP_FAILURE:
            actions, current_results = _generate_pump_failure_actions(
                original_network, modified_network, current_results,
                original_results, fault, impact, current_source_temps,
            )
            plan.actions.extend(actions)
        elif fault.fault_type == FAULT_TYPE_SOURCE_SHUTDOWN:
            actions, current_results, current_source_temps = _generate_source_shutdown_actions(
                original_network, modified_network, current_results,
                original_results, fault, impact, current_source_temps,
                min_temp_threshold,
            )
            plan.actions.extend(actions)

    fault_descriptions = [f.describe(original_network) for f in faults]
    plan.summary = (
        f"针对 {len(faults)} 项故障（" + "、".join(fault_descriptions) + "）"
        f"生成 {len(plan.actions)} 条应急调度措施"
    )

    if impact.total_heat_capacity_drop_pct > 30:
        plan.warnings.append(
            f"系统供热能力下降 {impact.total_heat_capacity_drop_pct:.1f}%，超过30%警戒线，需高度重视"
        )
    if impact.disconnected_user_ids:
        plan.warnings.append(
            f"{len(impact.disconnected_user_ids)} 户用户完全断供，需立即启动用户沟通预案"
        )

    return plan, current_results, current_source_temps
