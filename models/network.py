from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np

NODE_TYPE_SOURCE = "source"
NODE_TYPE_HEAT_EXCHANGER = "heat_exchanger"
NODE_TYPE_BRANCH = "branch"
NODE_TYPE_END_USER = "end_user"

PIPE_MATERIAL_STEEL = "steel"
PIPE_MATERIAL_CAST_IRON = "cast_iron"
PIPE_MATERIAL_PVC = "pvc"

INSULATION_MATERIAL_NONE = "none"
INSULATION_MATERIAL_POLYURETHANE = "polyurethane"
INSULATION_MATERIAL_ROCK_WOOL = "rock_wool"
INSULATION_MATERIAL_GLASS_WOOL = "glass_wool"

@dataclass
class Node:
    id: int
    name: str
    type: str = NODE_TYPE_BRANCH
    elevation: float = 0.0
    x: Optional[float] = None
    y: Optional[float] = None
    supply_pressure: Optional[float] = None
    return_pressure: Optional[float] = None
    design_flow: Optional[float] = None
    rated_capacity: Optional[float] = None

@dataclass
class Pipe:
    id: int
    name: str
    start_node_id: int
    end_node_id: int
    diameter: float = 0.2
    length: float = 100.0
    material: str = PIPE_MATERIAL_STEEL
    pipe_age: float = 10.0
    insulation_material: str = INSULATION_MATERIAL_POLYURETHANE
    insulation_thickness: float = 0.05
    burial_depth: float = 1.5
    has_valve: bool = False
    valve_kv: Optional[float] = None
    has_pump: bool = False
    rated_head: Optional[float] = None
    pump_efficiency_curve: Optional[List[Tuple[float, float]]] = None
    equivalent_local_length_ratio: float = 0.1

@dataclass
class PipeSectionResult:
    pipe_id: int
    flow_rate: float = 0.0
    velocity: float = 0.0
    reynolds: float = 0.0
    friction_factor: float = 0.0
    pressure_loss: float = 0.0
    local_pressure_loss: float = 0.0
    total_pressure_loss: float = 0.0
    pump_head: float = 0.0
    inlet_temperature: float = 0.0
    outlet_temperature: float = 0.0
    temperature_drop: float = 0.0
    heat_loss: float = 0.0
    heat_transfer_coefficient: float = 0.0
    power_consumption: float = 0.0

@dataclass
class NodeResult:
    node_id: int
    pressure: float = 0.0
    temperature: float = 0.0
    flow_in: float = 0.0
    flow_out: float = 0.0

@dataclass
class NetworkResults:
    pipe_results: Dict[int, PipeSectionResult] = field(default_factory=dict)
    node_results: Dict[int, NodeResult] = field(default_factory=dict)
    total_heat_supplied: float = 0.0
    total_heat_loss: float = 0.0
    heat_loss_rate: float = 0.0
    total_pump_power: float = 0.0
    specific_energy_consumption: float = 0.0
    iterations: int = 0
    converged: bool = False

class PipeNetwork:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.pipes: Dict[int, Pipe] = {}
        self.environment_temperature: float = 5.0
        self.water_specific_heat: float = 4186.0

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_pipe(self, pipe: Pipe) -> None:
        self.pipes[pipe.id] = pipe

    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        return [n for n in self.nodes.values() if n.type == node_type]

    def get_connected_pipes(self, node_id: int) -> List[Pipe]:
        return [p for p in self.pipes.values() if p.start_node_id == node_id or p.end_node_id == node_id]

    def get_upstream_pipes(self, node_id: int) -> List[Pipe]:
        return [p for p in self.pipes.values() if p.end_node_id == node_id]

    def get_downstream_pipes(self, node_id: int) -> List[Pipe]:
        return [p for p in self.pipes.values() if p.start_node_id == node_id]

    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        source_nodes = self.get_nodes_by_type(NODE_TYPE_SOURCE)
        end_users = self.get_nodes_by_type(NODE_TYPE_END_USER)
        if not source_nodes:
            errors.append("至少需要一个热源节点")
        if not end_users:
            errors.append("至少需要一个末端用户节点")
        for pipe in self.pipes.values():
            if pipe.start_node_id not in self.nodes:
                errors.append(f"管段{pipe.id}的起始节点{pipe.start_node_id}不存在")
            if pipe.end_node_id not in self.nodes:
                errors.append(f"管段{pipe.id}的终止节点{pipe.end_node_id}不存在")
            if pipe.diameter <= 0:
                errors.append(f"管段{pipe.id}的管径必须为正")
            if pipe.length <= 0:
                errors.append(f"管段{pipe.id}的长度必须为正")
            if pipe.has_pump and (pipe.rated_head is None or pipe.rated_head <= 0):
                errors.append(f"含泵站的管段{pipe.id}需要指定额定扬程")
        return (len(errors) == 0, errors)
