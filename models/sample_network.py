from models.network import (
    PipeNetwork, Node, Pipe,
    NODE_TYPE_SOURCE, NODE_TYPE_HEAT_EXCHANGER,
    NODE_TYPE_BRANCH, NODE_TYPE_END_USER,
    PIPE_MATERIAL_STEEL, PIPE_MATERIAL_CAST_IRON,
    INSULATION_MATERIAL_POLYURETHANE, INSULATION_MATERIAL_ROCK_WOOL,
)
import numpy as np


def create_sample_network() -> PipeNetwork:
    net = PipeNetwork()
    net.environment_temperature = 5.0

    node_positions = []
    theta = 0.0
    for i in range(30):
        if i < 2:
            r = 0
        elif i < 8:
            r = 3
            theta += np.pi / 3
        elif i < 16:
            r = 6
            theta += np.pi / 4
        elif i < 24:
            r = 9
            theta += np.pi / 4
        else:
            r = 12
            theta += np.pi / 3
        x = r * np.cos(theta) if r > 0 else (i == 0 and -1.5 or 1.5)
        y = r * np.sin(theta) if r > 0 else 0
        node_positions.append((x, y))

    node_configs = [
        (0, "热源1#", NODE_TYPE_SOURCE, 5.0, 110.0, None),
        (1, "热源2#", NODE_TYPE_SOURCE, 5.0, 108.0, None),
        (2, "主干分支点A", NODE_TYPE_BRANCH, 4.8, None, None),
        (3, "主干分支点B", NODE_TYPE_BRANCH, 4.6, None, None),
        (4, "主干分支点C", NODE_TYPE_BRANCH, 4.5, None, None),
        (5, "换热站1#", NODE_TYPE_HEAT_EXCHANGER, 4.2, None, None),
        (6, "换热站2#", NODE_TYPE_HEAT_EXCHANGER, 4.0, None, None),
        (7, "主干分支点D", NODE_TYPE_BRANCH, 4.3, None, None),
        (8, "分支节点1", NODE_TYPE_BRANCH, 3.8, None, None),
        (9, "分支节点2", NODE_TYPE_BRANCH, 3.6, None, None),
        (10, "分支节点3", NODE_TYPE_BRANCH, 3.5, None, None),
        (11, "末端用户1#", NODE_TYPE_END_USER, 3.0, None, 0.030),
        (12, "末端用户2#", NODE_TYPE_END_USER, 2.8, None, 0.025),
        (13, "末端用户3#", NODE_TYPE_END_USER, 3.2, None, 0.035),
        (14, "末端用户4#", NODE_TYPE_END_USER, 3.0, None, 0.028),
        (15, "分支节点4", NODE_TYPE_BRANCH, 3.3, None, None),
        (16, "分支节点5", NODE_TYPE_BRANCH, 2.8, None, None),
        (17, "分支节点6", NODE_TYPE_BRANCH, 2.5, None, None),
        (18, "分支节点7", NODE_TYPE_BRANCH, 2.6, None, None),
        (19, "末端用户5#", NODE_TYPE_END_USER, 2.2, None, 0.040),
        (20, "末端用户6#", NODE_TYPE_END_USER, 2.0, None, 0.045),
        (21, "末端用户7#", NODE_TYPE_END_USER, 2.3, None, 0.032),
        (22, "末端用户8#", NODE_TYPE_END_USER, 2.1, None, 0.038),
        (23, "分支节点8", NODE_TYPE_BRANCH, 2.0, None, None),
        (24, "末端用户9#", NODE_TYPE_END_USER, 1.5, None, 0.050),
        (25, "末端用户10#", NODE_TYPE_END_USER, 1.2, None, 0.055),
        (26, "末端用户11#", NODE_TYPE_END_USER, 1.8, None, 0.042),
        (27, "末端用户12#", NODE_TYPE_END_USER, 1.6, None, 0.048),
        (28, "末端用户13#", NODE_TYPE_END_USER, 1.4, None, 0.036),
        (29, "末端用户14#", NODE_TYPE_END_USER, 1.0, None, 0.060),
    ]

    for idx, name, ntype, elev, sup_p, dflow in node_configs:
        x, y = node_positions[idx]
        node = Node(
            id=idx, name=name, type=ntype, elevation=elev,
            x=x, y=y,
            supply_pressure=sup_p if ntype == NODE_TYPE_SOURCE else None,
            return_pressure=0.2 if ntype == NODE_TYPE_SOURCE else None,
            design_flow=dflow,
            rated_capacity=50.0e6 if idx == 0 else (40.0e6 if idx == 1 else None),
        )
        net.add_node(node)

    pipe_configs = [
        (0, "S1-主干线1", 0, 2, 0.6, 250, PIPE_MATERIAL_STEEL, 15, INSULATION_MATERIAL_POLYURETHANE, 0.08, 1.8, True, 1500, True, 45),
        (1, "S2-主干线2", 1, 3, 0.5, 280, PIPE_MATERIAL_STEEL, 12, INSULATION_MATERIAL_POLYURETHANE, 0.07, 1.7, True, 1200, True, 40),
        (2, "主干连接1", 2, 4, 0.6, 180, PIPE_MATERIAL_STEEL, 15, INSULATION_MATERIAL_POLYURETHANE, 0.08, 1.8, False, None, False, None),
        (3, "主干连接2", 3, 4, 0.5, 200, PIPE_MATERIAL_STEEL, 12, INSULATION_MATERIAL_POLYURETHANE, 0.07, 1.7, False, None, False, None),
        (4, "环网管段1", 2, 5, 0.4, 320, PIPE_MATERIAL_STEEL, 18, INSULATION_MATERIAL_ROCK_WOOL, 0.06, 2.0, True, 800, False, None),
        (5, "环网管段2", 5, 7, 0.35, 280, PIPE_MATERIAL_CAST_IRON, 25, INSULATION_MATERIAL_POLYURETHANE, 0.06, 1.9, False, None, False, None),
        (6, "环网管段3", 4, 7, 0.4, 240, PIPE_MATERIAL_STEEL, 10, INSULATION_MATERIAL_POLYURETHANE, 0.07, 1.8, False, None, False, None),
        (7, "环网管段4", 7, 6, 0.35, 300, PIPE_MATERIAL_CAST_IRON, 22, INSULATION_MATERIAL_ROCK_WOOL, 0.06, 2.1, False, None, False, None),
        (8, "环网管段5", 3, 6, 0.4, 350, PIPE_MATERIAL_STEEL, 14, INSULATION_MATERIAL_POLYURETHANE, 0.07, 1.8, True, 900, False, None),
        (9, "支线1", 4, 8, 0.3, 220, PIPE_MATERIAL_STEEL, 16, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.6, True, 600, False, None),
        (10, "支线2", 8, 9, 0.25, 180, PIPE_MATERIAL_STEEL, 16, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.6, False, None, False, None),
        (11, "支线3", 9, 10, 0.2, 150, PIPE_MATERIAL_CAST_IRON, 28, INSULATION_MATERIAL_ROCK_WOOL, 0.05, 1.7, False, None, False, None),
        (12, "用户供水管1", 8, 11, 0.15, 100, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 400, False, None),
        (13, "用户供水管2", 9, 12, 0.125, 90, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 300, False, None),
        (14, "用户供水管3", 10, 13, 0.15, 110, PIPE_MATERIAL_STEEL, 9, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 350, False, None),
        (15, "用户供水管4", 10, 14, 0.125, 95, PIPE_MATERIAL_CAST_IRON, 20, INSULATION_MATERIAL_ROCK_WOOL, 0.04, 1.6, True, 320, False, None),
        (16, "支线4", 5, 15, 0.3, 200, PIPE_MATERIAL_STEEL, 11, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.6, False, None, False, None),
        (17, "支线5", 15, 16, 0.25, 170, PIPE_MATERIAL_STEEL, 13, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.6, True, 500, False, None),
        (18, "支线6", 16, 17, 0.2, 140, PIPE_MATERIAL_CAST_IRON, 24, INSULATION_MATERIAL_ROCK_WOOL, 0.05, 1.7, False, None, False, None),
        (19, "支线7", 17, 18, 0.175, 120, PIPE_MATERIAL_STEEL, 10, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.6, False, None, False, None),
        (20, "用户供水管5", 16, 19, 0.15, 95, PIPE_MATERIAL_STEEL, 7, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 420, False, None),
        (21, "用户供水管6", 17, 20, 0.15, 105, PIPE_MATERIAL_STEEL, 7, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 450, False, None),
        (22, "用户供水管7", 18, 21, 0.125, 85, PIPE_MATERIAL_CAST_IRON, 19, INSULATION_MATERIAL_ROCK_WOOL, 0.04, 1.6, True, 330, False, None),
        (23, "用户供水管8", 18, 22, 0.125, 88, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 360, False, None),
        (24, "支线8", 6, 23, 0.3, 190, PIPE_MATERIAL_STEEL, 12, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.6, True, 550, False, None),
        (25, "支线9", 7, 23, 0.25, 260, PIPE_MATERIAL_CAST_IRON, 26, INSULATION_MATERIAL_ROCK_WOOL, 0.05, 1.9, False, None, False, None),
        (26, "支线10", 23, 24, 0.2, 160, PIPE_MATERIAL_STEEL, 9, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.5, False, None, False, None),
        (27, "支线11", 23, 25, 0.2, 175, PIPE_MATERIAL_STEEL, 14, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.6, False, None, False, None),
        (28, "支线12", 15, 26, 0.2, 130, PIPE_MATERIAL_CAST_IRON, 21, INSULATION_MATERIAL_ROCK_WOOL, 0.05, 1.7, True, 480, False, None),
        (29, "支线13", 8, 27, 0.15, 105, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 410, False, None),
        (30, "支线14", 9, 28, 0.125, 80, PIPE_MATERIAL_STEEL, 11, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, True, 340, False, None),
        (31, "支线15", 16, 29, 0.175, 115, PIPE_MATERIAL_CAST_IRON, 23, INSULATION_MATERIAL_ROCK_WOOL, 0.05, 1.6, True, 440, False, None),
        (32, "回水主干1", 11, 8, 0.15, 100, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
        (33, "回水主干2", 12, 9, 0.125, 90, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
        (34, "回水主干3", 13, 10, 0.15, 110, PIPE_MATERIAL_STEEL, 9, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
        (35, "回水主干4", 14, 10, 0.125, 95, PIPE_MATERIAL_CAST_IRON, 20, INSULATION_MATERIAL_ROCK_WOOL, 0.04, 1.6, False, None, False, None),
        (36, "回水主干5", 19, 16, 0.15, 95, PIPE_MATERIAL_STEEL, 7, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
        (37, "回水主干6", 20, 17, 0.15, 105, PIPE_MATERIAL_STEEL, 7, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
        (38, "回水主干7", 21, 18, 0.125, 85, PIPE_MATERIAL_CAST_IRON, 19, INSULATION_MATERIAL_ROCK_WOOL, 0.04, 1.6, False, None, False, None),
        (39, "回水主干8", 22, 18, 0.125, 88, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
        (40, "回水主干9", 24, 23, 0.2, 160, PIPE_MATERIAL_STEEL, 9, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.5, False, None, False, None),
        (41, "回水主干10", 25, 23, 0.2, 175, PIPE_MATERIAL_STEEL, 14, INSULATION_MATERIAL_POLYURETHANE, 0.05, 1.6, False, None, False, None),
        (42, "回水主干11", 26, 15, 0.2, 130, PIPE_MATERIAL_CAST_IRON, 21, INSULATION_MATERIAL_ROCK_WOOL, 0.05, 1.7, False, None, False, None),
        (43, "回水主干12", 27, 8, 0.15, 105, PIPE_MATERIAL_STEEL, 8, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
        (44, "回水主干13", 28, 9, 0.125, 80, PIPE_MATERIAL_STEEL, 11, INSULATION_MATERIAL_POLYURETHANE, 0.04, 1.5, False, None, False, None),
    ]

    for pid, pname, sn, en, d, l, mat, age, ins, thick, depth, hv, kv, hp, head in pipe_configs:
        pump_curve = None
        if hp and head:
            q_max = 0.4 if d > 0.5 else 0.32
            pump_curve = [
                (0.0, head * 1.2),
                (q_max * 0.2, head * 1.15),
                (q_max * 0.4, head * 1.05),
                (q_max * 0.6, head * 0.9),
                (q_max * 0.8, head * 0.7),
                (q_max, head * 0.4),
            ]
        eq_ratio = 0.15 if hv else 0.08
        pipe = Pipe(
            id=pid, name=pname, start_node_id=sn, end_node_id=en,
            diameter=d, length=l, material=mat, pipe_age=age,
            insulation_material=ins, insulation_thickness=thick,
            burial_depth=depth, has_valve=hv, valve_kv=kv,
            has_pump=hp, rated_head=head, pump_efficiency_curve=pump_curve,
            equivalent_local_length_ratio=eq_ratio,
        )
        net.add_pipe(pipe)

    return net
