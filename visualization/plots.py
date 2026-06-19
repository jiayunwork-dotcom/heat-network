import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from collections import deque

from models import (
    PipeNetwork, Node, Pipe, NetworkResults, PipeSectionResult,
    NODE_TYPE_SOURCE, NODE_TYPE_HEAT_EXCHANGER, NODE_TYPE_BRANCH, NODE_TYPE_END_USER,
    get_water_properties,
)
from calculations import compute_pump_head


def create_network_topology_figure(
    network: PipeNetwork,
    results: NetworkResults,
    color_by: str = "temperature",
) -> go.Figure:
    fig = go.Figure()
    max_flow = 0.0
    for pr in results.pipe_results.values():
        if abs(pr.flow_rate) > max_flow:
            max_flow = abs(pr.flow_rate)
    if max_flow < 1e-6:
        max_flow = 1.0
    pipe_vals = []
    all_mid_x = []
    all_mid_y = []
    for pid, pipe in network.pipes.items():
        pr = results.pipe_results.get(pid, PipeSectionResult(pipe_id=pid))
        sn = network.nodes[pipe.start_node_id]
        en = network.nodes[pipe.end_node_id]
        sx, sy = (sn.x, sn.y) if sn.x is not None else (0.0, 0.0)
        ex, ey = (en.x, en.y) if en.x is not None else (0.0, 0.0)
        if color_by == "temperature":
            val = (pr.inlet_temperature + pr.outlet_temperature) / 2.0
            colorbar_title = "温度 (°C)"
        elif color_by == "pressure":
            p_s = results.node_results.get(pipe.start_node_id)
            p_e = results.node_results.get(pipe.end_node_id)
            val = ((p_s.pressure if p_s else 0.0) + (p_e.pressure if p_e else 0.0)) / 2.0
            colorbar_title = "压力 (mH₂O)"
        else:
            val = pr.flow_rate
            colorbar_title = "流量 (m³/s)"
        val = float(val)
        pipe_vals.append(val)
        all_mid_x.append(float((sx + ex) / 2.0))
        all_mid_y.append(float((sy + ey) / 2.0))
        line_width = float(1.0 + 6.0 * abs(pr.flow_rate) / max(max_flow, 1e-6))
        hover_text = (
            f"<b>{pipe.name}</b><br>"
            f"管径: {pipe.diameter*1000:.0f}mm<br>"
            f"长度: {pipe.length:.0f}m<br>"
            f"流量: {pr.flow_rate*1000:.1f} L/s<br>"
            f"流速: {pr.velocity:.2f} m/s<br>"
            f"进口温度: {pr.inlet_temperature:.2f}°C<br>"
            f"出口温度: {pr.outlet_temperature:.2f}°C<br>"
            f"温降: {pr.temperature_drop:.2f}°C<br>"
            f"压力损失: {pr.total_pressure_loss/1000:.2f} kPa<br>"
            f"热损失: {pr.heat_loss/1000:.2f} kW"
        )
        if pipe.has_pump:
            hover_text += f"<br><b>泵站</b>: 扬程{pr.pump_head:.1f}m, 功率{pr.power_consumption:.1f}kW"
        if pipe.has_valve:
            hover_text += f"<br><b>阀门</b>: Kv={pipe.valve_kv}"
        fig.add_trace(go.Scatter(
            x=[float(sx), float(ex)],
            y=[float(sy), float(ey)],
            mode='lines',
            line=dict(
                width=line_width,
                color='#888888',
            ),
            hoverinfo='skip',
            showlegend=False,
        ))
    colorscale = 'RdYlBu_r' if color_by == "temperature" else ('Viridis' if color_by == "pressure" else 'Plasma')
    if pipe_vals:
        vmin = min(pipe_vals)
        vmax = max(pipe_vals)
        if abs(vmax - vmin) < 0.01:
            vmax = vmin + 1.0
        fig.add_trace(go.Scatter(
            x=all_mid_x, y=all_mid_y,
            mode='markers',
            marker=dict(
                size=4,
                color=pipe_vals,
                colorscale=colorscale,
                cmin=vmin, cmax=vmax,
                colorbar=dict(title=colorbar_title, thickness=15),
                showscale=True,
                opacity=0.0001,
            ),
            hoverinfo='skip',
            showlegend=False,
        ))
    node_type_colors = {
        NODE_TYPE_SOURCE: "#ff4444",
        NODE_TYPE_HEAT_EXCHANGER: "#ffaa00",
        NODE_TYPE_BRANCH: "#4488ff",
        NODE_TYPE_END_USER: "#44cc88",
    }
    node_type_sizes = {
        NODE_TYPE_SOURCE: 20,
        NODE_TYPE_HEAT_EXCHANGER: 16,
        NODE_TYPE_BRANCH: 10,
        NODE_TYPE_END_USER: 14,
    }
    for ntype in [NODE_TYPE_SOURCE, NODE_TYPE_HEAT_EXCHANGER, NODE_TYPE_BRANCH, NODE_TYPE_END_USER]:
        nodes_of_type = network.get_nodes_by_type(ntype)
        xs, ys, texts = [], [], []
        for node in nodes_of_type:
            nx, ny = (node.x, node.y) if node.x is not None else (0, 0)
            xs.append(nx)
            ys.append(ny)
            nr = results.node_results.get(node.id)
            type_label = {
                NODE_TYPE_SOURCE: "热源",
                NODE_TYPE_HEAT_EXCHANGER: "换热站",
                NODE_TYPE_BRANCH: "分支点",
                NODE_TYPE_END_USER: "末端用户",
            }[ntype]
            hover = (
                f"<b>{node.name}</b> [{type_label}]<br>"
                f"标高: {node.elevation:.1f}m<br>"
                f"压力: {nr.pressure:.2f} mH₂O<br>"
                f"温度: {nr.temperature:.2f}°C"
            )
            if node.design_flow:
                hover += f"<br>设计流量: {node.design_flow*1000:.1f} L/s"
                actual = nr.flow_in if nr else 0
                hover += f"<br>实际流量: {actual*1000:.1f} L/s"
            texts.append(hover)
        node_trace = go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(
                size=node_type_sizes[ntype],
                color=node_type_colors[ntype],
                line=dict(width=2, color='white'),
                symbol='circle',
            ),
            text=texts,
            hoverinfo='text',
            name={
                NODE_TYPE_SOURCE: "热源节点",
                NODE_TYPE_HEAT_EXCHANGER: "换热站",
                NODE_TYPE_BRANCH: "分支节点",
                NODE_TYPE_END_USER: "末端用户",
            }[ntype],
            showlegend=True,
        )
        fig.add_trace(node_trace)
    if pipe_vals:
        vmin, vmax = min(pipe_vals), max(pipe_vals)
        if abs(vmax - vmin) < 0.01:
            vmax = vmin + 1
        dummy = go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                colorscale='RdYlBu_r' if color_by == "temperature" else ('Viridis' if color_by == "pressure" else 'Plasma'),
                cmin=vmin, cmax=vmax, color=[(vmin+vmax)/2],
                colorbar=dict(title=colorbar_title, thickness=15),
                showscale=True,
            ),
            showlegend=False,
        )
        fig.add_trace(dummy)
    fig.update_layout(
        title=dict(text=f"供热管网拓扑图（按{color_by}着色，线宽=流量）", x=0.5),
        xaxis=dict(title="X坐标 (m)", showgrid=True, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y坐标 (m)", showgrid=True, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(240,248,255,0.3)',
        margin=dict(l=40, r=40, t=80, b=40),
        height=600,
    )
    return fig


def _find_longest_path_from_sources(
    network: PipeNetwork,
    results: NetworkResults,
) -> Tuple[List[int], List[PipeSectionResult]]:
    sources = network.get_nodes_by_type(NODE_TYPE_SOURCE)
    best_path = []
    best_total_drop = 0
    for src in sources:
        paths = {}
        q = deque()
        q.append((src.id, [src.id], 0.0))
        while q:
            nid, path, drop = q.popleft()
            if drop > best_total_drop:
                best_total_drop = drop
                best_path = path[:]
            for dp in network.get_downstream_pipes(nid):
                pr = results.pipe_results.get(dp.id)
                if pr and dp.end_node_id not in path:
                    new_drop = drop + pr.temperature_drop
                    q.append((dp.end_node_id, path + [dp.end_node_id], new_drop))
    path_pipes = []
    for i in range(len(best_path) - 1):
        s, e = best_path[i], best_path[i + 1]
        for pid, pipe in network.pipes.items():
            if pipe.start_node_id == s and pipe.end_node_id == e:
                pr = results.pipe_results.get(pid, PipeSectionResult(pipe_id=pid))
                path_pipes.append(pr)
                break
    return best_path, path_pipes


def create_temperature_drop_figure(
    network: PipeNetwork,
    results: NetworkResults,
) -> go.Figure:
    path, path_pipes = _find_longest_path_from_sources(network, results)
    if not path:
        path = list(network.nodes.keys())[:10]
        path_pipes = list(results.pipe_results.values())[:len(path)-1]
    cumulative_length = [0.0]
    temperatures = []
    pressure_list = []
    for i, nid in enumerate(path):
        nr = results.node_results.get(nid)
        temperatures.append(nr.temperature if nr else 0)
        pressure_list.append(nr.pressure if nr else 0)
        if i < len(path_pipes):
            pr = path_pipes[i]
            pipe = None
            for p in network.pipes.values():
                if p.id == pr.pipe_id:
                    pipe = p
                    break
            cumulative_length.append(cumulative_length[-1] + (pipe.length if pipe else 100))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=cumulative_length, y=temperatures,
            mode='lines+markers', name='供水温度',
            line=dict(color='red', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='距热源: %{x:.0f}m<br>温度: %{y:.2f}°C<extra></extra>',
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_length, y=pressure_list,
            mode='lines+markers', name='节点压力',
            line=dict(color='blue', width=2, dash='dash'),
            marker=dict(size=8, symbol='square'),
            hovertemplate='距热源: %{x:.0f}m<br>压力: %{y:.2f} mH₂O<extra></extra>',
        ),
        secondary_y=True,
    )
    node_labels = [network.nodes[nid].name for nid in path]
    for i, label in enumerate(node_labels):
        fig.add_annotation(
            x=cumulative_length[i], y=temperatures[i],
            text=label, showarrow=True, arrowhead=1,
            yshift=15, font=dict(size=9),
        )
    fig.update_layout(
        title=dict(text="沿最长路径温度衰减与压力分布曲线", x=0.5),
        xaxis=dict(title="距热源累计距离 (m)", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=60, r=60, t=100, b=50),
    )
    fig.update_yaxes(title_text="温度 (°C)", secondary_y=False, color='red')
    fig.update_yaxes(title_text="压力 (mH₂O)", secondary_y=True, color='blue')
    return fig


def create_pressure_contour_figure(
    network: PipeNetwork,
    results: NetworkResults,
) -> Optional[go.Figure]:
    xs = [n.x for n in network.nodes.values() if n.x is not None]
    ys = [n.y for n in network.nodes.values() if n.y is not None]
    if len(xs) < 4:
        return None
    coords = []
    pressures = []
    temps = []
    for nid, node in network.nodes.items():
        if node.x is not None and node.y is not None:
            nr = results.node_results.get(nid)
            coords.append((node.x, node.y))
            pressures.append(nr.pressure if nr else 0)
            temps.append(nr.temperature if nr else 0)
    coords_arr = np.array(coords)
    fig = go.Figure()
    try:
        from scipy.interpolate import griddata
        xi = np.linspace(coords_arr[:, 0].min() - 1, coords_arr[:, 0].max() + 1, 50)
        yi = np.linspace(coords_arr[:, 1].min() - 1, coords_arr[:, 1].max() + 1, 50)
        xi, yi = np.meshgrid(xi, yi)
        zi_p = griddata(coords_arr, np.array(pressures), (xi, yi), method='cubic', fill_value=np.mean(pressures))
        fig.add_trace(go.Contour(
            z=zi_p, x=xi[0], y=yi[:, 0],
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=12)),
            colorbar=dict(title='压力 (mH₂O)'),
            name='压力等值线',
        ))
    except Exception:
        pass
    for pid, pipe in network.pipes.items():
        sn = network.nodes[pipe.start_node_id]
        en = network.nodes[pipe.end_node_id]
        if sn.x is None or en.x is None:
            continue
        pr = results.pipe_results.get(pid)
        fig.add_trace(go.Scatter(
            x=[sn.x, en.x], y=[sn.y, en.y],
            mode='lines', line=dict(color='black', width=1.5, dash='solid'),
            showlegend=False, hoverinfo='skip',
        ))
    node_colors = temps
    fig.add_trace(go.Scatter(
        x=coords_arr[:, 0], y=coords_arr[:, 1],
        mode='markers',
        marker=dict(
            size=14, color=node_colors, colorscale='RdYlBu_r',
            line=dict(width=2, color='black'),
            colorbar=dict(title='节点温度 (°C)', x=1.08),
        ),
        text=[network.nodes[nid].name for nid in network.nodes if network.nodes[nid].x is not None],
        hovertemplate='<b>%{text}</b><br>T=%{marker.color:.1f}°C<extra></extra>',
        name='节点',
    ))
    fig.update_layout(
        title=dict(text="管网压力等值线与节点温度分布", x=0.5),
        xaxis=dict(title="X坐标 (m)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y坐标 (m)"),
        height=600,
    )
    return fig


def create_pump_operating_point_figures(
    network: PipeNetwork,
    results: NetworkResults,
) -> Dict[int, go.Figure]:
    figures = {}
    g = 9.81
    for pid, pipe in network.pipes.items():
        if not pipe.has_pump or pipe.rated_head is None:
            continue
        pr = results.pipe_results.get(pid)
        if not pr:
            continue
        if pipe.pump_efficiency_curve and len(pipe.pump_efficiency_curve) >= 2:
            q_points = [p[0] for p in pipe.pump_efficiency_curve]
            h_points = [p[1] for p in pipe.pump_efficiency_curve]
            q_pump = np.linspace(0, max(q_points) * 1.1, 100)
            h_pump = np.interp(q_pump, q_points, h_points, left=h_points[0], right=0)
            q_max_curve = max(q_points)
        else:
            q_max_curve = pipe.rated_head / 20.0
            q_pump = np.linspace(0, q_max_curve * 1.2, 100)
            h_pump = pipe.rated_head * (1.0 - (q_pump / max(q_max_curve, 1e-6)) ** 2)
        flow_op = abs(pr.flow_rate)
        head_op = abs(pr.pump_head)
        if flow_op > 1e-6 and head_op > 0:
            k_system = head_op / (flow_op ** 2)
        else:
            k_system = 1e6
        q_sys = np.linspace(0, max(q_max_curve, flow_op * 1.5), 100)
        h_sys = k_system * q_sys ** 2
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=q_pump * 1000, y=h_pump, mode='lines',
            name='泵特性曲线H-Q',
            line=dict(color='red', width=3),
        ))
        fig.add_trace(go.Scatter(
            x=q_sys * 1000, y=h_sys, mode='lines',
            name='管网阻力曲线',
            line=dict(color='blue', width=3, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=[flow_op * 1000], y=[head_op], mode='markers',
            name=f'工作点: Q={flow_op*1000:.1f}L/s, H={head_op:.1f}m',
            marker=dict(size=16, color='green', symbol='star', line=dict(width=2, color='black')),
        ))
        eta_points = []
        if pipe.pump_efficiency_curve and len(pipe.pump_efficiency_curve) >= 2:
            h_rated = pipe.rated_head
            for q, h in pipe.pump_efficiency_curve:
                ratio = h / max(h_rated, 1e-6)
                eta = 0.6 + 0.2 * ratio * (2 - ratio)
                eta_points.append((q, eta))
        if eta_points:
            qe = [p[0] * 1000 for p in eta_points]
            he = [p[1] * 100 for p in eta_points]
            fig.add_trace(go.Scatter(
                x=qe, y=he, mode='lines',
                name='泵效率(%)',
                line=dict(color='purple', width=2, dash='dot'),
                yaxis='y2',
            ))
        density, _ = get_water_properties((pr.inlet_temperature + pr.outlet_temperature) / 2.0)
        power_op = density * g * flow_op * head_op / (0.75 * 1000) if flow_op > 0 else 0
        fig.add_annotation(
            x=flow_op * 1000, y=head_op,
            text=f'功率: {pr.power_consumption:.1f} kW',
            showarrow=True, arrowhead=2,
            ax=80, ay=-40,
            font=dict(size=12, color='darkgreen'),
            bgcolor='rgba(255,255,255,0.8)',
        )
        fig.update_layout(
            title=dict(text=f"泵站工作点分析 - {pipe.name}", x=0.5),
            xaxis=dict(title="流量 Q (L/s)", showgrid=True),
            yaxis=dict(title="扬程 H (m)", color='red', showgrid=True),
            height=450,
            margin=dict(l=60, r=60, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        if eta_points:
            fig.update_layout(yaxis2=dict(
                title=dict(text="效率 (%)", font=dict(color='purple')),
                overlaying='y', side='right',
                range=[0, 100],
            ))
        figures[pid] = fig
    return figures


def create_heat_loss_pareto_figure(
    network: PipeNetwork,
    results: NetworkResults,
) -> go.Figure:
    pipe_hl = []
    for pid, pr in results.pipe_results.items():
        pipe = network.pipes[pid]
        pipe_hl.append((pid, pipe.name, pr.heat_loss))
    pipe_hl.sort(key=lambda x: -x[2])
    total_hl = sum(x[2] for x in pipe_hl) if pipe_hl else 1
    names = [x[1] for x in pipe_hl]
    losses_kw = [x[2] / 1000.0 for x in pipe_hl]
    cumulative_pct = []
    cum = 0.0
    for x in pipe_hl:
        cum += x[2]
        cumulative_pct.append(cum / max(total_hl, 1e-6) * 100)
    n_top20 = max(1, int(len(pipe_hl) * 0.2))
    top20_pct = cumulative_pct[n_top20 - 1] if n_top20 <= len(cumulative_pct) else cumulative_pct[-1]
    colors = ['#ff6b6b' if i < n_top20 else '#4ecdc4' for i in range(len(pipe_hl))]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=names, y=losses_kw,
            marker_color=colors,
            name='各管段热损失 (kW)',
            hovertemplate='<b>%{x}</b><br>热损失: %{y:.2f} kW<extra></extra>',
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=names, y=cumulative_pct,
            mode='lines+markers',
            name='累计占比 (%)',
            line=dict(color='#ffa500', width=3),
            marker=dict(size=8),
            hovertemplate='累计占比: %{y:.1f}%<extra></extra>',
        ),
        secondary_y=True,
    )
    if len(pipe_hl) > 0:
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(pipe_hl) - 0.5,
            y0=80, y1=80,
            line=dict(color="red", width=2, dash="dash"),
            secondary_y=True,
        )
        fig.add_annotation(
            x=len(pipe_hl) * 0.8, y=82,
            text="80% 基准线",
            showarrow=False,
            font=dict(color='red', size=11),
            secondary_y=True,
        )
    fig.update_layout(
        title=dict(text=f"各管段热损失Pareto分析（前20%管段贡献了{top20_pct:.1f}%的总热损失）", x=0.5),
        xaxis=dict(title="管段名称", tickangle=-45, showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
        margin=dict(l=60, r=60, t=120, b=150),
    )
    fig.update_yaxes(title_text="热损失 (kW)", secondary_y=False)
    fig.update_yaxes(title_text="累计占比 (%)", secondary_y=True, range=[0, 105])
    return fig


def create_energy_consumption_pie_figure(
    network: PipeNetwork,
    results: NetworkResults,
) -> go.Figure:
    pump_data = []
    for pid, pr in results.pipe_results.items():
        pipe = network.pipes[pid]
        if pipe.has_pump and pr.power_consumption > 0:
            pump_data.append((pipe.name, pr.power_consumption))
    if not pump_data:
        return go.Figure().add_annotation(text="无泵站数据", showarrow=False, font=dict(size=20))
    total_power = sum(x[1] for x in pump_data)
    total_hl = sum(pr.heat_loss / 1000.0 for pr in results.pipe_results.values())
    labels = [x[0] for x in pump_data]
    values = [x[1] for x in pump_data]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9',
              '#a29bfe', '#fd79a8', '#00b894', '#e17055', '#74b9ff', '#fab1a0']
    while len(colors) < len(labels):
        colors += colors
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=('各泵站耗电占比', '能量损失分布'),
    )
    fig.add_trace(
        go.Pie(
            labels=labels, values=values,
            hole=0.4,
            marker=dict(colors=colors[:len(labels)]),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>功率: %{value:.1f} kW<br>占比: %{percent}<extra></extra>',
            name='泵站电耗',
        ),
        row=1, col=1,
    )
    loss_labels = ['各泵站电耗合计', '管网热损失合计', '有效供热']
    heat_delivered = max(results.total_heat_supplied - results.total_heat_loss, 0) / 1000.0
    loss_values = [total_power, total_hl, heat_delivered]
    fig.add_trace(
        go.Pie(
            labels=loss_labels, values=loss_values,
            hole=0.4,
            marker=dict(colors=['#45b7d1', '#ff6b6b', '#00b894']),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>功率: %{value:.0f} kW<extra></extra>',
            name='能量分布',
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title=dict(text=f"能耗分解分析（总泵耗: {total_power:.1f}kW, 热损失率: {results.heat_loss_rate:.1f}%）", x=0.5),
        height=500,
        margin=dict(l=40, r=40, t=100, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


def create_fault_topology_figure(
    original_network: PipeNetwork,
    fault_results: Optional[NetworkResults],
    impact,
    faults,
    original_results: Optional[NetworkResults] = None,
) -> go.Figure:
    fig = go.Figure()

    burst_pipe_ids = set()
    failed_pump_ids = set()
    shutdown_source_ids = set()
    for f in faults:
        if f.fault_type == "pipe_burst":
            burst_pipe_ids.add(f.target_id)
        elif f.fault_type == "pump_failure":
            failed_pump_ids.add(f.target_id)
        elif f.fault_type == "source_shutdown":
            shutdown_source_ids.add(f.target_id)

    affected_user_set = set(impact.affected_user_ids) if impact else set()
    disconnected_user_set = set(impact.disconnected_user_ids) if impact else set()
    low_pressure_set = set(impact.low_pressure_users) if impact else set()

    display_network = original_network
    display_results = fault_results if fault_results else original_results

    max_flow = 0.0
    if display_results:
        for pr in display_results.pipe_results.values():
            if abs(pr.flow_rate) > max_flow:
                max_flow = abs(pr.flow_rate)
    if max_flow < 1e-6:
        max_flow = 1.0

    for pid, pipe in original_network.pipes.items():
        sn = original_network.nodes[pipe.start_node_id]
        en = original_network.nodes[pipe.end_node_id]
        sx, sy = (sn.x, sn.y) if sn.x is not None else (0.0, 0.0)
        ex, ey = (en.x, en.y) if en.x is not None else (0.0, 0.0)

        pr = None
        if display_results and pid in display_results.pipe_results:
            pr = display_results.pipe_results[pid]

        is_burst = pid in burst_pipe_ids
        is_failed_pump = pid in failed_pump_ids

        if is_burst:
            line_color = "#ff0000"
            line_width = 6.0
            line_dash = "dash"
        elif is_failed_pump:
            line_color = "#ff8800"
            line_width = 5.0
            line_dash = "dashdot"
        else:
            line_color = "#888888"
            flow = pr.flow_rate if pr else 0.001
            line_width = float(1.0 + 5.0 * abs(flow) / max_flow)
            line_dash = "solid"

        hover_text = f"<b>{pipe.name}</b><br>管径: {pipe.diameter*1000:.0f}mm<br>长度: {pipe.length:.0f}m"
        if pr:
            hover_text += (
                f"<br>流量: {pr.flow_rate*1000:.1f} L/s<br>"
                f"进口温度: {pr.inlet_temperature:.2f}°C<br>"
                f"出口温度: {pr.outlet_temperature:.2f}°C"
            )
        if is_burst:
            hover_text = "🚨 <b>[爆管故障]</b><br>" + hover_text
        if is_failed_pump:
            hover_text = "⚠️ <b>[泵站故障]</b><br>" + hover_text
        if pipe.has_pump and pr:
            hover_text += f"<br><b>泵站</b>: 扬程{abs(pr.pump_head):.1f}m"

        fig.add_trace(go.Scatter(
            x=[float(sx), float(ex)],
            y=[float(sy), float(ey)],
            mode='lines',
            line=dict(
                width=line_width,
                color=line_color,
                dash=line_dash,
            ),
            hoverinfo='skip',
            showlegend=False,
        ))
        mid_x = float((sx + ex) / 2.0)
        mid_y = float((sy + ey) / 2.0)
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode='markers',
            marker=dict(size=1, color=line_color, opacity=0.01),
            hoverinfo='text',
            text=hover_text,
            showlegend=False,
        ))

    node_type_colors = {
        NODE_TYPE_SOURCE: "#ff4444",
        NODE_TYPE_HEAT_EXCHANGER: "#ffaa00",
        NODE_TYPE_BRANCH: "#4488ff",
        NODE_TYPE_END_USER: "#44cc88",
    }
    node_type_sizes = {
        NODE_TYPE_SOURCE: 20,
        NODE_TYPE_HEAT_EXCHANGER: 16,
        NODE_TYPE_BRANCH: 10,
        NODE_TYPE_END_USER: 14,
    }
    type_labels = {
        NODE_TYPE_SOURCE: "热源",
        NODE_TYPE_HEAT_EXCHANGER: "换热站",
        NODE_TYPE_BRANCH: "分支点",
        NODE_TYPE_END_USER: "末端用户",
    }

    for ntype in [NODE_TYPE_SOURCE, NODE_TYPE_HEAT_EXCHANGER, NODE_TYPE_BRANCH, NODE_TYPE_END_USER]:
        nodes_of_type = original_network.get_nodes_by_type(ntype)
        for node in nodes_of_type:
            nx, ny = (node.x, node.y) if node.x is not None else (0, 0)

            is_shutdown = node.id in shutdown_source_ids
            is_disconnected = node.id in disconnected_user_set
            is_low_pressure = node.id in low_pressure_set
            is_affected = node.id in affected_user_set

            marker_color = node_type_colors[ntype]
            marker_size = node_type_sizes[ntype]
            marker_symbol = 'circle'
            marker_line_width = 2
            marker_line_color = 'white'

            prefix = ""
            if is_shutdown:
                prefix = "🛑 <b>[已停机]</b><br>"
                marker_color = "#333333"
                marker_symbol = 'x'
                marker_size = marker_size + 4
            elif is_disconnected:
                prefix = "❌ <b>[断供]</b><br>"
                marker_color = "#cc0000"
                marker_line_color = "#ff0000"
                marker_line_width = 3
                marker_size = marker_size + 4
            elif is_low_pressure:
                prefix = "⚠️ <b>[压力不足]</b><br>"
                marker_color = "#ff6600"
                marker_line_color = "#ff3300"
                marker_line_width = 3
            elif is_affected and ntype == NODE_TYPE_END_USER:
                prefix = "🔶 <b>[受影响]</b><br>"
                marker_line_color = "#ffaa00"
                marker_line_width = 3

            nr = None
            if display_results and node.id in display_results.node_results:
                nr = display_results.node_results[node.id]

            hover = prefix + f"<b>{node.name}</b> [{type_labels[ntype]}]<br>标高: {node.elevation:.1f}m"
            if nr:
                hover += f"<br>压力: {nr.pressure:.2f} mH₂O<br>温度: {nr.temperature:.2f}°C"
            if node.design_flow:
                hover += f"<br>设计流量: {node.design_flow*1000:.1f} L/s"
                if nr:
                    hover += f"<br>实际流量: {nr.flow_in*1000:.1f} L/s"

            legend_name = type_labels[ntype]
            if is_shutdown:
                legend_name = "已停机热源"
            elif is_disconnected:
                legend_name = "断供用户"
            elif is_low_pressure:
                legend_name = "压力不足用户"
            elif is_affected and ntype == NODE_TYPE_END_USER:
                legend_name = "受影响用户"

            show_legend = False
            if ntype == NODE_TYPE_SOURCE and is_shutdown:
                show_legend = True
            elif ntype == NODE_TYPE_END_USER and is_disconnected:
                show_legend = True
            elif ntype == NODE_TYPE_END_USER and is_low_pressure:
                show_legend = True
            elif ntype == NODE_TYPE_END_USER and is_affected and not is_low_pressure and not is_disconnected:
                show_legend = True
            elif not is_shutdown and not is_disconnected and not is_low_pressure and not (is_affected and ntype == NODE_TYPE_END_USER):
                show_legend = (ntype == NODE_TYPE_SOURCE or ntype == NODE_TYPE_END_USER)

            fig.add_trace(go.Scatter(
                x=[nx], y=[ny],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    line=dict(width=marker_line_width, color=marker_line_color),
                    symbol=marker_symbol,
                ),
                text=hover,
                hoverinfo='text',
                name=legend_name,
                showlegend=show_legend,
            ))

    fig.update_layout(
        title=dict(text="🚨 故障工况管网拓扑图（红色=爆管/断供，橙色=故障泵站/低压用户）", x=0.5),
        xaxis=dict(title="X坐标 (m)", showgrid=True, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y坐标 (m)", showgrid=True, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(240,248,255,0.3)',
        margin=dict(l=40, r=40, t=80, b=40),
        height=600,
    )
    return fig


def create_operating_cost_pie_figure(
    op_cost,
) -> go.Figure:
    labels = ['泵站电费', '热损失成本', '维护成本']
    values = [
        op_cost.electricity_cost_annual,
        op_cost.heat_loss_cost_annual,
        op_cost.maintenance_cost_annual,
    ]
    colors = ['#45b7d1', '#ff6b6b', '#ffd93d']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>年成本: %{value:,.0f} 元<br>占比: %{percent}<extra></extra>',
    )])

    total_annual = op_cost.total_annual_cost
    fig.update_layout(
        title=dict(text=f"年运行成本构成（合计：{total_annual:,.0f} 元/年）", x=0.5),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


def create_retrofit_investment_bar_figure(
    retrofit_items,
) -> go.Figure:
    if not retrofit_items:
        return go.Figure().add_annotation(
            text="无推荐改造项目",
            showarrow=False,
            font=dict(size=16),
        )

    names = [item.item_name for item in retrofit_items]
    investments = [item.investment for item in retrofit_items]
    savings = [item.annual_saving for item in retrofit_items]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=investments,
        name='投资额 (元)',
        marker_color='#ff6b6b',
        hovertemplate='<b>%{x}</b><br>投资额: %{y:,.0f} 元<extra></extra>',
    ))

    fig.add_trace(go.Bar(
        x=names,
        y=savings,
        name='年节省额 (元)',
        marker_color='#4ecdc4',
        hovertemplate='<b>%{x}</b><br>年节省: %{y:,.0f} 元<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text="改造项目投资额 vs 年节省额对比", x=0.5),
        xaxis=dict(title="改造项目", tickangle=-30, showgrid=True),
        yaxis=dict(title="金额 (元)", showgrid=True),
        barmode='group',
        height=450,
        margin=dict(l=60, r=40, t=60, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_sensitivity_analysis_figure(
    sensitivity_result,
    parameter_display_name: str,
    parameter_unit: str,
) -> go.Figure:
    if not sensitivity_result or not sensitivity_result.points:
        return go.Figure().add_annotation(
            text="暂无敏感性分析数据",
            showarrow=False,
            font=dict(size=16),
        )

    multipliers = [p.parameter_multiplier for p in sensitivity_result.points]
    parameter_values = [p.parameter_value for p in sensitivity_result.points]
    total_savings = [p.total_annual_saving for p in sensitivity_result.points]
    payback_periods = [p.overall_payback_period if p.overall_payback_period != float('inf') else None for p in sensitivity_result.points]

    x_labels = [f"{m:.1f}x\n({v:.2f}{unit})" for m, v, unit in zip(multipliers, parameter_values, [parameter_unit]*len(multipliers))]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=total_savings,
            mode='lines+markers',
            name='总年节省额 (元)',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>%{x}</b><br>总年节省额: %{y:,.0f} 元<extra></extra>',
        ),
        secondary_y=False,
    )

    valid_payback = []
    valid_x = []
    for x, pb in zip(x_labels, payback_periods):
        if pb is not None:
            valid_payback.append(pb)
            valid_x.append(x)

    fig.add_trace(
        go.Scatter(
            x=valid_x,
            y=valid_payback,
            mode='lines+markers',
            name='整体回报期 (年)',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=10, symbol='square'),
            hovertemplate='<b>%{x}</b><br>整体回报期: %{y:.2f} 年<extra></extra>',
        ),
        secondary_y=True,
    )

    base_mult = 1.0
    base_idx = min(range(len(multipliers)), key=lambda i: abs(multipliers[i] - base_mult))
    fig.add_vline(
        x=base_idx,
        line=dict(color='#ffa500', width=2, dash='dot'),
        annotation=dict(
            text=f"当前值 ({sensitivity_result.base_value:.2f} {parameter_unit})",
            font=dict(color='#ffa500', size=11),
            showarrow=True,
            arrowhead=1,
        )
    )

    fig.update_layout(
        title=dict(text=f"{parameter_display_name}敏感性分析", x=0.5),
        xaxis=dict(title=f"{parameter_display_name}变化倍率（括号内为实际价格）", showgrid=True),
        height=500,
        margin=dict(l=60, r=60, t=80, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="总年节省额 (元)", secondary_y=False, color='#4ecdc4')
    fig.update_yaxes(title_text="整体回报期 (年)", secondary_y=True, color='#ff6b6b')

    return fig


def create_cash_flow_figure(
    cash_flow_data,
    payback_year=None,
) -> go.Figure:
    if not cash_flow_data:
        return go.Figure().add_annotation(
            text="暂无现金流数据",
            showarrow=False,
            font=dict(size=16),
        )

    years = [cf.year for cf in cash_flow_data]
    cumulative_cash = [cf.cumulative_cash_flow for cf in cash_flow_data]
    investments = [cf.investment for cf in cash_flow_data]
    savings = [cf.saving for cf in cash_flow_data]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=cumulative_cash,
            mode='lines+markers',
            name='累计现金流',
            line=dict(color='#45b7d1', width=4),
            marker=dict(size=12, symbol='circle', line=dict(width=2, color='white')),
            hovertemplate='<b>第%{x}年</b><br>累计现金流: %{y:,.0f} 元<extra></extra>',
        )
    )

    fig.add_trace(
        go.Bar(
            x=years,
            y=[-inv for inv in investments],
            name='当年投资额',
            marker_color='#ff6b6b',
            opacity=0.6,
            hovertemplate='<b>第%{x}年</b><br>投资额: %{customdata:,.0f} 元<extra></extra>',
            customdata=investments,
        )
    )

    fig.add_trace(
        go.Bar(
            x=years,
            y=savings,
            name='当年节省额',
            marker_color='#4ecdc4',
            opacity=0.6,
            hovertemplate='<b>第%{x}年</b><br>节省额: %{y:,.0f} 元<extra></extra>',
        )
    )

    fig.add_hline(
        y=0,
        line=dict(color='#333', width=2, dash='solid'),
    )

    if payback_year is not None:
        payback_idx = years.index(payback_year) if payback_year in years else len(years) - 1
        payback_value = cumulative_cash[payback_idx] if payback_idx < len(cumulative_cash) else 0

        fig.add_vline(
            x=payback_year,
            line=dict(color='#ffa500', width=3, dash='dash'),
        )
        fig.add_annotation(
            x=payback_year,
            y=payback_value,
            text=f"🏆 投资回收期<br>第 {payback_year} 年",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor='#ffa500',
            font=dict(color='#ffa500', size=13, weight='bold'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ffa500',
            borderwidth=2,
            ax=80,
            ay=-60,
        )

    fig.update_layout(
        title=dict(text="分期改造累计现金流分析", x=0.5),
        xaxis=dict(title="年份", showgrid=True, tickmode='linear', dtick=1),
        yaxis=dict(title="累计现金流 (元)", showgrid=True),
        barmode='relative',
        height=550,
        margin=dict(l=60, r=40, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
