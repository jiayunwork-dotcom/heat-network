import io
from datetime import datetime
from typing import List, Dict, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

import numpy as np
from models import (
    PipeNetwork, NetworkResults, PipeSectionResult, NodeResult,
    NODE_TYPE_SOURCE, NODE_TYPE_HEAT_EXCHANGER, NODE_TYPE_END_USER, NODE_TYPE_BRANCH,
)


def generate_pdf_report(
    network: PipeNetwork,
    results: NetworkResults,
    flow_ratios: Optional[Dict[int, float]] = None,
    valve_suggestions: Optional[List[Dict]] = None,
    source_ratios: Optional[Dict[int, float]] = None,
    title: str = "城市集中供热管网运行分析报告",
) -> Optional[bytes]:
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=20, alignment=TA_CENTER,
        textColor=colors.darkblue,
    )
    h1_style = ParagraphStyle(
        'H1', parent=styles['Heading1'],
        fontSize=16, spaceBefore=15, spaceAfter=10,
        textColor=colors.darkblue, alignment=TA_LEFT,
        borderPadding=(0, 0, 5, 0),
        borderWidth=0, borderColor=colors.darkblue, borderPaddingBottom=5,
    )
    h2_style = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=13, spaceBefore=10, spaceAfter=8,
        textColor=colors.darkslategray,
    )
    normal_style = ParagraphStyle(
        'Normal', parent=styles['Normal'],
        fontSize=10, spaceAfter=6, alignment=TA_JUSTIFY, leading=15,
    )
    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 5 * mm))
    info_data = [
        ["报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["管网名称", f"{len(network.nodes)}节点 / {len(network.pipes)}管段"],
        ["环境温度", f"{network.environment_temperature:.1f} °C"],
        ["水的定压比热容", f"{network.water_specific_heat:.0f} J/(kg·°C)"],
        ["迭代收敛状态", f"{'已收敛' if results.converged else '未收敛'}（迭代{results.iterations}次）"],
    ]
    info_table = Table(info_data, colWidths=[4 * cm, 12 * cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("一、管网参数汇总", h1_style))
    node_types_count = {
        NODE_TYPE_SOURCE: len(network.get_nodes_by_type(NODE_TYPE_SOURCE)),
        NODE_TYPE_HEAT_EXCHANGER: len(network.get_nodes_by_type(NODE_TYPE_HEAT_EXCHANGER)),
        NODE_TYPE_BRANCH: len(network.get_nodes_by_type(NODE_TYPE_BRANCH)),
        NODE_TYPE_END_USER: len(network.get_nodes_by_type(NODE_TYPE_END_USER)),
    }
    type_labels = {
        NODE_TYPE_SOURCE: "热源", NODE_TYPE_HEAT_EXCHANGER: "换热站",
        NODE_TYPE_BRANCH: "分支点", NODE_TYPE_END_USER: "末端用户",
    }
    story.append(Paragraph("1.1 节点统计", h2_style))
    node_stats = [
        ["节点类型", "数量", "备注"],
    ]
    total_design_flow = 0.0
    for ntype, count in node_types_count.items():
        remark = ""
        if ntype == NODE_TYPE_SOURCE:
            sources = network.get_nodes_by_type(ntype)
            total_cap = sum((s.rated_capacity or 0) for s in sources) / 1e6
            remark = f"总额定容量: {total_cap:.1f} MW"
        elif ntype == NODE_TYPE_END_USER:
            users = network.get_nodes_by_type(ntype)
            total_design_flow = sum((u.design_flow or 0) for u in users)
            remark = f"总设计流量: {total_design_flow * 1000:.1f} L/s"
        node_stats.append([type_labels[ntype], str(count), remark])
    node_stats.append(["合计", str(len(network.nodes)), ""])
    story.append(_make_table(node_stats, [3 * cm, 3 * cm, 10 * cm]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("1.2 热源节点详情", h2_style))
    src_data = [["编号", "名称", "供水压力(mH₂O)", "标高(m)", "额定容量(MW)", "负荷分配比(%)"]]
    sources = network.get_nodes_by_type(NODE_TYPE_SOURCE)
    for s in sources:
        ratio = ((source_ratios or {}).get(s.id, 1.0 / max(len(sources), 1))) * 100
        src_data.append([
            str(s.id), s.name,
            f"{s.supply_pressure or 0:.2f}",
            f"{s.elevation:.1f}",
            f"{(s.rated_capacity or 0) / 1e6:.1f}",
            f"{ratio:.1f}",
        ])
    story.append(_make_table(src_data, [1.5 * cm, 3.5 * cm, 3 * cm, 2.5 * cm, 3 * cm, 2.5 * cm]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("1.3 管段统计", h2_style))
    pipes_list = list(network.pipes.values())
    total_len = sum(p.length for p in pipes_list)
    n_valve = sum(1 for p in pipes_list if p.has_valve)
    n_pump = sum(1 for p in pipes_list if p.has_pump)
    material_count = {}
    for p in pipes_list:
        material_count[p.material] = material_count.get(p.material, 0) + 1
    pipe_stats = [
        ["指标", "数值"],
        ["管段总数", str(len(pipes_list))],
        ["管网总长度", f"{total_len:.0f} m"],
        ["平均管段长度", f"{total_len / max(len(pipes_list), 1):.1f} m"],
        ["含阀门管段数", f"{n_valve}"],
        ["含泵站管段数", f"{n_pump}"],
        ["管径范围", f"{min(p.diameter for p in pipes_list) * 1000:.0f} ~ {max(p.diameter for p in pipes_list) * 1000:.0f} mm"],
        ["管材分布", ", ".join(f"{m}:{c}" for m, c in material_count.items())],
    ]
    story.append(_make_table(pipe_stats, [5 * cm, 11 * cm]))
    story.append(PageBreak())
    story.append(Paragraph("二、水力计算结果", h1_style))
    story.append(Paragraph("2.1 各管段水力参数", h2_style))
    hyd_header = ["编号", "管段名称", "管径(mm)", "流量(L/s)", "流速(m/s)", "雷诺数", "摩擦系数", "阻力损失(kPa)", "泵站扬程(m)"]
    hyd_data = [hyd_header]
    for pid in sorted(results.pipe_results.keys()):
        pr = results.pipe_results[pid]
        pipe = network.pipes[pid]
        hyd_data.append([
            str(pid),
            pipe.name,
            f"{pipe.diameter * 1000:.0f}",
            f"{pr.flow_rate * 1000:.1f}",
            f"{pr.velocity:.3f}",
            f"{pr.reynolds:.0f}",
            f"{pr.friction_factor:.5f}",
            f"{pr.total_pressure_loss / 1000:.2f}",
            f"{pr.pump_head:.2f}" if pr.pump_head > 0 else "-",
        ])
    hyd_cols = [1.2 * cm, 3.8 * cm, 1.8 * cm, 1.8 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 2 * cm, 1.8 * cm]
    story.append(_make_long_table(hyd_data, hyd_cols, h1_style, normal_style, story, "二、水力计算结果(续)"))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("2.2 各节点压力分布", h2_style))
    np_header = ["编号", "节点名称", "类型", "标高(m)", "压力(mH₂O)", "流入流量(L/s)", "流出流量(L/s)"]
    np_data = [np_header]
    type_ch = {NODE_TYPE_SOURCE: "热源", NODE_TYPE_HEAT_EXCHANGER: "换热站",
               NODE_TYPE_BRANCH: "分支", NODE_TYPE_END_USER: "用户"}
    for nid in sorted(results.node_results.keys()):
        nr = results.node_results[nid]
        node = network.nodes[nid]
        np_data.append([
            str(nid), node.name, type_ch.get(node.type, "-"),
            f"{node.elevation:.1f}",
            f"{nr.pressure:.3f}",
            f"{nr.flow_in * 1000:.1f}",
            f"{nr.flow_out * 1000:.1f}",
        ])
    story.append(_make_long_table(np_data, [1.2 * cm, 3.5 * cm, 1.5 * cm, 2 * cm, 2.3 * cm, 2.5 * cm, 2.5 * cm],
                                  h1_style, normal_style, story, "二、水力计算结果(续)"))
    story.append(PageBreak())
    story.append(Paragraph("三、热力计算结果", h1_style))
    story.append(Paragraph("3.1 各管段热力参数", h2_style))
    th_header = ["编号", "管段名称", "进口温度(°C)", "出口温度(°C)", "温降(°C)", "传热系数(W/m·K)", "热损失(kW)"]
    th_data = [th_header]
    for pid in sorted(results.pipe_results.keys()):
        pr = results.pipe_results[pid]
        pipe = network.pipes[pid]
        th_data.append([
            str(pid),
            pipe.name,
            f"{pr.inlet_temperature:.2f}",
            f"{pr.outlet_temperature:.2f}",
            f"{pr.temperature_drop:.3f}",
            f"{pr.heat_transfer_coefficient:.2f}",
            f"{pr.heat_loss / 1000:.2f}",
        ])
    story.append(_make_long_table(th_data, [1.2 * cm, 3.8 * cm, 2.2 * cm, 2.2 * cm, 1.8 * cm, 2.5 * cm, 2.5 * cm],
                                  h1_style, normal_style, story, "三、热力计算结果(续)"))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("3.2 各节点温度分布", h2_style))
    nt_header = ["编号", "节点名称", "类型", "温度(°C)"]
    nt_data = [nt_header]
    for nid in sorted(results.node_results.keys()):
        nr = results.node_results[nid]
        node = network.nodes[nid]
        nt_data.append([
            str(nid), node.name, type_ch.get(node.type, "-"),
            f"{nr.temperature:.2f}",
        ])
    story.append(_make_long_table(nt_data, [1.5 * cm, 4.5 * cm, 3 * cm, 3 * cm],
                                  h1_style, normal_style, story, "三、热力计算结果(续)"))
    story.append(PageBreak())
    story.append(Paragraph("四、能耗指标汇总", h1_style))
    heat_supplied_mw = results.total_heat_supplied / 1e6
    heat_loss_mw = results.total_heat_loss / 1e6
    heat_delivered_gj_h = (results.total_heat_supplied - results.total_heat_loss) / 1e9
    pump_kwh_per_gj = results.specific_energy_consumption
    total_pump_kw = results.total_pump_power
    energy_table = [
        ["指标名称", "数值", "单位"],
        ["总供热量", f"{heat_supplied_mw:.2f}", "MW (MJ/s)"],
        ["管网总热损失", f"{heat_loss_mw:.2f}", "MW"],
        ["有效供热功率", f"{heat_supplied_mw - heat_loss_mw:.2f}", "MW"],
        ["管网热损失率", f"{results.heat_loss_rate:.2f}", "%"],
        ["泵站总耗电功率", f"{total_pump_kw:.2f}", "kW"],
        ["供热单耗(输送电耗/供热量)", f"{pump_kwh_per_gj:.2f}", "kWh/GJ"],
        ["每小时泵耗电量", f"{total_pump_kw:.2f}", "kWh/h"],
        ["每日泵耗电量", f"{total_pump_kw * 24:.0f}", "kWh/天"],
    ]
    story.append(_make_table(energy_table, [6 * cm, 5 * cm, 5 * cm]))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("4.1 各泵站耗电明细", h2_style))
    pump_header = ["管段编号", "泵站名称", "额定扬程(m)", "工作流量(L/s)", "工作扬程(m)", "功率(kW)"]
    pump_data = [pump_header]
    for pid in sorted(results.pipe_results.keys()):
        pr = results.pipe_results[pid]
        pipe = network.pipes[pid]
        if pipe.has_pump:
            pump_data.append([
                str(pid), pipe.name,
                f"{pipe.rated_head or 0:.1f}",
                f"{pr.flow_rate * 1000:.1f}",
                f"{abs(pr.pump_head):.1f}",
                f"{pr.power_consumption:.2f}",
            ])
    if len(pump_data) > 1:
        story.append(_make_table(pump_data, [2 * cm, 3.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]))
    else:
        story.append(Paragraph("本管网不含泵站。", normal_style))
    story.append(PageBreak())
    story.append(Paragraph("五、水力平衡分析与优化建议", h1_style))
    story.append(Paragraph("5.1 末端用户流量不均匀度分析", h2_style))
    if flow_ratios:
        fr_header = ["用户编号", "用户名称", "设计流量(L/s)", "实际流量(L/s)", "流量不均匀度", "状态"]
        fr_data = [fr_header]
        for uid in sorted(flow_ratios.keys()):
            ratio = flow_ratios[uid]
            user = network.nodes[uid]
            nr = results.node_results.get(uid)
            actual = nr.flow_in if nr else 0
            status = "✓ 正常" if abs(ratio - 1.0) <= 0.15 else ("⚠ 偏大" if ratio > 1 else "⚠ 偏小")
            fr_data.append([
                str(uid), user.name,
                f"{(user.design_flow or 0) * 1000:.1f}",
                f"{actual * 1000:.1f}",
                f"{ratio * 100:.1f}%",
                status,
            ])
        story.append(_make_long_table(fr_data, [1.8 * cm, 4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm],
                                      h1_style, normal_style, story, "五、水力平衡分析(续)"))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("5.2 阀门调节建议", h2_style))
    if valve_suggestions:
        vs_header = ["用户", "管段", "动作", "调节幅度", "原因说明"]
        vs_data = [vs_header]
        for s in valve_suggestions:
            vs_data.append([
                s["user_name"],
                s["pipe_name"],
                s["action"],
                f"{s['adjustment']:.1f}%",
                s["reason"],
            ])
        story.append(_make_table(vs_data, [3.5 * cm, 3.5 * cm, 2 * cm, 2 * cm, 5 * cm]))
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(
            f"共发现 {len(valve_suggestions)} 个用户的流量偏差超过 ±15%，建议按照上表进行阀门调节，"
            f"调节过程中建议分多次逐步调整，每次调整后重新进行水力计算验证。", normal_style
        ))
    else:
        story.append(Paragraph("✓ 所有末端用户流量偏差均在 ±15% 以内，水力平衡状态良好，无需阀门调节。", normal_style))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("5.3 节能优化建议", h2_style))
    suggestions_text = []
    if results.heat_loss_rate > 5:
        worst_pipes = sorted(results.pipe_results.values(), key=lambda x: -x.heat_loss)[:5]
        worst_names = [network.pipes[p.pipe_id].name for p in worst_pipes]
        suggestions_text.append(
            f"<b>1. 保温改造建议：</b>管网热损失率为 {results.heat_loss_rate:.1f}%，高于5%的优良标准。"
            f"建议优先检查并升级以下高热损管段的保温层：{', '.join(worst_names)}。"
        )
    if total_pump_kw > 0:
        suggestions_text.append(
            f"<b>2. 泵站节能运行：</b>当前泵站总耗电 {total_pump_kw:.1f} kW，建议："
            f"(a) 分析泵工作点是否在高效区；"
            f"(b) 对长期低负荷运行的泵考虑变频调速；"
            f"(c) 多热源联供时采用最小能耗分配策略。"
        )
    if valve_suggestions:
        suggestions_text.append(
            f"<b>3. 水力平衡调节：</b>按 5.2 节建议进行阀门初调节后，可进一步采用全网平衡算法精细化调节。"
        )
    old_pipes = [p for p in network.pipes.values() if p.pipe_age > 25]
    if old_pipes:
        suggestions_text.append(
            f"<b>4. 管网更新建议：</b>共有 {len(old_pipes)} 条管段服役超过25年，建议按计划进行管网改造以降低结垢和泄漏风险。"
        )
    if not suggestions_text:
        suggestions_text.append("<b>✓ 当前管网各项指标运行良好，可继续保持现有运行策略。</b>")
    for idx, txt in enumerate(suggestions_text, 1):
        story.append(Paragraph(f"{idx}. {txt}" if idx > 1 else txt, normal_style))
        story.append(Spacer(1, 2 * mm))
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph("————————— 报告结束 —————————", ParagraphStyle(
        'End', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, textColor=colors.gray,
    )))
    doc.build(story)
    return buffer.getvalue()


def _make_table(data: List[List], col_widths: List[float]) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        bg = colors.whitesmoke if i % 2 == 0 else colors.white
        style_cmds.append(('BACKGROUND', (0, i), (-1, i), bg))
    table.setStyle(TableStyle(style_cmds))
    return table


def _make_long_table(
    data: List[List], col_widths: List[float],
    h1_style, normal_style, story, continue_title: str,
    max_rows_per_page: int = 22,
) -> Table:
    if len(data) <= max_rows_per_page + 1:
        return _make_table(data, col_widths)
    story.append(_make_table(data[:max_rows_per_page + 1], col_widths))
    remaining = [data[0]] + data[max_rows_per_page + 1:]
    page_idx = 1
    while len(remaining) > max_rows_per_page + 1:
        story.append(PageBreak())
        story.append(Paragraph(continue_title, h1_style))
        story.append(Paragraph(f"第 {page_idx} 部分", normal_style))
        story.append(_make_table(remaining[:max_rows_per_page + 1], col_widths))
        remaining = [data[0]] + remaining[max_rows_per_page + 1:]
        page_idx += 1
    if len(remaining) > 1:
        story.append(PageBreak())
        story.append(Paragraph(continue_title, h1_style))
        story.append(Paragraph(f"第 {page_idx} 部分", normal_style))
        story.append(_make_table(remaining, col_widths))
    return _make_table([data[0]], col_widths)
