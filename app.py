import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
import plotly.graph_objects as go

from models import (
    PipeNetwork, Node, Pipe, NetworkResults, PipeSectionResult,
    create_sample_network,
    NODE_TYPE_SOURCE, NODE_TYPE_HEAT_EXCHANGER, NODE_TYPE_BRANCH, NODE_TYPE_END_USER,
    PIPE_MATERIAL_STEEL, PIPE_MATERIAL_CAST_IRON, PIPE_MATERIAL_PVC,
    INSULATION_MATERIAL_NONE, INSULATION_MATERIAL_POLYURETHANE,
    INSULATION_MATERIAL_ROCK_WOOL, INSULATION_MATERIAL_GLASS_WOOL,
)
from calculations import (
    solve_coupled, analyze_hydraulic_balance,
    optimize_source_allocation_equal, optimize_source_allocation_min_energy,
    FaultConfig, ImpactAssessment, RiskAssessmentItem,
    FAULT_TYPE_PIPE_BURST, FAULT_TYPE_PUMP_FAILURE, FAULT_TYPE_SOURCE_SHUTDOWN,
    FAULT_TYPE_CN, MIN_OPERATING_PRESSURE,
    simulate_faults,
    get_available_pump_pipes, get_available_source_nodes, get_available_pipes,
    EmergencyAction, EmergencyPlan, RecoveryEffect,
    generate_emergency_plan, DEFAULT_TEMP_WARNING_THRESHOLD,
    calculate_risk_assessment, execute_emergency_action,
)
from visualization import (
    create_network_topology_figure,
    create_temperature_drop_figure,
    create_pressure_contour_figure,
    create_pump_operating_point_figures,
    create_heat_loss_pareto_figure,
    create_energy_consumption_pie_figure,
    create_fault_topology_figure,
)
from report import generate_pdf_report

st.set_page_config(
    page_title="城市集中供热管网水力热力耦合计算系统",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

NODE_TYPE_CN = {
    NODE_TYPE_SOURCE: "热源",
    NODE_TYPE_HEAT_EXCHANGER: "换热站",
    NODE_TYPE_BRANCH: "分支点",
    NODE_TYPE_END_USER: "末端用户",
}
PIPE_MATERIAL_CN = {
    PIPE_MATERIAL_STEEL: "钢管",
    PIPE_MATERIAL_CAST_IRON: "铸铁管",
    PIPE_MATERIAL_PVC: "PVC管",
}
INSULATION_CN = {
    INSULATION_MATERIAL_NONE: "无保温",
    INSULATION_MATERIAL_POLYURETHANE: "聚氨酯泡沫",
    INSULATION_MATERIAL_ROCK_WOOL: "岩棉",
    INSULATION_MATERIAL_GLASS_WOOL: "玻璃棉",
}

CN_TO_NODE_TYPE = {v: k for k, v in NODE_TYPE_CN.items()}
CN_TO_PIPE_MATERIAL = {v: k for k, v in PIPE_MATERIAL_CN.items()}
CN_TO_INSULATION = {v: k for k, v in INSULATION_CN.items()}


def init_session():
    if "network" not in st.session_state:
        st.session_state.network = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "flow_ratios" not in st.session_state:
        st.session_state.flow_ratios = None
    if "valve_suggestions" not in st.session_state:
        st.session_state.valve_suggestions = None
    if "source_ratios" not in st.session_state:
        st.session_state.source_ratios = None
    if "source_temps" not in st.session_state:
        st.session_state.source_temps = {}
    if "saved_conditions" not in st.session_state:
        st.session_state.saved_conditions = {}
    if "temp_threshold" not in st.session_state:
        st.session_state.temp_threshold = 65.0
    if "fault_list" not in st.session_state:
        st.session_state.fault_list = []
    if "fault_results" not in st.session_state:
        st.session_state.fault_results = None
    if "fault_impact" not in st.session_state:
        st.session_state.fault_impact = None
    if "fault_modified_network" not in st.session_state:
        st.session_state.fault_modified_network = None
    if "emergency_plan" not in st.session_state:
        st.session_state.emergency_plan = None
    if "fault_history" not in st.session_state:
        st.session_state.fault_history = []
    if "recovery_effects" not in st.session_state:
        st.session_state.recovery_effects = {}
    if "recovery_current_results" not in st.session_state:
        st.session_state.recovery_current_results = None
    if "recovery_current_impact" not in st.session_state:
        st.session_state.recovery_current_impact = None
    if "recovery_current_network" not in st.session_state:
        st.session_state.recovery_current_network = None
    if "recovery_current_source_temps" not in st.session_state:
        st.session_state.recovery_current_source_temps = None
    if "risk_assessment_results" not in st.session_state:
        st.session_state.risk_assessment_results = None
    if "risk_mode_enabled" not in st.session_state:
        st.session_state.risk_mode_enabled = False


init_session()

st.title("🔥 城市集中供热管网水力热力耦合计算与能耗分析系统")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 系统设置")
    st.subheader("1. 管网数据导入")
    data_source = st.radio(
        "选择数据来源",
        ["使用示例管网 (30节点/45管段)", "上传JSON文件", "手动编辑表格"],
        index=0,
    )
    if data_source == "使用示例管网 (30节点/45管段)":
        if st.button("🚀 加载示例管网", use_container_width=True):
            with st.spinner("正在加载示例管网..."):
                st.session_state.network = create_sample_network()
                src_nodes = st.session_state.network.get_nodes_by_type(NODE_TYPE_SOURCE)
                st.session_state.source_temps = {
                    s.id: (110.0 if i == 0 else 108.0) for i, s in enumerate(src_nodes)
                }
                st.session_state.results = None
                st.success(f"✅ 已加载示例管网：{len(st.session_state.network.nodes)} 节点 / {len(st.session_state.network.pipes)} 管段")
    elif data_source == "上传JSON文件":
        uploaded = st.file_uploader("上传管网JSON文件", type=["json"])
        if uploaded and st.button("📥 解析JSON", use_container_width=True):
            try:
                data = json.load(uploaded)
                net = PipeNetwork()
                net.environment_temperature = data.get("environment_temperature", 5.0)
                net.water_specific_heat = data.get("water_specific_heat", 4186.0)
                for nd in data.get("nodes", []):
                    ntype = nd.get("type", NODE_TYPE_BRANCH)
                    if ntype in CN_TO_NODE_TYPE:
                        ntype = CN_TO_NODE_TYPE[ntype]
                    node = Node(
                        id=int(nd["id"]), name=nd.get("name", f"节点{nd['id']}"),
                        type=ntype, elevation=float(nd.get("elevation", 0.0)),
                        x=nd.get("x"), y=nd.get("y"),
                        supply_pressure=nd.get("supply_pressure"),
                        return_pressure=nd.get("return_pressure"),
                        design_flow=nd.get("design_flow"),
                        rated_capacity=nd.get("rated_capacity"),
                    )
                    net.add_node(node)
                for pd_ in data.get("pipes", []):
                    mat = pd_.get("material", PIPE_MATERIAL_STEEL)
                    if mat in CN_TO_PIPE_MATERIAL:
                        mat = CN_TO_PIPE_MATERIAL[mat]
                    ins = pd_.get("insulation_material", INSULATION_MATERIAL_POLYURETHANE)
                    if ins in CN_TO_INSULATION:
                        ins = CN_TO_INSULATION[ins]
                    pump_curve = None
                    if pd_.get("pump_efficiency_curve"):
                        pump_curve = [(float(p[0]), float(p[1])) for p in pd_["pump_efficiency_curve"]]
                    pipe = Pipe(
                        id=int(pd_["id"]), name=pd_.get("name", f"管段{pd_['id']}"),
                        start_node_id=int(pd_["start_node_id"]),
                        end_node_id=int(pd_["end_node_id"]),
                        diameter=float(pd_.get("diameter", 0.2)),
                        length=float(pd_.get("length", 100.0)),
                        material=mat,
                        pipe_age=float(pd_.get("pipe_age", 10.0)),
                        insulation_material=ins,
                        insulation_thickness=float(pd_.get("insulation_thickness", 0.05)),
                        burial_depth=float(pd_.get("burial_depth", 1.5)),
                        has_valve=bool(pd_.get("has_valve", False)),
                        valve_kv=pd_.get("valve_kv"),
                        has_pump=bool(pd_.get("has_pump", False)),
                        rated_head=pd_.get("rated_head"),
                        pump_efficiency_curve=pump_curve,
                        equivalent_local_length_ratio=float(pd_.get("equivalent_local_length_ratio", 0.1)),
                    )
                    net.add_pipe(pipe)
                valid, errors = net.validate()
                if not valid:
                    st.error("❌ 管网数据校验失败：")
                    for e in errors:
                        st.write(f"  • {e}")
                else:
                    st.session_state.network = net
                    src_nodes = net.get_nodes_by_type(NODE_TYPE_SOURCE)
                    st.session_state.source_temps = {
                        s.id: (110.0 if i == 0 else 108.0) for i, s in enumerate(src_nodes)
                    }
                    st.session_state.results = None
                    st.success(f"✅ 已成功加载：{len(net.nodes)} 节点 / {len(net.pipes)} 管段")
            except Exception as e:
                st.error(f"❌ JSON解析失败：{e}")
    elif data_source == "手动编辑表格":
        st.info("💡 可在下方'管网数据编辑'页签手动创建/修改节点和管段")
    st.divider()
    st.subheader("2. 环境与物性参数")
    env_temp = st.slider("环境/土壤温度 (°C)", -20.0, 25.0, 5.0, 0.5)
    cp_water = st.number_input("水的定压比热容 J/(kg·°C)", 3800.0, 4500.0, 4186.0, 10.0)
    st.divider()
    st.subheader("3. 热源调度策略")
    dispatch_strategy = st.radio(
        "多热源负荷分配方式",
        ["等比例分配（按额定容量）", "最小能耗分配（优化求解）"],
        index=0,
        help="当有多个热源时生效",
    )
    balance_tolerance = st.slider("水力平衡容差 (%)", 5, 30, 15, 1) / 100.0
    st.divider()
    st.subheader("4. 执行计算")
    run_btn = st.button("▶️ 开始水力热力耦合计算", type="primary", use_container_width=True)
    st.divider()
    st.subheader("5. 工况管理")
    cond_count = len(st.session_state.saved_conditions)
    st.caption(f"已保存工况：{cond_count}/5")
    with st.form("save_condition_form", clear_on_submit=True):
        cond_name = st.text_input("工况名称", placeholder="例如：设计工况、极寒工况、检修工况")
        save_cond = st.form_submit_button("💾 保存当前工况", use_container_width=True)
    if save_cond:
        if not cond_name.strip():
            st.error("❌ 请输入工况名称")
        elif cond_name in st.session_state.saved_conditions:
            st.error(f"❌ 工况名称 '{cond_name}' 已存在")
        elif cond_count >= 5:
            st.error("❌ 最多只能保存5个工况，请先删除不需要的工况")
        elif st.session_state.results is None:
            st.error("❌ 请先运行计算后再保存工况")
        else:
            net = st.session_state.network
            r = st.session_state.results
            user_flow_demands = {}
            for uid, user in enumerate(net.get_nodes_by_type(NODE_TYPE_END_USER), 1):
                user_flow_demands[user.id] = user.design_flow or 0.0
            condition_data = {
                "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_params": {
                    "environment_temperature": net.environment_temperature,
                    "water_specific_heat": net.water_specific_heat,
                    "source_temps": dict(st.session_state.source_temps),
                    "dispatch_strategy": dispatch_strategy,
                    "balance_tolerance": balance_tolerance,
                    "user_flow_demands": user_flow_demands,
                },
                "results_summary": {
                    "total_heat_supplied": r.total_heat_supplied,
                    "total_heat_loss": r.total_heat_loss,
                    "heat_loss_rate": r.heat_loss_rate,
                    "total_pump_power": r.total_pump_power,
                    "specific_energy_consumption": r.specific_energy_consumption,
                    "iterations": r.iterations,
                    "converged": r.converged,
                },
                "source_ratios": dict(st.session_state.source_ratios) if st.session_state.source_ratios else {},
                "pipe_results": {str(pid): {
                    "flow_rate": pr.flow_rate,
                    "inlet_temperature": pr.inlet_temperature,
                    "outlet_temperature": pr.outlet_temperature,
                    "temperature_drop": pr.temperature_drop,
                    "heat_loss": pr.heat_loss,
                    "pressure_loss": pr.pressure_loss,
                    "total_pressure_loss": pr.total_pressure_loss,
                    "power_consumption": pr.power_consumption,
                    "pump_head": pr.pump_head,
                } for pid, pr in r.pipe_results.items()},
                "node_results": {str(nid): {
                    "pressure": nr.pressure,
                    "temperature": nr.temperature,
                    "flow_in": nr.flow_in,
                    "flow_out": nr.flow_out,
                } for nid, nr in r.node_results.items()},
            }
            st.session_state.saved_conditions[cond_name.strip()] = condition_data
            st.success(f"✅ 工况 '{cond_name.strip()}' 保存成功！")
    st.divider()
    if st.session_state.saved_conditions:
        st.markdown("###### 已保存工况列表")
        for cname in list(st.session_state.saved_conditions.keys()):
            cdata = st.session_state.saved_conditions[cname]
            with st.expander(f"📋 {cname}"):
                st.caption(f"保存时间: {cdata['saved_time']}")
                ip = cdata["input_params"]
                rs = cdata["results_summary"]
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.caption("**输入参数**")
                    st.write(f"环境温度: {ip['environment_temperature']}°C")
                    st.write(f"热源数: {len(ip['source_temps'])}")
                    for si, (sid, stv) in enumerate(ip["source_temps"].items()):
                        st.write(f"  热源{sid}: {stv}°C")
                with col_c2:
                    st.caption("**关键指标**")
                    st.write(f"总供热量: {rs['total_heat_supplied']/1e6:.2f} MW")
                    st.write(f"热损失率: {rs['heat_loss_rate']:.2f}%")
                    st.write(f"泵耗电: {rs['total_pump_power']:.1f} kW")
                if st.button(f"🗑️ 删除 '{cname}'", key=f"del_cond_{cname}", use_container_width=True):
                    del st.session_state.saved_conditions[cname]
                    st.success(f"✅ 工况 '{cname}' 已删除")
                    st.rerun()
    else:
        st.info("💡 运行计算后可在此保存工况，最多保存5个")

if run_btn and st.session_state.network is not None:
    st.session_state.network.environment_temperature = env_temp
    st.session_state.network.water_specific_heat = cp_water
    with st.spinner("🔄 正在进行水力热力耦合迭代计算..."):
        try:
            if dispatch_strategy.startswith("等比例"):
                ratios = optimize_source_allocation_equal(st.session_state.network)
                st.session_state.source_ratios = ratios
                results = solve_coupled(st.session_state.network, st.session_state.source_temps)
            else:
                ratios, results = optimize_source_allocation_min_energy(
                    st.session_state.network, st.session_state.source_temps
                )
                st.session_state.source_ratios = ratios
            st.session_state.results = results
            flow_ratios, suggestions = analyze_hydraulic_balance(
                st.session_state.network, results, tolerance=balance_tolerance
            )
            st.session_state.flow_ratios = flow_ratios
            st.session_state.valve_suggestions = suggestions
            st.success(
                f"✅ 计算完成！{'已收敛' if results.converged else '⚠️ 未完全收敛'} "
                f"(迭代{results.iterations}次)"
            )
        except Exception as e:
            st.error(f"❌ 计算失败：{e}")
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())

tabs = st.tabs([
    "📊 总览仪表盘", "🌐 管网拓扑与可视化", "💧 水力计算结果",
    "🌡️ 热力计算结果", "⚡ 能耗分析", "🔧 水力平衡与优化",
    "📈 工况对比", "🚨 故障模拟与应急预案", "📝 管网数据编辑", "📄 报告导出"
])

with tabs[0]:
    st.subheader("📊 运行总览仪表盘")
    if st.session_state.results is None:
        st.info("👈 请先加载示例管网或导入数据，然后点击侧边栏的计算按钮")
    else:
        r: NetworkResults = st.session_state.results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总供热量", f"{r.total_heat_supplied/1e6:.2f} MW")
        with col2:
            st.metric("管网总热损失", f"{r.total_heat_loss/1e6:.2f} MW",
                      delta=f"{r.heat_loss_rate:.1f}%")
        with col3:
            st.metric("泵站总耗电", f"{r.total_pump_power:.1f} kW")
        with col4:
            st.metric("供热单耗", f"{r.specific_energy_consumption:.2f} kWh/GJ",
                      delta="输送电耗/供热量", delta_color="off")
        st.divider()
        col5, col6, col7 = st.columns(3)
        net = st.session_state.network
        with col5:
            st.metric("节点总数", len(net.nodes))
            src_c = len(net.get_nodes_by_type(NODE_TYPE_SOURCE))
            usr_c = len(net.get_nodes_by_type(NODE_TYPE_END_USER))
            st.caption(f"其中热源 {src_c} 个，末端用户 {usr_c} 个")
        with col6:
            st.metric("管段总数", len(net.pipes))
            pump_c = sum(1 for p in net.pipes.values() if p.has_pump)
            valve_c = sum(1 for p in net.pipes.values() if p.has_valve)
            st.caption(f"其中泵站 {pump_c} 个，阀门 {valve_c} 个")
        with col7:
            total_len = sum(p.length for p in net.pipes.values())
            st.metric("管网总长度", f"{total_len:.0f} m")
            avg_age = sum(p.pipe_age for p in net.pipes.values()) / len(net.pipes)
            st.caption(f"平均管龄 {avg_age:.1f} 年")
        st.divider()
        if st.session_state.source_ratios:
            st.markdown("##### 🔋 热源负荷分配")
            ratio_df = []
            for sid, ratio in st.session_state.source_ratios.items():
                sn = net.nodes[sid]
                ratio_df.append({
                    "热源编号": sid, "热源名称": sn.name,
                    "额定容量(MW)": (sn.rated_capacity or 0) / 1e6,
                    "分配比例": f"{ratio * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(ratio_df), use_container_width=True, hide_index=True)
        st.divider()
        status_color = "🟢" if r.converged else "🟡"
        st.markdown(f"##### {status_color} 计算状态")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"耦合迭代次数：{r.iterations}")
        with c2:
            if r.converged:
                st.success("收敛判据满足：温度变化 < 0.01°C")
            else:
                st.warning("未完全收敛，可尝试增加迭代次数或检查边界条件")
        with c3:
            env_t = net.environment_temperature
            min_t = min(nr.temperature for nr in r.node_results.values())
            max_t = max(nr.temperature for nr in r.node_results.values())
            st.info(f"温度范围：{min_t:.1f} ~ {max_t:.1f}°C (环境 {env_t}°C)")

with tabs[1]:
    st.subheader("🌐 管网拓扑可视化")
    if st.session_state.results is None:
        st.info("请先运行计算")
    else:
        net = st.session_state.network
        r = st.session_state.results
        col_opt1, col_opt2 = st.columns([1, 3])
        with col_opt1:
            color_by = st.selectbox("管段着色方式", ["按温度", "按压力", "按流量"], index=0)
        color_map = {"按温度": "temperature", "按压力": "pressure", "按流量": "flow_rate"}
        with st.spinner("绘制拓扑图..."):
            fig_top = create_network_topology_figure(net, r, color_by=color_map[color_by])
            st.plotly_chart(fig_top, use_container_width=True)
        st.divider()
        fig_drop = create_temperature_drop_figure(net, r)
        st.plotly_chart(fig_drop, use_container_width=True)
        st.divider()
        with st.spinner("绘制压力等值线..."):
            fig_contour = create_pressure_contour_figure(net, r)
            if fig_contour is not None:
                st.plotly_chart(fig_contour, use_container_width=True)
            else:
                st.info("节点缺少空间坐标，跳过压力等值线绘制")
        st.divider()
        pump_figs = create_pump_operating_point_figures(net, r)
        if pump_figs:
            st.markdown("##### ⛽ 泵站工作点分析")
            pump_cols = st.columns(min(2, len(pump_figs)))
            for idx, (pid, fig) in enumerate(pump_figs.items()):
                with pump_cols[idx % len(pump_cols)]:
                    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("💧 水力计算详细结果")
    if st.session_state.results is None:
        st.info("请先运行计算")
    else:
        net = st.session_state.network
        r = st.session_state.results
        pipe_rows = []
        for pid, pr in r.pipe_results.items():
            pipe = net.pipes[pid]
            pipe_rows.append({
                "管段编号": pid,
                "管段名称": pipe.name,
                "管径(mm)": f"{pipe.diameter * 1000:.0f}",
                "长度(m)": f"{pipe.length:.0f}",
                "流量(L/s)": f"{pr.flow_rate * 1000:.2f}",
                "流速(m/s)": f"{pr.velocity:.3f}",
                "雷诺数": f"{pr.reynolds:.0f}",
                "摩擦系数": f"{pr.friction_factor:.5f}",
                "沿程阻力(kPa)": f"{pr.pressure_loss / 1000:.2f}",
                "局部阻力(kPa)": f"{pr.local_pressure_loss / 1000:.2f}",
                "总阻力(kPa)": f"{pr.total_pressure_loss / 1000:.2f}",
                "泵站扬程(m)": f"{pr.pump_head:.2f}" if pr.pump_head > 0 else "-",
            })
        st.markdown("##### 📋 各管段水力参数")
        df_pipes = pd.DataFrame(pipe_rows)
        st.dataframe(df_pipes, use_container_width=True, hide_index=True, height=480)
        st.divider()
        node_rows = []
        for nid, nr in r.node_results.items():
            node = net.nodes[nid]
            node_rows.append({
                "节点编号": nid,
                "节点名称": node.name,
                "类型": NODE_TYPE_CN.get(node.type, node.type),
                "标高(m)": f"{node.elevation:.1f}",
                "压力(mH₂O)": f"{nr.pressure:.3f}",
                "流入流量(L/s)": f"{nr.flow_in * 1000:.1f}",
                "流出流量(L/s)": f"{nr.flow_out * 1000:.1f}",
            })
        st.markdown("##### 📍 各节点水力参数")
        df_nodes = pd.DataFrame(node_rows)
        st.dataframe(df_nodes, use_container_width=True, hide_index=True, height=480)
        csv_pipes = df_pipes.to_csv(index=False).encode('utf-8-sig')
        csv_nodes = df_nodes.to_csv(index=False).encode('utf-8-sig')
        cc1, cc2 = st.columns(2)
        with cc1:
            st.download_button("📥 下载管段水力结果(CSV)", csv_pipes,
                               file_name="水力计算_管段.csv", mime="text/csv")
        with cc2:
            st.download_button("📥 下载节点水力结果(CSV)", csv_nodes,
                               file_name="水力计算_节点.csv", mime="text/csv")

with tabs[3]:
    st.subheader("🌡️ 热力计算详细结果")
    if st.session_state.results is None:
        st.info("请先运行计算")
    else:
        net = st.session_state.network
        r = st.session_state.results
        temp_threshold = st.number_input(
            "⚠️ 温度预警阈值 (°C)",
            min_value=30.0, max_value=100.0,
            value=st.session_state.temp_threshold,
            step=1.0,
            help="末端用户供水温度低于此值时触发告警",
        )
        st.session_state.temp_threshold = float(temp_threshold)

        end_user_nodes = net.get_nodes_by_type(NODE_TYPE_END_USER)
        low_temp_users = []
        for user in end_user_nodes:
            nr = r.node_results.get(user.id)
            if nr and nr.temperature < temp_threshold:
                low_temp_users.append({
                    "id": user.id,
                    "name": user.name,
                    "temperature": nr.temperature,
                })

        if low_temp_users:
            user_names = "、".join([f"{u['name']}(供水温度{u['temperature']:.1f}°C)" for u in low_temp_users])
            st.error(
                f"🚨 温度预警：共有 {len(low_temp_users)} 个末端用户供水温度低于 {temp_threshold}°C 阈值！\n\n"
                f"供温不达标用户：{user_names}",
                icon="⚠️"
            )

        th_rows = []
        total_hl = 0.0
        for pid, pr in r.pipe_results.items():
            pipe = net.pipes[pid]
            hl_kw = pr.heat_loss / 1000.0
            total_hl += hl_kw
            th_rows.append({
                "管段编号": pid,
                "管段名称": pipe.name,
                "保温材料": INSULATION_CN.get(pipe.insulation_material, pipe.insulation_material),
                "保温厚度(mm)": f"{pipe.insulation_thickness * 1000:.0f}",
                "进口温度(°C)": f"{pr.inlet_temperature:.2f}",
                "出口温度(°C)": f"{pr.outlet_temperature:.2f}",
                "温降(°C)": f"{pr.temperature_drop:.3f}",
                "传热系数(W/m·K)": f"{pr.heat_transfer_coefficient:.3f}",
                "热损失(kW)": f"{hl_kw:.2f}",
                "占比(%)": f"{hl_kw / max(total_hl, 1e-6) * 100:.1f}" if pid == list(r.pipe_results.keys())[-1] else "-",
            })
        if th_rows:
            th_rows[-1]["占比(%)"] = "100.0"
        st.markdown("##### 🔥 各管段热力参数与热损失")
        df_th = pd.DataFrame(th_rows)
        st.dataframe(df_th, use_container_width=True, hide_index=True, height=440)
        st.divider()
        tn_rows_data = []
        low_temp_ids = set(u["id"] for u in low_temp_users)
        for nid, nr in r.node_results.items():
            node = net.nodes[nid]
            tn_rows_data.append({
                "节点编号": nid,
                "节点名称": node.name,
                "类型": NODE_TYPE_CN.get(node.type, node.type),
                "温度(°C)": nr.temperature,
                "温差(与环境)": nr.temperature - net.environment_temperature,
                "_is_low_temp": (node.type == NODE_TYPE_END_USER and nid in low_temp_ids),
            })
        st.markdown("##### 🌡️ 各节点温度分布")

        def highlight_temp(row):
            if row["_is_low_temp"]:
                return ["background-color: #ffcccc; color: #cc0000; font-weight: bold" for _ in row]
            return ["" for _ in row]

        df_tn_display = pd.DataFrame([{
            "节点编号": r["节点编号"],
            "节点名称": r["节点名称"],
            "类型": r["类型"],
            "温度(°C)": f"{r['温度(°C)']:.2f}",
            "温差(与环境)": f"{r['温差(与环境)']:.1f}°C",
        } for r in tn_rows_data])

        df_tn_style = pd.DataFrame(tn_rows_data)

        def apply_temp_style(x):
            styles = pd.DataFrame("", index=x.index, columns=df_tn_display.columns)
            for idx, rdata in enumerate(tn_rows_data):
                if rdata["_is_low_temp"]:
                    styles.loc[idx, :] = "background-color: #ffcccc; color: #cc0000; font-weight: bold"
            return styles

        styled_tn = df_tn_display.style.apply(apply_temp_style, axis=None)
        st.dataframe(styled_tn, use_container_width=True, hide_index=True, height=380)

        if low_temp_users:
            st.info(f"💡 提示：红色标红行表示末端用户供水温度低于 {temp_threshold}°C 的预警阈值")

        csv_th = df_th.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载热力计算结果(CSV)", csv_th,
                           file_name="热力计算结果.csv", mime="text/csv")

with tabs[4]:
    st.subheader("⚡ 能耗分析与可视化")
    if st.session_state.results is None:
        st.info("请先运行计算")
    else:
        net = st.session_state.network
        r = st.session_state.results
        st.markdown("##### 📊 能耗指标汇总")
        ec_rows = [
            ["总供热量", f"{r.total_heat_supplied / 1e6:.2f}", "MW (MJ/s)"],
            ["管网总热损失", f"{r.total_heat_loss / 1e6:.2f}", "MW"],
            ["有效供热功率", f"{(r.total_heat_supplied - r.total_heat_loss) / 1e6:.2f}", "MW"],
            ["管网热损失率", f"{r.heat_loss_rate:.2f}", "%"],
            ["泵站总耗电功率", f"{r.total_pump_power:.2f}", "kW"],
            ["每小时泵耗电量", f"{r.total_pump_power:.2f}", "kWh/h"],
            ["每日泵耗电量", f"{r.total_pump_power * 24:.0f}", "kWh/天"],
            ["供热单耗(输送电耗/供热量)", f"{r.specific_energy_consumption:.2f}", "kWh/GJ"],
        ]
        st.table(pd.DataFrame(ec_rows, columns=["指标", "数值", "单位"]))
        st.divider()
        with st.spinner("绘制热损失Pareto图..."):
            fig_pareto = create_heat_loss_pareto_figure(net, r)
            st.plotly_chart(fig_pareto, use_container_width=True)
        st.divider()
        with st.spinner("绘制能耗饼图..."):
            fig_pie = create_energy_consumption_pie_figure(net, r)
            st.plotly_chart(fig_pie, use_container_width=True)
        st.divider()
        pump_rows = []
        for pid, pr in r.pipe_results.items():
            pipe = net.pipes[pid]
            if pipe.has_pump:
                daily_kwh = pr.power_consumption * 24
                monthly_yuan = daily_kwh * 30 * 0.7
                pump_rows.append({
                    "泵站编号": pid,
                    "所在管段": pipe.name,
                    "额定扬程(m)": pipe.rated_head or 0,
                    "工作流量(L/s)": f"{pr.flow_rate * 1000:.1f}",
                    "工作扬程(m)": f"{abs(pr.pump_head):.1f}",
                    "功率(kW)": f"{pr.power_consumption:.2f}",
                    "日耗电量(kWh)": f"{daily_kwh:.1f}",
                    "月电费估算(元)": f"{monthly_yuan:.0f}",
                })
        if pump_rows:
            st.markdown("##### ⛽ 各泵站能耗明细（电价按 0.7 元/kWh 估算）")
            st.dataframe(pd.DataFrame(pump_rows), use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("🔧 水力平衡分析与阀门调节建议")
    if st.session_state.results is None:
        st.info("请先运行计算")
    else:
        net = st.session_state.network
        r = st.session_state.results
        if st.session_state.flow_ratios:
            fr_rows = []
            n_ok = 0
            n_bad = 0
            for uid, ratio in st.session_state.flow_ratios.items():
                user = net.nodes[uid]
                nr = r.node_results.get(uid)
                actual = nr.flow_in if nr else 0
                design = user.design_flow or 0
                diff_pct = (ratio - 1.0) * 100
                if abs(diff_pct) <= balance_tolerance * 100:
                    status = "✅ 正常"
                    n_ok += 1
                elif diff_pct > 0:
                    status = "⚠️ 流量偏大"
                    n_bad += 1
                else:
                    status = "⚠️ 流量偏小"
                    n_bad += 1
                fr_rows.append({
                    "用户编号": uid,
                    "用户名称": user.name,
                    "设计流量(L/s)": f"{design * 1000:.1f}",
                    "实际流量(L/s)": f"{actual * 1000:.1f}",
                    "偏差(%)": f"{diff_pct:+.1f}",
                    "状态": status,
                })
            c_ok, c_bad, c_tot = st.columns(3)
            with c_ok:
                st.metric("✓ 平衡正常", n_ok, f"占比 {n_ok/max(n_ok+n_bad,1)*100:.0f}%")
            with c_bad:
                st.metric("⚠ 需调节", n_bad, delta_color="inverse")
            with c_tot:
                avg_dev = np.mean([abs(r - 1) for r in st.session_state.flow_ratios.values()]) * 100
                st.metric("平均偏差", f"{avg_dev:.1f}%")
            st.markdown("##### 📋 各末端用户流量不均匀度")
            st.dataframe(pd.DataFrame(fr_rows), use_container_width=True, hide_index=True, height=400)
        st.divider()
        st.markdown("##### 🎛️ 阀门调节建议")
        if st.session_state.valve_suggestions:
            vs_rows = []
            for s in st.session_state.valve_suggestions:
                vs_rows.append({
                    "用户": s["user_name"],
                    "所在管段": s["pipe_name"],
                    "建议动作": s["action"],
                    "调节幅度": f"{s['adjustment']:.1f}%",
                    "原因": s["reason"],
                })
            st.warning(f"⚠️ 共有 {len(vs_rows)} 个用户流量偏差超过 ±{balance_tolerance*100:.0f}%，建议按以下方案调节阀门：")
            st.dataframe(pd.DataFrame(vs_rows), use_container_width=True, hide_index=True)
            st.markdown("""
            > 💡 **操作建议**：
            > 1. 阀门调节应分步进行，每次调节 5-10%，间隔 15-30 分钟观察压力变化；
            > 2. 优先调节流量偏差较大的用户（>30%），再处理较小偏差；
            > 3. 调节完成后建议重新运行计算进行验证；
            > 4. 多热源联供系统需配合热源调度策略同步调整。
            """)
        else:
            st.success("✅ 所有末端用户流量偏差均在允许范围内，水力平衡状态良好，无需阀门调节。")

with tabs[6]:
    st.subheader("📈 工况对比分析")
    if not st.session_state.saved_conditions:
        st.info("💡 请先在侧边栏保存至少2个工况后再进行对比分析")
    else:
        cond_names = list(st.session_state.saved_conditions.keys())
        default_sel = cond_names[:min(2, len(cond_names))] if len(cond_names) >= 2 else cond_names
        selected_conds = st.multiselect(
            "选择需要对比的工况（2-3个）",
            cond_names,
            default=default_sel,
            max_selections=3,
        )
        if len(selected_conds) < 2:
            st.warning("⚠️ 请至少选择2个工况进行对比")
        else:
            def _calc_extra_metrics(cdata, net_ref=None):
                rs = cdata["results_summary"]
                node_res = cdata["node_results"]
                pipe_res = cdata["pipe_results"]
                temp_drops = [v["temperature_drop"] for v in pipe_res.values()]
                max_temp_drop = max(temp_drops) if temp_drops else 0.0
                end_user_pressures = []
                if net_ref is not None:
                    for n in net_ref.get_nodes_by_type(NODE_TYPE_END_USER):
                        nr = node_res.get(str(n.id))
                        if nr:
                            end_user_pressures.append(nr["pressure"])
                else:
                    for k, v in node_res.items():
                        end_user_pressures.append(v["pressure"])
                min_end_pressure = min(end_user_pressures) if end_user_pressures else 0.0
                return {
                    "总供热量(MW)": rs["total_heat_supplied"] / 1e6,
                    "热损失率(%)": rs["heat_loss_rate"],
                    "泵耗电(kW)": rs["total_pump_power"],
                    "供热单耗(kWh/GJ)": rs["specific_energy_consumption"],
                    "最大温降(°C)": max_temp_drop,
                    "最小末端压力(mH₂O)": min_end_pressure,
                }

            compare_data = {}
            for cn in selected_conds:
                compare_data[cn] = _calc_extra_metrics(
                    st.session_state.saved_conditions[cn],
                    st.session_state.network,
                )

            metric_names = list(list(compare_data.values())[0].keys())
            st.markdown("##### 📋 关键指标对比表（每行高亮差异最大的值）")

            def _highlight_row(row, metric):
                vals = [row[cn] for cn in selected_conds]
                vmin = min(vals)
                vmax = max(vals)
                if abs(vmax - vmin) < 1e-9:
                    return ["" for _ in selected_conds]
                if metric in ["热损失率(%)", "供热单耗(kWh/GJ)", "泵耗电(kW)", "最大温降(°C)"]:
                    target = vmin
                else:
                    target = vmax
                return [
                    "background-color: #fff3a8; font-weight: bold; color: #b8860b"
                    if abs(v - target) < 1e-9 else ""
                    for v in vals
                ]

            table_data = []
            for mn in metric_names:
                row = {"指标": mn}
                for cn in selected_conds:
                    val = compare_data[cn][mn]
                    if mn == "总供热量(MW)":
                        row[cn] = f"{val:.2f}"
                    elif mn == "热损失率(%)":
                        row[cn] = f"{val:.2f}"
                    elif mn == "泵耗电(kW)":
                        row[cn] = f"{val:.1f}"
                    elif mn == "供热单耗(kWh/GJ)":
                        row[cn] = f"{val:.2f}"
                    elif mn == "最大温降(°C)":
                        row[cn] = f"{val:.3f}"
                    elif mn == "最小末端压力(mH₂O)":
                        row[cn] = f"{val:.3f}"
                table_data.append(row)

            df_compare = pd.DataFrame(table_data)
            styled_rows = []
            for mn in metric_names:
                row = {"指标": mn}
                for cn in selected_conds:
                    row[cn] = compare_data[cn][mn]
                styled_rows.append(row)
            df_styled = pd.DataFrame(styled_rows)

            def apply_style(x):
                styles = pd.DataFrame("", index=x.index, columns=x.columns)
                for idx, mn in enumerate(metric_names):
                    row_vals = [compare_data[cn][mn] for cn in selected_conds]
                    vmin = min(row_vals)
                    vmax = max(row_vals)
                    if abs(vmax - vmin) < 1e-9:
                        continue
                    if mn in ["热损失率(%)", "供热单耗(kWh/GJ)", "泵耗电(kW)", "最大温降(°C)"]:
                        target = vmin
                    else:
                        target = vmax
                    for ci, cn in enumerate(selected_conds):
                        if abs(compare_data[cn][mn] - target) < 1e-9:
                            styles.loc[idx, cn] = "background-color: #fff3a8; font-weight: bold; color: #b8860b"
                return styles

            styled_df = df_compare.style.apply(apply_style, axis=None)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.divider()
            col_rad, col_bar = st.columns(2)

            with col_rad:
                st.markdown("##### 🕸️ 雷达图：6维度归一化得分")
                radar_fig = go.Figure()
                norm_metrics = {}
                for cn in selected_conds:
                    norm_metrics[cn] = {}
                for mn in metric_names:
                    vals = [compare_data[cn][mn] for cn in selected_conds]
                    vmin = min(vals)
                    vmax = max(vals)
                    for ci, cn in enumerate(selected_conds):
                        if abs(vmax - vmin) < 1e-9:
                            norm = 0.5
                        else:
                            if mn in ["热损失率(%)", "供热单耗(kWh/GJ)", "泵耗电(kW)", "最大温降(°C)"]:
                                norm = 1.0 - (vals[ci] - vmin) / (vmax - vmin)
                            else:
                                norm = (vals[ci] - vmin) / (vmax - vmin)
                        norm_metrics[cn][mn] = norm

                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                for ci, cn in enumerate(selected_conds):
                    radar_fig.add_trace(go.Scatterpolar(
                        r=[norm_metrics[cn][mn] for mn in metric_names],
                        theta=metric_names,
                        fill='toself',
                        name=cn,
                        line=dict(color=colors[ci % len(colors)], width=2),
                        fillcolor=f"rgba{tuple(int(colors[ci%len(colors)][i:i+2],16) for i in (1,3,5))+(0.25,)}",
                    ))
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1]),
                    ),
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                )
                st.plotly_chart(radar_fig, use_container_width=True)

            with col_bar:
                st.markdown("##### 📊 柱状图：各热源出力分配对比")
                all_source_ids = set()
                for cn in selected_conds:
                    sr = st.session_state.saved_conditions[cn].get("source_ratios", {})
                    all_source_ids.update(sr.keys())
                source_ids = sorted(all_source_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))
                if source_ids:
                    bar_fig = go.Figure()
                    source_names = {}
                    if st.session_state.network:
                        for sid in source_ids:
                            try:
                                sid_int = int(sid) if isinstance(sid, str) else sid
                                if sid_int in st.session_state.network.nodes:
                                    source_names[sid] = st.session_state.network.nodes[sid_int].name
                            except (ValueError, TypeError):
                                source_names[sid] = f"热源{sid}"
                    else:
                        for sid in source_ids:
                            source_names[sid] = f"热源{sid}"
                    for ci, cn in enumerate(selected_conds):
                        sr = st.session_state.saved_conditions[cn].get("source_ratios", {})
                        ratios = [sr.get(str(sid), sr.get(int(sid), 0.0)) * 100 for sid in source_ids]
                        bar_fig.add_trace(go.Bar(
                            name=cn,
                            x=[source_names.get(sid, f"热源{sid}") for sid in source_ids],
                            y=ratios,
                            text=[f"{r:.1f}%" for r in ratios],
                            textposition='auto',
                            marker_color=colors[ci % len(colors)],
                        ))
                    bar_fig.update_layout(
                        barmode='group',
                        yaxis=dict(title="负荷分配比例 (%)", range=[0, 100]),
                        height=500,
                        margin=dict(l=40, r=40, t=40, b=60),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                else:
                    st.info("所选工况中暂无热源分配数据")

            st.divider()
            with st.expander("📝 各工况输入参数详细对比"):
                param_rows = []
                param_list = [
                    ("环境温度(°C)", "environment_temperature", "°C", lambda x: f"{x:.1f}"),
                    ("水比热容(J/kg·°C)", "water_specific_heat", "", lambda x: f"{x:.0f}"),
                    ("调度策略", "dispatch_strategy", "", lambda x: x.split("（")[0]),
                    ("水力平衡容差(%)", "balance_tolerance", "%", lambda x: f"{x*100:.0f}"),
                ]
                for pname, pkey, punit, pfmt in param_list:
                    row = {"参数": pname}
                    for cn in selected_conds:
                        val = st.session_state.saved_conditions[cn]["input_params"].get(pkey, "-")
                        row[cn] = pfmt(val) if val != "-" else "-"
                    param_rows.append(row)
                n_sources_row = {"参数": "热源供水温度配置"}
                for cn in selected_conds:
                    sts = st.session_state.saved_conditions[cn]["input_params"]["source_temps"]
                    n_sources_row[cn] = f"{len(sts)}个热源"
                param_rows.append(n_sources_row)
                for si, (sid, _) in enumerate(st.session_state.source_temps.items()):
                    if st.session_state.network:
                        try:
                            sid_int = int(sid) if isinstance(sid, str) else sid
                            sname = st.session_state.network.nodes[sid_int].name if sid_int in st.session_state.network.nodes else f"热源{sid}"
                        except (ValueError, TypeError):
                            sname = f"热源{sid}"
                    else:
                        sname = f"热源{sid}"
                    row = {"参数": f"  {sname}供水温度(°C)"}
                    for cn in selected_conds:
                        sts = st.session_state.saved_conditions[cn]["input_params"]["source_temps"]
                        sv = sts.get(str(sid), sts.get(int(sid), "-"))
                        row[cn] = f"{sv:.1f}" if sv != "-" else "-"
                    param_rows.append(row)
                st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

with tabs[7]:
    st.subheader("🚨 管网故障模拟与应急预案")
    if st.session_state.network is None or st.session_state.results is None:
        st.info("👈 请先加载管网数据并运行正常工况计算，然后再进行故障模拟")
    else:
        net = st.session_state.network
        normal_results = st.session_state.results

        col_left, col_right = st.columns([1, 1.4])

        with col_left:
            st.markdown("##### 🎯 故障设置区")
            st.info("💡 可添加多个故障进行组合模拟（如：管段爆管 + 泵站故障同时发生）")

            fault_descriptions = []
            for i, f in enumerate(st.session_state.fault_list):
                fault_descriptions.append(f"{i+1}. {f.describe(net)}")
            if fault_descriptions:
                with st.expander(f"📋 已设置故障 ({len(st.session_state.fault_list)} 项)", expanded=True):
                    for fd in fault_descriptions:
                        st.write(fd)
                    if st.button("🗑️ 清空所有故障", use_container_width=True):
                        st.session_state.fault_list = []
                        st.session_state.fault_results = None
                        st.session_state.fault_impact = None
                        st.session_state.fault_modified_network = None
                        st.session_state.emergency_plan = None
                        st.rerun()
            else:
                st.caption("尚未设置任何故障")

            st.divider()
            st.markdown("###### ➕ 添加新故障")
            new_fault_type = st.selectbox(
                "故障类型",
                list(FAULT_TYPE_CN.keys()),
                format_func=lambda x: FAULT_TYPE_CN[x],
                key="new_fault_type_select",
            )

            if new_fault_type == FAULT_TYPE_PIPE_BURST:
                all_pipes = get_available_pipes(net)
                pipe_options = [(p.id, p.name) for p in all_pipes]
                existing_burst = {f.target_id for f in st.session_state.fault_list if f.fault_type == FAULT_TYPE_PIPE_BURST}
                pipe_options = [(pid, pname) for pid, pname in pipe_options if pid not in existing_burst]
                if pipe_options:
                    selected_pipe = st.selectbox(
                        "选择爆管管段",
                        pipe_options,
                        format_func=lambda x: f"{x[1]} (ID:{x[0]})",
                        key="burst_pipe_select",
                    )
                    if st.button("➕ 添加管段爆管故障", use_container_width=True):
                        st.session_state.fault_list.append(
                            FaultConfig(fault_type=FAULT_TYPE_PIPE_BURST, target_id=selected_pipe[0])
                        )
                        st.rerun()
                else:
                    st.warning("所有管段均已设置为爆管故障")

            elif new_fault_type == FAULT_TYPE_PUMP_FAILURE:
                pump_pipes = get_available_pump_pipes(net)
                if not pump_pipes:
                    st.warning("管网中没有泵站")
                else:
                    pump_options = [(p.id, p.name) for p in pump_pipes]
                    existing_pump_fail = {f.target_id for f in st.session_state.fault_list if f.fault_type == FAULT_TYPE_PUMP_FAILURE}
                    pump_options = [(pid, pname) for pid, pname in pump_options if pid not in existing_pump_fail]
                    if pump_options:
                        selected_pump = st.selectbox(
                            "选择故障泵站",
                            pump_options,
                            format_func=lambda x: f"{x[1]} (ID:{x[0]})",
                            key="failed_pump_select",
                        )
                        if st.button("➕ 添加泵站故障", use_container_width=True):
                            st.session_state.fault_list.append(
                                FaultConfig(fault_type=FAULT_TYPE_PUMP_FAILURE, target_id=selected_pump[0])
                            )
                            st.rerun()
                    else:
                        st.warning("所有泵站均已设置为故障")

            elif new_fault_type == FAULT_TYPE_SOURCE_SHUTDOWN:
                source_nodes = get_available_source_nodes(net)
                source_options = [(n.id, n.name) for n in source_nodes]
                existing_shutdown = {f.target_id for f in st.session_state.fault_list if f.fault_type == FAULT_TYPE_SOURCE_SHUTDOWN}
                source_options = [(sid, sname) for sid, sname in source_options if sid not in existing_shutdown]
                if source_options:
                    selected_source = st.selectbox(
                        "选择停机热源",
                        source_options,
                        format_func=lambda x: f"{x[1]} (ID:{x[0]})",
                        key="shutdown_source_select",
                    )
                    if st.button("➕ 添加热源停机故障", use_container_width=True):
                        st.session_state.fault_list.append(
                            FaultConfig(fault_type=FAULT_TYPE_SOURCE_SHUTDOWN, target_id=selected_source[0])
                        )
                        st.rerun()
                else:
                    st.warning("所有热源均已设置为停机")

            st.divider()
            risk_mode = st.checkbox(
                "☢️ 风险评估模式（自动遍历所有设备计算风险排名）",
                value=st.session_state.risk_mode_enabled,
                help="勾选后系统自动遍历所有管段/泵站/热源，计算每个潜在故障点的风险评分",
            )
            st.session_state.risk_mode_enabled = risk_mode

            if risk_mode:
                if st.button("🔍 开始风险评估计算", type="primary", use_container_width=True):
                    with st.spinner("🔬 正在遍历所有设备进行风险评估计算，这可能需要一些时间..."):
                        try:
                            risk_items = calculate_risk_assessment(
                                net, normal_results, st.session_state.source_temps,
                            )
                            st.session_state.risk_assessment_results = risk_items[:10]
                            st.success(f"✅ 风险评估完成，共评估 {len(risk_items)} 个潜在故障点")
                        except Exception as e:
                            st.error(f"❌ 风险评估失败：{e}")
                            import traceback
                            with st.expander("错误详情"):
                                st.code(traceback.format_exc())
                if st.session_state.risk_assessment_results:
                    st.divider()
                    st.markdown("###### 📊 风险排名 Top 10")
                    risk_rows = []
                    for item in st.session_state.risk_assessment_results:
                        risk_rows.append({
                            "设备名称": item.device_name,
                            "设备类型": item.device_type,
                            "故障概率": f"{item.fault_probability*100:.0f}%",
                            "供热能力下降(%)": f"{item.heat_capacity_drop_pct:.1f}%",
                            "风险评分": f"{item.risk_score:.2f}",
                        })
                    df_risk = pd.DataFrame(risk_rows)
                    st.dataframe(df_risk, use_container_width=True, hide_index=True, height=380)

                    fig_risk = go.Figure()
                    risk_names = [item.device_name for item in st.session_state.risk_assessment_results]
                    risk_scores = [item.risk_score for item in st.session_state.risk_assessment_results]
                    risk_types = [item.device_type for item in st.session_state.risk_assessment_results]
                    risk_probs = [item.fault_probability * 100 for item in st.session_state.risk_assessment_results]
                    risk_drops = [item.heat_capacity_drop_pct for item in st.session_state.risk_assessment_results]
                    risk_colors = []
                    for item in st.session_state.risk_assessment_results:
                        if item.device_type == "管段":
                            risk_colors.append("#e74c3c")
                        elif item.device_type == "泵站":
                            risk_colors.append("#f39c12")
                        else:
                            risk_colors.append("#3498db")
                    custom_hovertemplate = (
                        '<b>%{x}</b><br>' +
                        '风险评分: %{y:.2f}<br>' +
                        '设备类型: %{customdata[0]}<br>' +
                        '故障概率: %{customdata[1]:.0f}%<br>' +
                        '供热能力下降: %{customdata[2]:.1f}%' +
                        '<extra></extra>'
                    )
                    fig_risk.add_trace(go.Bar(
                        x=risk_names,
                        y=risk_scores,
                        marker_color=risk_colors,
                        text=[f"{s:.2f}" for s in risk_scores],
                        textposition='outside',
                        customdata=list(zip(risk_types, risk_probs, risk_drops)),
                        hovertemplate=custom_hovertemplate,
                    ))
                    fig_risk.update_layout(
                        xaxis_title="设备名称",
                        yaxis_title="风险评分",
                        yaxis=dict(tickformat='.2f'),
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=120),
                        xaxis_tickangle=-30,
                        hovermode='x unified',
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                    st.caption("🔴 红色=管段 | 🟡 橙色=泵站 | 🔵 蓝色=热源")
            else:
                col_run1, col_run2 = st.columns(2)
                with col_run1:
                    run_fault = st.button(
                        "🔄 运行故障模拟计算",
                        type="primary",
                        use_container_width=True,
                        disabled=len(st.session_state.fault_list) == 0,
                    )
                with col_run2:
                    if st.button("🔙 重置故障模拟", use_container_width=True):
                        st.session_state.fault_results = None
                        st.session_state.fault_impact = None
                        st.session_state.fault_modified_network = None
                        st.session_state.emergency_plan = None
                        st.session_state.recovery_effects = {}
                        st.session_state.recovery_current_results = None
                        st.session_state.recovery_current_impact = None
                        st.session_state.recovery_current_network = None
                        st.session_state.recovery_current_source_temps = None
                        st.rerun()

                if run_fault:
                    with st.spinner("🔬 正在进行故障工况水力热力耦合计算..."):
                        try:
                            fault_res, impact, modified_net = simulate_faults(
                                net, normal_results,
                                st.session_state.fault_list,
                                st.session_state.source_temps,
                            )
                            st.session_state.fault_results = fault_res
                            st.session_state.fault_impact = impact
                            st.session_state.fault_modified_network = modified_net
                            st.session_state.recovery_effects = {}
                            st.session_state.recovery_current_results = fault_res
                            st.session_state.recovery_current_impact = impact
                            st.session_state.recovery_current_network = modified_net
                            st.session_state.recovery_current_source_temps = dict(st.session_state.source_temps)

                            with st.spinner("📋 正在生成应急预案..."):
                                plan, _, _ = generate_emergency_plan(
                                    net, modified_net, normal_results,
                                    fault_res, st.session_state.fault_list,
                                    impact, st.session_state.source_temps,
                                    min_temp_threshold=st.session_state.temp_threshold,
                                )
                                st.session_state.emergency_plan = plan

                            history_record = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "fault_descriptions": [f.describe(net) for f in st.session_state.fault_list],
                                "impact": {
                                    "affected_user_count": len(impact.affected_user_ids),
                                    "disconnected_user_count": len(impact.disconnected_user_ids),
                                    "total_heat_capacity_drop_pct": impact.total_heat_capacity_drop_pct,
                                    "affected_user_ids": impact.affected_user_ids,
                                    "affected_user_names": impact.affected_user_names,
                                    "disconnected_user_ids": impact.disconnected_user_ids,
                                    "disconnected_user_names": impact.disconnected_user_names,
                                    "user_temp_changes": dict(impact.user_temp_changes),
                                    "user_pressure_changes": dict(impact.user_pressure_changes),
                                    "low_pressure_users": impact.low_pressure_users,
                                    "summary_text": impact.summary_text,
                                    "original_total_heat": impact.original_total_heat,
                                    "fault_total_heat": impact.fault_total_heat,
                                },
                                "emergency_plan_actions": [
                                    {
                                        "priority": a.priority,
                                        "action_type": a.action_type,
                                        "target_id": a.target_id,
                                        "target_name": a.target_name,
                                        "description": a.description,
                                        "detail": a.detail,
                                        "estimated_effect": a.estimated_effect,
                                    }
                                    for a in (plan.actions if plan else [])
                                ],
                                "emergency_plan_summary": plan.summary if plan else "",
                                "emergency_plan_warnings": plan.warnings if plan else [],
                            }
                            st.session_state.fault_history.insert(0, history_record)
                            if len(st.session_state.fault_history) > 10:
                                st.session_state.fault_history = st.session_state.fault_history[:10]

                            st.success("✅ 故障模拟与应急预案生成完成！")
                        except Exception as e:
                            st.error(f"❌ 故障模拟失败：{e}")
                            import traceback
                            with st.expander("错误详情"):
                                st.code(traceback.format_exc())

        with col_right:
            if st.session_state.fault_impact is None:
                st.info("👈 请先在左侧添加故障并运行模拟计算")
            else:
                display_impact: ImpactAssessment = (
                    st.session_state.recovery_current_impact
                    if st.session_state.recovery_current_impact is not None
                    else st.session_state.fault_impact
                )
                display_results = (
                    st.session_state.recovery_current_results
                    if st.session_state.recovery_current_results is not None
                    else st.session_state.fault_results
                )
                impact: ImpactAssessment = st.session_state.fault_impact
                fault_results = st.session_state.fault_results
                plan: EmergencyPlan = st.session_state.emergency_plan

                show_recovery = (
                    st.session_state.recovery_current_impact is not None
                    and st.session_state.recovery_effects
                )
                if show_recovery:
                    st.info("ℹ️ 当前显示应急措施执行后的恢复效果，点击下方'重置恢复效果'可返回故障初始状态")
                    if st.button("🔄 重置恢复效果", use_container_width=True):
                        st.session_state.recovery_effects = {}
                        st.session_state.recovery_current_results = st.session_state.fault_results
                        st.session_state.recovery_current_impact = st.session_state.fault_impact
                        st.session_state.recovery_current_network = st.session_state.fault_modified_network
                        st.session_state.recovery_current_source_temps = dict(st.session_state.source_temps)
                        st.rerun()

                st.markdown("##### 📊 故障影响评估汇总")
                m1, m2, m3, m4 = st.columns(4)
                total_users = len(net.get_nodes_by_type(NODE_TYPE_END_USER))
                with m1:
                    st.metric(
                        "受影响用户",
                        f"{len(display_impact.affected_user_ids)}/{total_users}",
                        delta=f"{len(display_impact.affected_user_ids)/max(total_users,1)*100:.0f}%"
                    )
                with m2:
                    st.metric(
                        "断供用户",
                        f"{len(display_impact.disconnected_user_ids)}",
                        delta_color="inverse"
                    )
                with m3:
                    st.metric(
                        "压力不达标",
                        f"{len(display_impact.low_pressure_users)}",
                        delta=f"<{MIN_OPERATING_PRESSURE}mH₂O",
                        delta_color="off"
                    )
                with m4:
                    st.metric(
                        "供热能力下降",
                        f"{display_impact.total_heat_capacity_drop_pct:.1f}%",
                        delta_color="inverse"
                    )

                st.divider()
                tab_compare, tab_topo, tab_plan, tab_history = st.tabs([
                    "📋 正常vs故障对比", "🗺️ 故障拓扑图", "🚒 应急预案", "📜 历史记录"
                ])

                with tab_compare:
                    st.markdown("###### 各末端用户温度与压力对比")
                    end_users = net.get_nodes_by_type(NODE_TYPE_END_USER)
                    compare_rows = []
                    for user in end_users:
                        nr_normal = normal_results.node_results.get(user.id)
                        nr_fault = display_results.node_results.get(user.id) if display_results else None
                        is_disconnected = user.id in display_impact.disconnected_user_ids
                        is_low_p = user.id in display_impact.low_pressure_users
                        is_affected = user.id in display_impact.affected_user_ids

                        temp_normal = nr_normal.temperature if nr_normal else 0.0
                        press_normal = nr_normal.pressure if nr_normal else 0.0

                        if is_disconnected:
                            temp_fault = 0.0
                            press_fault = 0.0
                            temp_delta = -999.0
                            press_delta = -999.0
                            status = "❌ 断供"
                        elif nr_fault:
                            temp_fault = nr_fault.temperature
                            press_fault = nr_fault.pressure
                            temp_delta = temp_fault - temp_normal
                            press_delta = press_fault - press_normal
                            if is_low_p:
                                status = "⚠️ 压力不足"
                            elif is_affected:
                                status = "🔶 受影响"
                            else:
                                status = "✅ 正常"
                        else:
                            temp_fault = 0.0
                            press_fault = 0.0
                            temp_delta = 0.0
                            press_delta = 0.0
                            status = "❓ 未知"

                        compare_rows.append({
                            "用户编号": user.id,
                            "用户名称": user.name,
                            "正常温度(°C)": f"{temp_normal:.1f}",
                            "故障温度(°C)": f"{temp_fault:.1f}" if not is_disconnected else "断供",
                            "温度变化(°C)": f"{temp_delta:+.1f}" if not is_disconnected else "--",
                            "正常压力(mH₂O)": f"{press_normal:.2f}",
                            "故障压力(mH₂O)": f"{press_fault:.2f}" if not is_disconnected else "断供",
                            "压力变化(mH₂O)": f"{press_delta:+.2f}" if not is_disconnected else "--",
                            "状态": status,
                            "_is_bad": is_disconnected or is_low_p or (abs(temp_delta) > 3 if not is_disconnected else False),
                        })

                    def apply_compare_style(x):
                        styles = pd.DataFrame("", index=x.index, columns=df_compare_display.columns)
                        for idx, rdata in enumerate(compare_rows):
                            if rdata["_is_bad"]:
                                if rdata["状态"] == "❌ 断供":
                                    bg_color = "#ffcccc"
                                    text_color = "#cc0000"
                                elif rdata["状态"] == "⚠️ 压力不足":
                                    bg_color = "#ffe0cc"
                                    text_color = "#cc5500"
                                else:
                                    bg_color = "#fff4cc"
                                    text_color = "#996600"
                                styles.loc[idx, :] = f"background-color: {bg_color}; color: {text_color}; font-weight: bold"
                        return styles

                    df_compare_display = pd.DataFrame([{
                        k: v for k, v in r.items() if not k.startswith("_")
                    } for r in compare_rows])
                    styled_compare = df_compare_display.style.apply(apply_compare_style, axis=None)
                    st.dataframe(
                        styled_compare,
                        use_container_width=True, hide_index=True, height=420,
                    )
                    st.caption("🔴 红色=断供 | 🟠 橙色=压力不足 | 🟡 黄色=温度/压力显著变化")

                    st.divider()
                    st.markdown("###### 📈 关键指标对比")
                    metric_compare = [
                        ["总供热量 (MW)",
                         f"{display_impact.original_total_heat/1e6:.2f}",
                         f"{display_impact.fault_total_heat/1e6:.2f}" if display_results else "计算失败",
                         f"{-display_impact.total_heat_capacity_drop_pct:.1f}%"],
                        ["受影响用户数", "0",
                         f"{len(display_impact.affected_user_ids)}",
                         f"+{len(display_impact.affected_user_ids)}"],
                        ["断供用户数", "0",
                         f"{len(display_impact.disconnected_user_ids)}",
                         f"+{len(display_impact.disconnected_user_ids)}"],
                        ["压力低于20mH₂O用户数",
                         f"{sum(1 for u in end_users if (normal_results.node_results.get(u.id) and normal_results.node_results[u.id].pressure < MIN_OPERATING_PRESSURE))}",
                         f"{len(display_impact.low_pressure_users)}",
                         f"+{max(0, len(display_impact.low_pressure_users) - sum(1 for u in end_users if (normal_results.node_results.get(u.id) and normal_results.node_results[u.id].pressure < MIN_OPERATING_PRESSURE)))}"],
                    ]
                    df_metric = pd.DataFrame(metric_compare, columns=["指标", "正常工况", "故障工况", "变化"])
                    st.dataframe(df_metric, use_container_width=True, hide_index=True)

                with tab_topo:
                    if st.session_state.fault_modified_network is not None:
                        with st.spinner("绘制故障拓扑图..."):
                            fig_fault = create_fault_topology_figure(
                                net, display_results,
                                display_impact, st.session_state.fault_list,
                                normal_results,
                            )
                            st.plotly_chart(fig_fault, use_container_width=True)
                    else:
                        st.info("故障拓扑图暂不可用")

                with tab_plan:
                    if plan and plan.actions:
                        st.markdown(f"###### 🚒 {plan.summary}")

                        if plan.warnings:
                            for w in plan.warnings:
                                st.warning(f"⚠️ {w}")

                        sorted_actions = plan.sorted_actions()
                        priority_labels = {1: "🔴 紧急", 2: "🟠 高", 3: "🟡 中", 4: "🔵 低"}
                        for action_idx, action in enumerate(sorted_actions):
                            p_label = priority_labels.get(action.priority, f"优先级{action.priority}")
                            executed = action_idx in st.session_state.recovery_effects
                            status_icon = "✅" if executed else "⏳"
                            with st.expander(f"{status_icon} {p_label} | {action.description}", expanded=(action.priority <= 2)):
                                st.markdown(f"**措施类型**: {action.action_type}")
                                st.markdown(f"**实施对象**: {action.target_name} (ID:{action.target_id})")
                                if action.detail:
                                    st.markdown(f"**详细说明**: {action.detail}")
                                if action.estimated_effect:
                                    st.success(f"💡 预期效果: {action.estimated_effect}")

                                can_simulate = (
                                    action.suggested_params
                                    and action.suggested_params.get("action", "") in [
                                        "increase_source_temp", "increase_pump_head",
                                        "switch_path", "isolate_pipe"
                                    ]
                                )
                                if can_simulate and not executed:
                                    if st.button(
                                        "▶️ 模拟执行",
                                        key=f"sim_action_{action_idx}",
                                        use_container_width=True,
                                        type="primary",
                                    ):
                                        with st.spinner("🔬 正在模拟执行该应急措施..."):
                                            curr_net = (
                                                st.session_state.recovery_current_network
                                                if st.session_state.recovery_current_network is not None
                                                else st.session_state.fault_modified_network
                                            )
                                            curr_res = (
                                                st.session_state.recovery_current_results
                                                if st.session_state.recovery_current_results is not None
                                                else st.session_state.fault_results
                                            )
                                            curr_impact = (
                                                st.session_state.recovery_current_impact
                                                if st.session_state.recovery_current_impact is not None
                                                else st.session_state.fault_impact
                                            )
                                            curr_st = (
                                                st.session_state.recovery_current_source_temps
                                                if st.session_state.recovery_current_source_temps is not None
                                                else st.session_state.source_temps
                                            )
                                            effect = execute_emergency_action(
                                                net, normal_results,
                                                curr_net, curr_res,
                                                curr_impact, curr_st,
                                                action,
                                                st.session_state.fault_list,
                                                action_idx,
                                            )
                                            if effect is not None:
                                                st.session_state.recovery_effects[action_idx] = effect
                                                st.session_state.recovery_current_results = effect.updated_results
                                                st.session_state.recovery_current_impact = effect.updated_impact
                                                st.session_state.recovery_current_network = effect.updated_network
                                                st.session_state.recovery_current_source_temps = effect.updated_source_temps
                                                st.success("✅ 模拟执行完成！上方影响评估已更新")
                                                st.rerun()
                                            else:
                                                st.error("❌ 该措施暂不支持模拟执行")

                                if executed:
                                    effect: RecoveryEffect = st.session_state.recovery_effects[action_idx]
                                    st.markdown("---")
                                    st.markdown("###### ✅ 执行后恢复效果")
                                    e1, e2 = st.columns(2)
                                    with e1:
                                        st.metric(
                                            "平均温度提升",
                                            f"{effect.recovered_temp_avg - effect.original_temp_avg:+.1f}°C"
                                        )
                                    with e2:
                                        st.metric(
                                            "与正常工况温度差距缩小",
                                            f"{effect.temp_gap_reduction_pct:.0f}%"
                                            if effect.temp_gap_reduction_pct > 0
                                            else "0%"
                                        )
                                    e3, e4 = st.columns(2)
                                    with e3:
                                        st.metric(
                                            "平均压力提升",
                                            f"{effect.recovered_pressure_avg - effect.original_pressure_avg:+.2f}mH₂O"
                                        )
                                    with e4:
                                        st.metric(
                                            "与正常工况压力差距缩小",
                                            f"{effect.pressure_gap_reduction_pct:.0f}%"
                                            if effect.pressure_gap_reduction_pct > 0
                                            else "0%"
                                        )

                                    if effect.user_temp_changes:
                                        st.markdown("###### 📋 各用户温度变化")
                                        user_changes = []
                                        end_user_map = {u.id: u.name for u in end_users}
                                        for uid, tchange in effect.user_temp_changes.items():
                                            if abs(tchange) > 0.05:
                                                user_changes.append({
                                                    "用户名称": end_user_map.get(uid, f"用户{uid}"),
                                                    "温度变化(°C)": f"{tchange:+.2f}",
                                                })
                                        if user_changes:
                                            st.dataframe(
                                                pd.DataFrame(user_changes),
                                                use_container_width=True, hide_index=True,
                                                height=200,
                                            )

                        st.divider()
                        st.markdown("""
                        > 📌 **调度操作注意事项**：
                        > 1. 优先执行优先级1、2的紧急措施；
                        > 2. 阀门操作应缓慢进行，避免管网水击；
                        > 3. 热源调整应逐步提升温度，每次不超过5°C；
                        > 4. 每项措施实施后应等待15-30分钟观察管网变化；
                        > 5. 建议实施应急措施后重新运行计算进行效果验证。
                        """)
                    else:
                        st.info("暂无应急预案内容")

                with tab_history:
                    if not st.session_state.fault_history:
                        st.info("💡 暂无历史记录，运行故障模拟后结果会自动保存（最多10条）")
                    else:
                        st.markdown(f"###### 📜 故障模拟历史记录（共 {len(st.session_state.fault_history)} 条）")
                        compare_mode = st.checkbox("🔀 并排对比模式（选择两条记录进行对比）", value=False)
                        st.divider()

                        if compare_mode:
                            history_options = [
                                f"[{h['timestamp']}] {' + '.join(h['fault_descriptions'])}"
                                for h in st.session_state.fault_history
                            ]
                            col_sel1, col_sel2 = st.columns(2)
                            with col_sel1:
                                sel_a = st.selectbox(
                                    "选择记录 A",
                                    range(len(st.session_state.fault_history)),
                                    format_func=lambda i: history_options[i],
                                    key="hist_compare_a",
                                )
                            with col_sel2:
                                default_b = min(1, len(st.session_state.fault_history) - 1)
                                sel_b = st.selectbox(
                                    "选择记录 B",
                                    range(len(st.session_state.fault_history)),
                                    format_func=lambda i: history_options[i],
                                    index=default_b if default_b != sel_a else 0,
                                    key="hist_compare_b",
                                )
                            if sel_a == sel_b:
                                st.warning("⚠️ 请选择两条不同的记录进行对比")
                            else:
                                rec_a = st.session_state.fault_history[sel_a]
                                rec_b = st.session_state.fault_history[sel_b]
                                st.divider()
                                st.markdown("###### 📊 关键指标对比")
                                comp_cols = st.columns(2)
                                with comp_cols[0]:
                                    st.info(f"**记录 A**: {rec_a['timestamp']}")
                                with comp_cols[1]:
                                    st.info(f"**记录 B**: {rec_b['timestamp']}")

                                metric_names = ["受影响用户数", "断供用户数", "供热能力下降(%)"]
                                val_a = [
                                    rec_a["impact"]["affected_user_count"],
                                    rec_a["impact"]["disconnected_user_count"],
                                    f"{rec_a['impact']['total_heat_capacity_drop_pct']:.1f}",
                                ]
                                val_b = [
                                    rec_b["impact"]["affected_user_count"],
                                    rec_b["impact"]["disconnected_user_count"],
                                    f"{rec_b['impact']['total_heat_capacity_drop_pct']:.1f}",
                                ]
                                comp_table = []
                                for i, mn in enumerate(metric_names):
                                    va = val_a[i]
                                    vb = val_b[i]
                                    try:
                                        diff = float(vb) - float(va)
                                        diff_str = f"{diff:+.1f}" if "." in str(va) else f"{int(diff):+d}"
                                    except (ValueError, TypeError):
                                        diff_str = "-"
                                    comp_table.append({
                                        "指标": mn,
                                        f"记录A ({rec_a['timestamp']})": va,
                                        f"记录B ({rec_b['timestamp']})": vb,
                                    })
                                df_comp = pd.DataFrame(comp_table)

                                def style_compare(row):
                                    res = [""] * len(df_comp.columns)
                                    try:
                                        va = float(row.iloc[1]) if isinstance(row.iloc[1], str) and row.iloc[1].replace('.', '', 1).lstrip('-').isdigit() else None
                                        vb = float(row.iloc[2]) if isinstance(row.iloc[2], str) and row.iloc[2].replace('.', '', 1).lstrip('-').isdigit() else None
                                        if va is not None and vb is not None:
                                            if va < vb:
                                                res[1] = "background-color: #ccffcc; color: #006600; font-weight: bold"
                                                res[2] = "background-color: #ffcccc; color: #cc0000; font-weight: bold"
                                            elif va > vb:
                                                res[1] = "background-color: #ffcccc; color: #cc0000; font-weight: bold"
                                                res[2] = "background-color: #ccffcc; color: #006600; font-weight: bold"
                                    except Exception:
                                        pass
                                    return res

                                styled_df_comp = df_comp.style.apply(style_compare, axis=1)
                                st.dataframe(styled_df_comp, use_container_width=True, hide_index=True)
                                st.caption("🟢 绿色=较优（影响较小） | 🔴 红色=较差（影响较大）")

                                st.divider()
                                st.markdown("###### 👥 各用户影响对比（红绿色标注差异）")
                                end_users_map = {u.id: u.name for u in net.get_nodes_by_type(NODE_TYPE_END_USER)}
                                user_diff_rows = []
                                all_uids = sorted(set(list(rec_a["impact"]["user_temp_changes"].keys()) +
                                                    list(rec_b["impact"]["user_temp_changes"].keys())))
                                for uid in all_uids:
                                    uname = end_users_map.get(uid, f"用户{uid}")
                                    disc_a = uid in rec_a["impact"]["disconnected_user_ids"]
                                    disc_b = uid in rec_b["impact"]["disconnected_user_ids"]
                                    t_a = rec_a["impact"]["user_temp_changes"].get(uid, 0.0)
                                    t_b = rec_b["impact"]["user_temp_changes"].get(uid, 0.0)
                                    p_a = rec_a["impact"]["user_pressure_changes"].get(uid, 0.0)
                                    p_b = rec_b["impact"]["user_pressure_changes"].get(uid, 0.0)

                                    if disc_a and disc_b:
                                        s_a = "❌ 断供"
                                        s_b = "❌ 断供"
                                        diff_flag = "same"
                                    elif disc_a:
                                        s_a = "❌ 断供"
                                        s_b = f"{t_b:+.1f}°C"
                                        diff_flag = "b_better"
                                    elif disc_b:
                                        s_a = f"{t_a:+.1f}°C"
                                        s_b = "❌ 断供"
                                        diff_flag = "a_better"
                                    else:
                                        s_a = f"{t_a:+.1f}°C"
                                        s_b = f"{t_b:+.1f}°C"
                                        if abs(t_a - t_b) < 0.1:
                                            diff_flag = "same"
                                        elif t_a > t_b:
                                            diff_flag = "a_better"
                                        else:
                                            diff_flag = "b_better"

                                    user_diff_rows.append({
                                        "用户名称": uname,
                                        "A温度变化": s_a,
                                        "B温度变化": s_b,
                                        "_diff": diff_flag,
                                    })

                                def style_user_diff(x):
                                    styles = pd.DataFrame("", index=x.index, columns=df_user_diff_display.columns)
                                    for idx, r in enumerate(user_diff_rows):
                                        if r["_diff"] == "a_better":
                                            styles.loc[idx, "A温度变化"] = "background-color: #ccffcc; color: #006600; font-weight: bold"
                                            styles.loc[idx, "B温度变化"] = "background-color: #ffcccc; color: #cc0000; font-weight: bold"
                                        elif r["_diff"] == "b_better":
                                            styles.loc[idx, "A温度变化"] = "background-color: #ffcccc; color: #cc0000; font-weight: bold"
                                            styles.loc[idx, "B温度变化"] = "background-color: #ccffcc; color: #006600; font-weight: bold"
                                    return styles

                                df_user_diff_display = pd.DataFrame([{
                                    k: v for k, v in r.items() if not k.startswith("_")
                                } for r in user_diff_rows])
                                styled_user_diff = df_user_diff_display.style.apply(style_user_diff, axis=None)
                                st.dataframe(styled_user_diff, use_container_width=True, hide_index=True, height=400)
                                st.caption("🟢 绿色=较优（温度下降更少或正常供热） | 🔴 红色=较差（温度下降更多或断供）")

                                st.divider()
                                col_plan_a, col_plan_b = st.columns(2)
                                with col_plan_a:
                                    st.markdown(f"###### 📋 记录A 应急预案")
                                    st.markdown(f"**摘要**: {rec_a.get('emergency_plan_summary', '')}")
                                    for w in rec_a.get("emergency_plan_warnings", []):
                                        st.warning(f"⚠️ {w}")
                                    for act in rec_a.get("emergency_plan_actions", [])[:10]:
                                        p_label = priority_labels.get(act["priority"], f"优先级{act['priority']}")
                                        with st.expander(f"{p_label} | {act['description']}", expanded=False):
                                            st.markdown(f"**措施类型**: {act['action_type']}")
                                            st.markdown(f"**实施对象**: {act['target_name']} (ID:{act['target_id']})")
                                            if act.get("detail"):
                                                st.markdown(f"**详细说明**: {act['detail']}")
                                with col_plan_b:
                                    st.markdown(f"###### 📋 记录B 应急预案")
                                    st.markdown(f"**摘要**: {rec_b.get('emergency_plan_summary', '')}")
                                    for w in rec_b.get("emergency_plan_warnings", []):
                                        st.warning(f"⚠️ {w}")
                                    for act in rec_b.get("emergency_plan_actions", [])[:10]:
                                        p_label = priority_labels.get(act["priority"], f"优先级{act['priority']}")
                                        with st.expander(f"{p_label} | {act['description']}", expanded=False):
                                            st.markdown(f"**措施类型**: {act['action_type']}")
                                            st.markdown(f"**实施对象**: {act['target_name']} (ID:{act['target_id']})")
                                            if act.get("detail"):
                                                st.markdown(f"**详细说明**: {act['detail']}")
                        else:
                            for hi, record in enumerate(st.session_state.fault_history):
                                with st.container():
                                    col_ts, col_recall = st.columns([3, 1])
                                    with col_ts:
                                        st.markdown(
                                            f"🕒 **{record['timestamp']}**  |  "
                                            f"{' + '.join(record['fault_descriptions'])}"
                                        )
                                    with col_recall:
                                        if st.button("📂 回显详情", key=f"recall_{hi}", use_container_width=True):
                                            st.session_state["_pending_recall"] = hi
                                            st.rerun()

                                    sum_c1, sum_c2, sum_c3 = st.columns(3)
                                    with sum_c1:
                                        st.caption(
                                            f"受影响用户: **{record['impact']['affected_user_count']}** 人"
                                        )
                                    with sum_c2:
                                        st.caption(
                                            f"断供用户: **{record['impact']['disconnected_user_count']}** 人"
                                        )
                                    with sum_c3:
                                        st.caption(
                                            f"供热能力下降: **{record['impact']['total_heat_capacity_drop_pct']:.1f}%**"
                                        )

                                    if st.session_state.get("_pending_recall") == hi:
                                        st.divider()
                                        st.markdown("###### 📋 回显详情")
                                        st.markdown(f"**摘要**: {record['impact']['summary_text']}")
                                        st.markdown("###### 👥 影响评估报告")
                                        recall_rows = []
                                        end_users_map = {u.id: u.name for u in net.get_nodes_by_type(NODE_TYPE_END_USER)}
                                        for uid in sorted(record["impact"]["user_temp_changes"].keys()):
                                            uname = end_users_map.get(uid, f"用户{uid}")
                                            disc = uid in record["impact"]["disconnected_user_ids"]
                                            lowp = uid in record["impact"]["low_pressure_users"]
                                            t_chg = record["impact"]["user_temp_changes"][uid]
                                            p_chg = record["impact"]["user_pressure_changes"][uid]
                                            if disc:
                                                status = "❌ 断供"
                                            elif lowp:
                                                status = "⚠️ 压力不足"
                                            elif abs(t_chg) > 0.1 or abs(p_chg) > 0.5:
                                                status = "🔶 受影响"
                                            else:
                                                status = "✅ 正常"
                                            recall_rows.append({
                                                "用户名称": uname,
                                                "温度变化(°C)": "--" if disc else f"{t_chg:+.1f}",
                                                "压力变化(mH₂O)": "--" if disc else f"{p_chg:+.2f}",
                                                "状态": status,
                                            })
                                        st.dataframe(
                                            pd.DataFrame(recall_rows),
                                            use_container_width=True, hide_index=True,
                                            height=300,
                                        )

                                        st.markdown("###### 🚒 应急预案")
                                        st.markdown(f"**摘要**: {record.get('emergency_plan_summary', '')}")
                                        for w in record.get("emergency_plan_warnings", []):
                                            st.warning(f"⚠️ {w}")
                                        priority_labels = {1: "🔴 紧急", 2: "🟠 高", 3: "🟡 中", 4: "🔵 低"}
                                        for act in record.get("emergency_plan_actions", []):
                                            p_label = priority_labels.get(act["priority"], f"优先级{act['priority']}")
                                            with st.expander(f"{p_label} | {act['description']}", expanded=False):
                                                st.markdown(f"**措施类型**: {act['action_type']}")
                                                st.markdown(f"**实施对象**: {act['target_name']} (ID:{act['target_id']})")
                                                if act.get("detail"):
                                                    st.markdown(f"**详细说明**: {act['detail']}")
                                                if act.get("estimated_effect"):
                                                    st.success(f"💡 预期效果: {act['estimated_effect']}")
                                        st.session_state["_pending_recall"] = None
                                    st.divider()

with tabs[8]:
    st.subheader("📝 管网数据编辑")
    net = st.session_state.network
    if net is None:
        st.info("💡 请先在左侧加载示例管网或上传JSON文件，然后可在此处编辑")
    else:
        tab_nodes, tab_pipes, tab_export = st.tabs(["节点编辑", "管段编辑", "导出JSON"])
        with tab_nodes:
            st.markdown("##### 节点列表（可直接在表格中修改）")
            node_data = []
            for nid, node in net.nodes.items():
                node_data.append({
                    "id": nid, "name": node.name,
                    "type": NODE_TYPE_CN.get(node.type, node.type),
                    "elevation": node.elevation,
                    "x": node.x, "y": node.y,
                    "supply_pressure": node.supply_pressure,
                    "return_pressure": node.return_pressure,
                    "design_flow": node.design_flow,
                    "rated_capacity_MW": (node.rated_capacity or 0) / 1e6,
                })
            df_nodes_edit = pd.DataFrame(node_data)
            edited_nodes = st.data_editor(df_nodes_edit, num_rows="dynamic",
                                          use_container_width=True, height=500,
                                          key="node_editor")
            if st.button("💾 保存节点修改", use_container_width=True, type="secondary"):
                try:
                    new_net = PipeNetwork()
                    new_net.environment_temperature = net.environment_temperature
                    new_net.water_specific_heat = net.water_specific_heat
                    for _, row in edited_nodes.iterrows():
                        ntype = str(row.get("type", NODE_TYPE_BRANCH))
                        if ntype in CN_TO_NODE_TYPE:
                            ntype = CN_TO_NODE_TYPE[ntype]
                        rated_cap = float(row.get("rated_capacity_MW", 0) or 0) * 1e6
                        n = Node(
                            id=int(row["id"]), name=str(row.get("name", "")),
                            type=ntype,
                            elevation=float(row.get("elevation", 0) or 0),
                            x=row.get("x"), y=row.get("y"),
                            supply_pressure=row.get("supply_pressure"),
                            return_pressure=row.get("return_pressure"),
                            design_flow=row.get("design_flow"),
                            rated_capacity=rated_cap if rated_cap > 0 else None,
                        )
                        new_net.add_node(n)
                    for pid, pipe in net.pipes.items():
                        new_net.add_pipe(pipe)
                    valid, errs = new_net.validate()
                    if valid:
                        st.session_state.network = new_net
                        st.success("✅ 节点修改已保存")
                    else:
                        st.error("❌ 数据校验失败：" + "; ".join(errs))
                except Exception as e:
                    st.error(f"❌ 保存失败：{e}")
        with tab_pipes:
            st.markdown("##### 管段列表（可直接在表格中修改）")
            pipe_data = []
            for pid, pipe in net.pipes.items():
                pipe_data.append({
                    "id": pid, "name": pipe.name,
                    "start_node_id": pipe.start_node_id,
                    "end_node_id": pipe.end_node_id,
                    "diameter_mm": pipe.diameter * 1000,
                    "length_m": pipe.length,
                    "material": PIPE_MATERIAL_CN.get(pipe.material, pipe.material),
                    "pipe_age_years": pipe.pipe_age,
                    "insulation": INSULATION_CN.get(pipe.insulation_material, pipe.insulation_material),
                    "insulation_thickness_mm": pipe.insulation_thickness * 1000,
                    "burial_depth_m": pipe.burial_depth,
                    "has_valve": pipe.has_valve, "valve_kv": pipe.valve_kv,
                    "has_pump": pipe.has_pump, "rated_head_m": pipe.rated_head,
                    "local_len_ratio": pipe.equivalent_local_length_ratio,
                })
            df_pipes_edit = pd.DataFrame(pipe_data)
            edited_pipes = st.data_editor(df_pipes_edit, num_rows="dynamic",
                                          use_container_width=True, height=500,
                                          key="pipe_editor")
            if st.button("💾 保存管段修改", use_container_width=True, type="secondary"):
                try:
                    new_net = PipeNetwork()
                    new_net.environment_temperature = net.environment_temperature
                    new_net.water_specific_heat = net.water_specific_heat
                    for nid, node in net.nodes.items():
                        new_net.add_node(node)
                    for _, row in edited_pipes.iterrows():
                        mat = str(row.get("material", PIPE_MATERIAL_STEEL))
                        if mat in CN_TO_PIPE_MATERIAL:
                            mat = CN_TO_PIPE_MATERIAL[mat]
                        ins = str(row.get("insulation", INSULATION_MATERIAL_POLYURETHANE))
                        if ins in CN_TO_INSULATION:
                            ins = CN_TO_INSULATION[ins]
                        has_p = bool(row.get("has_pump", False))
                        rh = row.get("rated_head_m")
                        pcurve = None
                        if has_p and rh:
                            rhf = float(rh)
                            qmax = rhf / 30.0
                            pcurve = [(0, rhf * 1.2), (qmax * 0.5, rhf), (qmax, rhf * 0.3)]
                        p = Pipe(
                            id=int(row["id"]), name=str(row.get("name", "")),
                            start_node_id=int(row["start_node_id"]),
                            end_node_id=int(row["end_node_id"]),
                            diameter=float(row.get("diameter_mm", 200) or 200) / 1000.0,
                            length=float(row.get("length_m", 100) or 100),
                            material=mat,
                            pipe_age=float(row.get("pipe_age_years", 10) or 10),
                            insulation_material=ins,
                            insulation_thickness=float(row.get("insulation_thickness_mm", 50) or 50) / 1000.0,
                            burial_depth=float(row.get("burial_depth_m", 1.5) or 1.5),
                            has_valve=bool(row.get("has_valve", False)),
                            valve_kv=row.get("valve_kv"),
                            has_pump=has_p, rated_head=float(rh) if rh and has_p else None,
                            pump_efficiency_curve=pcurve,
                            equivalent_local_length_ratio=float(row.get("local_len_ratio", 0.1) or 0.1),
                        )
                        new_net.add_pipe(p)
                    valid, errs = new_net.validate()
                    if valid:
                        st.session_state.network = new_net
                        st.success("✅ 管段修改已保存")
                    else:
                        st.error("❌ 数据校验失败：" + "; ".join(errs))
                except Exception as e:
                    st.error(f"❌ 保存失败：{e}")
        with tab_export:
            st.markdown("##### 导出当前管网为JSON")
            export_data = {
                "export_time": datetime.now().isoformat(),
                "environment_temperature": net.environment_temperature,
                "water_specific_heat": net.water_specific_heat,
                "nodes": [], "pipes": [],
            }
            for nid, node in net.nodes.items():
                export_data["nodes"].append({
                    "id": node.id, "name": node.name,
                    "type": NODE_TYPE_CN.get(node.type, node.type),
                    "elevation": node.elevation, "x": node.x, "y": node.y,
                    "supply_pressure": node.supply_pressure,
                    "return_pressure": node.return_pressure,
                    "design_flow": node.design_flow,
                    "rated_capacity": node.rated_capacity,
                })
            for pid, pipe in net.pipes.items():
                export_data["pipes"].append({
                    "id": pipe.id, "name": pipe.name,
                    "start_node_id": pipe.start_node_id,
                    "end_node_id": pipe.end_node_id,
                    "diameter": pipe.diameter, "length": pipe.length,
                    "material": PIPE_MATERIAL_CN.get(pipe.material, pipe.material),
                    "pipe_age": pipe.pipe_age,
                    "insulation_material": INSULATION_CN.get(pipe.insulation_material, pipe.insulation_material),
                    "insulation_thickness": pipe.insulation_thickness,
                    "burial_depth": pipe.burial_depth,
                    "has_valve": pipe.has_valve, "valve_kv": pipe.valve_kv,
                    "has_pump": pipe.has_pump, "rated_head": pipe.rated_head,
                    "pump_efficiency_curve": pipe.pump_efficiency_curve,
                    "equivalent_local_length_ratio": pipe.equivalent_local_length_ratio,
                })
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                "📥 下载管网JSON文件", json_str,
                file_name=f"管网数据_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True,
            )
            with st.expander("👁️ 预览JSON内容"):
                st.code(json_str, language="json")
    st.divider()
    st.subheader("🌡️ 各热源供水温度设置")
    if st.session_state.network is not None:
        src_nodes = st.session_state.network.get_nodes_by_type(NODE_TYPE_SOURCE)
        cols = st.columns(min(3, max(len(src_nodes), 1)))
        for i, sn in enumerate(src_nodes):
            with cols[i % len(cols)]:
                default_t = st.session_state.source_temps.get(sn.id, 110.0 if i == 0 else 108.0)
                t = st.slider(f"{sn.name} 供水温度 (°C)", 60.0, 150.0, float(default_t), 1.0, key=f"src_t_{sn.id}")
                st.session_state.source_temps[sn.id] = t

with tabs[9]:
    st.subheader("📄 供热运行分析报告导出")
    if st.session_state.results is None:
        st.info("请先运行计算后再导出报告")
    else:
        net = st.session_state.network
        r = st.session_state.results
        report_title = st.text_input("报告标题", "城市集中供热管网运行分析报告")
        col_a, col_b = st.columns(2)
        with col_a:
            gen_pdf = st.button("📄 生成PDF报告", type="primary", use_container_width=True)
        with col_b:
            if st.session_state.network is not None:
                summary_csv = io.StringIO()
                summary_data = {
                    "项目": [
                        "节点总数", "管段总数", "热源数", "末端用户数", "管网总长度(m)",
                        "总供热量(MW)", "总热损失(MW)", "热损失率(%)",
                        "泵站总功率(kW)", "供热单耗(kWh/GJ)", "计算迭代次数", "收敛状态",
                    ],
                    "数值": [
                        len(net.nodes), len(net.pipes),
                        len(net.get_nodes_by_type(NODE_TYPE_SOURCE)),
                        len(net.get_nodes_by_type(NODE_TYPE_END_USER)),
                        f"{sum(p.length for p in net.pipes.values()):.0f}",
                        f"{r.total_heat_supplied / 1e6:.2f}",
                        f"{r.total_heat_loss / 1e6:.2f}",
                        f"{r.heat_loss_rate:.2f}",
                        f"{r.total_pump_power:.2f}",
                        f"{r.specific_energy_consumption:.2f}",
                        r.iterations, "是" if r.converged else "否",
                    ],
                }
                pd.DataFrame(summary_data).to_csv(summary_csv, index=False, encoding="utf-8-sig")
                st.download_button(
                    "📊 下载汇总数据(CSV)", summary_csv.getvalue().encode('utf-8-sig'),
                    file_name=f"运行汇总_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True,
                )
        if gen_pdf:
            with st.spinner("🔨 正在生成PDF报告，可能需要几秒钟..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        network=net, results=r,
                        flow_ratios=st.session_state.flow_ratios,
                        valve_suggestions=st.session_state.valve_suggestions,
                        source_ratios=st.session_state.source_ratios,
                        title=report_title,
                    )
                    if pdf_bytes is None:
                        st.error("❌ PDF生成失败，请检查是否已安装 reportlab 库：pip install reportlab")
                    else:
                        st.success("✅ PDF报告生成成功！")
                        fname = f"供热运行分析报告_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(
                            "📥 下载PDF报告", pdf_bytes,
                            file_name=fname, mime="application/pdf",
                            use_container_width=True, type="primary",
                        )
                        with st.expander("📑 报告内容预览"):
                            st.info("PDF包含以下章节：\n"
                                    "1. 管网参数汇总（节点/管段/热源统计）\n"
                                    "2. 水力计算结果表（管段+节点）\n"
                                    "3. 热力计算结果表（管段+节点）\n"
                                    "4. 能耗指标汇总+各泵站耗电明细\n"
                                    "5. 水力平衡分析+阀门调节建议\n"
                                    "6. 节能优化建议")
                except Exception as e:
                    st.error(f"❌ PDF生成失败：{e}")
                    import traceback
                    with st.expander("错误详情"):
                        st.code(traceback.format_exc())

st.divider()
st.caption(
    "🔥 城市集中供热管网水力热力耦合计算系统 v1.0 | "
    "计算方法: 达西-魏斯巴赫 + Colebrook-White + Newton-Raphson + 多层圆柱壁传热模型 | "
    f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
