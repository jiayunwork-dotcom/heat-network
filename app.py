import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime

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
)
from visualization import (
    create_network_topology_figure,
    create_temperature_drop_figure,
    create_pressure_contour_figure,
    create_pump_operating_point_figures,
    create_heat_loss_pareto_figure,
    create_energy_consumption_pie_figure,
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
    "📝 管网数据编辑", "📄 报告导出"
])

with tabs[0]:
    st.subheader("📊 运行总览仪表盘")
    if st.session_state.results is None:
        st.info("👈 请先加载示例管网或导入数据，然后点击侧边栏的计算按钮")
    else:
        r: NetworkResults = st.session_state.results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总供热量", f"{r.total_heat_supplied/1e6:.2f}", "MW", delta_color="off")
        with col2:
            st.metric("管网总热损失", f"{r.total_heat_loss/1e6:.2f}", "MW",
                      delta=f"{r.heat_loss_rate:.1f}%")
        with col3:
            st.metric("泵站总耗电", f"{r.total_pump_power:.1f}", "kW", delta_color="off")
        with col4:
            st.metric("供热单耗", f"{r.specific_energy_consumption:.2f}", "kWh/GJ",
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
            st.metric("管网总长度", f"{total_len:.0f}", "m")
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
        tn_rows = []
        for nid, nr in r.node_results.items():
            node = net.nodes[nid]
            tn_rows.append({
                "节点编号": nid,
                "节点名称": node.name,
                "类型": NODE_TYPE_CN.get(node.type, node.type),
                "温度(°C)": f"{nr.temperature:.2f}",
                "温差(与环境)": f"{nr.temperature - net.environment_temperature:.1f}°C",
            })
        st.markdown("##### 🌡️ 各节点温度分布")
        df_tn = pd.DataFrame(tn_rows)
        st.dataframe(df_tn, use_container_width=True, hide_index=True, height=380)
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
                st.metric("平均偏差", f"{avg_dev:.1f}", "%")
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

with tabs[7]:
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
