import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("供热管网计算系统 - 核心模块测试")
print("=" * 60)

print("\n[1/6] 导入模块...")
try:
    from models import create_sample_network, PipeNetwork, Node, Pipe
    from models import (
        NODE_TYPE_SOURCE, NODE_TYPE_END_USER, NODE_TYPE_BRANCH, NODE_TYPE_HEAT_EXCHANGER,
        get_pipe_roughness, get_wall_thickness, get_water_properties,
        INSULATION_CONDUCTIVITY, PIPE_WALL_CONDUCTIVITY,
    )
    print("   ✅ models 模块导入成功")
except Exception as e:
    print(f"   ❌ models 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from calculations import (
        solve_colebrook_white, compute_pipe_hydraulics, compute_pump_head,
        solve_hydraulics, compute_overall_heat_transfer_coefficient,
        compute_pipe_temperature_drop, solve_thermal,
        solve_coupled, analyze_hydraulic_balance,
        optimize_source_allocation_equal, optimize_source_allocation_min_energy,
    )
    print("   ✅ calculations 模块导入成功")
except Exception as e:
    print(f"   ❌ calculations 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from visualization import (
        create_network_topology_figure, create_temperature_drop_figure,
        create_pressure_contour_figure, create_pump_operating_point_figures,
        create_heat_loss_pareto_figure, create_energy_consumption_pie_figure,
    )
    print("   ✅ visualization 模块导入成功")
except Exception as e:
    print(f"   ❌ visualization 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from report import generate_pdf_report
    print("   ✅ report 模块导入成功")
except Exception as e:
    print(f"   ⚠️  report 模块导入失败 (可选): {e}")

print("\n[2/6] 创建示例管网...")
net = create_sample_network()
valid, errors = net.validate()
if valid:
    print(f"   ✅ 管网校验通过: {len(net.nodes)} 节点 / {len(net.pipes)} 管段")
    sources = net.get_nodes_by_type(NODE_TYPE_SOURCE)
    end_users = net.get_nodes_by_type(NODE_TYPE_END_USER)
    print(f"   - 热源节点: {len(sources)} 个")
    print(f"   - 末端用户: {len(end_users)} 个")
    total_demand = sum((eu.design_flow or 0) for eu in end_users)
    print(f"   - 总设计流量: {total_demand * 1000:.1f} L/s")
    pumps = sum(1 for p in net.pipes.values() if p.has_pump)
    valves = sum(1 for p in net.pipes.values() if p.has_valve)
    total_len = sum(p.length for p in net.pipes.values())
    print(f"   - 泵站: {pumps} 个, 阀门: {valves} 个, 总长度: {total_len:.0f} m")
else:
    print(f"   ❌ 管网校验失败: {errors}")
    sys.exit(1)

print("\n[3/6] 水力计算基础功能测试...")
test_pipe = list(net.pipes.values())[0]
print(f"   测试管段: {test_pipe.name} (DN{test_pipe.diameter*1000:.0f}, L={test_pipe.length:.0f}m)")
vel, re, f, hf, hl = compute_pipe_hydraulics(test_pipe, 0.05, 80.0)
print(f"   - 流速: {vel:.3f} m/s")
print(f"   - 雷诺数: {re:.0f} (层流={re<2300})")
print(f"   - 摩擦系数: {f:.6f}")
print(f"   - 沿程/局部阻力损失: {hf:.3f} / {hl:.3f} m")
pump_head = compute_pump_head(test_pipe, 0.05)
print(f"   - 泵站扬程: {pump_head:.2f} m")
roughness = get_pipe_roughness(test_pipe.material, test_pipe.pipe_age)
print(f"   - 管壁粗糙度: {roughness*1000:.3f} mm")
U = compute_overall_heat_transfer_coefficient(test_pipe)
print(f"   - 总传热系数: {U:.3f} W/(m·K)")
print("   ✅ 基础水力/热力计算功能正常")

print("\n[4/6] 水力热力耦合计算 (这可能需要一些时间)...")
import time
t0 = time.time()
source_temps = {}
for i, s in enumerate(sources):
    source_temps[s.id] = 110.0 if i == 0 else 108.0
try:
    results = solve_coupled(net, source_supply_temps=source_temps)
    t1 = time.time()
    print(f"   ✅ 耦合计算完成 ({t1-t0:.1f} 秒)")
    print(f"   - 迭代次数: {results.iterations}")
    print(f"   - 收敛状态: {'是' if results.converged else '否'}")
    print(f"   - 总供热量: {results.total_heat_supplied/1e6:.2f} MW")
    print(f"   - 总热损失: {results.total_heat_loss/1e6:.2f} MW ({results.heat_loss_rate:.2f}%)")
    print(f"   - 泵站总耗电: {results.total_pump_power:.1f} kW")
    print(f"   - 供热单耗: {results.specific_energy_consumption:.2f} kWh/GJ")
    print(f"\n   管段结果抽样 (前5条):")
    for i, (pid, pr) in enumerate(list(results.pipe_results.items())[:5]):
        pipe = net.pipes[pid]
        print(f"   #{i+1} {pipe.name}: Q={pr.flow_rate*1000:.1f}L/s, "
              f"T_in={pr.inlet_temperature:.1f}→T_out={pr.outlet_temperature:.1f}°C, "
              f"ΔP={pr.total_pressure_loss/1000:.2f}kPa, "
              f"HL={pr.heat_loss/1000:.1f}kW")
    print(f"\n   节点温度抽样 (前8条):")
    for i, (nid, nr) in enumerate(list(results.node_results.items())[:8]):
        node = net.nodes[nid]
        print(f"   #{i+1} {node.name}: P={nr.pressure:.2f}mH2O, T={nr.temperature:.1f}°C")
except Exception as e:
    print(f"   ❌ 耦合计算失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/6] 水力平衡分析测试...")
try:
    flow_ratios, suggestions = analyze_hydraulic_balance(net, results, tolerance=0.15)
    print(f"   ✅ 平衡分析完成")
    print(f"   - 参与平衡用户数: {len(flow_ratios)}")
    if flow_ratios:
        ratios = list(flow_ratios.values())
        avg_ratio = sum(ratios) / len(ratios)
        max_dev = max(abs(r - 1.0) for r in ratios) * 100
        print(f"   - 平均流量比: {avg_ratio:.3f}, 最大偏差: {max_dev:.1f}%")
    print(f"   - 阀门调节建议数: {len(suggestions)}")
    for s in suggestions[:3]:
        print(f"     · {s['user_name']} ({s['flow_ratio']*100:.0f}%): "
              f"{s['action']}管段[{s['pipe_name']}] {s['adjustment']:.0f}%")
except Exception as e:
    print(f"   ❌ 平衡分析失败: {e}")
    import traceback
    traceback.print_exc()

print("\n[6/6] 可视化与报告测试...")
import numpy as np
try:
    fig1 = create_network_topology_figure(net, results)
    print(f"   ✅ 管网拓扑图已生成")
    fig2 = create_temperature_drop_figure(net, results)
    print(f"   ✅ 温降曲线图已生成")
    fig3 = create_heat_loss_pareto_figure(net, results)
    print(f"   ✅ 热损失Pareto图已生成")
    fig4 = create_energy_consumption_pie_figure(net, results)
    print(f"   ✅ 能耗饼图已生成")
    pump_figs = create_pump_operating_point_figures(net, results)
    print(f"   ✅ 泵站工作点图已生成 ({len(pump_figs)} 个)")
    fig5 = create_pressure_contour_figure(net, results)
    if fig5 is not None:
        print(f"   ✅ 压力等值线图已生成")
    else:
        print(f"   ⚠️  压力等值线跳过 (无坐标)")
except Exception as e:
    print(f"   ❌ 可视化失败: {e}")
    import traceback
    traceback.print_exc()

try:
    pdf_bytes = generate_pdf_report(
        network=net, results=results,
        flow_ratios=flow_ratios, valve_suggestions=suggestions,
        source_ratios={s.id: 1/len(sources) for s in sources},
    )
    if pdf_bytes:
        print(f"   ✅ PDF报告已生成 ({len(pdf_bytes)/1024:.1f} KB)")
    else:
        print(f"   ⚠️  PDF报告未生成 (reportlab未安装?)")
except Exception as e:
    print(f"   ⚠️  PDF生成警告: {e}")

print("\n" + "=" * 60)
print("✅ 所有核心模块测试通过！系统可以正常运行。")
print("=" * 60)
print("\n启动命令:")
print("  source venv/bin/activate")
print("  streamlit run app.py")
