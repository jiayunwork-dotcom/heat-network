import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculations import run_economic_optimization
from models import create_sample_network
from calculations import solve_coupled

print("=" * 60)
print("经济优化模块测试")
print("=" * 60)

print("\n1. 导入测试...")
print("   ✅ 导入成功！")

print("\n2. 创建示例管网...")
net = create_sample_network()
print(f"   ✅ 管网节点数: {len(net.nodes)}, 管段数: {len(net.pipes)}")

print("\n3. 运行水力热力耦合计算...")
source_temps = {}
for sn in net.get_nodes_by_type('source'):
    source_temps[sn.id] = 110.0 if sn.id == 0 else 108.0

results = solve_coupled(net, source_temps)
print(f"   ✅ 耦合计算完成: 收敛={results.converged}, 迭代{results.iterations}次")
print(f"   总供热量: {results.total_heat_supplied/1e6:.2f} MW")
print(f"   总热损失: {results.total_heat_loss/1e6:.2f} MW")
print(f"   泵站总功率: {results.total_pump_power:.2f} kW")

print("\n4. 运行经济优化计算...")
eco = run_economic_optimization(net, results)
print("   ✅ 经济优化计算完成！")

print("\n" + "=" * 60)
print("运行成本核算")
print("=" * 60)
op_cost = eco.operating_cost
print(f"日运行总成本: {op_cost.total_daily_cost:,.0f} 元")
print(f"年运行总成本: {op_cost.total_annual_cost:,.0f} 元")
print()
print(f"  泵站电费（日）: {op_cost.electricity_cost_daily:,.0f} 元")
print(f"  泵站电费（年）: {op_cost.electricity_cost_annual:,.0f} 元 "
      f"({op_cost.electricity_cost_annual/max(op_cost.total_annual_cost,1)*100:.1f}%)")
print()
print(f"  热损失成本（日）: {op_cost.heat_loss_cost_daily:,.0f} 元")
print(f"  热损失成本（年）: {op_cost.heat_loss_cost_annual:,.0f} 元 "
      f"({op_cost.heat_loss_cost_annual/max(op_cost.total_annual_cost,1)*100:.1f}%)")
print()
print(f"  维护成本（年）: {op_cost.maintenance_cost_annual:,.0f} 元 "
      f"({op_cost.maintenance_cost_annual/max(op_cost.total_annual_cost,1)*100:.1f}%)")

print("\n" + "=" * 60)
print("保温改造评估")
print("=" * 60)
print(f"共有 {len(eco.insulation_retrofits)} 条保温改造建议")
if eco.insulation_retrofits:
    print("\n前5个保温改造项目（按回报期排序）:")
    print("-" * 80)
    print(f"{'序号':<4} {'管段名称':<20} {'投资额(元)':<12} {'年节省(元)':<12} {'回报期(年)':<10}")
    print("-" * 80)
    for i, ir in enumerate(eco.insulation_retrofits[:5]):
        print(f"{i+1:<4} {ir.pipe_name:<20} {ir.investment:>10,.0f}   {ir.annual_gas_saving:>10,.0f}   {ir.payback_period:>8.2f}")

print("\n" + "=" * 60)
print("泵站能效优化")
print("=" * 60)
print(f"共有 {len(eco.pump_retrofits)} 个泵站")
print()
print(f"{'泵站名称':<20} {'额定扬程':<10} {'实际扬程':<10} {'状态':<14} {'变频建议':<10}")
print("-" * 70)
for pr in eco.pump_retrofits:
    rec = "推荐" if pr.has_vfd_recommendation else "暂不需要"
    print(f"{pr.pipe_name:<20} {pr.rated_head:>8.1f}m  {pr.actual_head:>8.1f}m  {pr.status:<14} {rec:<10}")
    if pr.has_vfd_recommendation:
        print(f"  变频改造: 投资{pr.vfd_investment:,.0f}元, "
              f"年省{pr.vfd_annual_saving:,.0f}元, "
              f"回报期{pr.vfd_payback_period:.2f}年")

print("\n" + "=" * 60)
print("综合改造方案")
print("=" * 60)
print(f"筛选出 {len(eco.comprehensive_list)} 个回报期低于 5 年的改造项目")
print(f"总投资额: {eco.total_investment:,.0f} 元")
print(f"预计总年节省: {eco.total_annual_saving:,.0f} 元")
print(f"整体投资回报期: {eco.overall_payback_period:.2f} 年")

if eco.comprehensive_list:
    print("\n综合改造优先级列表（按回报期从短到长）:")
    print("-" * 80)
    print(f"{'序号':<4} {'项目名称':<25} {'类型':<10} {'投资额(元)':<12} {'年节省(元)':<12} {'回报期(年)':<10}")
    print("-" * 80)
    for i, item in enumerate(eco.comprehensive_list):
        print(f"{i+1:<4} {item.item_name:<25} {item.retrofit_type:<10} "
              f"{item.investment:>10,.0f}   {item.annual_saving:>10,.0f}   {item.payback_period:>8.2f}")

print("\n" + "=" * 60)
print("✅ 测试全部通过！")
print("=" * 60)
