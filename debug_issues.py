import sys
sys.path.insert(0, '.')
from models import create_sample_network
from calculations import compute_pump_head, compute_pipe_temperature_drop, compute_overall_heat_transfer_coefficient
from models.material_properties import get_water_properties

net = create_sample_network()
pipe0 = net.pipes[0]
print('管段0 (S1主干线1):')
print(f'  管径: {pipe0.diameter} m')
print(f'  额定扬程: {pipe0.rated_head} m')
if pipe0.pump_efficiency_curve:
    print(f'  泵曲线点数: {len(pipe0.pump_efficiency_curve)}')
    print(f'  最后一点: {pipe0.pump_efficiency_curve[-1]}')
    print(f'  所有点:')
    for q, h in pipe0.pump_efficiency_curve:
        print(f'    Q={q:.4f} m3/s, H={h:.2f} m')

q = 0.417
print(f'\n流量 {q} m3/s ({q*1000:.0f} L/s) 时扬程: {compute_pump_head(pipe0, q)} m')
print(f'流量 0.38 m3/s (380 L/s) 时扬程: {compute_pump_head(pipe0, 0.38)} m')
print(f'流量 0.5 m3/s (500 L/s) 时扬程: {compute_pump_head(pipe0, 0.5)} m')

print('\n--- 热损失分析 ---')
U = compute_overall_heat_transfer_coefficient(pipe0)
print(f'传热系数U (每米): {U:.4f} W/(m*K)')
print(f'管长: {pipe0.length} m')

t_in = 110.0
t_env = 5.0
cp = 4186
density, _ = get_water_properties(t_in)
mass_flow = q * density
print(f'进水温度: {t_in}°C, 环境温度: {t_env}°C')
print(f'质量流量: {mass_flow:.1f} kg/s')

out_t, drop, hl = compute_pipe_temperature_drop(pipe0, q, t_in, t_env, cp)
print(f'出口温度: {out_t:.4f} °C')
print(f'温降: {drop:.4f} °C')
print(f'热损失: {hl/1000:.2f} kW')

heat_supplied = mass_flow * cp * (t_in - t_env)
print(f'总供热量: {heat_supplied/1e6:.2f} MW')
print(f'热损失率: {hl/heat_supplied*100:.4f} %')

# 手动计算一下热损失应该多少
wall_thick = 0.008  # 钢管DN600
d_inner = pipe0.diameter  # 0.6m
d_outer = d_inner + 2*wall_thick + 2*pipe0.insulation_thickness
print(f'\n--- 手动验证传热 ---')
print(f'内径: {d_inner} m, 外径: {d_outer:.4f} m')
print(f'保温厚度: {pipe0.insulation_thickness*1000:.0f} mm')
A_outer = 3.14159 * d_outer * pipe0.length
print(f'外表面积: {A_outer:.1f} m2')
delta_t = (t_in + out_t)/2 - t_env
print(f'平均温差: {delta_t:.1f} °C')
print(f'U*L = {U*pipe0.length:.1f} W/K')
hl_manual = U * pipe0.length * delta_t
print(f'手动计算热损失: {hl_manual/1000:.2f} kW')

# 热损失率低的根源：流量太大，质量热容太大
print(f'\n--- 分析: 为何热损失率低? ---')
print(f'm*Cp = {mass_flow*cp/1000:.0f} kW/K')
print(f'UA = {U*pipe0.length:.1f} W/K = {U*pipe0.length/1000:.2f} kW/K')
print(f'指数项 UA/(mCp) = {U*pipe0.length/(mass_flow*cp):.6f}')
print(f'exp(-UA/(mCp)) = {2.71828 ** (-U*pipe0.length/(mass_flow*cp)):.6f}')

# 如果希望热损失率达到10%，大约需要温降10°C（从110到100）
# 或者说 m*Cp*10 = UA*delta_T_log
target_loss_pct = 0.10
print(f'\n--- 达到10%热损失的估算 ---')
# 简化：目标热损失应该是多少
target_hl = heat_supplied * target_loss_pct
print(f'目标热损失: {target_hl/1000:.1f} kW')
print(f'实际热损失: {hl/1000:.1f} kW')
print(f'当前传热系数U: {U:.3f} W/(m*K)')
# 若要10%损失，需要多少传热系数？
# 简化估算：hl = U * A * delta_T => U = hl / (A * delta_T)
delta_T_avg = t_in - t_env
target_U_per_m = target_hl / (pipe0.length * delta_T_avg)
print(f'需要的U值: {target_U_per_m:.3f} W/(m*K)  (当前为{U:.3f})')
print(f'当前保温厚度: {pipe0.insulation_thickness*1000:.0f} mm 聚氨酯')

# 看看小直径管线的情况
print(f'\n--- 检查小直径管线热损失 ---')
# 用户供水管 pid=12 (DN150, L=100m)
pipe_small = net.pipes[12]
print(f'管段{pipe_small.id}: {pipe_small.name}')
print(f'  管径: {pipe_small.diameter*1000:.0f} mm, 长度: {pipe_small.length} m')
print(f'  保温: {pipe_small.insulation_material}, {pipe_small.insulation_thickness*1000:.0f} mm')
U_s = compute_overall_heat_transfer_coefficient(pipe_small)
print(f'  U (每米): {U_s:.3f} W/(m*K)')
# 设计流量
user_node = net.nodes[pipe_small.end_node_id]
q_s = user_node.design_flow or 0.03
print(f'  设计流量: {q_s*1000:.0f} L/s')
d_s, _ = get_water_properties(90.0)
m_s = q_s * d_s
out_ts, drops, hls = compute_pipe_temperature_drop(pipe_small, q_s, 85.0, 5.0, cp)
heat_s = m_s * cp * (85 - 5)
print(f'  质量流量: {m_s:.1f} kg/s')
print(f'  温降: {drops:.2f} °C')
print(f'  热损失: {hls:.1f} W = {hls/1000:.3f} kW')
print(f'  热损失率: {hls/heat_s*100:.2f} %')
