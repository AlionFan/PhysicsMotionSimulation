import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Any
import io

# ==================== 1. 配置与样式 (已适配暗色模式) ====================

def setup_page():
    st.set_page_config(
        page_title="3D物理仿真 Pro - 无人机投掷",
        page_icon="🚁",
        layout="wide"
    )

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    body { font-family: 'Inter', sans-serif; }
    .stTitle { display: none !important; }

    /* 顶部导航 */
    .compact-header {
        position: sticky; top: 0;
        background: var(--background-color);
        backdrop-filter: blur(20px);
        z-index: 999;
        padding: 0.8rem 1.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: var(--shadow-sm);
        display: flex; align-items: center; justify-content: space-between;
    }
    .compact-header h1 {
        font-size: 1.5rem; margin: 0; font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 卡片样式 */
    .feature-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
        padding: 1.5rem; border-radius: 12px;
        box-shadow: var(--shadow-sm);
        transition: transform 0.3s ease;
    }
    .feature-card h3 { color: var(--text-color) !important; font-size: 1.1rem; font-weight: 600; }
    .feature-card p { color: var(--text-color); opacity: 0.8; font-size: 0.9rem; }

    /* 指标与图表 */
    [data-testid="stMetricValue"] { color: var(--text-color) !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: var(--text-color) !important; opacity: 0.7; }
    .plotly-graph-div { border-radius: 12px; box-shadow: var(--shadow-sm); border: 1px solid rgba(128, 128, 128, 0.1); }

    /* 按钮 */
    .stButton > button[kind="primary"] { background: var(--primary-gradient) !important; border: none; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# ==================== 2. 物理引擎 (升级版) ====================

CONSTANTS = {'G': 9.81, 'AIR_DENSITY': 1.225, 'DRAG_COEFFICIENT': 0.47}

def calculate_ball_properties(radius_m: float, density: float) -> Tuple[float, float, float]:
    volume = (4/3) * np.pi * radius_m**3
    mass = density * volume
    area = np.pi * radius_m**2
    return volume, mass, area

def run_simulation(params: Dict[str, Any]):
    """核心仿真循环"""
    dt = params['dt']
    wind_vel = np.array([params['wind_x'], params['wind_y'], params['wind_z']])

    vol, mass, area = calculate_ball_properties(params['radius'], params['density'])
    pos = np.array([0.0, 0.0, params['height']])
    vel = np.array([params['v0_x'], params['v0_y'], params['v0_z']])

    # 历史记录
    trajectory = [pos.copy()]
    velocities = [vel.copy()]
    time_points = [0.0]
    energies = []

    t = 0.0
    gravity_vec = np.array([0.0, 0.0, -mass * CONSTANTS['G']])

    def get_energy(m, v, h):
        ke = 0.5 * m * np.linalg.norm(v)**2
        pe = m * CONSTANTS['G'] * h
        return ke, pe, ke + pe

    energies.append(get_energy(mass, vel, pos[2]))

    while pos[2] > 0 and t < 60:
        # 力学计算
        rel_vel = wind_vel - vel
        rel_speed = np.linalg.norm(rel_vel)
        drag_force = 0.5 * CONSTANTS['AIR_DENSITY'] * CONSTANTS['DRAG_COEFFICIENT'] * area * rel_speed * rel_vel if rel_speed > 0 else np.zeros(3)
        total_force = gravity_vec + drag_force

        # 欧拉积分
        acc = total_force / mass
        vel += acc * dt
        pos += vel * dt
        t += dt

        trajectory.append(pos.copy())
        velocities.append(vel.copy())
        time_points.append(t)
        energies.append(get_energy(mass, vel, pos[2]))

    # 数据转换
    traj_arr = np.array(trajectory)
    vel_arr = np.array(velocities)
    energy_arr = np.array(energies)
    time_arr = np.array(time_points)

    # 落地修正
    landing_point = traj_arr[-1]
    if traj_arr[-1, 2] < 0 and len(traj_arr) > 1:
        prev_z = traj_arr[-2, 2]
        curr_z = traj_arr[-1, 2]
        ratio = prev_z / (prev_z - curr_z)
        landing_point = traj_arr[-2] + ratio * (traj_arr[-1] - traj_arr[-2])
        landing_point[2] = 0.0
        flight_time = time_arr[-2] + ratio * dt
    else:
        landing_point[2] = max(0, landing_point[2])
        flight_time = time_arr[-1]

    return {
        'traj': traj_arr,
        'vel': vel_arr,
        'energy': energy_arr,
        'time': time_arr,
        'landing': landing_point,
        'flight_time': flight_time,
        'mass': mass,
        'area': area,
        'params': params # 保存参数以便对比
    }

# ==================== 3. 可视化组件 (含修复) ====================

def create_3d_plot(current_res: Dict, prev_res: Dict, anim_speed: float, show_anim: bool):
    """3D 绘图，支持双轨迹对比"""
    traj = current_res['traj']
    landing = current_res['landing']

    # 修复：从 current_res 中获取 dt，而不是直接使用未定义的 params
    dt = current_res['params']['dt']

    fig = make_subplots(specs=[[{'type': 'scatter3d'}]])

    # 1. 绘制上一条轨迹 (如果有)
    if prev_res:
        p_traj = prev_res['traj']
        fig.add_trace(go.Scatter3d(
            x=p_traj[:, 0], y=p_traj[:, 1], z=p_traj[:, 2],
            mode='lines', name='上次轨迹',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=3, dash='dash')
        ))
        fig.add_trace(go.Scatter3d(
            x=[prev_res['landing'][0]], y=[prev_res['landing'][1]], z=[prev_res['landing'][2]],
            mode='markers', name='上次落点', marker=dict(color='grey', size=5, symbol='x')
        ))

    # 2. 地面
    grid_size = max(abs(landing[0]), abs(landing[1]), 20) * 1.5
    x = np.linspace(-grid_size, grid_size, 20)
    X, Y = np.meshgrid(x, x)
    fig.add_trace(go.Surface(x=X, y=Y, z=X*0, colorscale='Greys', showscale=False, opacity=0.2, name='地面'))

    # 3. 当前轨迹
    fig.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
        mode='lines', name='当前轨迹', line=dict(color='#667eea', width=5)
    ))

    # 起点终点
    fig.add_trace(go.Scatter3d(x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]], mode='markers', name='起点', marker=dict(color='green', size=6)))
    fig.add_trace(go.Scatter3d(x=[landing[0]], y=[landing[1]], z=[landing[2]], mode='markers', name='落地点', marker=dict(color='red', size=8)))

    # 动画
    if show_anim:
        step = max(1, len(traj) // 80)
        frames = [go.Frame(data=[go.Scatter3d(x=[traj[i,0]], y=[traj[i,1]], z=[traj[i,2]], mode='markers', marker=dict(color='orange', size=10))], name=f'f{i}') for i in range(0, len(traj), step)]
        fig.frames = frames
        # 修复：使用上面提取的 dt
        fig.update_layout(updatemenus=[{'type': 'buttons', 'buttons': [{'label': '▶', 'method': 'animate', 'args': [None, {'frame': {'duration': dt*1000/anim_speed}}]}]}])

        # 初始球
        fig.add_trace(go.Scatter3d(x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]], mode='markers', name='小球', marker=dict(color='orange', size=10), showlegend=False))

    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=500)
    return fig

def create_kinematics_charts(res: Dict):
    """生成运动学图表：速度与能量"""
    t = res['time']
    v_mag = np.linalg.norm(res['vel'], axis=1)
    ke, pe, te = res['energy'][:, 0], res['energy'][:, 1], res['energy'][:, 2]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("速度随时间变化", "机械能分析"))

    # 速度图
    fig.add_trace(go.Scatter(x=t, y=v_mag, name='合速度 (m/s)', line=dict(color='#4facfe')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=res['vel'][:, 2], name='垂直速度 (m/s)', line=dict(dash='dot', color='#f5576c')), row=1, col=1)

    # 能量图
    fig.add_trace(go.Scatter(x=t, y=ke, name='动能 (J)', line=dict(color='#ff9a9e')), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=pe, name='势能 (J)', line=dict(color='#a18cd1')), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=te, name='机械能 (J)', line=dict(color='white', width=2)), row=1, col=2)

    fig.update_layout(height=350, margin=dict(t=30, b=0), hovermode="x unified", template="plotly_dark")
    return fig

# ==================== 4. 蒙特卡洛与数据导出 ====================

def run_monte_carlo(base_params: Dict, n_runs: int, noise_std: float):
    """蒙特卡洛模拟：风力扰动"""
    landings = []

    progress_bar = st.progress(0)
    for i in range(n_runs):
        # 复制参数并添加随机扰动
        p = base_params.copy()
        p['wind_x'] += np.random.normal(0, noise_std)
        p['wind_y'] += np.random.normal(0, noise_std)

        # 快速运行（不返回完整轨迹，省内存）
        res = run_simulation(p)
        landings.append(res['landing'])
        progress_bar.progress((i + 1) / n_runs)

    progress_bar.empty()
    return np.array(landings)

def convert_df(res: Dict):
    """将结果转换为 CSV"""
    df = pd.DataFrame({
        'Time (s)': res['time'],
        'Pos_X (m)': res['traj'][:, 0],
        'Pos_Y (m)': res['traj'][:, 1],
        'Pos_Z (m)': res['traj'][:, 2],
        'Vel_X (m/s)': res['vel'][:, 0],
        'Vel_Y (m/s)': res['vel'][:, 1],
        'Vel_Z (m/s)': res['vel'][:, 2],
        'Kinetic_E (J)': res['energy'][:, 0],
        'Potential_E (J)': res['energy'][:, 1]
    })
    return df.to_csv(index=False).encode('utf-8')

# ==================== 5. 主程序逻辑 ====================

def main():
    setup_page()
    load_css()

    # 顶部导航
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="compact-header"><h1>🚁 3D物理仿真 Pro</h1></div>', unsafe_allow_html=True)

    # 侧边栏
    st.sidebar.header("⚙️ 参数控制")

    # --- 参数输入区域 (保持原有的双向绑定逻辑) ---
    mode = st.sidebar.selectbox("运动模式", ["自由落体", "平抛运动", "斜抛运动"])

    st.sidebar.subheader("📏 物体属性")
    # 高度双向绑定
    if 'drone_height' not in st.session_state: st.session_state.drone_height = 50.0
    def sync_h(k_src, k_tgt): st.session_state[k_tgt] = st.session_state[k_src]

    c1, c2 = st.sidebar.columns([3, 1])
    c1.slider("高度 (m)", 0.0, 200.0, key="h_s", value=float(st.session_state.drone_height), on_change=sync_h, args=("h_s", "drone_height"))
    c2.number_input("H", 0.0, 200.0, key="h_i", value=float(st.session_state.drone_height), on_change=sync_h, args=("h_i", "drone_height"), label_visibility="collapsed")

    # 半径双向绑定
    if 'radius_cm' not in st.session_state: st.session_state.radius_cm = 10.0
    c3, c4 = st.sidebar.columns([3, 1])
    c3.slider("半径 (cm)", 1.0, 100.0, key="r_s", value=float(st.session_state.radius_cm), on_change=sync_h, args=("r_s", "radius_cm"))
    c4.number_input("R", 1.0, 100.0, key="r_i", value=float(st.session_state.radius_cm), on_change=sync_h, args=("r_i", "radius_cm"), label_visibility="collapsed")

    density = st.sidebar.number_input("密度 (kg/m³)", value=1000.0)

    st.sidebar.subheader("🚀 初始状态")
    v0_x = st.sidebar.slider("水平速度 X", 0.0, 50.0, 20.0) if mode != "自由落体" else 0.0
    v0_y = st.sidebar.slider("侧向速度 Y", 0.0, 50.0, 0.0) if mode == "斜抛运动" else 0.0
    v0_z = st.sidebar.slider("垂直速度 Z", 0.0, 50.0, 10.0) if mode == "斜抛运动" else 0.0

    st.sidebar.subheader("💨 环境")
    wx = st.sidebar.slider("风速 X", -20.0, 20.0, 5.0)
    wy = st.sidebar.slider("风速 Y", -20.0, 20.0, 0.0)
    wz = st.sidebar.slider("风速 Z", -10.0, 10.0, 0.0)
    dt = st.sidebar.slider("步长 (s)", 0.001, 0.1, 0.01)

    # 运行按钮
    if st.sidebar.button("🚀 开始仿真", type="primary"):
        st.session_state.run_sim = True

    # --- 参数字典 ---
    params = {
        'height': st.session_state.drone_height,
        'radius': st.session_state.radius_cm / 100.0,
        'density': density,
        'v0_x': v0_x, 'v0_y': v0_y, 'v0_z': v0_z,
        'wind_x': wx, 'wind_y': wy, 'wind_z': wz,
        'dt': dt
    }

    # ==================== 主界面渲染 ====================

    if 'run_sim' in st.session_state and st.session_state.run_sim:

        # 1. 运行当前仿真
        current_res = run_simulation(params)

        # 2. 处理历史对比逻辑
        if 'prev_res' not in st.session_state:
            st.session_state.prev_res = None

        # 3. 结果指标栏
        st.markdown("### 📊 实时数据")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("飞行时间", f"{current_res['flight_time']:.2f} s")
        m2.metric("落点 X", f"{current_res['landing'][0]:.2f} m")
        m3.metric("落点 Y", f"{current_res['landing'][1]:.2f} m")

        # 计算动能 (0.5mv^2)
        final_v = np.linalg.norm(current_res['vel'][-1])
        final_ke = 0.5 * current_res['mass'] * final_v**2
        m4.metric("落地动能", f"{final_ke:.1f} J")
        m5.metric("质量", f"{current_res['mass']:.2f} kg")

        # 4. 3D 可视化 (含对比)
        col_viz, col_ctrl = st.columns([3, 1])
        with col_ctrl:
            st.markdown("#### 🎮 视图控制")
            show_anim = st.toggle("播放动画", value=True)
            anim_speed = st.slider("动画倍速", 0.5, 5.0, 1.0)

            st.divider()
            st.info("💡 **提示**: 灰色虚线为上一次运行的轨迹，可用于调整参数后的对比。")

            # --- 功能：CSV 导出 ---
            csv = convert_df(current_res)
            st.download_button(
                label="📥 导出数据 (CSV)",
                data=csv,
                file_name='sim_data.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col_viz:
            st.plotly_chart(create_3d_plot(current_res, st.session_state.prev_res, anim_speed, show_anim), use_container_width=True)

        # 5. --- 功能：运动学图表 ---
        st.markdown("### 📈 运动学分析")
        chart_fig = create_kinematics_charts(current_res)
        st.plotly_chart(chart_fig, use_container_width=True)

        # 6. --- 功能：蒙特卡洛模拟 ---
        st.markdown("---")
        with st.expander("🎲 蒙特卡洛模拟 (Monte Carlo Simulation) - 落点概率分析", expanded=False):
            mc_col1, mc_col2 = st.columns([1, 2])

            with mc_col1:
                st.markdown("**参数设置**")
                n_mc = st.number_input("模拟次数", 10, 500, 100)
                noise = st.slider("风速扰动标准差 (m/s)", 0.1, 5.0, 1.0, help="风速的不确定性大小")
                run_mc = st.button("开始概率模拟")

            with mc_col2:
                if run_mc:
                    with st.spinner(f"正在进行 {n_mc} 次并行仿真..."):
                        mc_landings = run_monte_carlo(params, n_mc, noise)

                        # 绘制散点图
                        mc_fig = go.Figure()
                        # 散点
                        mc_fig.add_trace(go.Scatter(
                            x=mc_landings[:, 0], y=mc_landings[:, 1],
                            mode='markers', marker=dict(color='rgba(102, 126, 234, 0.6)', size=8),
                            name='模拟落点'
                        ))
                        # 平均落点
                        mean_x, mean_y = np.mean(mc_landings[:, 0]), np.mean(mc_landings[:, 1])
                        mc_fig.add_trace(go.Scatter(
                            x=[mean_x], y=[mean_y],
                            mode='markers', marker=dict(color='red', size=15, symbol='cross'),
                            name='平均落点'
                        ))
                        # 靶心 (当前单次运行的落点)
                        mc_fig.add_trace(go.Scatter(
                            x=[current_res['landing'][0]], y=[current_res['landing'][1]],
                            mode='markers', marker=dict(color='gold', size=12, symbol='star'),
                            name='当前主落点'
                        ))

                        mc_fig.update_layout(
                            title="落点散布图 (俯视 XY平面)",
                            xaxis_title="X (m)", yaxis_title="Y (m)",
                            template="plotly_dark", height=400,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(mc_fig, use_container_width=True)
                        st.success(f"模拟完成！落点标准差: X={np.std(mc_landings[:,0]):.2f}m, Y={np.std(mc_landings[:,1]):.2f}m")

        # 仿真结束后，将当前结果存为“上次结果”，供下一次对比使用
        st.session_state.prev_res = current_res

    else:
        st.info("👈 请点击左侧 **开始仿真** 按钮运行程序")

if __name__ == "__main__":
    main()
