import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Any

# ==================== 1. 配置与样式 ====================

def setup_page():
    """配置页面基本设置"""
    st.set_page_config(
        page_title="3D物理仿真 - 无人机投掷小球",
        page_icon="🚁",
        layout="wide"
    )

def load_css():
    """加载自定义CSS样式 (已适配暗色/亮色模式)"""
    st.markdown("""
    <style>
    /* 导入现代字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* 使用 Streamlit 原生变量进行适配:
       var(--text-color): 自动随主题变化的文字颜色
       var(--background-color): 自动随主题变化的背景色
       var(--secondary-background-color): 侧边栏或卡片背景色
    */

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

        /* 默认阴影（通用） */
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    body { font-family: 'Inter', sans-serif; }

    /* 隐藏默认标题 */
    .stTitle { display: none !important; }

    /* ----------------------------------------------------
       核心修复：适配暗色模式的配色方案
       ----------------------------------------------------
    */

    /* 顶部导航栏：使用原生背景色并增加模糊，边框使用半透明 */
    .compact-header {
        position: sticky; top: 0;
        background: var(--background-color); /* 自动适配背景 */
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        z-index: 999;
        padding: 0.8rem 1.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2); /* 通用边框色 */
        box-shadow: var(--shadow-sm);
        display: flex; align-items: center;
    }

    .compact-header h1 {
        font-size: 1.5rem; margin: 0; font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 欢迎横幅：保持原样，因为文字是强制白色的 */
    .welcome-banner {
        background: var(--primary-gradient);
        padding: 2rem; color: white;
        margin-bottom: 2rem;
        border-radius: 12px;
        box-shadow: var(--shadow-md);
    }

    /* 功能卡片：
       背景色改为 var(--secondary-background-color)
       这样在亮色模式是浅灰/白，暗色模式是深灰
    */
    .feature-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: rgba(128, 128, 128, 0.3);
    }

    /* 文字颜色修复：
       强制使用 var(--text-color)
       确保在暗色模式下自动变白
    */
    .feature-card h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-color) !important;
    }

    .feature-card p {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.8; /* 稍微透明一点作为次级文本 */
        margin: 0;
    }

    /* Streamlit 组件优化 */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: var(--text-color) !important; /* 强制适配颜色 */
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: var(--text-color) !important;
        opacity: 0.7;
    }

    .stButton > button { border-radius: 10px; font-weight: 600; transition: all 0.2s; }
    .stButton > button[kind="primary"] { background: var(--primary-gradient) !important; border: none; color: white !important; }

    /* 侧边栏背景微调 */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }

    /* 图表容器 */
    .plotly-graph-div {
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }

    /* Expander 样式适配 */
    .streamlit-expanderHeader {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== 2. 物理引擎 ====================

# 物理常量
CONSTANTS = {
    'G': 10.0,              # 重力加速度 (m/s²)
    'AIR_DENSITY': 1.225,   # 空气密度 (kg/m³)
    'DRAG_COEFFICIENT': 0.42 # 球体阻力系数
}

def calculate_ball_properties(radius: float, density: float) -> Tuple[float, float, float]:
    """计算小球物理属性：体积、质量、迎风面积"""
    volume = (4/3) * np.pi * radius**3
    mass = density * volume
    area = np.pi * radius**2
    return volume, mass, area

def calculate_forces(velocity: np.ndarray, wind_velocity: np.ndarray,
                    mass: float, area: float) -> Tuple[np.ndarray, np.ndarray]:
    """计算合力和阻力"""
    # 1. 重力
    gravity_force = np.array([0.0, 0.0, -mass * CONSTANTS['G']])

    # 2. 空气阻力
    rel_vel = wind_velocity - velocity
    rel_speed = np.linalg.norm(rel_vel)

    if rel_speed > 0:
        drag_force = 0.5 * CONSTANTS['AIR_DENSITY'] * CONSTANTS['DRAG_COEFFICIENT'] * area * rel_speed * rel_vel
    else:
        drag_force = np.zeros(3)

    return gravity_force + drag_force, drag_force

def run_simulation(params: Dict[str, Any]):
    """核心仿真循环"""
    dt = params['dt']
    wind_vel = np.array([params['wind_x'], params['wind_y'], params['wind_z']])

    vol, mass, area = calculate_ball_properties(params['radius'], params['density'])
    pos = np.array([0.0, 0.0, params['height']])
    vel = np.array([params['v0_x'], params['v0_y'], params['v0_z']])

    trajectory = [pos.copy()]
    time_points = [0.0]
    velocities = [vel.copy()]  # 记录速度变化

    t = 0.0

    gravity_work = 0.0
    wind_work = 0.0
    gravity_vec_const = np.array([0.0, 0.0, -mass * CONSTANTS['G']])

    while pos[2] > 0 and t < 100:
        total_force, drag_force = calculate_forces(vel, wind_vel, mass, area)

        acc = total_force / mass
        displacement = vel * dt

        gravity_work += np.dot(gravity_vec_const, displacement)
        wind_work += np.dot(drag_force, displacement)

        vel += acc * dt
        pos += vel * dt
        t += dt

        trajectory.append(pos.copy())
        time_points.append(t)
        velocities.append(vel.copy())

    trajectory = np.array(trajectory)
    time_points = np.array(time_points)
    velocities = np.array(velocities)

    if trajectory[-1, 2] < 0 and len(trajectory) > 1:
        prev_p = trajectory[-2]
        last_p = trajectory[-1]
        ratio = prev_p[2] / (prev_p[2] - last_p[2])
        landing_point = prev_p + ratio * (last_p - prev_p)
        landing_point[2] = 0.0
        flight_time = time_points[-2] + ratio * dt
    else:
        landing_point = trajectory[-1]
        landing_point[2] = max(0.0, landing_point[2])
        flight_time = time_points[-1]

    # 计算速度大小
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)

    # 计算机械能
    # 势能 PE = m * g * h
    potential_energy = mass * CONSTANTS['G'] * trajectory[:, 2]
    # 动能 KE = 0.5 * m * v^2
    kinetic_energy = 0.5 * mass * velocity_magnitudes**2
    # 总机械能 ME = PE + KE
    mechanical_energy = potential_energy + kinetic_energy

    return {
        'traj': trajectory,
        'time': time_points,
        'velocities': velocities,
        'velocity_magnitudes': velocity_magnitudes,
        'potential_energy': potential_energy,
        'kinetic_energy': kinetic_energy,
        'mechanical_energy': mechanical_energy,
        'landing': landing_point,
        'flight_time': flight_time,
        'mass': mass,
        'volume': vol,
        'area': area,
        'work_g': gravity_work,
        'work_w': wind_work,
        'wind_acc_max': np.linalg.norm(wind_vel)**2 * 0.5 * CONSTANTS['AIR_DENSITY'] * CONSTANTS['DRAG_COEFFICIENT'] * area / mass if mass > 0 else 0
    }

# ==================== 3. 可视化组件 ====================

def create_3d_plot(res: Dict, params: Dict, anim_speed: float, show_anim: bool):
    traj = res['traj']
    landing = res['landing']

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    grid_size = max(abs(landing[0]), abs(landing[1]), 50)
    x = np.linspace(-grid_size, grid_size, 20)
    X, Y = np.meshgrid(x, x)
    fig.add_trace(go.Surface(x=X, y=Y, z=X*0, colorscale='Greys', showscale=False, opacity=0.3, name='地面'))

    fig.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
        mode='lines', name='轨迹', line=dict(color='#667eea', width=4)
    ))

    fig.add_trace(go.Scatter3d(x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]], mode='markers', name='起点', marker=dict(color='green', size=8)))
    fig.add_trace(go.Scatter3d(x=[landing[0]], y=[landing[1]], z=[landing[2]], mode='markers', name='落地点', marker=dict(color='red', size=8)))

    if show_anim:
        total_points = len(traj)
        step = max(1, total_points // 100)
        frames = []
        for i in range(0, total_points, step):
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=[traj[i,0]], y=[traj[i,1]], z=[traj[i,2]],
                    mode='markers', marker=dict(color='orange', size=12)
                )],
                name=f'f{i}'
            ))
        fig.frames = frames

        frame_dur = params['dt'] * 1000 / anim_speed
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'x': 0.1, 'y': 0,
                'buttons': [{
                    'label': '▶ 播放',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': frame_dur, 'redraw': True}, 'fromcurrent': True}]
                }, {
                    'label': '⏸ 暂停',
                    'method': 'animate',
                    'args': [[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}}]
                }]
            }]
        )

    fig.add_trace(go.Scatter3d(x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]], mode='markers', name='小球', marker=dict(color='orange', size=12), showlegend=False))

    fig.update_layout(
        title="3D 轨迹视图",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5)),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    return fig

def create_2d_projections(traj, landing):
    plots = []
    views = [
        ('XZ 平面 (侧视)', 0, 2, 'X', 'Z'),
        ('XY 平面 (俯视)', 0, 1, 'X', 'Y'),
        ('YZ 平面 (正视)', 1, 2, 'Y', 'Z')
    ]

    for title, idx_x, idx_y, label_x, label_y in views:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traj[:, idx_x], y=traj[:, idx_y], mode='lines', line=dict(color='#667eea', width=2)))
        fig.add_trace(go.Scatter(x=[landing[idx_x]], y=[landing[idx_y]], mode='markers', marker=dict(color='red', size=8)))
        fig.update_layout(
            title=title, xaxis_title=f'{label_x} (m)', yaxis_title=f'{label_y} (m)',
            margin=dict(l=20, r=20, b=20, t=40), height=300
        )
        plots.append(fig)
    return plots

def create_kinematics_charts(res: Dict):
    """创建运动学图表：速度分析和机械能分析"""
    time = res['time']
    velocity_magnitudes = res['velocity_magnitudes']
    potential_energy = res['potential_energy']
    kinetic_energy = res['kinetic_energy']
    mechanical_energy = res['mechanical_energy']

    # 左图：速度分析
    fig_velocity = go.Figure()
    fig_velocity.add_trace(go.Scatter(
        x=time,
        y=velocity_magnitudes,
        mode='lines',
        name='总速度',
        line=dict(color='#667eea', width=3)
    ))

    # 检查是否达到终端速度（速度变化率趋近于0）
    if len(velocity_magnitudes) > 10:
        # 计算最后10%时间段的平均速度变化率
        last_10_percent = int(len(velocity_magnitudes) * 0.1)
        if last_10_percent > 0:
            velocity_change_rate = np.abs(np.diff(velocity_magnitudes[-last_10_percent:])).mean()
            terminal_velocity_threshold = 0.01  # 速度变化率小于此值认为达到终端速度

            if velocity_change_rate < terminal_velocity_threshold and velocity_magnitudes[-1] > 0:
                terminal_velocity = velocity_magnitudes[-1]
                # 在图上标注终端速度
                fig_velocity.add_hline(
                    y=terminal_velocity,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"终端速度: {terminal_velocity:.2f} m/s",
                    annotation_position="top right"
                )

    fig_velocity.update_layout(
        title="速度分析",
        xaxis_title="时间 (s)",
        yaxis_title="速度 (m/s)",
        margin=dict(l=20, r=20, b=20, t=40),
        height=400,
        hovermode='x unified'
    )

    # 右图：机械能分析
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(
        x=time,
        y=potential_energy,
        mode='lines',
        name='势能 (PE)',
        line=dict(color='#9b59b6', width=2)
    ))
    fig_energy.add_trace(go.Scatter(
        x=time,
        y=kinetic_energy,
        mode='lines',
        name='动能 (KE)',
        line=dict(color='#e91e63', width=2)
    ))
    fig_energy.add_trace(go.Scatter(
        x=time,
        y=mechanical_energy,
        mode='lines',
        name='总机械能 (ME)',
        line=dict(color='white', width=3)
    ))

    fig_energy.update_layout(
        title="机械能分析",
        xaxis_title="时间 (s)",
        yaxis_title="能量 (J)",
        margin=dict(l=20, r=20, b=20, t=40),
        height=400,
        hovermode='x unified',
        template='plotly_dark'
    )

    return fig_velocity, fig_energy

# ==================== 4. 侧边栏逻辑 (已修复同步问题) ====================

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("⚙️ 参数设置")

    # 1. 模式选择
    mode = st.sidebar.selectbox("运动模式", ["自由落体", "平抛运动", "斜抛运动"])
    st.sidebar.divider()

    # 2. 高度控制 (双向绑定修复)
    st.sidebar.subheader("📏 基础参数")
    h_mode = st.sidebar.radio("高度模式", ["无人机高度", "小球高度"], horizontal=True)

    # 确定当前的 Key (存储在 session_state 中的变量名)
    current_h_key = "drone_height" if h_mode == "无人机高度" else "ball_height"

    # 初始化变量（如果不存在）
    if current_h_key not in st.session_state:
        st.session_state[current_h_key] = 50.0

    # 定义回调函数：当组件变化时，更新 session_state 中的值
    def sync_height_slider():
        st.session_state[current_h_key] = st.session_state.h_slider_widget

    def sync_height_input():
        st.session_state[current_h_key] = st.session_state.h_input_widget

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        # 注意：value 直接读取 session_state[current_h_key]，key 是唯一的 widget key
        st.slider(f"{h_mode} (m)", 0.0, 200.0,
                 key="h_slider_widget",
                 value=float(st.session_state[current_h_key]), # 强制 float
                 on_change=sync_height_slider) # 绑定回调
    with col2:
        st.number_input("数值", 0.0, 200.0,
                       key="h_input_widget",
                       value=float(st.session_state[current_h_key]), # 强制 float
                       label_visibility="collapsed",
                       on_change=sync_height_input) # 绑定回调

    # 计算用于物理计算的实际高度
    raw_h = st.session_state[current_h_key]
    actual_ball_height = raw_h - 0.1 if h_mode == "无人机高度" else raw_h

    # 3. 半径控制 (双向绑定修复)
    if 'radius_cm' not in st.session_state:
        st.session_state.radius_cm = 10.0

    # 定义半径的回调函数
    def sync_radius_slider():
        st.session_state.radius_cm = st.session_state.r_slider_widget

    def sync_radius_input():
        st.session_state.radius_cm = st.session_state.r_input_widget

    col_r1, col_r2 = st.sidebar.columns([3, 1])
    with col_r1:
        st.slider("小球半径 (cm)", 1.0, 100.0,
                 key="r_slider_widget",
                 value=float(st.session_state.radius_cm),
                 on_change=sync_radius_slider)
    with col_r2:
        st.number_input("R_input", 1.0, 100.0,
                       key="r_input_widget",
                       value=float(st.session_state.radius_cm),
                       label_visibility="collapsed",
                       on_change=sync_radius_input)

    # 密度预设
    density_presets = {"空气": 1.29, "水": 1000.0, "塑料": 935.0, "木材": 500.0, "铁": 7860.0, "金": 19320.0}
    with st.sidebar.expander("🎯 密度预设"):
        sel_preset = st.selectbox("选择材质", list(density_presets.keys()), index=1)
        if st.button("应用预设"):
            st.session_state.ball_density = density_presets[sel_preset]
            st.rerun()

    if 'ball_density' not in st.session_state: st.session_state.ball_density = 1000.0
    density = st.sidebar.number_input("小球密度 (kg/m³)", 1.0, 20000.0, key="ball_density")

    st.sidebar.divider()

    # 4. 速度设置
    st.sidebar.subheader("🚀 初速度设置")
    v0_x, v0_y, v0_z = 0.0, 0.0, 0.0

    if mode != "自由落体":
        v0_x = st.sidebar.slider("水平速度 v₀x", 0.0, 50.0, 20.0)
    if mode == "斜抛运动":
        v0_y = st.sidebar.slider("侧向速度 v₀y", 0.0, 50.0, 0.0)
        v0_z = st.sidebar.slider("垂直速度 v₀z", 0.0, 50.0, 10.0)

    st.sidebar.divider()

    # 5. 环境参数
    st.sidebar.subheader("💨 风力与仿真")
    wx = st.sidebar.slider("风速 X (右+)", -20.0, 20.0, 0.0)
    wy = st.sidebar.slider("风速 Y (前+)", -20.0, 20.0, 0.0)
    wz = st.sidebar.slider("风速 Z (上+)", -10.0, 10.0, 0.0)

    w_total = np.sqrt(wx**2 + wy**2 + wz**2)
    st.sidebar.caption(f"当前总风速: {w_total:.1f} m/s")

    dt = st.sidebar.slider("时间步长 (s)", 0.001, 0.1, 0.01, 0.001, help="越小越精确")

    # 动作按钮
    reset = st.sidebar.button("🔄 重置参数")
    if reset:
        st.session_state.clear()
        st.rerun()

    start = st.sidebar.button("🚀 开始仿真", type="primary")

    return {
        'run': start,
        'mode': mode,
        'height': actual_ball_height,
        'radius': st.session_state.radius_cm / 100.0,
        'density': density,
        'v0_x': v0_x, 'v0_y': v0_y, 'v0_z': v0_z,
        'wind_x': wx, 'wind_y': wy, 'wind_z': wz,
        'dt': dt
    }

# ==================== 5. 结果页面渲染 ====================

def render_formulas(res: Dict = None):
    """渲染详细的物理公式说明"""
    st.markdown("---")
    st.subheader("📚 物理原理详解")

    with st.expander("点击展开完整计算公式", expanded=True):
        st.markdown(r"""
        #### 1. 几何属性
        - **体积**: $V = \frac{4}{3}\pi r^3$
        - **质量**: $m = \rho_{\text{ball}} \cdot V$
        - **迎风面积**: $A = \pi r^2$

        #### 2. 动力学模型 (Forces)
        物体受到**重力**和**空气阻力**的共同作用：

        - **重力**: $\vec{F}_g = m\vec{g} = (0, 0, -mg)$
        - **相对空气速度**: $\vec{v}_{rel} = \vec{v}_{wind} - \vec{v}_{ball}$
        - **空气阻力**: $\vec{F}_{drag} = \frac{1}{2} C_d \rho_{\text{air}} A |\vec{v}_{rel}| \vec{v}_{rel}$
        - **合外力**: $\vec{F}_{total} = \vec{F}_g + \vec{F}_{drag}$
        - **瞬时加速度**: $\vec{a} = \frac{\vec{F}_{total}}{m}$

        #### 3. 数值积分 (Numerical Integration)
        采用**半隐式欧拉法 (Semi-implicit Euler Method)** 进行离散时间步迭代：

        $$
        \begin{aligned}
        \vec{v}_{t+\Delta t} &= \vec{v}_t + \vec{a}_t \cdot \Delta t \\
        \vec{p}_{t+\Delta t} &= \vec{p}_t + \vec{v}_{t+\Delta t} \cdot \Delta t
        \end{aligned}
        $$

        #### 4. 能量与做功
        - **功的计算**: $W = \sum (\vec{F} \cdot \Delta \vec{p})$
          (其中 $\Delta \vec{p} = \vec{v} \cdot \Delta t$ 为每个时间步的位移矢量)
        """)

        if res:
            st.info(f"""
            **当前仿真参数代入：**
            - 质量 $m = {res['mass']:.4f}$ kg
            - 迎风面积 $A = {res['area']:.6f}$ m²
            - 空气密度 $\\rho_{{air}} = {CONSTANTS['AIR_DENSITY']}$ kg/m³
            - 阻力系数 $C_d = {CONSTANTS['DRAG_COEFFICIENT']}$
            """)

def render_results(res: Dict, params: Dict):
    st.success("✅ 仿真计算完成")

    # 1. 核心指标卡片
    st.markdown("### 📊 核心指标")
    cols = st.columns(5)
    metrics = [
        ("初始高度", f"{params['height']:.2f} m"),
        ("飞行时间", f"{res['flight_time']:.4f} s"),
        ("落地点 X", f"{res['landing'][0]:.4f} m"),
        ("落地点 Y", f"{res['landing'][1]:.4f} m"),
        ("落地点 Z", f"{res['landing'][2]:.4f} m"),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)

    # 2. 物理分析
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>💨 风力与受力分析</h3>
            <p><strong>风速大小:</strong> {:.2f} m/s</p>
            <p><strong>最大风力加速度:</strong> {:.4f} m/s² ({:.1f}% g)</p>
            <p><strong>说明:</strong> {}</p>
        </div>
        """.format(
            np.linalg.norm([params['wind_x'], params['wind_y'], params['wind_z']]),
            res['wind_acc_max'],
            (res['wind_acc_max']/CONSTANTS['G'])*100,
            "风力显著改变落点" if res['wind_acc_max'] > 0.5 else "风力影响较小" if res['wind_acc_max'] < 0.1 else "风力影响适中"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ 能量与做功</h3>
            <p><strong>重力做功:</strong> {:.4f} J</p>
            <p><strong>风力做功:</strong> {:.4f} J</p>
            <p><strong>总做功:</strong> {:.4f} J</p>
        </div>
        """.format(res['work_g'], res['work_w'], res['work_g'] + res['work_w']), unsafe_allow_html=True)

    # 3. 3D 可视化
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.subheader("🎯 3D 轨迹演示")

    # 动画控制
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        show_anim = st.checkbox("启用动画渲染", value=True)
    with col_ctrl2:
        speed = st.slider("动画播放倍速", 0.1, 10.0, 1.0, disabled=not show_anim)

    fig_3d = create_3d_plot(res, params, speed, show_anim)
    st.plotly_chart(fig_3d, width="stretch")

    # 4. 2D 投影
    st.subheader("📐 多视图投影")
    proj_figs = create_2d_projections(res['traj'], res['landing'])
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.plotly_chart(proj_figs[0], width="stretch")
    col_p2.plotly_chart(proj_figs[1], width="stretch")
    col_p3.plotly_chart(proj_figs[2], width="stretch")

    # 5. 运动学图表
    st.subheader("📈 运动学分析")
    fig_velocity, fig_energy = create_kinematics_charts(res)
    col_k1, col_k2 = st.columns(2)
    col_k1.plotly_chart(fig_velocity, width="stretch")
    col_k2.plotly_chart(fig_energy, width="stretch")

    # 6. 底部公式展示
    render_formulas(res)

# ==================== 6. 主程序入口 ====================

def main():
    setup_page()
    load_css()

    # 头部
    st.markdown('<div class="compact-header"><h1>🚁 3D物理仿真实验室</h1></div>', unsafe_allow_html=True)

    # 默认欢迎页
    if 'has_run' not in st.session_state:
        st.session_state.has_run = False

    # 侧边栏
    params = render_sidebar()

    # 主逻辑
    if params['run']:
        st.session_state.has_run = True
        with st.spinner("正在进行物理结算..."):
            results = run_simulation(params)
        render_results(results, params)

    elif not st.session_state.has_run:
        # 初始欢迎界面
        st.markdown("""
        <div class="welcome-banner">
            <h2>👋 欢迎使用物理仿真系统</h2>
            <p>这是一个基于 Streamlit 和 Plotly 的高精度运动学仿真工具。支持模拟重力、空气阻力和复杂风场环境下的物体运动。</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="feature-card"><h3>🌬️ 风场模拟</h3><p>支持三维矢量风场设置，实时计算气动阻力。</p></div>', unsafe_allow_html=True)
        c2.markdown('<div class="feature-card"><h3>📐 多维视图</h3><p>提供3D交互式视图及XZ/XY/YZ三个正交投影平面。</p></div>', unsafe_allow_html=True)
        c3.markdown('<div class="feature-card"><h3>⚡ 能量分析</h3><p>实时计算重力做功与风力做功，辅助物理教学分析。</p></div>', unsafe_allow_html=True)

        render_formulas()
        st.info("👈 请在左侧侧边栏设置参数并点击「开始仿真」")

if __name__ == "__main__":
    main()
