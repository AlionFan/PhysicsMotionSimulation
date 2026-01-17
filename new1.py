"""
基于 Python + Streamlit 的 3D 物理仿真程序
模拟无人机投放小球的运动轨迹，考虑重力和风力作用。

优化版特性：
1. 模块化结构 (Config, Physics, UI, Plotting)
2. 类型安全 (Type Hinting)
3. 代码复用 (通用组件封装)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

# ==================== 1. 常量定义 (Constants) ====================

# 物理常量
CONST_G: float = 10.0                # 重力加速度 (m/s²)
CONST_AIR_DENSITY: float = 1.225     # 空气密度 (kg/m³)
CONST_DRAG_COEFF: float = 0.42       # 球体空气阻力系数
CONST_DEFAULT_RADIUS_CM: float = 10.0
CONST_DEFAULT_DENSITY: float = 100.0

# 样式常量
STYLE_PRIMARY_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
STYLE_GLASS_BG = "rgba(255, 255, 255, 0.85)"

# ==================== 2. 数据结构 (Data Structures) ====================

@dataclass
class SimulationParams:
    """仿真参数容器"""
    ball_height: float
    ball_radius: float
    ball_density: float
    mode: str
    v0: np.ndarray          # [vx, vy, vz]
    wind_velocity: np.ndarray # [wx, wy, wz]
    dt: float
    max_time: float = 100.0

@dataclass
class SimulationResult:
    """仿真结果容器"""
    trajectory: np.ndarray
    time_points: np.ndarray
    landing_point: np.ndarray
    flight_time: float
    mass: float
    volume: float
    gravity_work: float
    wind_work: float
    max_drag_force: float = 0.0 # 补充计算字段

# ==================== 3. 核心逻辑 (Core Logic) ====================

class PhysicsEngine:
    """物理计算引擎"""

    @staticmethod
    def calculate_properties(radius: float, density: float) -> Tuple[float, float, float]:
        """计算物理属性: 体积, 质量, 迎风面积"""
        volume = (4/3) * np.pi * radius**3
        mass = density * volume
        area = np.pi * radius**2
        return volume, mass, area

    @staticmethod
    def calculate_forces(velocity: np.ndarray, wind_velocity: np.ndarray, 
                         mass: float, area: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算合力和风力"""
        # 1. 重力
        gravity_force = np.array([0.0, 0.0, -mass * CONST_G])
        
        # 2. 空气阻力 (相对速度)
        relative_velocity = wind_velocity - velocity
        relative_speed = np.linalg.norm(relative_velocity)
        
        drag_force = np.zeros(3)
        if relative_speed > 0:
            coeff = 0.5 * CONST_AIR_DENSITY * CONST_DRAG_COEFF * area
            drag_force = coeff * relative_speed * relative_velocity
            
        return gravity_force + drag_force, drag_force

    @staticmethod
    def run_simulation(params: SimulationParams) -> SimulationResult:
        """执行物理仿真"""
        # 准备初始条件
        volume, mass, area = PhysicsEngine.calculate_properties(params.ball_radius, params.ball_density)
        
        position = np.array([0.0, 0.0, params.ball_height], dtype=float)
        velocity = params.v0.astype(float).copy()
        
        # 轨迹记录
        trajectory: List[np.ndarray] = [position.copy()]
        time_points: List[float] = [0.0]
        
        # 状态累加器
        t = 0.0
        gravity_work = 0.0
        wind_work = 0.0

        # 主循环
        while position[2] > 0 and t < params.max_time:
            #受力分析
            total_force, drag_force_vec = PhysicsEngine.calculate_forces(
                velocity, params.wind_velocity, mass, area
            )
            gravity_force_vec = np.array([0, 0, -mass * CONST_G])

            # 欧拉积分
            acceleration = total_force / mass
            displacement = velocity * params.dt
            
            # 做功计算 (W = F · d)
            gravity_work += np.dot(gravity_force_vec, displacement)
            wind_work += np.dot(drag_force_vec, displacement)

            # 状态更新
            velocity += acceleration * params.dt
            position += velocity * params.dt
            t += params.dt

            trajectory.append(position.copy())
            time_points.append(t)

        # 结果转换
        traj_np = np.array(trajectory)
        times_np = np.array(time_points)
        
        # 落地点修正 (线性插值)
        landing_point = traj_np[-1]
        flight_time = times_np[-1]

        if traj_np[-1, 2] < 0 and len(traj_np) > 1:
            prev_p = traj_np[-2]
            last_p = traj_np[-1]
            prev_t = times_np[-2]
            last_t = times_np[-1]
            
            # 计算Z轴穿过0点的比例 ratio = z_prev / (z_prev - z_last)
            ratio = prev_p[2] / (prev_p[2] - last_p[2])
            
            landing_point = prev_p + ratio * (last_p - prev_p)
            landing_point[2] = 0.0 # 强制修正为地面
            flight_time = prev_t + ratio * (last_t - prev_t)
        else:
            landing_point[2] = max(0.0, landing_point[2])

        return SimulationResult(
            trajectory=traj_np,
            time_points=times_np,
            landing_point=landing_point,
            flight_time=flight_time,
            mass=mass,
            volume=volume,
            gravity_work=gravity_work,
            wind_work=wind_work
        )

# ==================== 4. UI 组件 (UI Components) ====================

def load_css():
    """加载自定义CSS样式"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --glass-bg: rgba(255, 255, 255, 0.85);
        --glass-border: rgba(255, 255, 255, 0.3);
        --text-primary: #1a1a2e;
        --text-secondary: #4a4a6a;
    }
    body { font-family: 'Inter', sans-serif; }
    .stTitle { display: none !important; }
    
    /* 紧凑标题 */
    .compact-header {
        position: sticky; top: 0; background: var(--glass-bg); z-index: 999;
        padding: 0.8rem 1.5rem; border-bottom: 1px solid var(--glass-border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); backdrop-filter: blur(20px);
    }
    .compact-header h1 {
        font-size: 1.5rem; margin: 0; background: var(--primary-gradient);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    /* 通用卡片样式 */
    .feature-card {
        background: var(--glass-bg); border: 1px solid var(--glass-border);
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08); transition: transform 0.3s;
    }
    .feature-card:hover { transform: translateY(-4px); }
    .welcome-banner {
        background: var(--primary-gradient); color: white; padding: 2rem;
        border-radius: 16px; margin-bottom: 2rem;
    }
    
    /* Streamlit组件覆写 */
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; }
    .stButton > button { border-radius: 12px; font-weight: 600; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%); }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """渲染顶部标题和横幅"""
    st.markdown("""
    <div class="compact-header"><h1>🚁 3D物理仿真 - 无人机投掷小球</h1></div>
    <div class="welcome-banner">
        <h2 style="margin:0">🎯 探索物理世界的奥秘</h2>
        <p style="opacity:0.9; margin-top:0.5rem">通过交互式3D仿真，深入理解重力、空气阻力和风力对物体运动的影响</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("🌬️ 真实风力模拟", "精确计算三维风力对小球轨迹的影响"),
        ("📐 3D可视化", "实时3D轨迹动画，支持多视角2D投影"),
        ("🔬 精确物理计算", "基于牛顿运动定律和空气阻力方程")
    ]
    for col, (title, desc) in zip(cols, features):
        col.markdown(f"""
        <div class="feature-card">
            <h3 style="font-size:1.1rem; margin-bottom:0.5rem">{title}</h3>
            <p style="font-size:0.9rem; color:#4a4a6a; margin:0">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    st.divider()

def render_synced_input(label: str, session_key: str, min_v: float, max_v: float, 
                        step: float, default: float, help_text: str = "") -> float:
    """
    渲染同步的滑块和数字输入框
    """
    if session_key not in st.session_state:
        st.session_state[session_key] = default

    col1, col2 = st.sidebar.columns([3, 1])
    
    # 回调函数
    def _sync_slider():
        st.session_state[session_key] = st.session_state[f"{session_key}_slider"]
        
    def _sync_input():
        st.session_state[session_key] = st.session_state[f"{session_key}_input"]

    with col1:
        st.slider(
            f"{label}", min_v, max_v, st.session_state[session_key], step,
            key=f"{session_key}_slider", help=help_text, on_change=_sync_slider
        )
    with col2:
        st.number_input(
            "Value", min_v, max_v, st.session_state[session_key], step,
            key=f"{session_key}_input", label_visibility="collapsed", on_change=_sync_input
        )
        
    return st.session_state[session_key]

# ==================== 5. 绘图逻辑 (Plotting) ====================

def create_3d_figure(res: SimulationResult, dt: float, speed: float) -> go.Figure:
    """创建3D轨迹和动画"""
    fig = make_subplots(specs=[[{'type': 'scatter3d'}]])
    
    # 1. 地面网格
    grid_range = max(abs(res.landing_point[0]), abs(res.landing_point[1]), 50)
    grid = np.linspace(-grid_range, grid_range, 20)
    X, Y = np.meshgrid(grid, grid)
    fig.add_trace(go.Surface(x=X, y=Y, z=np.zeros_like(X), colorscale='Greys', showscale=False, opacity=0.3))
    
    # 2. 静态元素 (轨迹线、起点、终点)
    fig.add_trace(go.Scatter3d(
        x=res.trajectory[:, 0], y=res.trajectory[:, 1], z=res.trajectory[:, 2],
        mode='lines', name='运动轨迹', line=dict(color='blue', width=4)
    ))
    fig.add_trace(go.Scatter3d(
        x=[res.trajectory[0, 0]], y=[res.trajectory[0, 1]], z=[res.trajectory[0, 2]],
        mode='markers', name='起点', marker=dict(color='green', size=10)
    ))
    fig.add_trace(go.Scatter3d(
        x=[res.landing_point[0]], y=[res.landing_point[1]], z=[res.landing_point[2]],
        mode='markers', name='落地点', marker=dict(color='red', size=10)
    ))

    # 3. 动画帧生成
    frames = []
    # 智能采样：根据点数动态调整，保证动画流畅且不过大
    num_points = len(res.trajectory)
    step_size = max(1, int(num_points / (100 if speed > 2.0 else 200))) 
    
    for i in range(0, num_points, step_size):
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=[res.trajectory[i, 0]], y=[res.trajectory[i, 1]], z=[res.trajectory[i, 2]],
                mode='markers', marker=dict(color='orange', size=15)
            )],
            name=f'f{i}'
        ))
    
    # 初始动画点
    fig.add_trace(go.Scatter3d(
        x=[res.trajectory[0, 0]], y=[res.trajectory[0, 1]], z=[res.trajectory[0, 2]],
        mode='markers', name='小球', marker=dict(color='orange', size=15), showlegend=False
    ))
    
    fig.frames = frames
    frame_duration = dt * 1000 / speed

    # 动画按钮配置
    anim_buttons = [dict(
        label='▶ 播放', method='animate',
        args=[None, {'frame': {'duration': int(frame_duration), 'redraw': True}, 'fromcurrent': True}]
    ), dict(
        label='⏸ 暂停', method='animate',
        args=[[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
    )]

    fig.update_layout(
        title="小球运动轨迹 (3D视图)", height=700,
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5)),
        updatemenus=[dict(type='buttons', showactive=False, y=0, x=0.1, xanchor='right', direction='left', buttons=anim_buttons)]
    )
    return fig

def plot_2d_projection(res: SimulationResult):
    """绘制三个平面的2D投影"""
    col1, col2, col3 = st.columns(3)
    planes = [
        ("XZ 平面 (侧视图)", 0, 2, 'X (m)', 'Z (m)', col1),
        ("XY 平面 (俯视图)", 0, 1, 'X (m)', 'Y (m)', col2),
        ("YZ 平面 (正视图)", 1, 2, 'Y (m)', 'Z (m)', col3)
    ]
    
    for title, idx_x, idx_y, x_lab, y_lab, col in planes:
        fig = go.Figure()
        # 轨迹
        fig.add_trace(go.Scatter(x=res.trajectory[:, idx_x], y=res.trajectory[:, idx_y], mode='lines', line=dict(color='blue', width=3), name='轨迹'))
        # 起点
        fig.add_trace(go.Scatter(x=[res.trajectory[0, idx_x]], y=[res.trajectory[0, idx_y]], mode='markers', marker=dict(color='green', size=10), name='起点'))
        # 落地点
        fig.add_trace(go.Scatter(x=[res.landing_point[idx_x]], y=[res.landing_point[idx_y]], mode='markers', marker=dict(color='red', size=10), name='落地点'))
        
        fig.update_layout(title=title, xaxis_title=x_lab, yaxis_title=y_lab, height=400, margin=dict(l=20, r=20, t=40, b=20))
        col.plotly_chart(fig, width='stretch')

# ==================== 6. 主程序 (Main App) ====================

def main():
    st.set_page_config(page_title="3D物理仿真 - 无人机投掷小球", page_icon="🚁", layout="wide")
    load_css()
    render_header()

    # --- Sidebar: 参数设置 ---
    st.sidebar.header("⚙️ 参数设置")
    
    # 1. 模式选择
    mode = st.sidebar.selectbox("运动模式", ["自由落体", "平抛运动", "斜抛运动"])
    st.sidebar.divider()

    # 2. 高度设置 (逻辑保留)
    st.sidebar.subheader("📏 基础参数")
    height_mode = st.sidebar.radio("高度模式", ["无人机高度", "小球高度"], horizontal=True)
    
    if height_mode == "无人机高度":
        drone_h = render_synced_input("无人机高度 (m)", "drone_height", 0.0, 100.0, 0.1, 5.0)
        ball_h = drone_h - 0.1
    else:
        ball_h = render_synced_input("小球高度 (m)", "ball_height", 0.0, 100.0, 0.1, 5.0)
        drone_h = ball_h + 0.1

    # 半径与密度
    ball_r_cm = render_synced_input("小球半径 (cm)", "ball_radius_cm", 1.0, 1000.0, 1.0, CONST_DEFAULT_RADIUS_CM)
    
    if 'ball_density' not in st.session_state:
        st.session_state.ball_density = CONST_DEFAULT_DENSITY
        
    st.sidebar.number_input("小球密度 (kg/m³)", 1.0, 20000.0, key="ball_density")
    
    with st.sidebar.expander("🎯 快速选择密度"):
        presets = {"空气": 1.29, "水": 1000.0, "铁": 7860.0, "金": 19320.0, "塑料": 935.0, "木材": 500.0}
        sel_preset = st.selectbox("选择预设", list(presets.keys()), index=1)
        if st.button("应用预设"):
            st.session_state.ball_density = presets[sel_preset]
            st.rerun()

    # 3. 速度设置 (逻辑合并优化)
    st.sidebar.divider()
    st.sidebar.subheader("🚀 初速度设置")
    
    # 默认值
    v0_defaults = {'x': 20.0, 'y': 0.0, 'z': 10.0}
    
    # 根据模式显示不同的滑块，未显示的设为0
    vx, vy, vz = 0.0, 0.0, 0.0
    
    if mode in ["平抛运动", "斜抛运动"]:
        vx = st.sidebar.slider("水平初速度 v₀x", 0.0, 50.0, st.session_state.get('v0_x', v0_defaults['x']), key='v0_x')
    
    if mode == "斜抛运动":
        vy = st.sidebar.slider("侧向初速度 v₀y", 0.0, 50.0, st.session_state.get('v0_y', v0_defaults['y']), key='v0_y')
        vz = st.sidebar.slider("垂直初速度 v₀z", 0.0, 50.0, st.session_state.get('v0_z', v0_defaults['z']), key='v0_z')
    
    # 4. 风力设置
    st.sidebar.divider()
    st.sidebar.subheader("💨 风力设置")
    wx = st.sidebar.slider("X风速 (m/s)", -20.0, 20.0, 0.0, key="wind_x", help="正值向右")
    wy = st.sidebar.slider("Y风速 (m/s)", -20.0, 20.0, 0.0, key="wind_y", help="正值向前")
    wz = st.sidebar.slider("Z风速 (m/s)", -10.0, 10.0, 0.0, key="wind_z", help="正值向上")
    
    wind_total = np.sqrt(wx**2 + wy**2 + wz**2)
    st.sidebar.caption(f"💡 总风速: {wind_total:.1f} m/s")

    # 5. 仿真控制
    st.sidebar.divider()
    dt = st.sidebar.slider("时间步长 (s)", 0.001, 0.1, 0.01, 0.001)
    
    if st.sidebar.button("🔄 重置所有参数"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    # --- Main: 运行仿真 ---
    if st.sidebar.button("🚀 开始仿真", type="primary", use_container_width=True):
        
        # 封装参数
        params = SimulationParams(
            ball_height=ball_h,
            ball_radius=ball_r_cm / 100.0,
            ball_density=st.session_state.ball_density,
            mode=mode,
            v0=np.array([vx, vy, vz]),
            wind_velocity=np.array([wx, wy, wz]),
            dt=dt
        )
        
        with st.spinner("正在计算轨迹..."):
            result = PhysicsEngine.run_simulation(params)
        
        st.success("✅ 仿真完成！")

        # 参数回顾
        st.markdown('<div class="feature-card"><h3>⚙️ 仿真参数摘要</h3></div>', unsafe_allow_html=True)
        with st.expander("查看详细参数"):
            c1, c2 = st.columns(2)
            c1.write(f"**基本**: 模式={mode}, 高度={ball_h}m, 半径={ball_r_cm}cm, 密度={params.ball_density}kg/m³")
            c2.write(f"**速度**: V0=({vx}, {vy}, {vz})m/s, 风速=({wx}, {wy}, {wz})m/s")

        # 结果指标
        fmt = ".4f"
        cols = st.columns(5)
        metrics = [
            ("初始高度", ball_h, "m"), ("飞行时间", result.flight_time, "s"),
            ("落地点 X", result.landing_point[0], "m"), ("落地点 Y", result.landing_point[1], "m"), 
            ("落地点 Z", result.landing_point[2], "m")
        ]
        for c, (lbl, val, unit) in zip(cols, metrics):
            c.metric(lbl, f"{val:{fmt}} {unit}")

        # 物理属性与风力
        st.subheader("📊 详细物理数据")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("质量", f"{result.mass:.4f} kg")
        c2.metric("体积", f"{result.volume:.6f} m³")
        c3.metric("迎风面积", f"{np.pi * params.ball_radius**2:.6f} m²")
        
        # 风力加速度估算 (最大值)
        max_drag = 0.5 * CONST_AIR_DENSITY * CONST_DRAG_COEFF * (np.pi * params.ball_radius**2) * wind_total**2
        wind_acc = max_drag / result.mass
        c4.metric("最大风力加速度", f"{wind_acc:.4f} m/s²")
        
        # 智能提示 (Info Boxes)
        wind_msg = f"**风力分析**: 风速 {wind_total:.2f} m/s. "
        if wind_total == 0: wind_msg += "无风环境。"
        elif wind_acc > 0.5: wind_msg += "🌪️ 风力影响显著，落地点偏移较大。"
        else: wind_msg += "✅ 风力影响适中。"
        st.info(wind_msg)
        
        work_msg = f"**能量分析**: 重力做功 {result.gravity_work:.2f}J (势能转化), 风力做功 {result.wind_work:.2f}J."
        if result.wind_work > 0: work_msg += " (风力助推)"
        elif result.wind_work < 0: work_msg += " (风力阻碍)"
        st.info(work_msg)

        # 3D 可视化
        st.subheader("🎯 3D 轨迹可视化")
        c_anim1, c_anim2 = st.columns([1, 3])
        show_anim = c_anim1.checkbox("🎬 显示动画", value=True)
        speed = c_anim2.slider("播放倍速", 0.1, 10.0, 1.0, disabled=not show_anim)
        
        if show_anim:
            st.plotly_chart(create_3d_figure(result, dt, speed), use_container_width=True)
        
        # 2D 投影
        st.subheader("📐 2D 投影图")
        plot_2d_projection(result)

        # 底部参考资料
        with st.expander("📚 密度参考与物理公式"):
            st.dataframe(pd.DataFrame([
                ["水", 1000, "液体"], ["铁", 7860, "金属"], ["空气", 1.29, "气体"]
            ], columns=["物体", "密度(kg/m³)", "类别"]), hide_index=True, use_container_width=True)
            st.markdown(r"""
            $$ \vec{F}_{total} = m\vec{g} + \frac{1}{2}\rho C_d A |\vec{v}_{rel}|\vec{v}_{rel} $$
            """)

if __name__ == "__main__":
    main()
