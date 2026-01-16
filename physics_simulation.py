"""
基于Python + Streamlit的3D物理仿真程序
模拟无人机投放小球的运动轨迹，考虑重力和风力作用
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(
    page_title="3D物理仿真 - 无人机投掷小球",
    page_icon="🚁",
    layout="wide"
)

# 自定义CSS样式，隐藏默认标题，创建紧凑标题
st.markdown("""
<style>
/* 隐藏Streamlit默认标题 */
.stTitle {
    display: none !important;
}

/* 创建紧凑标题样式 */
.compact-header {
    position: sticky;
    top: 0;
    background-color: #ffffff;
    z-index: 999;
    padding: 0.3rem 1rem;
    margin: 0;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    align-items: center;
}

.compact-header h1 {
    font-size: 1.4rem;
    margin: 0;
    padding: 0;
    line-height: 1.3;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# 使用HTML创建紧凑标题
st.markdown("""
<div class="compact-header">
    <h1>🚁 3D物理仿真 - 无人机投掷小球</h1>
</div>
""", unsafe_allow_html=True)

# ==================== 物理常量定义 ====================
G = 10.0  # 重力加速度 (m/s²)
AIR_DENSITY = 1.225  # 空气密度 (kg/m³)
DRAG_COEFFICIENT = 0.42  # 球体空气阻力系数
BALL_DENSITY = 1000  # 小球密度 (kg/m³)，假设为水的密度

# ==================== 物理计算函数 ====================

def calculate_ball_properties(radius, density):
    """
    计算小球的物理属性
    
    参数:
        radius: 小球半径 (m)
        density: 小球密度 (kg/m³)
    """
    # 体积 V = 4/3 * π * r³
    volume = (4/3) * np.pi * radius**3
    
    # 质量 m = ρ * V
    mass = density * volume
    
    # 迎风面积 A = π * r²
    cross_sectional_area = np.pi * radius**2
    
    return volume, mass, cross_sectional_area

def calculate_forces(ball_velocity, wind_velocity, mass, cross_sectional_area):
    """
    计算作用在小球上的合力
    
    参数:
        ball_velocity: 小球速度向量 [vx, vy, vz] (m/s)
        wind_velocity: 风速向量 [wx, wy, wz] (m/s)
        mass: 小球质量 (kg)
        cross_sectional_area: 迎风面积 (m²)
    
    返回:
        total_force: 合力向量 [Fx, Fy, Fz] (N)
    """
    # 1. 重力 F_g = m * g (沿Z轴负方向)
    gravity_force = np.array([0, 0, -mass * G])
    
    # 2. 风力/空气阻力
    # 相对速度 v_rel = v_wind - v_ball
    relative_velocity = wind_velocity - ball_velocity
    
    # 相对速度的大小
    relative_speed = np.linalg.norm(relative_velocity)
    
    if relative_speed > 0:
        # 空气阻力方程: F_wind = 0.5 * ρ * Cd * A * |v_rel| * v_rel
        drag_force = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * cross_sectional_area * relative_speed * relative_velocity
    else:
        drag_force = np.array([0, 0, 0])
    
    # 3. 合力
    total_force = gravity_force + drag_force
    
    return total_force

def simulate_trajectory(params):
    """
    模拟小球运动轨迹
    
    参数:
        params: 包含所有仿真参数的字典
    
    返回:
        trajectory: 轨迹点数组 (N x 3)
        time: 时间数组
        landing_point: 落地点坐标
        flight_time: 飞行时间
    """
    # 提取参数
    ball_height = params['ball_height']
    ball_radius = params['ball_radius']
    ball_density = params['ball_density']
    mode = params['mode']
    v0_x = params['v0_x']
    v0_y = params['v0_y']
    v0_z = params['v0_z']
    wind_speed_x = params['wind_speed_x']
    wind_speed_y = params['wind_speed_y']
    wind_speed_z = params['wind_speed_z']
    dt = params['dt']
    
    # 计算小球属性
    volume, mass, cross_sectional_area = calculate_ball_properties(ball_radius, ball_density)
    
    # 初始位置（小球高度）
    initial_position = np.array([0.0, 0.0, ball_height], dtype=float)
    
    # 初始速度
    if mode == '自由落体':
        initial_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
    elif mode == '平抛运动':
        initial_velocity = np.array([float(v0_x), 0.0, 0.0], dtype=float)
    elif mode == '斜抛运动':
        initial_velocity = np.array([float(v0_x), float(v0_y), float(v0_z)], dtype=float)
    
    # 风速向量
    wind_velocity = np.array([float(wind_speed_x), float(wind_speed_y), float(wind_speed_z)], dtype=float)
    
    # 数值积分（使用半隐式欧拉法）
    position = initial_position.copy()
    velocity = initial_velocity.copy()
    
    trajectory = [position.copy()]
    time_points = [0]
    t = 0
    
    # 模拟循环
    while position[2] > 0:  # 直到小球落地（Z坐标等于0，即地面）
        # 计算合力
        force = calculate_forces(velocity, wind_velocity, mass, cross_sectional_area)
        
        # 计算加速度 a = F / m
        acceleration = force / mass
        
        # 更新速度 v = v + a * dt
        velocity += acceleration * dt
        
        # 更新位置 p = p + v * dt
        position += velocity * dt
        
        # 更新时间
        t += dt
        
        # 记录轨迹
        trajectory.append(position.copy())
        time_points.append(t)
        
        # 安全检查：防止无限循环
        if t > 100:  # 最多模拟100秒
            break
    
    # 转换为numpy数组
    trajectory = np.array(trajectory)
    time_points = np.array(time_points)
    
    # 修正落地点坐标
    # 如果最后一个点的Z坐标小于0（穿过了地面），进行线性插值修正
    if trajectory[-1, 2] < 0:
        # 获取倒数第二个点（在地面以上）
        prev_point = trajectory[-2]
        last_point = trajectory[-1]
        
        # 计算穿过地面的比例
        # prev_z > 0, last_z < 0
        # 我们要找到z=0的点
        z_prev = prev_point[2]
        z_last = last_point[2]
        
        # 线性插值比例：从prev_point到last_point，z从正数变为负数
        # 我们要找到z=0的位置的比例
        ratio = z_prev / (z_prev - z_last)  # 0 < ratio < 1
        
        # 插值计算落地点
        landing_point = prev_point + ratio * (last_point - prev_point)
        landing_point[2] = 0.0  # 确保Z坐标为0
        
        # 插值计算飞行时间
        flight_time = time_points[-2] + ratio * (time_points[-1] - time_points[-2])
    else:
        # 如果最后一个点的Z坐标刚好为0或大于0（理论上不应该发生）
        landing_point = trajectory[-1]
        landing_point[2] = max(0.0, landing_point[2])  # 确保Z坐标不为负
        flight_time = time_points[-1]
    
    return trajectory, time_points, landing_point, flight_time, mass, volume

# ==================== Streamlit界面 ====================

# 侧边栏参数设置
st.sidebar.header("⚙️ 参数设置")

# 运动模式选择
mode = st.sidebar.selectbox(
    "运动模式",
    ["自由落体", "平抛运动", "斜抛运动"],
    help="选择小球的初始运动模式"
)

# 基础参数
st.sidebar.subheader("📏 基础参数")

# 高度模式选择
height_mode = st.sidebar.radio(
    "高度模式",
    ["无人机高度", "小球高度"],
    help="选择设置的是无人机高度还是小球初始高度"
)

if height_mode == "无人机高度":
    # 初始化 session_state
    if 'drone_height' not in st.session_state:
        st.session_state.drone_height = 5.0
    
    # 回调函数：从滑动条更新输入框
    def update_drone_height_from_slider():
        st.session_state.drone_height_input = st.session_state.drone_height_slider
        st.session_state.drone_height = st.session_state.drone_height_slider
    
    # 回调函数：从输入框更新滑动条
    def update_drone_height_from_input():
        st.session_state.drone_height_slider = st.session_state.drone_height_input
        st.session_state.drone_height = st.session_state.drone_height_input
    
    col_h1, col_h2 = st.sidebar.columns([3, 1])
    with col_h1:
        st.slider(
            "无人机高度 (m)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.drone_height,
            step=0.1,
            key="drone_height_slider",
            on_change=update_drone_height_from_slider,
            help="无人机距离地面的高度"
        )
    with col_h2:
        st.number_input(
            "无人机高度 (m)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.drone_height,
            step=0.1,
            key="drone_height_input",
            on_change=update_drone_height_from_input,
            label_visibility="collapsed"
        )
    drone_height = st.session_state.drone_height
    ball_height = drone_height - 0.1  # 小球在无人机下方10cm
else:
    # 初始化 session_state
    if 'ball_height' not in st.session_state:
        st.session_state.ball_height = 5.0
    
    # 回调函数：从滑动条更新输入框
    def update_ball_height_from_slider():
        st.session_state.ball_height_input = st.session_state.ball_height_slider
        st.session_state.ball_height = st.session_state.ball_height_slider
    
    # 回调函数：从输入框更新滑动条
    def update_ball_height_from_input():
        st.session_state.ball_height_slider = st.session_state.ball_height_input
        st.session_state.ball_height = st.session_state.ball_height_input
    
    col_h1, col_h2 = st.sidebar.columns([3, 1])
    with col_h1:
        st.slider(
            "小球高度 (m)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.ball_height,
            step=0.1,
            key="ball_height_slider",
            on_change=update_ball_height_from_slider,
            help="小球初始距离地面的高度"
        )
    with col_h2:
        st.number_input(
            "小球高度 (m)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.ball_height,
            step=0.1,
            key="ball_height_input",
            on_change=update_ball_height_from_input,
            label_visibility="collapsed"
        )
    ball_height = st.session_state.ball_height
    drone_height = ball_height + 0.1  # 无人机在小球上方10cm

# 小球半径（cm转换为m）
# 初始化 session_state
if 'ball_radius_cm' not in st.session_state:
    st.session_state.ball_radius_cm = 10.0  # 增大半径，让风力效果更明显

# 回调函数：从滑动条更新输入框
def update_ball_radius_from_slider():
    st.session_state.ball_radius_input = st.session_state.ball_radius_slider
    st.session_state.ball_radius_cm = st.session_state.ball_radius_slider

# 回调函数：从输入框更新滑动条
def update_ball_radius_from_input():
    st.session_state.ball_radius_slider = st.session_state.ball_radius_input
    st.session_state.ball_radius_cm = st.session_state.ball_radius_input

col_r1, col_r2 = st.sidebar.columns([3, 1])
with col_r1:
    st.slider(
        "小球半径 (cm)",
        min_value=1.0,
        max_value=1000.0,
        value=st.session_state.ball_radius_cm,
        step=1.0,
        key="ball_radius_slider",
        on_change=update_ball_radius_from_slider,
        help="小球的半径，用于计算体积和质量"
    )
with col_r2:
    st.number_input(
        "小球半径 (cm)",
        min_value=1.0,
        max_value=1000.0,
        value=st.session_state.ball_radius_cm,
        step=1.0,
        key="ball_radius_input",
        on_change=update_ball_radius_from_input,
        label_visibility="collapsed"
    )
ball_radius = st.session_state.ball_radius_cm / 100  # 转换为米

# 小球密度
if 'ball_density' not in st.session_state:
    st.session_state.ball_density = 100.0  # 改为较小的密度，让风力效果更明显
if 'ball_density_input' not in st.session_state:
    st.session_state.ball_density_input = st.session_state.ball_density

# 回调函数：同步输入值到主值
def update_ball_density():
    st.session_state.ball_density = st.session_state.ball_density_input

st.sidebar.number_input(
    "小球密度 (kg/m³)",
    min_value=1.0,
    max_value=20000.0,
    value=st.session_state.ball_density_input,
    step=1.0,
    key="ball_density_input",
    on_change=update_ball_density
)
ball_density = st.session_state.ball_density

# 常见物体密度参考
with st.sidebar.expander("📚 常见物体密度参考"):
    # 密度数据
    density_data = [
        # 金属
        ["金", 19320, "金属"],
        ["银", 10500, "金属"],
        ["铜", 8930, "金属"],
        ["铁/钢", 7860, "金属"],
        ["铝", 2700, "金属"],
        # 液体
        ["汞(水银)", 13550, "液体"],
        ["海水", 1025, "液体"],
        ["水(纯水)", 1000, "液体"],
        ["酒精", 789, "液体"],
        ["汽油", 730, "液体"],
        # 建筑材料
        ["花岗岩", 2700, "建筑材料"],
        ["混凝土", 2400, "建筑材料"],
        ["玻璃", 2600, "建筑材料"],
        ["砖", 1800, "建筑材料"],
        # 气体
        ["二氧化碳", 1.98, "气体"],
        ["空气", 1.29, "气体"],
        ["氦气", 0.178, "气体"],
        # 其他
        ["冰", 917, "其他"],
        ["干松木", 500, "其他"],
        ["塑料", 935, "其他"],
        ["人体平均", 1002, "其他"],
    ]
    
    # 创建DataFrame
    import pandas as pd
    density_df = pd.DataFrame(density_data, columns=["物体", "密度 (kg/m³)", "类别"])
    
    # 按类别分组显示
    categories = density_df["类别"].unique()
    
    for category in categories:
        st.write(f"**{category}**")
        category_data = density_df[density_df["类别"] == category]
        
        # 创建列来显示物体名称和按钮
        for idx, row in category_data.iterrows():
            col_name, col_btn = st.columns([2, 1])
            with col_name:
                st.write(f"{row['物体']}: {row['密度 (kg/m³)']}")
            with col_btn:
                if st.button("使用", key=f"density_btn_{idx}"):
                    density_value = float(row['密度 (kg/m³)'])
                    st.session_state.ball_density = density_value
                    st.session_state.ball_density_input = density_value
                    st.rerun()
        st.divider()
    
    # 显示完整表格
    st.write("**完整密度表格**")
    st.dataframe(density_df, width='stretch', hide_index=True)
    
    # 显示单位换算说明
    st.info("💡 **单位换算**: 1 g/cm³ = 1000 kg/m³")

# 初速度设置
st.sidebar.subheader("🚀 初速度设置")

if mode == "平抛运动":
    if 'v0_x' not in st.session_state:
        st.session_state.v0_x = 20.0
    
    # 回调函数：更新初速度
    def update_v0_x():
        st.session_state.v0_x = st.session_state.v0_x_slider
    
    st.sidebar.slider("水平初速度 v₀x (m/s)", 0.0, 50.0, st.session_state.v0_x, 1.0, key="v0_x_slider", on_change=update_v0_x)
    v0_x = st.session_state.v0_x
    v0_y = 0.0
    v0_z = 0.0
elif mode == "斜抛运动":
    if 'v0_x' not in st.session_state:
        st.session_state.v0_x = 20.0
    if 'v0_y' not in st.session_state:
        st.session_state.v0_y = 0.0
    if 'v0_z' not in st.session_state:
        st.session_state.v0_z = 10.0
    
    # 回调函数：更新初速度
    def update_v0_x():
        st.session_state.v0_x = st.session_state.v0_x_slider
    
    def update_v0_y():
        st.session_state.v0_y = st.session_state.v0_y_slider
    
    def update_v0_z():
        st.session_state.v0_z = st.session_state.v0_z_slider
    
    st.sidebar.slider("水平初速度 v₀x (m/s)", 0.0, 50.0, st.session_state.v0_x, 1.0, key="v0_x_slider", on_change=update_v0_x)
    st.sidebar.slider("侧向初速度 v₀y (m/s)", 0.0, 50.0, st.session_state.v0_y, 1.0, key="v0_y_slider", on_change=update_v0_y)
    st.sidebar.slider("垂直初速度 v₀z (m/s)", 0.0, 50.0, st.session_state.v0_z, 1.0, key="v0_z_slider", on_change=update_v0_z)
    v0_x = st.session_state.v0_x
    v0_y = st.session_state.v0_y
    v0_z = st.session_state.v0_z
else:  # 自由落体
    v0_x = 0.0
    v0_y = 0.0
    v0_z = 0.0

# 风力设置
st.sidebar.subheader("💨 风力设置")
if 'wind_speed_x' not in st.session_state:
    st.session_state.wind_speed_x = 0.0
if 'wind_speed_y' not in st.session_state:
    st.session_state.wind_speed_y = 0.0
if 'wind_speed_z' not in st.session_state:
    st.session_state.wind_speed_z = 0.0

# 回调函数：更新风速
def update_wind_speed_x():
    st.session_state.wind_speed_x = st.session_state.wind_speed_x_slider

def update_wind_speed_y():
    st.session_state.wind_speed_y = st.session_state.wind_speed_y_slider

def update_wind_speed_z():
    st.session_state.wind_speed_z = st.session_state.wind_speed_z_slider

st.sidebar.slider(
    "X方向风速 (m/s)",
    min_value=-20.0,
    max_value=20.0,
    value=st.session_state.wind_speed_x,
    step=1.0,
    key="wind_speed_x_slider",
    on_change=update_wind_speed_x,
    help="正值向右，负值向左"
)
st.sidebar.slider(
    "Y方向风速 (m/s)",
    min_value=-20.0,
    max_value=20.0,
    value=st.session_state.wind_speed_y,
    step=1.0,
    key="wind_speed_y_slider",
    on_change=update_wind_speed_y,
    help="正值向前，负值向后"
)
st.sidebar.slider(
    "Z方向风速 (m/s)",
    min_value=-10.0,
    max_value=10.0,
    value=st.session_state.wind_speed_z,
    step=1.0,
    key="wind_speed_z_slider",
    on_change=update_wind_speed_z,
    help="正值向上，负值向下"
)
wind_speed_x = st.session_state.wind_speed_x
wind_speed_y = st.session_state.wind_speed_y
wind_speed_z = st.session_state.wind_speed_z

# 风力效果提示
st.sidebar.info("💡 **增强风力效果的方法**：\n- 增加风速（最大20m/s）\n- 增大小球半径\n- 减小球球密度\n- 增加初始高度")

# 仿真参数
st.sidebar.subheader("🔬 仿真参数")
if 'dt' not in st.session_state:
    st.session_state.dt = 0.01

# 回调函数：更新时间步长
def update_dt():
    st.session_state.dt = st.session_state.dt_slider

st.sidebar.slider(
    "时间步长 (s)",
    min_value=0.001,
    max_value=0.1,
    value=st.session_state.dt,
    step=0.001,
    key="dt_slider",
    on_change=update_dt,
    help="数值积分的时间步长"
)
dt = st.session_state.dt

# 开始仿真按钮
if st.sidebar.button("🚀 开始仿真", type="primary"):
    # 构建参数字典
    params = {
        'ball_height': ball_height,
        'ball_radius': ball_radius,
        'ball_density': ball_density,
        'mode': mode,
        'v0_x': v0_x,
        'v0_y': v0_y,
        'v0_z': v0_z,
        'wind_speed_x': wind_speed_x,
        'wind_speed_y': wind_speed_y,
        'wind_speed_z': wind_speed_z,
        'dt': dt
    }
    
    # 运行仿真
    with st.spinner("正在计算轨迹..."):
        trajectory, time_points, landing_point, flight_time, mass, volume = simulate_trajectory(params)
    
    # 显示结果
    st.success("✅ 仿真完成！")
    
    # 根据精确度格式化数据
    format_str = ".4f"
    
    # 分栏显示结果
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("初始高度", f"{ball_height:{format_str}} m")
    col2.metric("飞行时间", f"{flight_time:{format_str}} s")
    col3.metric("落地点 X", f"{landing_point[0]:{format_str}} m")
    col4.metric("落地点 Y", f"{landing_point[1]:{format_str}} m")
    col5.metric("落地点 Z", f"{landing_point[2]:{format_str}} m")
    
    # 显示小球属性
    st.subheader("📊 小球物理属性")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("小球质量", f"{mass:{format_str}} kg")
    col6.metric("小球体积", f"{volume:{format_str}} m³")
    col7.metric("小球密度", f"{ball_density:{format_str}} kg/m³")
    col8.metric("迎风面积", f"{np.pi * ball_radius**2:{format_str}} m²")
    
    # 显示风力信息
    st.subheader("💨 风力影响")
    wind_speed = np.sqrt(wind_speed_x**2 + wind_speed_y**2 + wind_speed_z**2)
    col_wind1, col_wind2, col_wind3, col_wind4 = st.columns(4)
    col_wind1.metric("风速大小", f"{wind_speed:{format_str}} m/s")
    col_wind2.metric("X方向风速", f"{wind_speed_x:{format_str}} m/s")
    col_wind3.metric("Y方向风速", f"{wind_speed_y:{format_str}} m/s")
    col_wind4.metric("Z方向风速", f"{wind_speed_z:{format_str}} m/s")
    
    # 计算最大风力产生的加速度（假设小球静止）
    max_drag_force = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * (np.pi * ball_radius**2) * wind_speed**2
    max_acceleration = max_drag_force / mass if mass > 0 else 0
    col_wind5, col_wind6 = st.columns(2)
    col_wind5.metric("最大空气阻力", f"{max_drag_force:{format_str}} N")
    col_wind6.metric("风力加速度", f"{max_acceleration:{format_str}} m/s²")
    
    # 风力影响说明
    wind_info = f"""
    **风力影响分析：**
    - 当前风速：{wind_speed:{format_str}} m/s
    - 风力加速度：{max_acceleration:{format_str}} m/s²
    - 相对重力加速度：{max_acceleration/G:.2%}
    """
    if wind_speed > 0:
        if max_acceleration < 0.1:
            wind_info += "\n⚠️ **风力影响较小**：建议增加风速、增大小球半径或减小小球密度以增强风力效果。"
        elif max_acceleration < 0.5:
            wind_info += "\n✅ **风力影响适中**：风力会对落地点产生明显影响。"
        else:
            wind_info += "\n🌪️ **风力影响显著**：风力将大幅改变落地点位置。"
    else:
        wind_info += "\nℹ️ **无风**：小球将按自由落体或抛物线运动。"
    
    st.info(wind_info)
    
    # 3D轨迹可视化
    st.subheader("🎯 3D轨迹可视化")
    
    # 初始化session_state
    if 'animation_speed' not in st.session_state:
        st.session_state.animation_speed = 1.0  # 默认1倍速（真实时间）
    if 'show_animation' not in st.session_state:
        st.session_state.show_animation = True
    
    # 计算动画帧持续时间（毫秒）
    # 真实时间播放：每帧的时间间隔应该等于dt（时间步长）
    # duration = dt * 1000 / speed（毫秒）
    frame_duration = dt * 1000 / st.session_state.animation_speed
    
    # 动画控制面板
    col_anim1, col_anim2, col_anim3 = st.columns([2, 2, 2])
    with col_anim1:
        st.checkbox("🎬 显示动画", value=st.session_state.show_animation, key="show_anim_checkbox", help="勾选后显示小球运动动画")
    with col_anim2:
        if st.session_state.show_animation:
            st.slider(
                "播放速度倍率", 
                0.1, 10.0, 
                st.session_state.animation_speed, 0.1, 
                help=f"1.0x = 真实时间（每帧{dt*1000:{format_str}}ms），当前每帧{frame_duration:{format_str}}ms",
                key="anim_speed_slider",
                disabled=True
            )
    with col_anim3:
        if st.session_state.show_animation:
            auto_play = st.checkbox("自动播放", value=True, help="勾选后自动播放动画", key="auto_play_checkbox")
    
    # 显示时间信息
    if st.session_state.show_animation:
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            st.info(f"📊 真实飞行时间: {flight_time:{format_str}} 秒")
        with col_time2:
            st.info(f"🎬 动画播放时间: {flight_time / st.session_state.animation_speed:{format_str}} 秒")
    
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scatter3d'}]]
    )
    
    # 添加地面网格
    grid_range = max(abs(landing_point[0]), abs(landing_point[1]), 50)
    x_grid = np.linspace(-grid_range, grid_range, 20)
    y_grid = np.linspace(-grid_range, grid_range, 20)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = np.zeros_like(X_grid)
    
    fig.add_trace(
        go.Surface(
            x=X_grid,
            y=Y_grid,
            z=Z_grid,
            colorscale='Greys',
            showscale=False,
            opacity=0.3,
            name='地面'
        )
    )
    
    # 添加轨迹线
    fig.add_trace(
        go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines',
            name='运动轨迹',
            line=dict(color='blue', width=4)
        )
    )
    
    # 添加起点
    fig.add_trace(
        go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            name='起点',
            marker=dict(color='green', size=10)
        )
    )
    
    # 添加落地点
    fig.add_trace(
        go.Scatter3d(
            x=[landing_point[0]],
            y=[landing_point[1]],
            z=[landing_point[2]],
            mode='markers',
            name='落地点',
            marker=dict(color='red', size=10)
        )
    )
    
    # 添加动画小球
    frames = []
    # 采样率：根据轨迹长度和播放速度动态调整
    # 目标：保持动画流畅，同时帧数不过多
    base_sampling_rate = max(1, int(len(trajectory) * 0.01))  # 基础采样率1%
    
    # 高速播放时增加采样间隔，减少帧数
    if st.session_state.animation_speed > 2.0:
        sampling_rate = max(1, int(len(trajectory) * 0.005))  # 0.5%采样
    elif st.session_state.animation_speed > 5.0:
        sampling_rate = max(1, int(len(trajectory) * 0.002))  # 0.2%采样
    else:
        sampling_rate = base_sampling_rate
    
    for i in range(0, len(trajectory), sampling_rate):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=[trajectory[i, 0]],
                    y=[trajectory[i, 1]],
                    z=[trajectory[i, 2]],
                    mode='markers',
                    name='小球',
                    marker=dict(color='orange', size=15),
                    showlegend=False
                )
            ],
            name=f'frame_{i}'
        )
        frames.append(frame)
    
    # 添加初始小球位置
    fig.add_trace(
        go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            name='小球',
            marker=dict(color='orange', size=15),
            showlegend=False
        )
    )
    
    fig.frames = frames
    
    # 设置布局
    fig.update_layout(
        title="小球运动轨迹 (3D视图)",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        height=700,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': int(frame_duration), 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }],
                    'label': '▶ 播放',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': '⏸ 暂停',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': '⏮ 重放',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': '进度: '},
            'len': 0.9,
            'steps': [
                {
                    'args': [[f'frame_{i}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f'{int(i / len(frames) * 100)}%',
                    'method': 'animate'
                }
                for i in range(0, len(frames), max(1, len(frames) // 10))
            ],
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # 2D投影图
    st.subheader("📐 2D投影图")
    
    col9, col10 = st.columns(2)
    
    # XZ平面投影
    fig_xz = go.Figure()
    fig_xz.add_trace(go.Scatter(
        x=trajectory[:, 0],
        y=trajectory[:, 2],
        mode='lines',
        name='轨迹',
        line=dict(color='blue', width=3)
    ))
    fig_xz.add_trace(go.Scatter(
        x=[trajectory[0, 0]],
        y=[trajectory[0, 2]],
        mode='markers',
        name='起点',
        marker=dict(color='green', size=10)
    ))
    fig_xz.add_trace(go.Scatter(
        x=[landing_point[0]],
        y=[landing_point[2]],
        mode='markers',
        name='落地点',
        marker=dict(color='red', size=10)
    ))
    fig_xz.update_layout(
        title="XZ平面投影 (侧视图)",
        xaxis_title='X (m)',
        yaxis_title='Z (m)',
        height=400
    )
    col9.plotly_chart(fig_xz, width='stretch')
    
    # XY平面投影
    fig_xy = go.Figure()
    fig_xy.add_trace(go.Scatter(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        mode='lines',
        name='轨迹',
        line=dict(color='blue', width=3)
    ))
    fig_xy.add_trace(go.Scatter(
        x=[trajectory[0, 0]],
        y=[trajectory[0, 1]],
        mode='markers',
        name='起点',
        marker=dict(color='green', size=10)
    ))
    fig_xy.add_trace(go.Scatter(
        x=[landing_point[0]],
        y=[landing_point[1]],
        mode='markers',
        name='落地点',
        marker=dict(color='red', size=10)
    ))
    fig_xy.update_layout(
        title="XY平面投影 (俯视图)",
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        height=400
    )
    col10.plotly_chart(fig_xy, width='stretch')
    
    # 显示物理公式说明
    st.subheader("📚 物理公式说明")
    st.markdown("""
    **1. 小球属性计算：**
    - 体积：$V = \\frac{4}{3}\\pi r^3$
    - 质量：$m = \\rho_{ball} \\times V$
    - 迎风面积：$A = \\pi r^2$
    
    **2. 受力分析：**
    - 重力：$\\vec{F}_g = m \\times \\vec{g} = (0, 0, -mg)$
    - 空气阻力：$\\vec{F}_{drag} = \\frac{1}{2} \\rho C_d A |\\vec{v}_{rel}| \\vec{v}_{rel}$
    - 相对速度：$\\vec{v}_{rel} = \\vec{v}_{wind} - \\vec{v}_{ball}$
    
    **3. 运动方程：**
    - 合力：$\\vec{F}_{total} = \\vec{F}_g + \\vec{F}_{drag}$
    - 加速度：$\\vec{a} = \\frac{\\vec{F}_{total}}{m}$
    - 速度更新：$\\vec{v}_{new} = \\vec{v}_{old} + \\vec{a} \\times \\Delta t$
    - 位置更新：$\\vec{p}_{new} = \\vec{p}_{old} + \\vec{v}_{new} \\times \\Delta t$
    """)

# 页脚
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>💡 提示：调整侧边栏参数后点击'开始仿真'按钮查看结果</div>", unsafe_allow_html=True)
