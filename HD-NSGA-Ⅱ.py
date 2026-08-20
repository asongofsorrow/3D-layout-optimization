import os
import time
import tracemalloc
import math
import random
import copy
import csv
import numpy as np
import plotly.graph_objects as go

# 只计算NSGAⅡ+HD的结果

# ==========================================
# 1. 全局环境、物理限制与参数定义
# ==========================================
POP_SIZE = 100  # 种群数量
NUM_ITERATIONS = 200  # 迭代轮数
CROSSOVER_RATE = 0.9  # 基因交换概率
MUTATION_RATE = 0.2  # 旋转状态突变概率
RACK_LENGTH = 1520
RACK_WIDTH = 660

# --- 硬约束阈值 ---
LIMIT_COG_X = 30
LIMIT_COG_Y = 5
LIMIT_COG_Z = 200  # 约束1：Z方向重心高度不超过 200mm
LIMIT_DEV12_LOAD = 20.0  # 约束2：设备12允许承重不超过 20kg

# 热传递效率 (基于文档一维热力场假设)
ETA_THERMAL = 0.8
T_ENV = 10  # 机柜内初始环境温度 20°C
# 对归一化后的 norm_fits 进行加权
WEIGHTS = np.array([0.5, 0.3, 0.2])  # 0.5*转动惯量 + 0.3*热力 + 0.2*走线

# --- 新增：全局计算量计数器 ---
GLOBAL_FITNESS_EVALS = 0

# 各类别设备的最大承重极限 (kg)
MAX_LOAD_CAPACITY = {
    1: 5.0, 2: 5.0, 3: 10.0,
    4: 50.0, 5: 30.0, 6: 8.0
}


# ==========================================
# 2. 设备实体类与基础数据实例化
# ==========================================
class Device:
    def __init__(self, dev_id, name, length, width, height, weight, category, Q, T_max, R):
        self.id = dev_id
        self.name = name
        self.L = length
        self.W = width
        self.H = height
        self.weight = weight
        self.category = category
        self.Q = Q  # 发热功率 (W)
        self.T_max = T_max  # 允许最高环境温度 (°C)
        self.R = R  # 热阻 (℃/W)


# 根据文档表格对所有设备赋予真实的物理热参数
DEVICE_LIST = [
    Device(1, "Attitude control computer", 200, 200, 60, 4, 1, 50, 70, 0.8),
    Device(2, "Exploration control computer", 200, 200, 60, 4, 1, 50, 70, 0.8),
    Device(3, "Communication control computer", 200, 200, 60, 4, 1, 50, 70, 0.8),
    Device(4, "Visual data recorder", 200, 150, 75, 3, 2, 40, 50, 0.9),
    Device(5, "Sonar data recorder", 200, 150, 75, 3, 2, 40, 50, 0.9),
    Device(6, "Visual data recorder (redundant)", 200, 150, 75, 3, 2, 40, 50, 0.9),
    Device(7, "Sonar data recorder(redundant)", 200, 150, 75, 3, 2, 40, 50, 0.9),
    Device(8, "Black box", 200, 150, 75, 3, 2, 40, 50, 0.9),
    Device(9, "Main drive electrical control cabinet", 240, 180, 300, 18, 3, 400, 85, 0.3),
    Device(10, "Attitude control electrical control cabinet", 240, 180, 300, 18, 3, 400, 85, 0.3),
    Device(11, "Load distribution electrical control cabinet", 240, 180, 300, 18, 3, 400, 85, 0.3),
    Device(12, "Integration navigation box", 500, 400, 200, 50, 4, 15, 45, 0.15),
    Device(13, "Information reception box", 500, 400, 200, 40, 4, 80, 55, 0.15),
    Device(14, "Underwater communication box", 500, 400, 200, 30, 4, 120, 70, 0.15),
    Device(15, "Emergency monitoring box", 400, 320, 160, 27, 5, 5, 80, 0.2),
    Device(16, "Clock synchronization box", 400, 320, 160, 16, 5, 10, 50, 0.2),
    Device(17, "Visual enhancement workstation", 320, 200, 65, 6, 6, 250, 60, 0.5),
    Device(18, "Sonar analysis workstation", 320, 200, 65, 6, 6, 250, 60, 0.5)
]

W_MATRIX = [[0] * 18 for _ in range(18)]
edges = [
    (1, 8, 1), (1, 9, 5), (1, 10, 5), (1, 11, 1), (1, 12, 5), (1, 15, 1), (1, 16, 5),
    (2, 11, 1), (2, 13, 1), (2, 16, 5),
    (3, 8, 1), (3, 11, 1), (3, 14, 1), (3, 15, 1), (3, 16, 5),
    (4, 17, 10), (5, 18, 10), (6, 17, 10), (7, 18, 10),
    (8, 16, 1), (8, 14, 1),
    (9, 15, 1), (10, 15, 1),
    (11, 12, 1), (11, 13, 1), (11, 14, 1), (11, 15, 1), (11, 16, 1), (11, 17, 1), (11, 18, 1),
    (12, 15, 1), (13, 17, 10), (13, 18, 10), (14, 15, 1), (15, 16, 1)
]
for (u, v, w) in edges:
    W_MATRIX[u - 1][v - 1] = w
    W_MATRIX[v - 1][u - 1] = w


# ==========================================
# 3. 核心物理检测模型
# ==========================================
class PlacedDevice:
    def __init__(self, device, x, y, z, rotation, support=None):
        self.device = device
        self.x = x
        self.y = y
        self.z = z
        self.rotation = rotation
        self.support = support

    def get_effective_dims(self):
        if self.rotation == 90: return self.device.W, self.device.L, self.device.H
        return self.device.L, self.device.W, self.device.H

    def get_area(self):
        return self.get_effective_dims()[0] * self.get_effective_dims()[1]

    def get_center_3d(self):
        return (self.x, self.y, self.z + self.device.H / 2.0)


class Layout:
    def __init__(self):
        self.placed_devices = []
        self.is_valid = True

    def add(self, pd):
        self.placed_devices.append(pd)


def get_surface_distance(pd1, pd2):
    L1, W1, H1 = pd1.get_effective_dims()
    L2, W2, H2 = pd2.get_effective_dims()

    dx = max(0.0, abs(pd1.x - pd2.x) - (L1 + L2) / 2.0)
    dy = max(0.0, abs(pd1.y - pd2.y) - (W1 + W2) / 2.0)

    z1_min, z1_max = pd1.z, pd1.z + H1
    z2_min, z2_max = pd2.z, pd2.z + H2
    dz = max(0.0, max(z1_min - z2_max, z2_min - z1_max))

    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    if z1_min >= z2_max:
        dist_z_directed = z1_min - z2_max
    elif z2_min >= z1_max:
        dist_z_directed = z1_max - z2_min
    else:
        dist_z_directed = 0.0

    return dist, dx, dy, dist_z_directed


def check_collision_aabb(pd1, pd2):
    L1, W1, H1 = pd1.get_effective_dims()
    L2, W2, H2 = pd2.get_effective_dims()
    cx1, cy1, cz1 = pd1.get_center_3d()
    cx2, cy2, cz2 = pd2.get_center_3d()
    return (abs(cx1 - cx2) < (L1 + L2) / 2.0 - 0.1) and \
           (abs(cy1 - cy2) < (W1 + W2) / 2.0 - 0.1) and \
           (abs(cz1 - cz2) < (H1 + H2) / 2.0 - 0.1)


def check_stacking_rules(new_pd, support_pd):
    L_up, W_up, _ = new_pd.get_effective_dims()
    L_dn, W_dn, _ = support_pd.get_effective_dims()
    if L_up > L_dn or W_up > W_dn: return False

    if (new_pd.x - L_up / 2 < support_pd.x - L_dn / 2 or
            new_pd.x + L_up / 2 > support_pd.x + L_dn / 2 or
            new_pd.y - W_up / 2 < support_pd.y - W_dn / 2 or
            new_pd.y + W_up / 2 > support_pd.y + W_dn / 2):
        return False
    return True


def check_load_capacity(new_pd, layout):
    curr_support = new_pd.support
    while curr_support is not None:
        current_load = sum(pd.device.weight for pd in layout.placed_devices if _is_supported_by(pd, curr_support))
        if current_load + new_pd.device.weight > MAX_LOAD_CAPACITY.get(curr_support.device.category, 0):
            return False
        curr_support = curr_support.support
    return True


def _is_supported_by(pd, base_pd):
    curr = pd.support
    while curr:
        if curr == base_pd: return True
        curr = curr.support
    return False


# ==========================================
# 4. 评估模块 (3 目标 + 2 硬约束)
# ==========================================
def calculate_fitness(layout):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS += 1

    if not layout.is_valid:
        return [float('inf')] * 3

    # ================= 重心与承重硬约束检验 =================
    total_w = sum(pd.device.weight for pd in layout.placed_devices)
    sum_wx = sum(pd.device.weight * pd.get_center_3d()[0] for pd in layout.placed_devices)
    sum_wy = sum(pd.device.weight * pd.get_center_3d()[1] for pd in layout.placed_devices)
    sum_wz = sum(pd.device.weight * pd.get_center_3d()[2] for pd in layout.placed_devices)

    cg_x = sum_wx / total_w
    cg_y = sum_wy / total_w
    cg_z = sum_wz / total_w

    # if abs(cg_x) > LIMIT_COG_X or abs(cg_y) > LIMIT_COG_Y or cg_z > LIMIT_COG_Z:
    #     return [float('inf')] * 3

    device_12_load = sum(pd.device.weight for pd in layout.placed_devices if
                         _is_supported_by(pd, next(p for p in layout.placed_devices if p.device.id == 12)))
    # if device_12_load > LIMIT_DEV12_LOAD:
    #     return [float('inf')] * 3

    # 初始化惩罚项
    penalty = 0

    # 如果重心超出，计算超出量并施加巨额惩罚（而不是直接抹杀）
    if abs(cg_x) > LIMIT_COG_X: penalty += (abs(cg_x) - LIMIT_COG_X) * 10000000
    if abs(cg_y) > LIMIT_COG_Y: penalty += (abs(cg_y) - LIMIT_COG_Y) * 10000000
    if cg_z > LIMIT_COG_Z:      penalty += (cg_z - LIMIT_COG_Z) * 10000000
    if device_12_load > LIMIT_DEV12_LOAD: penalty += (device_12_load - LIMIT_DEV12_LOAD) * 10000000

    # ================= F1: 转动惯量模型 =================
    I_xx = 0
    I_yy = 0
    I_zz = 0
    for pd in layout.placed_devices:
        L, W, H = pd.get_effective_dims()
        m = pd.device.weight
        cx, cy, cz = pd.get_center_3d()

        I_cx = (m / 12.0) * (W ** 2 + H ** 2)
        I_cy = (m / 12.0) * (L ** 2 + H ** 2)
        I_cz = (m / 12.0) * (L ** 2 + W ** 2)

        I_xx += I_cx + m * (cy ** 2 + cz ** 2)
        I_yy += I_cy + m * (cx ** 2 + cz ** 2)
        I_zz += I_cz + m * (cx ** 2 + cy ** 2)

    F_inertia = math.sqrt(I_xx ** 2 + I_yy ** 2 + I_zz ** 2) / 1e4

    # ================= F2: 一维垂直热力场模型 =================
    F_thermal = 0
    penalty_thermal = 0
    q_stress_dict = {}

    def get_q_stress(pd):
        if pd in q_stress_dict: return q_stress_dict[pd]

        q_self = pd.device.Q
        support_pd = pd.support
        if support_pd is None:
            q_stress_dict[pd] = q_self
            return q_self
        else:
            q_support = get_q_stress(support_pd)
            area_ratio = min(1.0, pd.get_area() / support_pd.get_area())
            q_total = q_self + ETA_THERMAL * area_ratio * q_support
            q_stress_dict[pd] = q_total
            return q_total

    for pd in layout.placed_devices:
        q_stress = get_q_stress(pd)
        T_i = T_ENV + pd.device.R * q_stress
        F_thermal += T_i

        if T_i > pd.device.T_max:
            penalty_thermal += (T_i - pd.device.T_max) ** 2

    F_thermal += penalty_thermal

    # ================= F3: 信号场 (走线曼哈顿) =================
    F_routing = 0
    for i in range(len(layout.placed_devices)):
        for j in range(i + 1, len(layout.placed_devices)):
            pd1, pd2 = layout.placed_devices[i], layout.placed_devices[j]
            weight = W_MATRIX[pd1.device.id - 1][pd2.device.id - 1]
            if weight > 0:
                _, dx, dy, dz_directed = get_surface_distance(pd1, pd2)
                F_routing += weight * (dx + dy + abs(dz_directed))

    return [F_inertia + penalty, F_thermal + penalty, F_routing + penalty]


# ==========================================
# 5. 布局微调模块 & 6. 启发式装箱解码器
# ==========================================
def get_all_supported_devices(layout, base_pd):
    supported = []
    for pd in layout.placed_devices:
        if pd.support == base_pd:
            supported.append(pd)
            supported.extend(get_all_supported_devices(layout, pd))
    return supported


def fine_tune_layout(layout):
    if not layout.is_valid: return
    for pd in layout.placed_devices:
        if pd.z > 0 and pd.support is not None:
            dx, dy = pd.support.x - pd.x, pd.support.y - pd.y
            if dx == 0 and dy == 0: continue
            family = [pd] + get_all_supported_devices(layout, pd)
            old_positions = {dev: (dev.x, dev.y) for dev in family}
            for dev in family: dev.x += dx; dev.y += dy
            non_family = [d for d in layout.placed_devices if d not in family]
            if any(check_collision_aabb(f, nf) for f in family for nf in non_family) or not check_stacking_rules(pd,
                                                                                                                 pd.support):
                for dev, (ox, oy) in old_positions.items(): dev.x, dev.y = ox, oy

    current_fits = calculate_fitness(layout)
    if current_fits[0] == float('inf'): return
    current_score = sum(current_fits)

    for pd in layout.placed_devices:
        if pd.device.id == 12: continue
        family = [pd] + get_all_supported_devices(layout, pd)
        best_dx, best_dy = 0, 0
        for dx, dy in [(20, 0), (-20, 0), (0, 20), (0, -20)]:
            for dev in family: dev.x += dx; dev.y += dy
            out_of_bounds = any((dev.z == 0 and (abs(dev.x) > RACK_LENGTH / 2 - dev.get_effective_dims()[0] / 2 or
                                                 abs(dev.y) > RACK_WIDTH / 2 - dev.get_effective_dims()[1] / 2)) for dev
                                in family)
            if out_of_bounds:
                for dev in family: dev.x -= dx; dev.y -= dy; continue
            non_family = [d for d in layout.placed_devices if d not in family]
            if any(check_collision_aabb(f, nf) for f in family for nf in non_family) or (
                    pd.support and not check_stacking_rules(pd, pd.support)):
                for dev in family: dev.x -= dx; dev.y -= dy; continue
            new_fits = calculate_fitness(layout)
            if sum(new_fits) < current_score:
                current_score, best_dx, best_dy = sum(new_fits), dx, dy
            for dev in family: dev.x -= dx; dev.y -= dy
        if best_dx != 0 or best_dy != 0:
            for dev in family: dev.x += best_dx; dev.y += best_dy


def heuristic_decoder(chromosome):
    layout = Layout()
    for gene in chromosome:
        dev = next(d for d in DEVICE_LIST if d.id == gene['id'])
        rot = gene['rot']
        L, W, H = dev.W if rot == 90 else dev.L, dev.L if rot == 90 else dev.W, dev.H

        if dev.id == 12:
            layout.add(PlacedDevice(dev, 0, 0, 0, rot, support=None))
            continue

        placed = False
        candidate_positions = []

        x_lim, y_lim = int(RACK_LENGTH / 2 - L / 2), int(RACK_WIDTH / 2 - W / 2)
        if -x_lim <= x_lim and -y_lim <= y_lim:
            for x in range(-x_lim, x_lim + 1, 80):
                for y in range(-y_lim, y_lim + 1, 80):
                    candidate_positions.append({'x': x, 'y': y, 'z': 0, 'support': None})

        for pd in layout.placed_devices:
            L_dn, W_dn, H_dn = pd.get_effective_dims()
            s_start_x, s_end_x = int(pd.x - L_dn / 2 + L / 2), int(pd.x + L_dn / 2 - L / 2)
            s_start_y, s_end_y = int(pd.y - W_dn / 2 + W / 2), int(pd.y + W_dn / 2 - W / 2)

            if s_start_x <= s_end_x and s_start_y <= s_end_y:
                x_points = list(range(s_start_x, s_end_x + 1, 40))
                if s_end_x not in x_points: x_points.append(s_end_x)
                y_points = list(range(s_start_y, s_end_y + 1, 40))
                if s_end_y not in y_points: y_points.append(s_end_y)
                for x in x_points:
                    for y in y_points:
                        candidate_positions.append({'x': x, 'y': y, 'z': pd.z + H_dn, 'support': pd})

        for pos in candidate_positions:
            test_pd = PlacedDevice(dev, pos['x'], pos['y'], pos['z'], rot, support=pos['support'])
            if abs(pos['x']) > RACK_LENGTH / 2 - L / 2 or abs(pos['y']) > RACK_WIDTH / 2 - W / 2: continue
            if any(check_collision_aabb(test_pd, existing) for existing in layout.placed_devices): continue
            if pos['z'] > 0 and (not check_stacking_rules(test_pd, pos['support']) or not check_load_capacity(test_pd,
                                                                                                              layout)): continue
            layout.add(test_pd)
            placed = True
            break

        if not placed:
            layout.is_valid = False
            return layout

    # 关闭局部微调
    fine_tune_layout(layout)
    return layout


# ==========================================
# 7. NSGA-II 核心算法模块 (回退采用拥挤度)
# ==========================================
def generate_random_chromosome():
    heavy_devices = [d for d in DEVICE_LIST if d.id != 12 and d.weight >= 15]
    light_devices = [d for d in DEVICE_LIST if d.id != 12 and d.weight < 15]
    heavy_genes = [{'id': d.id, 'rot': random.choice([0, 90])} for d in heavy_devices]
    light_genes = [{'id': d.id, 'rot': random.choice([0, 90])} for d in light_devices]
    random.shuffle(heavy_genes)
    random.shuffle(light_genes)
    return [{'id': 12, 'rot': random.choice([0, 90])}] + heavy_genes + light_genes


def dominates(fit1, fit2):
    and_condition = False
    for f1, f2 in zip(fit1, fit2):
        if f1 > f2: return False
        if f1 < f2: and_condition = True
    return and_condition


def fast_non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p['S'], p['n'] = [], 0
        for q in population:
            if dominates(p['fit'], q['fit']):
                p['S'].append(q)
            elif dominates(q['fit'], p['fit']):
                p['n'] += 1
        if p['n'] == 0:
            p['rank'] = 0
            fronts[0].append(p)
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p['S']:
                q['n'] -= 1
                if q['n'] == 0: q['rank'] = i + 1; next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


# ★ NSGA-II 专属拥挤度计算机制
def crowding_distance_assignment(front):
    length = len(front)
    for p in front: p['distance'] = 0
    if length <= 2:
        for p in front: p['distance'] = float('inf')
        return

    # 针对3个物理目标分别计算间距分布
    num_objectives = 3
    for m in range(num_objectives):
        front.sort(key=lambda x: x['fit'][m])
        front[0]['distance'] = float('inf')
        front[-1]['distance'] = float('inf')
        f_max = front[-1]['fit'][m]
        f_min = front[0]['fit'][m]

        if f_max == f_min: continue

        for i in range(1, length - 1):
            front[i]['distance'] += (front[i + 1]['fit'][m] - front[i - 1]['fit'][m]) / (f_max - f_min)


def save_detailed_csv(layout, filename, fits, cg, load12):
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['设备名称', 'X坐标', 'Y坐标', 'Z坐标', '旋转状态'])
            for pd in layout.placed_devices:
                writer.writerow([f"{pd.device.id}-{pd.device.name}", pd.x, pd.y, pd.z, pd.rotation])
            writer.writerow([])
            writer.writerow(['--- 方案综合评估指标 ---'])
            writer.writerow(['转动惯量(F1)', '垂直热力(F2)', '信号走线(F3)', '重心X', '重心Y', '重心Z', '设备12承重'])
            writer.writerow([round(fits[0], 3), round(fits[1], 2), round(fits[2], 2),
                             round(cg[0], 2), round(cg[1], 2), round(cg[2], 2), round(load12, 2)])
    except Exception as e:
        print(f"保存文件出错: {e}")


# ==========================================
# ★ 新增：解质量评估模块 (HV & IGD)
# ==========================================
def calculate_hypervolume_monte_carlo(pareto_front_fits, ref_point=[1.1, 1.1, 1.1], num_samples=50000):
    """
    使用蒙特卡洛法近似计算3D超体积 (HV)
    假设输入的 pareto_front_fits 已经是归一化到 [0, 1] 区间的数据
    """
    if len(pareto_front_fits) == 0: return 0.0

    # 在 [0, ref_point] 范围内生成随机采样点
    samples = np.random.uniform(low=0.0, high=ref_point[0], size=(num_samples, 3))
    dominated_count = 0

    # 检查采样点是否被帕累托前沿中的任意一个解支配 (因为是最小化问题，前沿点需小于等于采样点)
    for sample in samples:
        for fit in pareto_front_fits:
            if fit[0] <= sample[0] and fit[1] <= sample[1] and fit[2] <= sample[2]:
                dominated_count += 1
                break

    # 计算体积占比
    total_volume = ref_point[0] * ref_point[1] * ref_point[2]
    hv = (dominated_count / num_samples) * total_volume
    return hv


def calculate_igd(pareto_front_fits, ideal_front):
    """
    计算倒世代距离 (IGD)
    由于真实前沿未知，通常传入一个合成的理想前沿或多次运行的最优解集
    """
    if len(pareto_front_fits) == 0 or len(ideal_front) == 0: return float('inf')

    total_distance = 0.0
    for ideal_point in ideal_front:
        # 计算理想点到当前算法求出前沿的最短欧氏距离
        distances = np.linalg.norm(pareto_front_fits - ideal_point, axis=1)
        total_distance += np.min(distances)

    return total_distance / len(ideal_front)


# ==========================================
# 8. 主控运行与交互式可视化展示
# ==========================================
def run_nsga2(pop_size=50, max_gen=50, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0  # 重置计数器

    # --- 启动性能监控 ---
    tracemalloc.start()
    start_time = time.time()

    population = []
    print("🚀 正在初始化种群，寻找符合硬约束的合法解...")
    while len(population) < pop_size:
        chromo = generate_random_chromosome()
        layout = heuristic_decoder(chromo)
        if layout.is_valid:
            fit = calculate_fitness(layout)
            if fit[0] != float('inf'):
                population.append({'gene': chromo, 'fit': fit, 'layout': layout})

    print(f"\n⚡ 开始 NSGA-II 3目标进化迭代 (共 {max_gen} 代)...")
    for gen in range(max_gen):
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(population, 2)

            if random.random() < CROSSOVER_RATE:
                # 发生交叉：互换两个随机位置的基因
                child_gene = copy.deepcopy(p1['gene'])
                idx1, idx2 = random.sample(range(1, len(child_gene)), 2)
                child_gene[idx1], child_gene[idx2] = child_gene[idx2], child_gene[idx1]
            else:
                # 不发生交叉：直接复制父代 1
                child_gene = copy.deepcopy(p1['gene'])

            if random.random() < MUTATION_RATE:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['rot'] = 90 if child_gene[mut_idx]['rot'] == 0 else 0

            layout = heuristic_decoder(child_gene)
            if layout.is_valid:
                fit = calculate_fitness(layout)
                if fit[0] != float('inf'):
                    offspring.append({'gene': child_gene, 'fit': fit, 'layout': layout})

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)

        population = []
        i = 0
        # ★ NSGA-II 截断与精英保留机制
        while len(population) + len(fronts[i]) <= pop_size:
            crowding_distance_assignment(fronts[i])
            population.extend(fronts[i])
            i += 1
            if i >= len(fronts): break

        if len(population) < pop_size and i < len(fronts):
            crowding_distance_assignment(fronts[i])
            # 依照拥挤度从大到小排序，优先保留拥挤度大（孤立、稀缺）的解
            fronts[i].sort(key=lambda x: x['distance'], reverse=True)
            population.extend(fronts[i][:pop_size - len(population)])

        rank0_count = len([p for p in population if p['rank'] == 0])
        print(f"迭代 {gen + 1}/{max_gen} | 3目标帕累托前沿解数量: {rank0_count}")

    # --- 停止性能监控 ---
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    computation_time = end_time - start_time
    peak_memory_mb = peak_mem / 10 ** 6

    # ================= 提取指标与打印 =================
    final_front = [p for p in population if p.get('rank', 0) == 0]
    final_front.sort(key=lambda x: x['fit'][0])

    # 归一化处理用于 HV 和 IGD 计算
    all_fits = np.array([ind['fit'] for ind in final_front])
    f_min = np.min(all_fits, axis=0)
    f_max = np.max(all_fits, axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1e-6
    norm_fits = (all_fits - f_min) / denom

    # 1. 计算 HV (基于归一化空间)
    hv_value = calculate_hypervolume_monte_carlo(norm_fits)

    # 2. ★ 计算真实的 IGD ★
    if true_pf is not None:
        # 传入未归一化的 all_fits 和真实的 true_pf
        igd_value = calculate_igd_normalized(all_fits, true_pf)
    else:
        # 如果没传 True PF，就给个无效值，避免报错
        igd_value = float('inf')

    print("\n" + "=" * 110)
    print("📊 算法综合性能评估报告 (Algorithm Performance Metrics)")
    print("-" * 110)
    print(f"⏱️  计算耗时 (Computation Time)       : {computation_time:.2f} 秒")
    print(f"💾 内存占用峰值 (Peak Memory Usage)  : {peak_memory_mb:.2f} MB")
    print(f"⚙️  实际适应度评估次数 (Fitness Evals) : {GLOBAL_FITNESS_EVALS} 次 (利用启发式解码大幅减少了无效搜索)")
    print(f"📐 超体积 HV (Hypervolume)           : {hv_value:.4f} (基于归一化空间 [1.1,1.1,1.1] 参考点，值越大越好)")
    print(f"🎯 倒世代距离 IGD (Inverted Gen Dist): {igd_value:.4f} (基于30 次聚合 True PF，值越小越好)")

    print("\n" + "=" * 110)
    print(f"🏆 NSGA-II 优化完成！输出 {len(final_front)} 个帕累托最优解。")
    print(
        f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'重心X':<7} | {'重心Y':<7} | {'重心Z':<7} | {'12号承重'}")
    print("-" * 110)

    detailed_info = []

    for idx, ind in enumerate(final_front):
        pd_list = ind['layout'].placed_devices
        total_w = sum(p.device.weight for p in pd_list)
        cg_x = sum(p.device.weight * p.get_center_3d()[0] for p in pd_list) / total_w
        cg_y = sum(p.device.weight * p.get_center_3d()[1] for p in pd_list) / total_w
        cg_z = sum(p.device.weight * p.get_center_3d()[2] for p in pd_list) / total_w

        dev12_load = sum(
            pd.device.weight for pd in pd_list if _is_supported_by(pd, next(p for p in pd_list if p.device.id == 12)))

        fit = ind['fit']
        info_dict = {'id': idx + 1, 'fit': fit, 'cg': [cg_x, cg_y, cg_z], 'load12': dev12_load}
        detailed_info.append(info_dict)

        print(
            f"{idx + 1:<6} | {fit[0]:<10.3f} | {fit[1]:<10.1f} | {fit[2]:<10.1f} | {cg_x:<8.1f} | {cg_y:<8.1f} | {cg_z:<8.1f} | {dev12_load:<8.1f}")
        save_detailed_csv(ind['layout'], f"nsga2_solution_{idx + 1}.csv", fit, [cg_x, cg_y, cg_z], dev12_load)

    # # ================= 计算绝对均衡解 =================
    # distances_to_ideal = np.linalg.norm(norm_fits, axis=1)
    # best_balanced_idx = np.argmin(distances_to_ideal)

    # 矩阵乘法：计算每个方案在归一化空间下的加权得分
    weighted_scores = np.sum(norm_fits * WEIGHTS, axis=1)

    # 选出得分最小（即损失最小）的方案
    best_balanced_idx = np.argmin(weighted_scores)
    best_balanced_score = weighted_scores[best_balanced_idx]

    print("-" * 110)
    print(f"⭐ 基于自定义权重 (惯量0.5:热力0.3:走线0.2) 推荐的绝对最优解为: [方案 {best_balanced_idx + 1}]")
    print(f"   加权综合得分 (基于归一化空间): {best_balanced_score:.4f}")

    # ================= 交互式 Plotly 3D 可视化 =================
    hover_texts = []
    for i, info in enumerate(detailed_info):
        n_f1, n_f2, n_f3 = norm_fits[i][0], norm_fits[i][1], norm_fits[i][2]

        text = (f"<b>方案 ID: {info['id']}</b><br>"
                f"--------------------<br>"
                f"⚙️ 转动惯量 (F1): {info['fit'][0]:.3f} <span style='color:gray;'>(归一: {n_f1:.3f})</span><br>"
                f"🔥 垂直热力 (F2): {info['fit'][1]:.1f} <span style='color:gray;'>(归一: {n_f2:.3f})</span><br>"
                f"🔌 信号走线 (F3): {info['fit'][2]:.1f} <span style='color:gray;'>(归一: {n_f3:.3f})</span><br>"
                f"⚖️ 重心偏移(X,Y,Z): ({info['cg'][0]:.1f}, {info['cg'][1]:.1f}, {info['cg'][2]:.1f})<br>"
                f"📦 导航箱承重: {info['load12']: .1f} kg")
        hover_texts.append(text)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=norm_fits[:, 0], y=norm_fits[:, 1], z=norm_fits[:, 2],
        mode='markers',
        marker=dict(size=6, color=norm_fits[:, 2], colorscale='Viridis', opacity=0.8),
        text=hover_texts,
        hoverinfo='text',
        name='Pareto Solutions'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='cross'),
        name='Ideal Point [0,0,0]',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter3d(
        x=[norm_fits[best_balanced_idx, 0]],
        y=[norm_fits[best_balanced_idx, 1]],
        z=[norm_fits[best_balanced_idx, 2]],
        mode='markers',
        marker=dict(size=14, color='red', symbol='diamond', line=dict(color='yellow', width=2)),
        text=[f"🏆 <b>【绝对最均衡解】</b><br>" + hover_texts[best_balanced_idx]],
        hoverinfo='text',
        name='Best Balanced Solution'
    ))

    fig.update_layout(
        title='Interactive Normalized 3D Pareto Front (F1: Inertia, F2: Thermal, F3: Routing)',
        scene=dict(
            xaxis_title='Norm Inertia F1',
            yaxis_title='Norm Thermal F2',
            zaxis_title='Norm Routing F3',
            xaxis=dict(range=[-0.1, 1.1]),
            yaxis=dict(range=[-0.1, 1.1]),
            zaxis=dict(range=[-0.1, 1.1])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    print("\n🌐 正在默认浏览器中生成交互式 3D 帕累托图...")
    fig.show()


# ==========================================
# 9. 批量实验与 True PF 构建模块
# ==========================================

def get_global_non_dominated_set(fits_array):
    """
    对汇总后的超级目标空间矩阵进行全局非支配排序，提取 True PF
    """
    # 1. 去重，保留唯一的适应度坐标
    unique_fits = np.unique(fits_array, axis=0)
    n = unique_fits.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    # 2. 暴力两两比较，筛选绝对非支配解
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # 若 i 支配 j (所有目标不大于j，且至少一个小于j)
            if np.all(unique_fits[i] <= unique_fits[j]) and np.any(unique_fits[i] < unique_fits[j]):
                is_dominated[j] = True

    # 3. 提取最终前沿
    true_pf = unique_fits[~is_dominated]
    return true_pf


def run_nsga2_silent(pop_size=50, max_gen=50):
    """
    静默版本的 NSGA-II：不打印日志，不绘图，直接返回当前独立运行的帕累托前沿适应度 (N, 3)
    """
    population = []
    while len(population) < pop_size:
        chromo = generate_random_chromosome()
        layout = heuristic_decoder(chromo)
        if layout.is_valid:
            fit = calculate_fitness(layout)
            if fit[0] != float('inf'):
                population.append({'gene': chromo, 'fit': fit, 'layout': layout})

    for gen in range(max_gen):
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(population, 2)
            child_gene = copy.deepcopy(p1['gene'])
            idx1, idx2 = random.sample(range(1, len(child_gene)), 2)
            child_gene[idx1], child_gene[idx2] = child_gene[idx2], child_gene[idx1]

            if random.random() < 0.2:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['rot'] = 90 if child_gene[mut_idx]['rot'] == 0 else 0

            layout = heuristic_decoder(child_gene)
            if layout.is_valid:
                fit = calculate_fitness(layout)
                if fit[0] != float('inf'):
                    offspring.append({'gene': child_gene, 'fit': fit, 'layout': layout})

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)

        population = []
        i = 0
        while len(population) + len(fronts[i]) <= pop_size:
            crowding_distance_assignment(fronts[i])
            population.extend(fronts[i])
            i += 1
            if i >= len(fronts): break

        if len(population) < pop_size and i < len(fronts):
            crowding_distance_assignment(fronts[i])
            fronts[i].sort(key=lambda x: x['distance'], reverse=True)
            population.extend(fronts[i][:pop_size - len(population)])

    final_front = [p for p in population if p.get('rank', 0) == 0]
    return np.array([ind['fit'] for ind in final_front])


def build_true_pf(num_runs=30, pop_size=50, max_gen=50, save_path="True_PF_raw.npy"):
    """
    执行多次独立实验，构建并保存 True PF
    """
    print(f"\n🚀 开始执行 {num_runs} 次独立批处理实验以构建 True PF...")
    all_runs_fits = []

    start_time = time.time()
    for i in range(num_runs):
        print(f"   ⏳ 正在运行第 {i + 1}/{num_runs} 次实验...", end="\r")
        pf_fits = run_nsga2_silent(pop_size, max_gen)
        all_runs_fits.append(pf_fits)

    print(f"\n✅ {num_runs} 次实验运行完毕，耗时 {time.time() - start_time:.2f} 秒。")

    # 合并为一个超级池
    pooled_fits = np.vstack(all_runs_fits)
    print(f"🔍 汇总池中共有 {pooled_fits.shape[0]} 个解，正在提取 True PF...")

    # 提取全局非支配解集
    true_pf_fits = get_global_non_dominated_set(pooled_fits)
    print(f"⭐ 成功构建 True PF！包含 {true_pf_fits.shape[0]} 个绝对最优解。")

    # 保存到本地
    np.save(save_path, true_pf_fits)
    print(f"💾 True PF 已保存至本地文件: {save_path}")
    return true_pf_fits


# ==========================================
# 10. 修改原评估工具，引入全局归一化 IGD
# ==========================================

def calculate_igd_normalized(current_pf, true_pf):
    """
    计算基于全局边界归一化后的 IGD。
    重要：必须使用 True PF 的 min 和 max 作为全局边界来归一化！
    """
    global_min = np.min(true_pf, axis=0)
    global_max = np.max(true_pf, axis=0)
    denom = global_max - global_min
    denom[denom == 0] = 1e-6  # 防止除以0

    # 统一归一化到 [0, 1] 空间
    norm_true_pf = (true_pf - global_min) / denom
    norm_current_pf = (current_pf - global_min) / denom

    total_distance = 0.0
    for ideal_point in norm_true_pf:
        # 寻找当前前沿中距离理想点最近的欧氏距离
        distances = np.linalg.norm(norm_current_pf - ideal_point, axis=1)
        total_distance += np.min(distances)

    return total_distance / len(norm_true_pf)


# ==========================================
# ★ 对比算法模块: Vanilla NSGA / 传统 NSGA-II / 传统 NSGA-III
# ==========================================
def generate_continuous_chromosome():
    """传统基础编码：随机生成所有设备的绝对坐标(X, Y, Z)和旋转状态"""
    chromo = []
    for dev in DEVICE_LIST:
        x = random.uniform(-RACK_LENGTH / 2, RACK_LENGTH / 2)
        y = random.uniform(-RACK_WIDTH / 2, RACK_WIDTH / 2)
        z = random.uniform(0, 500)  # 假设机柜可用高度 500mm
        rot = random.choice([0, 90])
        chromo.append({'id': dev.id, 'x': x, 'y': y, 'z': z, 'rot': rot})
    return chromo


def calculate_continuous_fitness(chromosome):
    """传统评估：暴力坐标解析与干涉重罚（代替启发式解码器）"""
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS += 1

    layout = Layout()
    penalty_bounds = 0
    penalty_collision = 0

    for gene in chromosome:
        dev = next(d for d in DEVICE_LIST if d.id == gene['id'])
        pd = PlacedDevice(dev, gene['x'], gene['y'], gene['z'], gene['rot'], support=None)
        layout.add(pd)

        # 1. 梯度边界惩罚：超出边界多少毫米，就叠加多少惩罚
        L, W, H = pd.get_effective_dims()
        dx = max(0, abs(pd.x) + L / 2 - RACK_LENGTH / 2)
        dy = max(0, abs(pd.y) + W / 2 - RACK_WIDTH / 2)
        dz_under = max(0, -pd.z)  # 钻入地下
        dz_over = max(0, pd.z + H - 500)  # 假设机柜限高 500mm
        penalty_bounds += (dx + dy + dz_under + dz_over) * 1000  # 越界惩罚系数

    # 2. 梯度干涉惩罚：精确计算两两之间的“重叠体积”
    for i in range(len(layout.placed_devices)):
        for j in range(i + 1, len(layout.placed_devices)):
            pd1, pd2 = layout.placed_devices[i], layout.placed_devices[j]
            L1, W1, H1 = pd1.get_effective_dims()
            L2, W2, H2 = pd2.get_effective_dims()
            cx1, cy1, cz1 = pd1.get_center_3d()
            cx2, cy2, cz2 = pd2.get_center_3d()

            overlap_x = max(0, (L1 + L2) / 2 - abs(cx1 - cx2))
            overlap_y = max(0, (W1 + W2) / 2 - abs(cy1 - cy2))
            overlap_z = max(0, (H1 + H2) / 2 - abs(cz1 - cz2))

            if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
                # 如果长宽高都有重叠，说明发生了空间相交
                overlap_vol = overlap_x * overlap_y * overlap_z
                penalty_collision += overlap_vol * 1.0  # 重叠体积越大，惩罚越重

    # 借用基础物理公式，叠加传统算法的巨额干涉惩罚
    fits = calculate_fitness(layout)
    total_penalty = penalty_collision + penalty_bounds
    # 若无干涉则 is_valid 为 True
    return [fits[0] + total_penalty, fits[1] + total_penalty, fits[2] + total_penalty], (total_penalty == 0)


def run_vanilla_nsga(pop_size=50, max_gen=50):
    """基准 1: Vanilla NSGA (无精英保留 + 适应度共享)"""
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    print(f"\n⚡ 运行基准 1: Vanilla NSGA (1994) [无精英保留机制]...")
    population = [{'gene': generate_continuous_chromosome()} for _ in range(pop_size)]
    for p in population: p['fit'], p['is_valid'] = calculate_continuous_fitness(p['gene'])

    for gen in range(max_gen):
        offspring = []
        # Vanilla NSGA 特征：子代完全替换父代，无精英保留
        for _ in range(pop_size):
            p1, p2 = random.sample(population, 2)
            child_gene = copy.deepcopy(p1['gene'])
            if random.random() < CROSSOVER_RATE:
                swap_indices = random.sample(range(len(child_gene)), len(child_gene) // 2)
                for idx in swap_indices: child_gene[idx] = copy.deepcopy(p2['gene'][idx])
            if random.random() < MUTATION_RATE:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['x'] += random.uniform(-50, 50)
                child_gene[mut_idx]['y'] += random.uniform(-50, 50)

            fit, is_valid = calculate_continuous_fitness(child_gene)
            offspring.append({'gene': child_gene, 'fit': fit, 'is_valid': is_valid})
        population = offspring  # 纯粹的代际更替

    fronts = fast_non_dominated_sort(population)
    final_front = fronts[0] if fronts else []

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    valid_count = sum(1 for p in final_front if p['is_valid'])
    return np.array([ind['fit'] for ind in
                     final_front]), end_time - start_time, peak_mem / 10 ** 6, GLOBAL_FITNESS_EVALS, valid_count


def run_traditional_nsga2(pop_size=50, max_gen=50):
    """基准 2: 传统 NSGA-II (精英保留 + 拥挤度)"""
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    print(f"⚡ 运行基准 2: Traditional NSGA-II (2002) [坐标连续编码]...")
    population = [{'gene': generate_continuous_chromosome()} for _ in range(pop_size)]
    for p in population: p['fit'], p['is_valid'] = calculate_continuous_fitness(p['gene'])

    for gen in range(max_gen):
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(population, 2)
            child_gene = copy.deepcopy(p1['gene'])
            if random.random() < CROSSOVER_RATE:
                swap_indices = random.sample(range(len(child_gene)), len(child_gene) // 2)
                for idx in swap_indices: child_gene[idx] = copy.deepcopy(p2['gene'][idx])
            if random.random() < MUTATION_RATE:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['x'] += random.uniform(-50, 50)
                child_gene[mut_idx]['y'] += random.uniform(-50, 50)
            fit, is_valid = calculate_continuous_fitness(child_gene)
            offspring.append({'gene': child_gene, 'fit': fit, 'is_valid': is_valid})

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        population = []
        i = 0
        while len(population) + len(fronts[i]) <= pop_size:
            crowding_distance_assignment(fronts[i])
            population.extend(fronts[i])
            i += 1
            if i >= len(fronts): break
        if len(population) < pop_size and i < len(fronts):
            crowding_distance_assignment(fronts[i])
            fronts[i].sort(key=lambda x: x['distance'], reverse=True)
            population.extend(fronts[i][:pop_size - len(population)])

    final_front = [p for p in population if p.get('rank', 0) == 0]

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    valid_count = sum(1 for p in final_front if p['is_valid'])
    return np.array([ind['fit'] for ind in
                     final_front]), end_time - start_time, peak_mem / 10 ** 6, GLOBAL_FITNESS_EVALS, valid_count


def run_traditional_nsga3(pop_size=50, max_gen=50):
    """基准 3: 传统 NSGA-III (精英保留 + 参考点机制)"""
    # 模拟 NSGA-III 的参考点生成机制来维持多样性
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    print(f"⚡ 运行基准 3: Traditional NSGA-III (2014) [参考点代替拥挤度]...")
    population = [{'gene': generate_continuous_chromosome()} for _ in range(pop_size)]
    for p in population: p['fit'], p['is_valid'] = calculate_continuous_fitness(p['gene'])

    for gen in range(max_gen):
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(population, 2)
            child_gene = copy.deepcopy(p1['gene'])
            if random.random() < CROSSOVER_RATE:
                swap_indices = random.sample(range(len(child_gene)), len(child_gene) // 2)
                for idx in swap_indices: child_gene[idx] = copy.deepcopy(p2['gene'][idx])
            if random.random() < MUTATION_RATE:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['x'] += random.uniform(-50, 50)
                child_gene[mut_idx]['y'] += random.uniform(-50, 50)
            fit, is_valid = calculate_continuous_fitness(child_gene)
            offspring.append({'gene': child_gene, 'fit': fit, 'is_valid': is_valid})

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        population = []
        i = 0
        while len(population) + len(fronts[i]) <= pop_size:
            population.extend(fronts[i])
            i += 1
            if i >= len(fronts): break

        # 截断层模拟 NSGA-III 的参考点截断 (此处简化表示为随机均匀截断，体现其不依赖拥挤度的特性)
        if len(population) < pop_size and i < len(fronts):
            random.shuffle(fronts[i])
            population.extend(fronts[i][:pop_size - len(population)])

    final_front = [p for p in population if p.get('rank', 0) == 0]

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    valid_count = sum(1 for p in final_front if p['is_valid'])
    return np.array([ind['fit'] for ind in
                     final_front]), end_time - start_time, peak_mem / 10 ** 6, GLOBAL_FITNESS_EVALS, valid_count


# ==========================================
# ★ 真正的学术对比基准：经典最低最左装箱法 (Bottom-Left-Fill, BLF)
# ==========================================

def standard_blf_decoder(chromosome):
    """
    通用底左优先解码器 (Standard BLF Decoder)
    学术界传统3D装箱问题的标准解码器。
    它只考虑几何不重叠，盲目追求最高空间利用率，无视AUV的承重、重心和特定设备(12号)规则。
    """
    layout = Layout()
    for gene in chromosome:
        dev = next(d for d in DEVICE_LIST if d.id == gene['id'])
        rot = gene['rot']
        L, W, H = dev.get_effective_dims() if hasattr(dev, 'get_effective_dims') else (
            (dev.W, dev.L, dev.H) if rot == 90 else (dev.L, dev.W, dev.H)
        )

        placed = False
        # BLF 核心逻辑：Z轴优先(从底向上)，X轴次之(从左向右)，Y轴最后
        # 为保持运算时间与HD一致，采用相同的粗网格扫描步长
        for z in range(0, 500, 40):
            if placed: break

            x_start = int(-RACK_LENGTH / 2 + L / 2)
            x_end = int(RACK_LENGTH / 2 - L / 2)
            for x in range(x_start, x_end + 1, 80):
                if placed: break

                y_start = int(-RACK_WIDTH / 2 + W / 2)
                y_end = int(RACK_WIDTH / 2 - W / 2)
                for y in range(y_start, y_end + 1, 80):
                    # 在BLF中，设备只是堆叠，无法建立精细的工程 support 支撑树
                    test_pd = PlacedDevice(dev, x, y, z, rot, support=None)

                    # 仅进行纯几何碰撞检测 (无视 UUV 承重限制和堆叠规则)
                    if not any(check_collision_aabb(test_pd, existing) for existing in layout.placed_devices):
                        layout.add(test_pd)
                        placed = True
                        break

        # 如果连通用 BLF 都放不下，说明该序列空间无解
        if not placed:
            layout.is_valid = False
            return layout

    return layout


def run_blf_nsga2(pop_size=50, max_gen=50):
    """
    基准算法 2：BLF-NSGA-II (传统通用装箱算法)
    """
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    print(f"\n⚡ 运行对比基准: BLF-NSGA-II (经典 Bottom-Left-Fill 通用装箱算法)...")
    population = []

    # 初始化
    while len(population) < pop_size:
        chromo = generate_random_chromosome()
        layout = standard_blf_decoder(chromo)
        if layout.is_valid:
            fit = calculate_fitness(layout)
            if fit[0] != float('inf'):
                population.append({'gene': chromo, 'fit': fit, 'layout': layout})

    # 主循环
    for gen in range(max_gen):
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(population, 2)
            child_gene = copy.deepcopy(p1['gene'])
            if random.random() < CROSSOVER_RATE:
                idx1, idx2 = random.sample(range(1, len(child_gene)), 2)
                child_gene[idx1], child_gene[idx2] = child_gene[idx2], child_gene[idx1]
            if random.random() < MUTATION_RATE:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['rot'] = 90 if child_gene[mut_idx]['rot'] == 0 else 0

            layout = standard_blf_decoder(child_gene)
            if layout.is_valid:
                fit = calculate_fitness(layout)
                if fit[0] != float('inf'):
                    offspring.append({'gene': child_gene, 'fit': fit, 'layout': layout})

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        population = []
        i = 0
        while len(population) + len(fronts[i]) <= pop_size:
            crowding_distance_assignment(fronts[i])
            population.extend(fronts[i])
            i += 1
            if i >= len(fronts): break
        if len(population) < pop_size and i < len(fronts):
            crowding_distance_assignment(fronts[i])
            fronts[i].sort(key=lambda x: x['distance'], reverse=True)
            population.extend(fronts[i][:pop_size - len(population)])

    final_front = [p for p in population if p.get('rank', 0) == 0]

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return np.array(
        [ind['fit'] for ind in final_front]), end_time - start_time, peak_mem / 10 ** 6, GLOBAL_FITNESS_EVALS, len(
        final_front)


# ==========================================
# ★ 顶级学术对比基准：RL-NSGA-II-HD (结合 Q-learning 的自适应演化算法)
# 参考文献: 3D print orientation optimization and comparative analysis of NSGA-II versus NSGA-II with Q-learning
# ==========================================
def run_rl_nsga2_hd(pop_size=50, max_gen=50):
    """
    RL-NSGA-II-HD: 利用 Q-learning 动态调整交叉率、变异率并注入多样性。
    """
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    print(f"\n⚡ 运行顶级对比基准: RL-NSGA-II-HD (Q-learning 动态参数自适应)...")

    # --- 1. Q-learning 参数与环境初始化 ---
    alpha_lr = 0.1  # 学习率
    gamma_df = 0.9  # 折扣因子
    epsilon = 0.2  # 探索概率 (Epsilon-greedy)
    num_states = 4  # 状态空间: 0(双降), 1(多样性升), 2(收敛性升), 3(双升)
    num_actions = 5  # 动作空间: 0(加交叉), 1(减交叉), 2(加变异), 3(减变异), 4(随机解注入)
    Q_table = np.zeros((num_states, num_actions))

    # 初始演化参数
    p_c = CROSSOVER_RATE
    p_m = MUTATION_RATE

    # --- 2. 种群初始化 ---
    population = []
    while len(population) < pop_size:
        chromo = generate_random_chromosome()
        layout = heuristic_decoder(chromo)
        if layout.is_valid:
            fit = calculate_fitness(layout)
            if fit[0] != float('inf'):
                population.append({'gene': chromo, 'fit': fit, 'layout': layout})

    # 计算初始状态指标 (Pareto数量 与 平均拥挤度)
    fronts = fast_non_dominated_sort(population)
    prev_pf_size = len(fronts[0]) if fronts else 0
    crowding_distance_assignment(fronts[0])
    prev_cd = sum(p['distance'] for p in fronts[0] if p['distance'] != float('inf')) / max(1, prev_pf_size)
    current_state = 0

    # --- 3. RL-NSGA-II 主循环 ---
    for gen in range(max_gen):
        # [Step A]: 动作选择 (Epsilon-greedy)
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            action = np.argmax(Q_table[current_state])

        # [Step B]: 执行动作，更新演化参数
        if action == 0:
            p_c = min(1.0, p_c + 0.1)
        elif action == 1:
            p_c = max(0.5, p_c - 0.1)
        elif action == 2:
            p_m = min(0.5, p_m + 0.1)
        elif action == 3:
            p_m = max(0.05, p_m - 0.1)
        elif action == 4:
            # 引入随机多样性：替换最差的 10% 个体
            num_replace = max(1, int(pop_size * 0.1))
            population.sort(key=lambda x: x.get('rank', float('inf')))
            for i in range(num_replace):
                chromo = generate_random_chromosome()
                layout = heuristic_decoder(chromo)
                if layout.is_valid:
                    fit = calculate_fitness(layout)
                    if fit[0] != float('inf'):
                        population[-(i + 1)] = {'gene': chromo, 'fit': fit, 'layout': layout}

        # [Step C]: 标准子代生成 (使用当前动态调整的 p_c 和 p_m)
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(population, 2)
            if random.random() < p_c:
                child_gene = copy.deepcopy(p1['gene'])
                idx1, idx2 = random.sample(range(1, len(child_gene)), 2)
                child_gene[idx1], child_gene[idx2] = child_gene[idx2], child_gene[idx1]
            else:
                child_gene = copy.deepcopy(p1['gene'])

            if random.random() < p_m:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['rot'] = 90 if child_gene[mut_idx]['rot'] == 0 else 0

            layout = heuristic_decoder(child_gene)
            if layout.is_valid:
                fit = calculate_fitness(layout)
                if fit[0] != float('inf'):
                    offspring.append({'gene': child_gene, 'fit': fit, 'layout': layout})

        # [Step D]: 拥挤度与非支配排序淘汰
        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)
        population = []
        i = 0
        while len(population) + len(fronts[i]) <= pop_size:
            crowding_distance_assignment(fronts[i])
            population.extend(fronts[i])
            i += 1
            if i >= len(fronts): break
        if len(population) < pop_size and i < len(fronts):
            crowding_distance_assignment(fronts[i])
            fronts[i].sort(key=lambda x: x['distance'], reverse=True)
            population.extend(fronts[i][:pop_size - len(population)])

        # [Step E]: 计算新状态并计算 Reward
        curr_pf_size = len([p for p in population if p.get('rank', 0) == 0])
        front0 = [p for p in population if p.get('rank', 0) == 0]
        crowding_distance_assignment(front0)
        curr_cd = sum(p['distance'] for p in front0 if p['distance'] != float('inf')) / max(1, curr_pf_size)

        delta_N = curr_pf_size - prev_pf_size
        delta_CD = curr_cd - prev_cd

        # 状态划分：0(均退化) 1(多样性升) 2(收敛性升) 3(均升)
        if delta_N <= 0 and delta_CD <= 0:
            next_state = 0
            reward = -1.0
        elif delta_N <= 0 and delta_CD > 0:
            next_state = 1
            reward = 0.5
        elif delta_N > 0 and delta_CD <= 0:
            next_state = 2
            reward = 0.5
        else:
            next_state = 3
            reward = 1.0

        # [Step F]: 更新 Q-Table (Bellman Equation)
        Q_table[current_state, action] = Q_table[current_state, action] + alpha_lr * (
                reward + gamma_df * np.max(Q_table[next_state]) - Q_table[current_state, action]
        )

        current_state = next_state
        prev_pf_size = curr_pf_size
        prev_cd = curr_cd

    # --- 4. 结果提取 ---
    final_front = [p for p in population if p.get('rank', 0) == 0]

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return np.array(
        [ind['fit'] for ind in final_front]), end_time - start_time, peak_mem / 10 ** 6, GLOBAL_FITNESS_EVALS, len(
        final_front)


# ==========================================
# ★ 3D 帕累托前沿可视化模块 (Plotly)
# ==========================================
def plot_pareto_front_3d(all_fits, best_fit, algo_name="HD-NSGA-II"):
    """
    使用 Plotly 绘制 3D 帕累托前沿，并用醒目的红色钻石高亮最佳加权解。
    :param all_fits: numpy array, 形状为 (N, 3)，包含所有合法的帕累托解
    :param best_fit: numpy array, 形状为 (3,)，最佳均衡解的三个目标值
    :param algo_name: 算法名称，用于图例显示
    """
    if len(all_fits) == 0:
        print(f"⚠️ {algo_name} 没有合法的帕累托解，无法绘图。")
        return

    # 1. 基础散点图：所有的帕累托解
    # 使用 Z 轴（走线损失）或综合数值作为颜色映射，增加 3D 层次感
    colors = all_fits[:, 0] + all_fits[:, 1] + all_fits[:, 2]

    trace_all = go.Scatter3d(
        x=all_fits[:, 0],
        y=all_fits[:, 1],
        z=all_fits[:, 2],
        mode='markers',
        name=f'Pareto Solutions ({algo_name})',
        marker=dict(
            size=6,
            color=colors,
            colorscale='Viridis',  # 高级渐变色系
            opacity=0.8,
            line=dict(width=0.5, color='white')  # 加一点白边让点更清晰
        ),
        hovertemplate="<b>方案</b><br>惯量(F1): %{x:.2f}<br>热力(F2): %{y:.2f}<br>走线(F3): %{z:.2f}<extra></extra>"
    )

    # 2. 高亮散点图：最佳加权均衡解
    trace_best = go.Scatter3d(
        x=[best_fit[0]] if best_fit is not None else [],
        y=[best_fit[1]] if best_fit is not None else [],
        z=[best_fit[2]] if best_fit is not None else [],
        mode='markers+text',
        name='Best Compromise Solution (Optimal)',
        marker=dict(
            size=12,  # 更大的尺寸
            color='red',  # 醒目的红色
            symbol='diamond',  # 钻石形状
            line=dict(color='yellow', width=2)
        ),
        text=["★ Best Solution"],
        textposition="top center",
        textfont=dict(color='red', size=14, family='Arial Black'),
        hovertemplate="<b>★ 最佳均衡方案 ★</b><br>惯量(F1): %{x:.2f}<br>热力(F2): %{y:.2f}<br>走线(F3): %{z:.2f}<extra></extra>"
    )

    # 3. 设置 3D 场景布局与背景
    layout = go.Layout(
        title=dict(
            text=f"3D Pareto Front & Best Compromise Solution ({algo_name})",
            x=0.5, y=0.95,
            xanchor='center', yanchor='top',
            font=dict(size=20, family='Arial')
        ),
        scene=dict(
            xaxis=dict(title='F1: Moment of Inertia', backgroundcolor="rgb(240, 240, 240)",
                       gridcolor="white"),
            yaxis=dict(title='F2: Thermal Loss', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            zaxis=dict(title='F3: Routing Loss', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # 默认的观察视角
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    fig = go.Figure(data=[trace_all, trace_best], layout=layout)

    # 将图表保存为高清晰度 HTML 文件，方便你在浏览器中随时交互查看和截图
    html_filename = f"Pareto_Front_3D_{algo_name}.html"
    fig.write_html(html_filename)
    print(f"\n📊 3D 帕累托前沿图已生成并保存为: {html_filename}")

    # 自动在浏览器中弹出显示
    fig.show()


# ==========================================
# 11. 最终主程序入口
# ==========================================
# ==========================================
# 11. 最终主程序入口 (终极三向基准测试)
# ==========================================
if __name__ == "__main__":
    ######训练多次求平均
    PF_FILE = "True_PF_raw.npy"

    # 1. 检查并构建 True PF (如果不存在)
    if not os.path.exists(PF_FILE):
        print("⚠️ 未检测到 True PF 文件，将自动启动批量实验进行构建（可能需要几分钟）...")
        build_true_pf(num_runs=30, pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, save_path=PF_FILE)
    else:
        print("✅ 检测到本地已存在 True PF 数据。")

    true_pf_raw = np.load(PF_FILE)

    print("\n" + "=" * 110)
    print("🚀 开始终极多维对比实验：BLF (经典通用) vs RL (前沿自适应) vs HD (本文所提)")
    print("=" * 110)

    # ================= 计算综合评估指标 =================
    global_min = np.min(true_pf_raw, axis=0)
    global_max = np.max(true_pf_raw, axis=0)
    denom = global_max - global_min
    denom[denom == 0] = 1e-6


    def get_metrics(fits, valid_cnt):
        if valid_cnt == 0 or len(fits) == 0: return 0.0, float('inf')
        valid_fits = np.array([f for f in fits if f[0] < 1e5])  # 剔除由于严重干涉带来的极端劣解
        if len(valid_fits) == 0: return 0.0, float('inf')
        norm_fits = (valid_fits - global_min) / denom
        hv = calculate_hypervolume_monte_carlo(norm_fits)
        igd = calculate_igd_normalized(valid_fits, true_pf_raw)
        return hv, igd


    for i in range(1):  # 反复测试3遍
        # ================= 运行三个算法 =================
        # 1. 运行 BLF-NSGA-II (经典最低最左通用装箱)
        # res_blf = run_blf_nsga2(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS)
        #
        # # 2. 运行 RL-NSGA-II-HD (Q-learning 强化自适应算法)
        # res_rl = run_rl_nsga2_hd(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS)

        # 3. 运行 提出的 HD-NSGA-II
        GLOBAL_FITNESS_EVALS = 0
        tracemalloc.start()
        t_start = time.time()
        print(f"\n⚡ 运行本研究提出算法HD-NSGA-II...")
        proposed_fits = run_nsga2_silent(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS)
        t_end = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res_prop = (proposed_fits, t_end - t_start, peak / 10 ** 6, GLOBAL_FITNESS_EVALS, len(proposed_fits))

        metrics = [get_metrics(res[0], res[4]) for res in [res_prop]]

        # 理论最大评估次数 (初始种群 + 种群数 * 迭代代数)
        THEORETICAL_MAX_EVALS = POP_SIZE + (POP_SIZE * NUM_ITERATIONS)

        # ================= 打印权威对比表格 =================
        print("\n\n" + "*" * 125)
        print("🏆 终极算法对比评估结果汇总表 (Comprehensive Comparative Results)")
        print("*" * 125)
        # 将表头改为 Total Evals
        print(
            f"{'Algorithm Approach':<30} | {'HV (↑)':<10} | {'IGD (↓)':<10} | {'Time (s) (↓)':<12} | {'Mem (MB) (↓)':<12} | {'Total Evals (Search Depth)'}")
        print("-" * 125)
        names = ["HD-NSGA-II (Proposed)"]

        for name, res, (hv, igd) in zip(names, [res_prop], metrics):
            # res[3] 就是 GLOBAL_FITNESS_EVALS 的真实调用次数
            actual_evals = res[3]
            print(f"{name:<30} | {hv:<10.4f} | {igd:<10.4f} | {res[1]:<12.2f} | {res[2]:<12.2f} | {actual_evals:<12}")
        print("*" * 125)
    #######训练多次求平均

    ######训练一次对比最佳结果
    # PF_FILE = "True_PF_raw.npy"
    # if not os.path.exists(PF_FILE):
    #     build_true_pf(num_runs=30, pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, save_path=PF_FILE)
    # true_pf_raw = np.load(PF_FILE)
    #
    # print("\n" + "=" * 110)
    # print("🚀 开始单次深度运行：提取并对比各算法的最佳加权决策方案")
    # print("=" * 110)
    #
    # # ================= 运行三个算法 =================
    # res_blf = run_blf_nsga2(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS)
    # res_rl = run_rl_nsga2_hd(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS)
    #
    # GLOBAL_FITNESS_EVALS = 0
    # proposed_fits = run_nsga2_silent(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS)
    # res_prop = (proposed_fits, 0, 0, GLOBAL_FITNESS_EVALS, len(proposed_fits))
    #
    # # ================= 全局边界与权重设置 =================
    # global_min = np.min(true_pf_raw, axis=0)
    # global_max = np.max(true_pf_raw, axis=0)
    # denom = global_max - global_min
    # denom[denom == 0] = 1e-6
    #
    # # 专家偏好权重设定
    # WEIGHTS = np.array([0.5, 0.3, 0.2])
    #
    #
    # # 提取最佳加权解的内部函数
    # def get_best_weighted_solution(fits, valid_cnt):
    #     if valid_cnt == 0 or len(fits) == 0:
    #         return None, float('inf')
    #
    #     valid_fits = np.array([f for f in fits if f[0] < 1e5])  # 剔除严重受罚解
    #     if len(valid_fits) == 0:
    #         return None, float('inf')
    #
    #     # 必须在归一化空间计算加权得分，消除量纲影响
    #     norm_fits = (valid_fits - global_min) / denom
    #     weighted_scores = np.sum(norm_fits * WEIGHTS, axis=1)
    #
    #     best_idx = np.argmin(weighted_scores)
    #     best_raw_fit = valid_fits[best_idx]
    #     best_score = weighted_scores[best_idx]
    #
    #     return best_raw_fit, best_score
    #
    #
    # # ================= 计算并提取最佳解 =================
    # best_fit_blf, score_blf = get_best_weighted_solution(res_blf[0], res_blf[4])
    # best_fit_rl, score_rl = get_best_weighted_solution(res_rl[0], res_rl[4])
    # best_fit_prop, score_prop = get_best_weighted_solution(res_prop[0], res_prop[4])
    #
    # # ================= 打印最佳方案对比表 =================
    # print("\n\n" + "*" * 120)
    # print("🏆 专家偏好驱动下的最佳均衡方案对比表 (Best Compromise Solution Comparison)")
    # print(f"偏好权重设定: 惯量(0.5) : 热力(0.3) : 走线(0.2)")
    # print("*" * 120)
    # print(
    #     f"{'Algorithm Approach':<28} | {'加权综合得分(↓)':<16} | {'惯量 F1 (绝对值)':<18} | {'热力 F2 (绝对值)':<18} | {'走线 F3 (绝对值)'}")
    # print("-" * 120)
    #
    #
    # # ================= 绘制 3D 帕累托前沿并高亮最佳解 =================
    # # 提取提出算法 (HD-NSGA-II) 的所有非严重受罚的合法解用于绘图
    # valid_fits_prop = np.array([f for f in res_prop[0] if f[0] < 1e5])
    #
    # print("\n正在渲染 3D 帕累托前沿图表，请稍候...")
    # plot_pareto_front_3d(valid_fits_prop, best_fit_prop, algo_name="HD-NSGA-II")