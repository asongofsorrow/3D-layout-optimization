import os
import time
import tracemalloc
import math
import random
import copy
import csv
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 全局环境、物理限制与参数定义
# ==========================================
POP_SIZE = 50  # 种群数量
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
        self.Q = Q
        self.T_max = T_max
        self.R = R


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

    if z1_min >= z2_max:
        dist_z_directed = z1_min - z2_max
    elif z2_min >= z1_max:
        dist_z_directed = z1_max - z2_min
    else:
        dist_z_directed = 0.0

    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
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

    total_w = sum(pd.device.weight for pd in layout.placed_devices)
    sum_wx = sum(pd.device.weight * pd.get_center_3d()[0] for pd in layout.placed_devices)
    sum_wy = sum(pd.device.weight * pd.get_center_3d()[1] for pd in layout.placed_devices)
    sum_wz = sum(pd.device.weight * pd.get_center_3d()[2] for pd in layout.placed_devices)

    cg_x = sum_wx / total_w
    cg_y = sum_wy / total_w
    cg_z = sum_wz / total_w

    device_12_load = sum(pd.device.weight for pd in layout.placed_devices if
                         _is_supported_by(pd, next(p for p in layout.placed_devices if p.device.id == 12)))

    penalty = 0
    if abs(cg_x) > LIMIT_COG_X: penalty += (abs(cg_x) - LIMIT_COG_X) * 10000000
    if abs(cg_y) > LIMIT_COG_Y: penalty += (abs(cg_y) - LIMIT_COG_Y) * 10000000
    if cg_z > LIMIT_COG_Z:      penalty += (cg_z - LIMIT_COG_Z) * 10000000
    if device_12_load > LIMIT_DEV12_LOAD: penalty += (device_12_load - LIMIT_DEV12_LOAD) * 10000000

    I_xx, I_yy, I_zz = 0, 0, 0
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

    fine_tune_layout(layout)
    return layout


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


# ==========================================
# 7. MOGA 核心算法模块 (含共享适应度机制)
# ==========================================

def moga_rank_and_share(population, sigma_share=0.5):
    """
    MOGA 核心机制：支配计数定级与同 Rank 内的小生境适应度共享
    """
    # 1. 支配计数定级 (Rank): rank 越小越好 (0为前沿)
    for p in population:
        n_dom = sum(1 for q in population if dominates(q['fit'], p['fit']))
        p['rank'] = n_dom

    # 按 Rank 从优到劣排序
    population.sort(key=lambda x: x['rank'])

    # 2. 线性插值分配原始适应度 (Rank 越靠前，适应度越高)
    pop_len = len(population)
    for i, p in enumerate(population):
        p['raw_fit'] = pop_len - i

    # 同等级求平均适应度，确保同一前沿的解原始适应度相同
    ranks = set(p['rank'] for p in population)
    for r in ranks:
        inds = [p for p in population if p['rank'] == r]
        avg_fit = sum(p['raw_fit'] for p in inds) / len(inds)
        for p in inds:
            p['raw_fit'] = avg_fit

    # 3. 小生境适应度共享 (Fitness Sharing)
    fits = np.array([p['fit'] for p in population])
    f_min = fits.min(axis=0)
    f_max = fits.max(axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1e-6

    for r in ranks:
        inds = [p for p in population if p['rank'] == r]
        for p1 in inds:
            nc = 0.0  # Niche count (生境计数)
            for p2 in inds:
                d = np.linalg.norm((np.array(p1['fit']) - np.array(p2['fit'])) / denom)
                if d < sigma_share:
                    nc += 1.0 - (d / sigma_share)

            # 适应度共享衰减：拥挤的区域共享适应度降低
            p1['shared_fit'] = p1['raw_fit'] / nc if nc > 0 else p1['raw_fit']


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
# 8. 解质量评估模块 (HV & IGD)
# ==========================================
def calculate_hypervolume_monte_carlo(pareto_front_fits, ref_point=[1.1, 1.1, 1.1], num_samples=50000):
    if len(pareto_front_fits) == 0: return 0.0
    samples = np.random.uniform(low=0.0, high=ref_point[0], size=(num_samples, 3))
    dominated_count = 0
    for sample in samples:
        for fit in pareto_front_fits:
            if fit[0] <= sample[0] and fit[1] <= sample[1] and fit[2] <= sample[2]:
                dominated_count += 1
                break
    total_volume = ref_point[0] * ref_point[1] * ref_point[2]
    return (dominated_count / num_samples) * total_volume


def calculate_igd_normalized(current_pf, true_pf):
    global_min = np.min(true_pf, axis=0)
    global_max = np.max(true_pf, axis=0)
    denom = global_max - global_min
    denom[denom == 0] = 1e-6
    norm_true_pf = (true_pf - global_min) / denom
    norm_current_pf = (current_pf - global_min) / denom

    total_distance = 0.0
    for ideal_point in norm_true_pf:
        distances = np.linalg.norm(norm_current_pf - ideal_point, axis=1)
        total_distance += np.min(distances)
    return total_distance / len(norm_true_pf)


# ==========================================
# 9. HD-MOGA 主控运行与交互式可视化展示
# ==========================================
def run_moga(pop_size=50, max_gen=50, true_pf=None, sigma_share=0.5):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0

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

    print(f"\n⚡ 开始 HD-MOGA 3目标进化迭代 (共 {max_gen} 代)...")
    for gen in range(max_gen):
        # MOGA 适应度共享与排序
        moga_rank_and_share(population, sigma_share)

        # 轮盘赌概率
        total_fit = sum(p['shared_fit'] for p in population)
        if total_fit == 0:
            probs = [1.0 / len(population)] * len(population)
        else:
            probs = [p['shared_fit'] / total_fit for p in population]

        offspring = []
        while len(offspring) < pop_size:
            idx1 = np.random.choice(len(population), p=probs)
            idx2 = np.random.choice(len(population), p=probs)
            p1, p2 = population[idx1], population[idx2]

            child_gene = copy.deepcopy(p1['gene'])
            if random.random() < CROSSOVER_RATE:
                gene_idx1, gene_idx2 = random.sample(range(1, len(child_gene)), 2)
                child_gene[gene_idx1], child_gene[gene_idx2] = child_gene[gene_idx2], child_gene[gene_idx1]

            if random.random() < MUTATION_RATE:
                mut_idx = random.randint(0, len(child_gene) - 1)
                child_gene[mut_idx]['rot'] = 90 if child_gene[mut_idx]['rot'] == 0 else 0

            layout = heuristic_decoder(child_gene)
            if layout.is_valid:
                fit = calculate_fitness(layout)
                if fit[0] != float('inf'):
                    offspring.append({'gene': child_gene, 'fit': fit, 'layout': layout})

        # MOGA 为代际更替，子代直接替换父代
        population = offspring

        # 为日志打印提取一下目前的 rank 0 数量
        for p in population:
            p['rank'] = sum(1 for q in population if dominates(q['fit'], p['fit']))
        rank0_count = len([p for p in population if p['rank'] == 0])
        print(f"迭代 {gen + 1}/{max_gen} | 3目标帕累托前沿解数量: {rank0_count}")

    # --- 停止性能监控 ---
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    computation_time = end_time - start_time
    peak_memory_mb = peak_mem / 10 ** 6

    # ================= 提取最终前沿指标与打印 =================
    # 最后一代再计算一次支配关系
    for p in population:
        p['rank'] = sum(1 for q in population if dominates(q['fit'], p['fit']))

    final_front = [p for p in population if p['rank'] == 0]
    final_front.sort(key=lambda x: x['fit'][0])

    if not final_front:
        print("未找到任何有效的前沿解！")
        return

    # 归一化处理用于 HV 和 IGD 计算
    all_fits = np.array([ind['fit'] for ind in final_front])
    f_min = np.min(all_fits, axis=0)
    f_max = np.max(all_fits, axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1e-6
    norm_fits = (all_fits - f_min) / denom

    # 1. 计算 HV
    hv_value = calculate_hypervolume_monte_carlo(norm_fits)

    # 2. 计算 IGD
    if true_pf is not None:
        igd_value = calculate_igd_normalized(all_fits, true_pf)
    else:
        igd_value = float('inf')

    print("\n" + "=" * 110)
    print("📊 HD-MOGA 综合性能评估报告 (Algorithm Performance Metrics)")
    print("-" * 110)
    print(f"⏱️  计算耗时 (Computation Time)       : {computation_time:.2f} 秒")
    print(f"💾 内存占用峰值 (Peak Memory Usage)  : {peak_memory_mb:.2f} MB")
    print(f"⚙️  实际适应度评估次数 (Fitness Evals) : {GLOBAL_FITNESS_EVALS} 次")
    print(f"📐 超体积 HV (Hypervolume)           : {hv_value:.4f} (参考点 [1.1,1.1,1.1])")
    print(f"🎯 倒世代距离 IGD (Inverted Gen Dist): {igd_value:.4f}")

    print("\n" + "=" * 110)
    print(f"🏆 HD-MOGA 优化完成！输出 {len(final_front)} 个帕累托最优解。")
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
        save_detailed_csv(ind['layout'], f"moga_solution_{idx + 1}.csv", fit, [cg_x, cg_y, cg_z], dev12_load)

    # ================= 计算最佳加权和 (绝对均衡解) =================
    # 在归一化空间下计算各方案加权得分
    weighted_scores = np.sum(norm_fits * WEIGHTS, axis=1)
    best_balanced_idx = np.argmin(weighted_scores)
    best_balanced_score = weighted_scores[best_balanced_idx]

    print("-" * 110)
    print(f"⭐ 基于自定义权重 (惯量0.5 : 热力0.3 : 走线0.2) 推荐的加权绝对最优解为: [方案 {best_balanced_idx + 1}]")
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
        x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2],
        mode='markers',
        marker=dict(size=6, color=all_fits[:, 2], colorscale='Viridis', opacity=0.8,
                    line=dict(width=0.5, color='white')),
        text=hover_texts,
        hoverinfo='text',
        name='MOGA Pareto Solutions'
    ))

    fig.add_trace(go.Scatter3d(
        x=[all_fits[best_balanced_idx, 0]],
        y=[all_fits[best_balanced_idx, 1]],
        z=[all_fits[best_balanced_idx, 2]],
        mode='markers+text',
        marker=dict(size=14, color='red', symbol='diamond', line=dict(color='yellow', width=2)),
        text=["★ Best Solution"],
        textposition="top center",
        textfont=dict(color='red', size=14, family='Arial Black'),
        hovertemplate="<b>★【加权最佳均衡解】★</b><br>" + hover_texts[best_balanced_idx] + "<extra></extra>",
        name='Best Weighted Solution'
    ))

    fig.update_layout(
        title=dict(
            text="Interactive 3D Pareto Front (HD-MOGA) & Best Compromise Solution",
            x=0.5, y=0.95,
            xanchor='center', yanchor='top',
            font=dict(size=20, family='Arial')
        ),
        scene=dict(
            xaxis=dict(title='F1: Moment of Inertia', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            yaxis=dict(title='F2: Thermal Loss', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            zaxis=dict(title='F3: Routing Loss', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    html_filename = "Pareto_Front_3D_HD_MOGA.html"
    fig.write_html(html_filename)
    print(f"\n🌐 3D 帕累托图已生成并保存为: {html_filename}，正在默认浏览器中打开...")
    fig.show()


# ==========================================
# 10. True PF 构建模块 (如无提供则通过快速搜索模拟)
# ==========================================
def get_global_non_dominated_set(fits_array):
    unique_fits = np.unique(fits_array, axis=0)
    n = unique_fits.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]: continue
        for j in range(n):
            if i == j or is_dominated[j]: continue
            if np.all(unique_fits[i] <= unique_fits[j]) and np.any(unique_fits[i] < unique_fits[j]):
                is_dominated[j] = True
    return unique_fits[~is_dominated]


# 快速用于生成粗略 True PF 的极简版求解器
def run_simple_search(pop_size=50, max_gen=30):
    population = [{'gene': generate_random_chromosome()} for _ in range(pop_size)]
    for p in population:
        l = heuristic_decoder(p['gene'])
        p['fit'] = calculate_fitness(l) if l.is_valid else [float('inf')] * 3
    for _ in range(max_gen):
        for p in population:
            child = copy.deepcopy(p['gene'])
            mut_idx = random.randint(0, len(child) - 1)
            child[mut_idx]['rot'] = 90 if child[mut_idx]['rot'] == 0 else 0
            l = heuristic_decoder(child)
            f = calculate_fitness(l) if l.is_valid else [float('inf')] * 3
            if sum(f) < sum(p['fit']): p['gene'], p['fit'] = child, f
    return np.array([p['fit'] for p in population if p['fit'][0] != float('inf')])


def build_true_pf(num_runs=10, save_path="True_PF_raw.npy"):
    all_runs_fits = []
    for i in range(num_runs):
        pf_fits = run_simple_search(50, 30)
        all_runs_fits.append(pf_fits)
    pooled_fits = np.vstack(all_runs_fits)
    true_pf_fits = get_global_non_dominated_set(pooled_fits)
    np.save(save_path, true_pf_fits)
    return true_pf_fits


# ==========================================
# 11. 最终主程序入口
# ==========================================
if __name__ == "__main__":
    PF_FILE = "True_PF_raw.npy"

    # 如果没有 True PF 数据则构建一次
    if not os.path.exists(PF_FILE):
        print("⚠️ 未检测到 True PF 文件，将自动生成以计算 IGD...")
        build_true_pf(num_runs=10, save_path=PF_FILE)

    true_pf_raw = np.load(PF_FILE)

    # 直接执行完整的 HD-MOGA 算法流程并渲染结果
    run_moga(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw, sigma_share=0.5)