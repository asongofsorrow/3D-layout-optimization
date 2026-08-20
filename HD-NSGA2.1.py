import os
import time
import tracemalloc
import math
import random
import copy
import csv
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.linalg import solve

# ==========================================
# 1. 全局环境、物理限制与参数定义 (严格匹配原始版本)
# ==========================================
POP_SIZE = 100
NUM_ITERATIONS = 200
RACK_LENGTH, RACK_WIDTH = 1520, 660

# --- IEBA 算法特定参数 ---
F_MIN, F_MAX = 0, 2.0  # 频率范围
A_0, R_0 = 0.9, 0.1  # 初始响度与脉冲发射率
ALPHA, GAMMA = 0.9, 0.9  # 衰减系数
ARCHIVE_SIZE = 50

# --- MOEAD 算法特定参数 ---
NEIGHBOR_SIZE = 10
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2

# --- SPEA2 算法特定参数 ---
ARCHIVE_SIZE = POP_SIZE  # SPEA2 外部存档数量 (通常与种群一致)

# --- 硬约束阈值 ---
LIMIT_COG_X, LIMIT_COG_Y, LIMIT_COG_Z = 30, 5, 200
LIMIT_DEV12_LOAD = 20.0
ETA_THERMAL, T_ENV = 0.8, 10
WEIGHTS = np.array([0.5, 0.3, 0.2])

GLOBAL_FITNESS_EVALS = 0
MAX_LOAD_CAPACITY = {1: 5.0, 2: 5.0, 3: 10.0, 4: 50.0, 5: 30.0, 6: 8.0}


# ==========================================
# 2. 设备实体与物理模型 (恢复原始逻辑)
# ==========================================
class Device:
    def __init__(self, dev_id, name, length, width, height, weight, category, Q, T_max, R):
        self.id, self.name, self.L, self.W, self.H = dev_id, name, length, width, height
        self.weight, self.category, self.Q, self.T_max, self.R = weight, category, Q, T_max, R


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
edges = [(1, 8, 1), (1, 9, 5), (1, 10, 5), (1, 11, 1), (1, 12, 5), (1, 15, 1), (1, 16, 5), (2, 11, 1), (2, 13, 1),
         (2, 16, 5), (3, 8, 1), (3, 11, 1), (3, 14, 1), (3, 15, 1), (3, 16, 5), (4, 17, 10), (5, 18, 10), (6, 17, 10),
         (7, 18, 10), (8, 16, 1), (8, 14, 1), (9, 15, 1), (10, 15, 1), (11, 12, 1), (11, 13, 1), (11, 14, 1),
         (11, 15, 1), (11, 16, 1), (11, 17, 1), (11, 18, 1), (12, 15, 1), (13, 17, 10), (13, 18, 10), (14, 15, 1),
         (15, 16, 1)]
for u, v, w in edges: W_MATRIX[u - 1][v - 1] = w; W_MATRIX[v - 1][u - 1] = w


class PlacedDevice:
    def __init__(self, device, x, y, z, rotation, support=None):
        self.device, self.x, self.y, self.z, self.rotation, self.support = device, x, y, z, rotation, support

    def get_effective_dims(self):
        return (self.device.W, self.device.L, self.device.H) if self.rotation == 90 else (
        self.device.L, self.device.W, self.device.H)

    def get_area(self): d = self.get_effective_dims(); return d[0] * d[1]

    def get_center_3d(self): return (self.x, self.y, self.z + self.device.H / 2.0)


class Layout:
    def __init__(self): self.placed_devices = []; self.is_valid = True

    def add(self, pd): self.placed_devices.append(pd)


def get_surface_distance(pd1, pd2):
    L1, W1, H1 = pd1.get_effective_dims()
    L2, W2, H2 = pd2.get_effective_dims()
    dx = max(0.0, abs(pd1.x - pd2.x) - (L1 + L2) / 2.0)
    dy = max(0.0, abs(pd1.y - pd2.y) - (W1 + W2) / 2.0)
    z1_min, z1_max, z2_min, z2_max = pd1.z, pd1.z + H1, pd2.z, pd2.z + H2
    dz_dir = (z1_min - z2_max) if z1_min >= z2_max else (z1_max - z2_min) if z2_min >= z1_max else 0.0
    return math.sqrt(dx ** 2 + dy ** 2 + dz_dir ** 2), dx, dy, dz_dir

def check_stacking_rules(new_pd, support_pd):
    L_up, W_up, _ = new_pd.get_effective_dims();
    L_dn, W_dn, _ = support_pd.get_effective_dims()
    if L_up > L_dn or W_up > W_dn: return False
    return not (
                new_pd.x - L_up / 2 < support_pd.x - L_dn / 2 or new_pd.x + L_up / 2 > support_pd.x + L_dn / 2 or new_pd.y - W_up / 2 < support_pd.y - W_dn / 2 or new_pd.y + W_up / 2 > support_pd.y + W_dn / 2)

def check_load_capacity(new_pd, layout):
    curr_support = new_pd.support
    while curr_support is not None:
        current_load = sum(pd.device.weight for pd in layout.placed_devices if _is_supported_by(pd, curr_support))
        if current_load + new_pd.device.weight > MAX_LOAD_CAPACITY.get(curr_support.device.category, 0): return False
        curr_support = curr_support.support
    return True

def check_collision_aabb(pd1, pd2):
    L1, W1, H1 = pd1.get_effective_dims()
    L2, W2, H2 = pd2.get_effective_dims()
    cx1, cy1, cz1 = pd1.get_center_3d()
    cx2, cy2, cz2 = pd2.get_center_3d()
    return (abs(cx1 - cx2) < (L1 + L2) / 2 - 0.1) and (abs(cy1 - cy2) < (W1 + W2) / 2 - 0.1) and (
                abs(cz1 - cz2) < (H1 + H2) / 2 - 0.1)


def _is_supported_by(pd, base_pd):
    curr = pd.support
    while curr:
        if curr == base_pd: return True
        curr = curr.support
    return False


# ==========================================
# 3. 核心评估模块 (严格恢复原始数值计算)
# ==========================================
def calculate_fitness(layout):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS += 1
    if not layout.is_valid: return [float('inf')] * 3

    # 1. 重心与硬约束
    total_w = sum(pd.device.weight for pd in layout.placed_devices)
    cx = sum(pd.device.weight * pd.get_center_3d()[0] for pd in layout.placed_devices) / total_w
    cy = sum(pd.device.weight * pd.get_center_3d()[1] for pd in layout.placed_devices) / total_w
    cz = sum(pd.device.weight * pd.get_center_3d()[2] for pd in layout.placed_devices) / total_w

    dev12 = next(p for p in layout.placed_devices if p.device.id == 12)
    dev12_load = sum(pd.device.weight for pd in layout.placed_devices if _is_supported_by(pd, dev12))

    penalty = 0
    if abs(cx) > LIMIT_COG_X: penalty += (abs(cx) - LIMIT_COG_X) * 1e7
    if abs(cy) > LIMIT_COG_Y: penalty += (abs(cy) - LIMIT_COG_Y) * 1e7
    if cz > LIMIT_COG_Z: penalty += (cz - LIMIT_COG_Z) * 1e7
    if dev12_load > LIMIT_DEV12_LOAD: penalty += (dev12_load - LIMIT_DEV12_LOAD) * 1e7

    # F1: 转动惯量 (Parallel Axis Theorem)
    I_xx, I_yy, I_zz = 0, 0, 0
    for pd in layout.placed_devices:
        L, W, H = pd.get_effective_dims()
        m = pd.device.weight
        px, py, pz = pd.get_center_3d()
        I_xx += (m / 12.0) * (W ** 2 + H ** 2) + m * (py ** 2 + pz ** 2)
        I_yy += (m / 12.0) * (L ** 2 + H ** 2) + m * (px ** 2 + pz ** 2)
        I_zz += (m / 12.0) * (L ** 2 + W ** 2) + m * (px ** 2 + py ** 2)
    F_inertia = math.sqrt(I_xx ** 2 + I_yy ** 2 + I_zz ** 2) / 1e4

    # F2: 热传递 (基于递归热应力)
    F_thermal, penalty_thermal, q_cache = 0, 0, {}

    def get_q(pd):
        if pd in q_cache: return q_cache[pd]
        q_self = pd.device.Q
        if pd.support is None:
            res = q_self
        else:
            res = q_self + ETA_THERMAL * min(1.0, pd.get_area() / pd.support.get_area()) * get_q(pd.support)
        q_cache[pd] = res
        return res

    for pd in layout.placed_devices:
        T_i = T_ENV + pd.device.R * get_q(pd)
        F_thermal += T_i
        if T_i > pd.device.T_max: penalty_thermal += (T_i - pd.device.T_max) ** 2
    F_thermal += penalty_thermal

    # F3: 布线 (权重 * 表面距离)
    F_routing = 0
    for i in range(len(layout.placed_devices)):
        for j in range(i + 1, len(layout.placed_devices)):
            w = W_MATRIX[layout.placed_devices[i].device.id - 1][layout.placed_devices[j].device.id - 1]
            if w > 0:
                _, dx, dy, dz_d = get_surface_distance(layout.placed_devices[i], layout.placed_devices[j])
                F_routing += w * (dx + dy + abs(dz_d))

    return [F_inertia + penalty, F_thermal + penalty, F_routing + penalty]

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
# ==========================================
# 4. 解码器与指标工具 (保持原始精度)
# ==========================================
def heuristic_decoder(chromo):
    layout = Layout()
    for g in chromo:
        dev = next(d for d in DEVICE_LIST if d.id == g['id'])
        L, W, H = (dev.W, dev.L, dev.H) if g['rot'] == 90 else (dev.L, dev.W, dev.H)
        if dev.id == 12: layout.add(PlacedDevice(dev, 0, 0, 0, g['rot'])); continue

        placed, candidates = False, []
        xl, yl = int(RACK_LENGTH / 2 - L / 2), int(RACK_WIDTH / 2 - W / 2)
        for x in range(-xl, xl + 1, 100):
            for y in range(-yl, yl + 1, 100): candidates.append({'x': x, 'y': y, 'z': 0, 's': None})
        for pd in layout.placed_devices:
            Ld, Wd, Hd = pd.get_effective_dims()
            sx, ex, sy, ey = int(pd.x - Ld / 2 + L / 2), int(pd.x + Ld / 2 - L / 2), int(pd.y - Wd / 2 + W / 2), int(
                pd.y + Wd / 2 - W / 2)
            if sx <= ex and sy <= ey:
                for x in [sx, ex]:
                    for y in [sy, ey]: candidates.append({'x': x, 'y': y, 'z': pd.z + Hd, 's': pd})

        # random.shuffle(candidates)
        for pos in candidates:
            t = PlacedDevice(dev, pos['x'], pos['y'], pos['z'], g['rot'], pos['s'])
            if abs(pos['x']) > xl or abs(pos['y']) > yl: continue
            if any(check_collision_aabb(t, ex) for ex in layout.placed_devices): continue
            if pos['z'] > 0 and (not check_stacking_rules(t, pos['s']) or not check_load_capacity(t, layout)): continue
            layout.add(t)
            placed = True
            break
        if not placed: layout.is_valid = False; return layout
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


def calculate_igd(curr_pf, true_pf):
    if true_pf is None or len(curr_pf) == 0: return 0.0
    mn, mx = true_pf.min(0), true_pf.max(0)
    den = mx - mn
    den[den == 0] = 1e-6
    n_true, n_curr = (true_pf - mn) / den, (curr_pf - mn) / den
    return np.mean([np.min(np.linalg.norm(n_curr - p, axis=1)) for p in n_true])


def calculate_hv(n_fits, ref=[1.1, 1.1, 1.1], samples=50000):
    if len(n_fits) == 0: return 0.0
    s = np.random.uniform(0, ref[0], (samples, 3))
    count = sum(1 for smp in s if any(all(f <= smp) for f in n_fits))
    return (count / samples) * np.prod(ref)


# ==========================================
# 5. IEBA 算法核心与性能监控
# ==========================================
# def dominates(f1, f2):
#     return all(np.array(f1) <= np.array(f2)) and any(np.array(f1) < np.array(f2))
#NSGA2特供
def dominates(fit1, fit2):
    and_condition = False
    for f1, f2 in zip(fit1, fit2):
        if f1 > f2: return False
        if f1 < f2: and_condition = True
    return and_condition

# ==========================================
# 6. nsga2 核心算法模块
# ==========================================
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


def fast_non_dominated_sort(pop):
    n = len(pop)
    dom_sets = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]

    # 1. Compare the 'fit' lists correctly instead of comparing dictionaries
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(pop[i]['fit'], pop[j]['fit']):
                dom_sets[i].append(j)
                dom_count[j] += 1
            elif dominates(pop[j]['fit'], pop[i]['fit']):
                dom_sets[j].append(i)
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)
            pop[i]['rank'] = 0  # Assign rank 0 for the first front

    curr = 0
    while len(fronts[curr]) > 0:
        next_front = []
        for i in fronts[curr]:
            for j in dom_sets[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
                    pop[j]['rank'] = curr + 1  # Assign ranks to sub-fronts
        curr += 1

        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    # 2. Return lists of actual dictionary objects instead of just integer indices
    actual_fronts = [[pop[idx] for idx in front] for front in fronts]
    return actual_fronts

def run_nsga2(pop_size=50, max_gen=50, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0  # 重置计数器

    # --- 启动性能监控 ---
    tracemalloc.start()
    start_time = time.time()

    population = []
    # print("🚀 正在初始化种群，寻找符合硬约束的合法解...")
    while len(population) < pop_size:
        chromo = generate_random_chromosome()
        layout = heuristic_decoder(chromo)
        if layout.is_valid:
            fit = calculate_fitness(layout)
            if fit[0] != float('inf'):
                population.append({'gene': chromo, 'fit': fit, 'layout': layout})

    # print(f"\n⚡ 开始 NSGA-II 3目标进化迭代 (共 {max_gen} 代)...")
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
        # print(f"迭代 {gen + 1}/{max_gen} | 3目标帕累托前沿解数量: {rank0_count}")

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
    hv_value = calculate_hv(norm_fits)

    # 2. ★ 计算真实的 IGD ★
    if true_pf is not None:
        # 传入未归一化的 all_fits 和真实的 true_pf
        igd_value = calculate_igd(all_fits, true_pf)
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

    # print("\n" + "=" * 110)
    # print(f"🏆 NSGA-II 优化完成！输出 {len(final_front)} 个帕累托最优解。")
    # print(
    #     f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'重心X':<7} | {'重心Y':<7} | {'重心Z':<7} | {'12号承重'}")
    # print("-" * 110)

    detailed_info = []

    for idx, ind in enumerate(final_front):
        pd_list = ind['layout'].placed_devices
        total_w = sum(p.device.weight for p in pd_list)
        cg_x = sum(p.device.weight * p.get_center_3d()[0] for p in pd_list) / total_w
        cg_y = sum(p.device.weight * p.get_center_3d()[1] for p in pd_list) / total_w
        cg_z = sum(p.device.weight * p.get_center_3d()[2] for p in pd_list) / total_w

        dev12_load = sum(
            pd.device.weight for pd in pd_list if
            _is_supported_by(pd, next(p for p in pd_list if p.device.id == 12)))

        fit = ind['fit']
        info_dict = {'id': idx + 1, 'fit': fit, 'cg': [cg_x, cg_y, cg_z], 'load12': dev12_load}
        detailed_info.append(info_dict)

        # print(
        #     f"{idx + 1:<6} | {fit[0]:<10.3f} | {fit[1]:<10.1f} | {fit[2]:<10.1f} | {cg_x:<8.1f} | {cg_y:<8.1f} | {cg_z:<8.1f} | {dev12_load:<8.1f}")
        # save_detailed_csv(ind['layout'], f"nsga2_solution_{idx + 1}.csv", fit, [cg_x, cg_y, cg_z], dev12_load)

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

    # # ================= 交互式 Plotly 3D 可视化 =================
    # hover_texts = []
    # for i, info in enumerate(detailed_info):
    #     n_f1, n_f2, n_f3 = norm_fits[i][0], norm_fits[i][1], norm_fits[i][2]
    #
    #     text = (f"<b>方案 ID: {info['id']}</b><br>"
    #             f"--------------------<br>"
    #             f"⚙️ 转动惯量 (F1): {info['fit'][0]:.3f} <span style='color:gray;'>(归一: {n_f1:.3f})</span><br>"
    #             f"🔥 垂直热力 (F2): {info['fit'][1]:.1f} <span style='color:gray;'>(归一: {n_f2:.3f})</span><br>"
    #             f"🔌 信号走线 (F3): {info['fit'][2]:.1f} <span style='color:gray;'>(归一: {n_f3:.3f})</span><br>"
    #             f"⚖️ 重心偏移(X,Y,Z): ({info['cg'][0]:.1f}, {info['cg'][1]:.1f}, {info['cg'][2]:.1f})<br>"
    #             f"📦 导航箱承重: {info['load12']: .1f} kg")
    #     hover_texts.append(text)
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatter3d(
    #     x=norm_fits[:, 0], y=norm_fits[:, 1], z=norm_fits[:, 2],
    #     mode='markers',
    #     marker=dict(size=6, color=norm_fits[:, 2], colorscale='Viridis', opacity=0.8),
    #     text=hover_texts,
    #     hoverinfo='text',
    #     name='Pareto Solutions'
    # ))
    #
    # fig.add_trace(go.Scatter3d(
    #     x=[0], y=[0], z=[0],
    #     mode='markers',
    #     marker=dict(size=8, color='black', symbol='cross'),
    #     name='Ideal Point [0,0,0]',
    #     hoverinfo='skip'
    # ))
    #
    # fig.add_trace(go.Scatter3d(
    #     x=[norm_fits[best_balanced_idx, 0]],
    #     y=[norm_fits[best_balanced_idx, 1]],
    #     z=[norm_fits[best_balanced_idx, 2]],
    #     mode='markers',
    #     marker=dict(size=14, color='red', symbol='diamond', line=dict(color='yellow', width=2)),
    #     text=[f"🏆 <b>【绝对最均衡解】</b><br>" + hover_texts[best_balanced_idx]],
    #     hoverinfo='text',
    #     name='Best Balanced Solution'
    # ))
    #
    # fig.update_layout(
    #     title='Interactive Normalized 3D Pareto Front (F1: Inertia, F2: Thermal, F3: Routing)',
    #     scene=dict(
    #         xaxis_title='Norm Inertia F1',
    #         yaxis_title='Norm Thermal F2',
    #         zaxis_title='Norm Routing F3',
    #         xaxis=dict(range=[-0.1, 1.1]),
    #         yaxis=dict(range=[-0.1, 1.1]),
    #         zaxis=dict(range=[-0.1, 1.1])
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=40)
    # )
    #
    # print("\n🌐 正在默认浏览器中生成交互式 3D 帕累托图...")
    # fig.show()

if __name__ == "__main__":
    PF_FILE = "True_PF_raw.npy"
    true_pf_raw = np.load(PF_FILE) if os.path.exists(PF_FILE) else None
    for i in range(8):
        run_nsga2(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)