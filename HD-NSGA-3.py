import os
import time
import tracemalloc
import math
import random
import copy
import csv
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import solve

# ==========================================
# 1. 全局环境、物理限制与参数定义
# ==========================================
POP_SIZE = 100  # 种群数量
NUM_ITERATIONS = 200  # 迭代轮数
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
RACK_LENGTH = 1520
RACK_WIDTH = 660

# --- 硬约束阈值 ---
LIMIT_COG_X = 30
LIMIT_COG_Y = 5
LIMIT_COG_Z = 200
LIMIT_DEV12_LOAD = 20.0

# 热传递参数
ETA_THERMAL = 0.8
T_ENV = 10
# 权重定义：[转动惯量(F1), 垂直热力(F2), 信号走线(F3)]
WEIGHTS = np.array([0.5, 0.3, 0.2])

GLOBAL_FITNESS_EVALS = 0

MAX_LOAD_CAPACITY = {
    1: 5.0, 2: 5.0, 3: 10.0,
    4: 50.0, 5: 30.0, 6: 8.0
}


# ==========================================
# 2. 设备实体类与基础数据 (保持不变)
# ==========================================
class Device:
    def __init__(self, dev_id, name, length, width, height, weight, category, Q, T_max, R):
        self.id = dev_id;
        self.name = name;
        self.L = length;
        self.W = width;
        self.H = height
        self.weight = weight;
        self.category = category;
        self.Q = Q;
        self.T_max = T_max;
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
edges = [(1, 8, 1), (1, 9, 5), (1, 10, 5), (1, 11, 1), (1, 12, 5), (1, 15, 1), (1, 16, 5), (2, 11, 1), (2, 13, 1),
         (2, 16, 5), (3, 8, 1), (3, 11, 1), (3, 14, 1), (3, 15, 1), (3, 16, 5), (4, 17, 10), (5, 18, 10), (6, 17, 10),
         (7, 18, 10), (8, 16, 1), (8, 14, 1), (9, 15, 1), (10, 15, 1), (11, 12, 1), (11, 13, 1), (11, 14, 1),
         (11, 15, 1), (11, 16, 1), (11, 17, 1), (11, 18, 1), (12, 15, 1), (13, 17, 10), (13, 18, 10), (14, 15, 1),
         (15, 16, 1)]
for (u, v, w) in edges:
    W_MATRIX[u - 1][v - 1] = w;
    W_MATRIX[v - 1][u - 1] = w


# ==========================================
# 3. 核心物理检测模型 (保持不变)
# ==========================================
class PlacedDevice:
    def __init__(self, device, x, y, z, rotation, support=None):
        self.device = device;
        self.x = x;
        self.y = y;
        self.z = z;
        self.rotation = rotation;
        self.support = support

    def get_effective_dims(self):
        return (self.device.W, self.device.L, self.device.H) if self.rotation == 90 else (
        self.device.L, self.device.W, self.device.H)

    def get_area(self): d = self.get_effective_dims(); return d[0] * d[1]

    def get_center_3d(self): return (self.x, self.y, self.z + self.device.H / 2.0)


class Layout:
    def __init__(self): self.placed_devices = []; self.is_valid = True

    def add(self, pd): self.placed_devices.append(pd)


def get_surface_distance(pd1, pd2):
    L1, W1, H1 = pd1.get_effective_dims();
    L2, W2, H2 = pd2.get_effective_dims()
    dx = max(0.0, abs(pd1.x - pd2.x) - (L1 + L2) / 2.0)
    dy = max(0.0, abs(pd1.y - pd2.y) - (W1 + W2) / 2.0)
    z1_min, z1_max, z2_min, z2_max = pd1.z, pd1.z + H1, pd2.z, pd2.z + H2
    dz = max(0.0, max(z1_min - z2_max, z2_min - z1_max))
    dz_dir = (z1_min - z2_max) if z1_min >= z2_max else (z1_max - z2_min) if z2_min >= z1_max else 0.0
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2), dx, dy, dz_dir


def check_collision_aabb(pd1, pd2):
    L1, W1, H1 = pd1.get_effective_dims();
    L2, W2, H2 = pd2.get_effective_dims()
    cx1, cy1, cz1 = pd1.get_center_3d();
    cx2, cy2, cz2 = pd2.get_center_3d()
    return (abs(cx1 - cx2) < (L1 + L2) / 2.0 - 0.1) and (abs(cy1 - cy2) < (W1 + W2) / 2.0 - 0.1) and (
                abs(cz1 - cz2) < (H1 + H2) / 2.0 - 0.1)


def check_stacking_rules(new_pd, support_pd):
    L_up, W_up, _ = new_pd.get_effective_dims();
    L_dn, W_dn, _ = support_pd.get_effective_dims()
    if L_up > L_dn or W_up > W_dn: return False
    return not (
                new_pd.x - L_up / 2 < support_pd.x - L_dn / 2 or new_pd.x + L_up / 2 > support_pd.x + L_dn / 2 or new_pd.y - W_up / 2 < support_pd.y - W_dn / 2 or new_pd.y + W_up / 2 > support_pd.y + W_dn / 2)


def _is_supported_by(pd, base_pd):
    curr = pd.support
    while curr:
        if curr == base_pd: return True
        curr = curr.support
    return False


def check_load_capacity(new_pd, layout):
    curr_support = new_pd.support
    while curr_support is not None:
        current_load = sum(pd.device.weight for pd in layout.placed_devices if _is_supported_by(pd, curr_support))
        if current_load + new_pd.device.weight > MAX_LOAD_CAPACITY.get(curr_support.device.category, 0): return False
        curr_support = curr_support.support
    return True


# ==========================================
# 4. 评估模块与解码器 (保持不变)
# ==========================================
def calculate_fitness(layout):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS += 1
    if not layout.is_valid: return [float('inf')] * 3
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

    I_xx, I_yy, I_zz = 0, 0, 0
    for pd in layout.placed_devices:
        L, W, H = pd.get_effective_dims();
        m = pd.device.weight;
        cx_i, cy_i, cz_i = pd.get_center_3d()
        I_xx += (m / 12.0) * (W ** 2 + H ** 2) + m * (cy_i ** 2 + cz_i ** 2)
        I_yy += (m / 12.0) * (L ** 2 + H ** 2) + m * (cx_i ** 2 + cz_i ** 2)
        I_zz += (m / 12.0) * (L ** 2 + W ** 2) + m * (cx_i ** 2 + cy_i ** 2)
    F_inertia = math.sqrt(I_xx ** 2 + I_yy ** 2 + I_zz ** 2) / 1e4

    F_thermal, penalty_thermal, q_cache = 0, 0, {}

    def get_q(pd):
        if pd in q_cache: return q_cache[pd]
        q_self = pd.device.Q
        if pd.support is None:
            res = q_self
        else:
            res = q_self + ETA_THERMAL * min(1.0, pd.get_area() / pd.support.get_area()) * get_q(pd.support)
        q_cache[pd] = res;
        return res

    for pd in layout.placed_devices:
        T_i = T_ENV + pd.device.R * get_q(pd)
        F_thermal += T_i
        if T_i > pd.device.T_max: penalty_thermal += (T_i - pd.device.T_max) ** 2
    F_thermal += penalty_thermal

    F_routing = 0
    for i in range(len(layout.placed_devices)):
        for j in range(i + 1, len(layout.placed_devices)):
            pd1, pd2 = layout.placed_devices[i], layout.placed_devices[j]
            w = W_MATRIX[pd1.device.id - 1][pd2.device.id - 1]
            if w > 0:
                _, dx, dy, dz_d = get_surface_distance(pd1, pd2)
                F_routing += w * (dx + dy + abs(dz_d))
    return [F_inertia + penalty, F_thermal + penalty, F_routing + penalty]


def heuristic_decoder(chromosome):
    layout = Layout()
    for gene in chromosome:
        dev = next(d for d in DEVICE_LIST if d.id == gene['id']);
        rot = gene['rot']
        L, W, H = (dev.W, dev.L, dev.H) if rot == 90 else (dev.L, dev.W, dev.H)
        if dev.id == 12: layout.add(PlacedDevice(dev, 0, 0, 0, rot, None)); continue
        placed, candidates = False, []
        xl, yl = int(RACK_LENGTH / 2 - L / 2), int(RACK_WIDTH / 2 - W / 2)
        for x in range(-xl, xl + 1, 80):
            for y in range(-yl, yl + 1, 80): candidates.append({'x': x, 'y': y, 'z': 0, 's': None})
        for pd in layout.placed_devices:
            Ld, Wd, Hd = pd.get_effective_dims()
            sx, ex = int(pd.x - Ld / 2 + L / 2), int(pd.x + Ld / 2 - L / 2)
            sy, ey = int(pd.y - Wd / 2 + W / 2), int(pd.y + Wd / 2 - W / 2)
            if sx <= ex and sy <= ey:
                for x in list(range(sx, ex + 1, 40)) + [ex]:
                    for y in list(range(sy, ey + 1, 40)) + [ey]: candidates.append(
                        {'x': x, 'y': y, 'z': pd.z + Hd, 's': pd})
        for pos in candidates:
            t = PlacedDevice(dev, pos['x'], pos['y'], pos['z'], rot, pos['s'])
            if abs(pos['x']) > xl or abs(pos['y']) > yl: continue
            if any(check_collision_aabb(t, ex) for ex in layout.placed_devices): continue
            if pos['z'] > 0 and (not check_stacking_rules(t, pos['s']) or not check_load_capacity(t, layout)): continue
            layout.add(t);
            placed = True;
            break
        if not placed: layout.is_valid = False; return layout
    return layout


def generate_random_chromosome():
    h = [{'id': d.id, 'rot': random.choice([0, 90])} for d in DEVICE_LIST if d.id != 12 and d.weight >= 15]
    l = [{'id': d.id, 'rot': random.choice([0, 90])} for d in DEVICE_LIST if d.id != 12 and d.weight < 15]
    random.shuffle(h);
    random.shuffle(l)
    return [{'id': 12, 'rot': random.choice([0, 90])}] + h + l


def dominates(f1, f2):
    cond = False
    for a, b in zip(f1, f2):
        if a > b: return False
        if a < b: cond = True
    return cond


# ==========================================
# 5. NSGA-III 核心算法模块
# ==========================================

def fast_non_dominated_sort(fits):
    n = len(fits)
    dom_sets = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(fits[i], fits[j]):
                dom_sets[i].append(j);
                dom_count[j] += 1
            elif dominates(fits[j], fits[i]):
                dom_sets[j].append(i);
                dom_count[i] += 1
        if dom_count[i] == 0: fronts[0].append(i)

    curr = 0
    while len(fronts[curr]) > 0:
        next_front = []
        for i in fronts[curr]:
            for j in dom_sets[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0: next_front.append(j)
        curr += 1
        fronts.append(next_front)
    return fronts[:-1]


def uniform_reference_points(n_obj, n_div):
    def generate_recursive(pts, curr, left, n_obj, depth):
        if depth == n_obj - 1:
            curr[depth] = left / n_div
            pts.append(curr.copy())
        else:
            for i in range(left + 1):
                curr[depth] = i / n_div
                generate_recursive(pts, curr, left - i, n_obj, depth + 1)

    pts = []
    generate_recursive(pts, np.zeros(n_obj), n_div, n_obj, 0)
    return np.array(pts)


def get_extreme_points(fits, ideal_point):
    n_obj = fits.shape[1]
    weights = np.eye(n_obj) + 1e-6
    extreme_indices = []
    for w in weights:
        asf = np.max((fits - ideal_point) / w, axis=1)
        extreme_indices.append(np.argmin(asf))
    return extreme_indices


def get_intercepts(fits, extreme_indices, ideal_point):
    n_obj = fits.shape[1]
    extreme_fits = fits[extreme_indices] - ideal_point
    try:
        b = np.ones(n_obj)
        a = solve(extreme_fits, b)
        intercepts = 1 / a
        if np.any(intercepts <= 1e-6): raise Exception()
    except:
        intercepts = np.max(fits, axis=0) - ideal_point
    return intercepts


def nsga3_environmental_selection(combined_pop, n_select, ref_points):
    fits = np.array([p['fit'] for p in combined_pop])
    fronts = fast_non_dominated_sort(fits)
    next_pop_indices = []
    l = 0
    while len(next_pop_indices) + len(fronts[l]) <= n_select:
        next_pop_indices.extend(fronts[l])
        l += 1
        if l >= len(fronts): break
    if len(next_pop_indices) == n_select: return [combined_pop[i] for i in next_pop_indices]

    last_front = fronts[l];
    num_needed = n_select - len(next_pop_indices)
    all_indices = next_pop_indices + last_front;
    all_fits = fits[all_indices]
    ideal_point = np.min(fits, axis=0)
    extreme_idx = get_extreme_points(all_fits, ideal_point)
    intercepts = get_intercepts(all_fits, extreme_idx, ideal_point)
    norm_fits = (fits - ideal_point) / (intercepts + 1e-6)

    def associate(pop_idx):
        assoc_ref, dists = [], []
        for i in pop_idx:
            f = norm_fits[i]
            ref_dists = []
            for r in ref_points:
                norm_r = np.linalg.norm(r)
                if norm_r < 1e-6:
                    d = np.linalg.norm(f)
                else:
                    d = np.linalg.norm(f - (np.dot(f, r) / (norm_r ** 2)) * r)
                ref_dists.append(d)
            best_r = np.argmin(ref_dists);
            assoc_ref.append(best_r);
            dists.append(ref_dists[best_r])
        return assoc_ref, dists

    assoc_indices, _ = associate(next_pop_indices)
    niche_count = [0] * len(ref_points)
    for a in assoc_indices: niche_count[a] += 1
    lf_assoc, lf_dists = associate(last_front)

    chosen_lf = []
    while len(chosen_lf) < num_needed:
        min_indices = [i for i, c in enumerate(niche_count) if c == min(niche_count)]
        r_bar = random.choice(min_indices)
        candidates = [i for i, r in enumerate(lf_assoc) if r == r_bar and i not in chosen_lf]
        if candidates:
            sel = candidates[np.argmin([lf_dists[c] for c in candidates])] if niche_count[
                                                                                  r_bar] == 0 else random.choice(
                candidates)
            chosen_lf.append(last_front[sel]);
            niche_count[r_bar] += 1
        else:
            niche_count[r_bar] = float('inf')
    next_pop_indices.extend(chosen_lf)
    return [combined_pop[i] for i in next_pop_indices]


# ==========================================
# 6. 绩效评估指标 (HV & IGD)
# ==========================================
def calculate_hypervolume_monte_carlo(pf_fits, ref=[1.1, 1.1, 1.1], samples=50000):
    """
    计算超体积 (HV): 使用蒙特卡洛法估算
    """
    if len(pf_fits) == 0: return 0.0
    s = np.random.uniform(0, ref[0], (samples, 3))
    count = 0
    for smp in s:
        for f in pf_fits:
            if all(f <= smp): count += 1; break
    return (count / samples) * np.prod(ref)


def calculate_igd_normalized(curr, true):
    mn, mx = true.min(0), true.max(0);
    den = mx - mn;
    den[den == 0] = 1e-6
    n_true, n_curr = (true - mn) / den, (curr - mn) / den
    return np.mean([np.min(np.linalg.norm(n_curr - p, axis=1)) for p in n_true])


# ==========================================
# 7. NSGA-III 主流程与输出
# ==========================================
def run_nsga3(pop_size=50, max_gen=200, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start();
    start_time = time.time()

    ref_points = uniform_reference_points(3, 8)  # 45个参考点

    population = []
    while len(population) < pop_size:
        c = generate_random_chromosome();
        l = heuristic_decoder(c)
        if l.is_valid:
            f = calculate_fitness(l)
            if f[0] != float('inf'): population.append({'gene': c, 'fit': f, 'layout': l})

    print(f"⚡ 开始 NSGA-III 3目标优化 (共 {max_gen} 代)...")
    for gen in range(max_gen):
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(population, 2)
            c_gene = copy.deepcopy(p1['gene'])
            if random.random() < CROSSOVER_RATE:
                i1, i2 = random.sample(range(1, len(c_gene)), 2)
                c_gene[i1], c_gene[i2] = c_gene[i2], c_gene[i1]
            if random.random() < MUTATION_RATE:
                mi = random.randint(0, len(c_gene) - 1)
                c_gene[mi]['rot'] = 90 if c_gene[mi]['rot'] == 0 else 0
            l = heuristic_decoder(c_gene)
            if l.is_valid:
                f = calculate_fitness(l);
                offspring.append({'gene': c_gene, 'fit': f, 'layout': l})

        population = nsga3_environmental_selection(population + offspring, pop_size, ref_points)
        if (gen + 1) % 10 == 0 or gen == 0:
            fits = np.array([p['fit'] for p in population])
            pf_count = len(fast_non_dominated_sort(fits)[0])
            print(f"迭代 {gen + 1}/{max_gen} | 当前帕累托前沿解数量: {pf_count}")

    end_time = time.time();
    _, peak_mem = tracemalloc.get_traced_memory();
    tracemalloc.stop()

    # 提取帕累托前沿
    fits_all = np.array([p['fit'] for p in population])
    p_idx = fast_non_dominated_sort(fits_all)[0]
    final_front = [population[i] for i in p_idx];
    final_front.sort(key=lambda x: x['fit'][0])
    all_fits = np.array([p['fit'] for p in final_front])

    # 归一化用于指标计算
    f_min, f_max = all_fits.min(0), all_fits.max(0)
    den = f_max - f_min;
    den[den == 0] = 1e-6
    norm_fits = (all_fits - f_min) / den

    # 计算 HV 和 IGD
    hv_val = calculate_hypervolume_monte_carlo(norm_fits)
    igd_val = calculate_igd_normalized(all_fits, true_pf) if true_pf is not None else 0.0

    # 计算加权综合得分
    weighted_sums = np.sum(norm_fits * WEIGHTS, axis=1)
    best_balanced_idx = np.argmin(weighted_sums)

    print("\n" + "=" * 110 + "\n📊 NSGA-III 综合性能评估报告")
    print(
        f"⏱️ 耗时: {end_time - start_time:.2f}s | 💾 内存峰值: {peak_mem / 1e6:.2f}MB | ⚙️ 评估次数: {GLOBAL_FITNESS_EVALS}")
    print(f"📐 HV (超体积): {hv_val:.4f} (参考点[1.1,1.1,1.1]) | 🎯 IGD (倒世代距离): {igd_val:.4f}")
    print("-" * 110)
    print(
        f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'重心X':<7} | {'重心Y':<7} | {'重心Z':<7} | {'12号承重'}")

    detailed = []
    for i, ind in enumerate(final_front):
        pd_l = ind['layout'].placed_devices;
        tw = sum(p.device.weight for p in pd_l)
        cx = sum(p.device.weight * p.get_center_3d()[0] for p in pd_l) / tw
        cy = sum(p.device.weight * p.get_center_3d()[1] for p in pd_l) / tw
        cz = sum(p.device.weight * p.get_center_3d()[2] for p in pd_l) / tw
        l12 = sum(pd.device.weight for pd in pd_l if _is_supported_by(pd, next(p for p in pd_l if p.device.id == 12)))
        print(
            f"{i + 1:<6} | {ind['fit'][0]:<10.3f} | {ind['fit'][1]:<10.1f} | {ind['fit'][2]:<10.1f} | {cx:<8.1f} | {cy:<8.1f} | {cz:<8.1f} | {l12:<8.1f}")
        detailed.append({'id': i + 1, 'fit': ind['fit'], 'cg': [cx, cy, cz], 'load12': l12, 'w_sum': weighted_sums[i]})

    print("-" * 110)
    print(f"⭐ 基于自定义权重 (惯量{WEIGHTS[0]} : 热力{WEIGHTS[1]} : 走线{WEIGHTS[2]}) 推荐的最佳加权和均衡解为:")
    print(
        f"   [方案 {best_balanced_idx + 1}] | 归一化空间加权得分 (Weighted Sum): {weighted_sums[best_balanced_idx]:.4f}")

    # 可视化 (Plotly)
    hts = [
        f"<b>方案 {d['id']}</b><br>F1: {d['fit'][0]:.3f}<br>F2: {d['fit'][1]:.1f}<br>F3: {d['fit'][2]:.1f}<br>加权得分: {d['w_sum']:.4f}"
        for d in detailed]
    fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
                                       marker=dict(size=6, color=weighted_sums, colorscale='Viridis',
                                                   colorbar=dict(title="Weighted Sum")), text=hts, hoverinfo='text')])
    fig.add_trace(go.Scatter3d(x=[all_fits[best_balanced_idx, 0]], y=[all_fits[best_balanced_idx, 1]],
                               z=[all_fits[best_balanced_idx, 2]], mode='markers+text',
                               marker=dict(size=12, color='red', symbol='diamond'), text=["★ Best Solution"],
                               textposition="top center"))
    fig.update_layout(title="NSGA-III Pareto Front 3D (HV & Weighted Sum Included)",
                      scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
    fig.write_html("Pareto_Front_NSGA3_Final.html");
    fig.show()


if __name__ == "__main__":
    PF_FILE = "True_PF_raw.npy"
    true_pf_raw = np.load(PF_FILE) if os.path.exists(PF_FILE) else None
    run_nsga3(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)