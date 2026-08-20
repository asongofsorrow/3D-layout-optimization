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

# ==========================================
# 1. 全局环境、物理限制与参数定义
# ==========================================
POP_SIZE = 100
NUM_ITERATIONS = 200
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
NEIGHBOR_SIZE = 10
RACK_LENGTH = 1520
RACK_WIDTH = 660

LIMIT_COG_X, LIMIT_COG_Y, LIMIT_COG_Z = 30, 5, 200
LIMIT_DEV12_LOAD = 20.0

ETA_THERMAL, T_ENV = 0.8, 10
WEIGHTS = np.array([0.5, 0.3, 0.2])

GLOBAL_FITNESS_EVALS = 0
MAX_LOAD_CAPACITY = {1: 5.0, 2: 5.0, 3: 10.0, 4: 50.0, 5: 30.0, 6: 8.0}


# ==========================================
# 2. 设备实体类与物理模型 (保持不变)
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
# 3. 评估模块与解码器 (保持不变)
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
        L, W, H = pd.get_effective_dims()
        m = pd.device.weight
        cx_i, cy_i, cz_i = pd.get_center_3d()
        I_xx += (m / 12.0) * (W ** 2 + H ** 2) + m * (cy_i ** 2 + cz_i ** 2)
        I_yy += (m / 12.0) * (L ** 2 + H ** 2) + m * (cx_i ** 2 + cz_i ** 2)
        I_zz += (m / 12.0) * (L ** 2 + W ** 2) + m * (cx_i ** 2 + cy_i ** 2)
    F_inertia = math.sqrt(I_xx ** 2 + I_yy ** 2 + I_zz ** 2) / 1e4

    F_thermal, q_cache = 0, {}

    def get_q(pd):
        if pd in q_cache: return q_cache[pd]
        q_self = pd.device.Q
        res = q_self if pd.support is None else q_self + ETA_THERMAL * min(1.0,
                                                                           pd.get_area() / pd.support.get_area()) * get_q(
            pd.support)
        q_cache[pd] = res
        return res

    for pd in layout.placed_devices:
        T_i = T_ENV + pd.device.R * get_q(pd)
        F_thermal += T_i + (max(0, T_i - pd.device.T_max) ** 2)

    F_routing = 0
    for i in range(len(layout.placed_devices)):
        for j in range(i + 1, len(layout.placed_devices)):
            pd1, pd2 = layout.placed_devices[i], layout.placed_devices[j]
            w = W_MATRIX[pd1.device.id - 1][pd2.device.id - 1]
            if w > 0:
                L1, W1, H1 = pd1.get_effective_dims()
                L2, W2, H2 = pd2.get_effective_dims()
                dx, dy = max(0.0, abs(pd1.x - pd2.x) - (L1 + L2) / 2), max(0.0, abs(pd1.y - pd2.y) - (W1 + W2) / 2)
                dz = abs(pd1.z - pd2.z)
                F_routing += w * (dx + dy + dz)
    return [F_inertia + penalty, F_thermal + penalty, F_routing + penalty]


def heuristic_decoder(chromosome):
    layout = Layout()
    for gene in chromosome:
        dev = next(d for d in DEVICE_LIST if d.id == gene['id'])
        rot = gene['rot']
        L, W, H = (dev.W, dev.L, dev.H) if rot == 90 else (dev.L, dev.W, dev.H)
        if dev.id == 12: layout.add(PlacedDevice(dev, 0, 0, 0, rot, None)); continue
        placed, candidates = False, []
        xl, yl = int(RACK_LENGTH / 2 - L / 2), int(RACK_WIDTH / 2 - W / 2)
        for x in range(-xl, xl + 1, 80):
            for y in range(-yl, yl + 1, 80): candidates.append({'x': x, 'y': y, 'z': 0, 's': None})
        for pd in layout.placed_devices:
            Ld, Wd, Hd = pd.get_effective_dims()
            sx, ex, sy, ey = int(pd.x - Ld / 2 + L / 2), int(pd.x + Ld / 2 - L / 2), int(pd.y - Wd / 2 + W / 2), int(
                pd.y + Wd / 2 - W / 2)
            if sx <= ex and sy <= ey:
                for x in [sx, (sx + ex) // 2, ex]:
                    for y in [sy, (sy + ey) // 2, ey]: candidates.append({'x': x, 'y': y, 'z': pd.z + Hd, 's': pd})
        for pos in candidates:
            t = PlacedDevice(dev, pos['x'], pos['y'], pos['z'], rot, pos['s'])
            if abs(pos['x']) > xl or abs(pos['y']) > yl: continue
            if any(check_collision_aabb(t, ex) for ex in layout.placed_devices): continue
            layout.add(t)
            placed = True
            break
        if not placed: layout.is_valid = False; return layout
    return layout


def generate_random_chromosome():
    ids = [d.id for d in DEVICE_LIST if d.id != 12]
    random.shuffle(ids)
    return [{'id': 12, 'rot': random.choice([0, 90])}] + [{'id': i, 'rot': random.choice([0, 90])} for i in ids]


# ==========================================
# 4. 评价指标计算 (IGD & HV)
# ==========================================
def calculate_igd_normalized(curr_pf, true_pf):
    """
    计算归一化倒世代距离 (IGD)
    """
    if true_pf is None or len(curr_pf) == 0: return 0.0
    # 归一化处理
    mn, mx = true_pf.min(0), true_pf.max(0)
    den = mx - mn
    den[den == 0] = 1e-6
    n_true, n_curr = (true_pf - mn) / den, (curr_pf - mn) / den
    # 计算每个真实点到当前前沿的最短距离
    dists = [np.min(np.linalg.norm(n_curr - p, axis=1)) for p in n_true]
    return np.mean(dists)


def calculate_hv_monte_carlo(n_fits, ref=[1.1, 1.1, 1.1], samples=50000):
    if len(n_fits) == 0: return 0.0
    s = np.random.uniform(0, ref[0], (samples, 3))
    count = sum(1 for smp in s if any(all(f <= smp) for f in n_fits))
    return (count / samples) * np.prod(ref)


# ==========================================
# 5. MOEA/D 核心逻辑
# ==========================================
def init_weight_vectors(n_obj, n_div):
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


def tchebycheff_aggregation(fits, weight, ideal_point):
    return np.max(weight * np.abs(np.array(fits) - ideal_point))


def update_ep(ep, candidate):
    if candidate['fit'][0] == float('inf'): return ep
    new_ep = []
    is_dominated = False
    for member in ep:
        if all(candidate['fit'] <= member['fit']) and any(candidate['fit'] < member['fit']): continue
        if all(member['fit'] <= candidate['fit']) and any(member['fit'] < candidate['fit']):
            is_dominated = True
            new_ep.append(member)
        else:
            new_ep.append(member)
    if not is_dominated: new_ep.append(candidate)
    return new_ep


def run_moead(max_gen=200, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    weights_vectors = init_weight_vectors(3, 8)
    n_subproblems = len(weights_vectors)
    neighbors = np.argsort(cdist(weights_vectors, weights_vectors), axis=1)[:, :NEIGHBOR_SIZE]

    population, ep = [], []
    ideal_point = np.array([float('inf')] * 3)

    print(f"🚀 初始化 MOEA/D 种群 ({n_subproblems} 子问题)...")
    for i in range(n_subproblems):
        valid = False
        while not valid:
            chromo = generate_random_chromosome()
            layout = heuristic_decoder(chromo)
            if layout.is_valid:
                fit = np.array(calculate_fitness(layout))
                if fit[0] != float('inf'):
                    ind = {'gene': chromo, 'fit': fit, 'layout': layout}
                    population.append(ind)
                    ideal_point = np.minimum(ideal_point, ind['fit'])
                    ep = update_ep(ep, ind)
                    valid = True

    for gen in range(max_gen):
        for i in range(n_subproblems):
            p1_idx, p2_idx = random.sample(list(neighbors[i]), 2)
            p1, p2 = population[p1_idx], population[p2_idx]
            child_gene = copy.deepcopy(p1['gene'])
            if random.random() < CROSSOVER_RATE:
                idx1, idx2 = random.sample(range(1, len(child_gene)), 2)
                child_gene[idx1], child_gene[idx2] = child_gene[idx2], child_gene[idx1]
            if random.random() < MUTATION_RATE:
                mut_i = random.randint(0, len(child_gene) - 1)
                child_gene[mut_i]['rot'] = 90 if child_gene[mut_i]['rot'] == 0 else 0

            l_child = heuristic_decoder(child_gene)
            if not l_child.is_valid: continue
            f_child = np.array(calculate_fitness(l_child))
            if f_child[0] == float('inf'): continue

            ideal_point = np.minimum(ideal_point, f_child)
            for j in neighbors[i]:
                if tchebycheff_aggregation(f_child, weights_vectors[j], ideal_point) <= tchebycheff_aggregation(
                        population[j]['fit'], weights_vectors[j], ideal_point):
                    population[j] = {'gene': child_gene, 'fit': f_child, 'layout': l_child}
            ep = update_ep(ep, {'gene': child_gene, 'fit': f_child, 'layout': l_child})
        if (gen + 1) % 50 == 0: print(f"迭代 {gen + 1}/{max_gen} | EP 数量: {len(ep)}")

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    final_front = ep
    final_front.sort(key=lambda x: x['fit'][0])
    all_fits = np.array([p['fit'] for p in final_front])
    f_min, f_max = all_fits.min(0), all_fits.max(0)
    den = f_max - f_min
    den[den == 0] = 1e-6
    norm_fits = (all_fits - f_min) / den

    # 指标计算
    hv_val = calculate_hv_monte_carlo(norm_fits)
    igd_val = calculate_igd_normalized(all_fits, true_pf)
    weighted_sums = np.sum(norm_fits * WEIGHTS, axis=1)
    best_idx = np.argmin(weighted_sums)

    print("\n" + "=" * 110 + "\n📊 MOEA/D 综合性能评估报告")
    print(
        f"⏱️ 耗时: {end_time - start_time:.2f}s | 💾 内存峰值: {peak_mem / 1e6:.2f}MB | ⚙️ 评估次数: {GLOBAL_FITNESS_EVALS}")
    print(f"📐 HV (超体积): {hv_val:.4f} | 🎯 IGD (倒世代距离): {igd_val:.4f}")
    print("-" * 110)
    print(
        f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'重心X':<7} | {'重心Y':<7} | {'重心Z':<7} | {'12号承重'}")

    for i, ind in enumerate(final_front):
        pd_l = ind['layout'].placed_devices
        tw = sum(p.device.weight for p in pd_l)
        cx, cy, cz = [sum(p.device.weight * p.get_center_3d()[j] for p in pd_l) / tw for j in range(3)]
        l12 = sum(pd.device.weight for pd in pd_l if _is_supported_by(pd, next(p for p in pd_l if p.device.id == 12)))
        print(
            f"{i + 1:<6} | {ind['fit'][0]:<10.3f} | {ind['fit'][1]:<10.1f} | {ind['fit'][2]:<10.1f} | {cx:<8.1f} | {cy:<8.1f} | {cz:<8.1f} | {l12:<8.1f}")

    print("-" * 110)
    print(f"⭐ 推荐均衡解 (惯量0.5:热力0.3:走线0.2): [方案 {best_idx + 1}] | 加权得分: {weighted_sums[best_idx]:.4f}")

    # 3D 可视化
    fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
                                       marker=dict(size=5, color=weighted_sums, colorscale='Viridis'))])
    fig.update_layout(title="MOEA/D Pareto Front 3D", scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
    fig.write_html("Pareto_Front_MOEAD_IGD.html")
    fig.show()


if __name__ == "__main__":
    PF_FILE = "True_PF_raw.npy"
    true_pf_raw = np.load(PF_FILE) if os.path.exists(PF_FILE) else None
    run_moead(max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)