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

def update_archive(archive, ind):
    if ind['fit'][0] == float('inf'): return archive
    new_archive = [ind]
    for a in archive:
        if dominates(ind['fit'], a['fit']): continue
        if dominates(a['fit'], ind['fit']): return archive
        new_archive.append(a)
    return new_archive[:ARCHIVE_SIZE]


def run_ieba(max_gen=200, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    # 1. 初始化
    pop = [{'gene': generate_random_chromosome(), 'v': 0, 'A': A_0, 'r': R_0} for _ in range(POP_SIZE)]
    for p in pop: p['fit'] = np.array(calculate_fitness(heuristic_decoder(p['gene'])))

    archive = []
    for p in pop: archive = update_archive(archive, p)

    # print(f"⚡ 开始 IEBA 3目标布局优化 (共 {max_gen} 代)...")
    for gen in range(max_gen):
        # 获取平衡池 (Equilibrium Pool)
        if not archive:
            pool = pop
        else:
            pool = random.sample(archive, min(4, len(archive)))
            while len(pool) < 4: pool.append(random.choice(archive))
            avg_ind = copy.deepcopy(pool[0])
            # 简化：均值解通过随机重组模拟
            pool.append(avg_ind)

        for i in range(POP_SIZE):
            freq = F_MIN + (F_MAX - F_MIN) * random.random()
            target = random.choice(pool)['gene']

            # 位置更新 (蝙蝠飞向平衡状态)
            new_gene = copy.deepcopy(pop[i]['gene'])
            for j in range(1, len(new_gene)):
                if random.random() < freq * 0.2:
                    # 交换算子
                    tid = target[j]['id']
                    for k in range(1, len(new_gene)):
                        if new_gene[k]['id'] == tid:
                            new_gene[j], new_gene[k] = new_gene[k], new_gene[j]
                            break
                    new_gene[j]['rot'] = target[j]['rot']

            # 局部搜索
            if random.random() > pop[i]['r']:
                idx = random.randint(1, len(new_gene) - 1)
                new_gene[idx]['rot'] = 90 if new_gene[idx]['rot'] == 0 else 0

            l_new = heuristic_decoder(new_gene)
            if l_new.is_valid:
                f_new = np.array(calculate_fitness(l_new))
                if dominates(f_new, pop[i]['fit']) or random.random() < pop[i]['A']:
                    pop[i]['gene'], pop[i]['fit'] = new_gene, f_new
                    pop[i]['A'] *= ALPHA
                    pop[i]['r'] = R_0 * (1 - math.exp(-GAMMA * gen))
                archive = update_archive(archive, {'gene': new_gene, 'fit': f_new})
        #
        # if (gen + 1) % 50 == 0:
        #     print(f"迭代 {gen + 1}/{max_gen} | 存档非支配解数量: {len(archive)}")

    # 停止监控并获取内存
    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 整理结果
    archive.sort(key=lambda x: x['fit'][0])
    all_fits = np.array([p['fit'] for p in archive])
    f_min, f_max = all_fits.min(0), all_fits.max(0)
    den = f_max - f_min
    den[den == 0] = 1e-6
    norm_fits = (all_fits - f_min) / den

    hv = calculate_hv(norm_fits)
    igd = calculate_igd(all_fits, true_pf)
    weighted_sums = np.sum(norm_fits * WEIGHTS, axis=1)
    best_idx = np.argmin(weighted_sums)

    # ==========================================
    # 6. 报告输出 (包含内存占用)
    # ==========================================
    print("\n" + "=" * 110)
    print("📊 IEBA (Improved Equilibrium Bat Algorithm) 综合性能评估报告")
    print("-" * 110)
    print(f"⏱️ 计算耗时 (Time)         : {end_time - start_time:.2f} 秒")
    print(f"💾 内存占用峰值 (Peak Mem) : {peak_mem / 10 ** 6:.2f} MB")
    print(f"⚙️ 适应度评估次数 (Evals)  : {GLOBAL_FITNESS_EVALS} 次")
    print(f"📐 超体积 HV (Hypervolume) : {hv:.4f}")
    print(f"🎯 倒世代距离 IGD (IGD)    : {igd:.4f}")
    # print("-" * 110)
    # print(f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'综合得分'}")

    # for i, ind in enumerate(archive):
    #     print(
    #         f"{i + 1:<6} | {ind['fit'][0]:<10.3f} | {ind['fit'][1]:<10.1f} | {ind['fit'][2]:<10.1f} | {weighted_sums[i]:.4f}")

    print("-" * 110)
    print(f"⭐ 推荐均衡解: [方案 {best_idx + 1}] | 加权得分: {weighted_sums[best_idx]:.4f}")

    # # 3D 可视化
    # fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
    #                                    marker=dict(size=5, color=weighted_sums, colorscale='Viridis', showscale=True))])
    # fig.add_trace(go.Scatter3d(x=[all_fits[best_idx, 0]], y=[all_fits[best_idx, 1]], z=[all_fits[best_idx, 2]],
    #                            mode='markers+text', marker=dict(size=12, color='red', symbol='diamond'),
    #                            text=["★ BEST"]))
    # fig.update_layout(title="IEBA Pareto Front 3D Visualization",
    #                   scene=dict(xaxis_title='F1: Inertia', yaxis_title='F2: Thermal', zaxis_title='F3: Routing'))
    # fig.show()

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

    # print(f"🚀 初始化 MOEA/D 种群 ({n_subproblems} 子问题)...")
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
        # if (gen + 1) % 50 == 0: print(f"迭代 {gen + 1}/{max_gen} | EP 数量: {len(ep)}")

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
    hv_val = calculate_hv(norm_fits)
    igd_val = calculate_igd(all_fits, true_pf)
    weighted_sums = np.sum(norm_fits * WEIGHTS, axis=1)
    best_idx = np.argmin(weighted_sums)

    print("\n" + "=" * 110 + "\n📊 MOEA/D 综合性能评估报告")
    print(
        f"⏱️ 耗时: {end_time - start_time:.2f}s | 💾 内存峰值: {peak_mem / 1e6:.2f}MB | ⚙️ 评估次数: {GLOBAL_FITNESS_EVALS}")
    print(f"📐 HV (超体积): {hv_val:.4f} | 🎯 IGD (倒世代距离): {igd_val:.4f}")
    # print("-" * 110)
    # print(
    #     f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'重心X':<7} | {'重心Y':<7} | {'重心Z':<7} | {'12号承重'}")
    #
    # for i, ind in enumerate(final_front):
    #     pd_l = ind['layout'].placed_devices
    #     tw = sum(p.device.weight for p in pd_l)
    #     cx, cy, cz = [sum(p.device.weight * p.get_center_3d()[j] for p in pd_l) / tw for j in range(3)]
    #     l12 = sum(pd.device.weight for pd in pd_l if _is_supported_by(pd, next(p for p in pd_l if p.device.id == 12)))
    #     print(
    #         f"{i + 1:<6} | {ind['fit'][0]:<10.3f} | {ind['fit'][1]:<10.1f} | {ind['fit'][2]:<10.1f} | {cx:<8.1f} | {cy:<8.1f} | {cz:<8.1f} | {l12:<8.1f}")

    print("-" * 110)
    print(f"⭐ 推荐均衡解 (惯量0.5:热力0.3:走线0.2): [方案 {best_idx + 1}] | 加权得分: {weighted_sums[best_idx]:.4f}")

    # # 3D 可视化
    # fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
    #                                    marker=dict(size=5, color=weighted_sums, colorscale='Viridis'))])
    # fig.update_layout(title="MOEA/D Pareto Front 3D", scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
    # fig.write_html("Pareto_Front_MOEAD_IGD.html")
    # fig.show()

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

def run_nsga3(pop_size=50, max_gen=200, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    ref_points = uniform_reference_points(3, 8)  # 45个参考点

    population = []
    while len(population) < pop_size:
        c = generate_random_chromosome()
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
                f = calculate_fitness(l)
                offspring.append({'gene': c_gene, 'fit': f, 'layout': l})

        population = nsga3_environmental_selection(population + offspring, pop_size, ref_points)
        if (gen + 1) % 10 == 0 or gen == 0:
            fits = np.array([p['fit'] for p in population])
            pf_count = len(fast_non_dominated_sort(fits)[0])
            # print(f"迭代 {gen + 1}/{max_gen} | 当前帕累托前沿解数量: {pf_count}")

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 提取帕累托前沿
    fits_all = np.array([p['fit'] for p in population])
    p_idx = fast_non_dominated_sort(fits_all)[0]
    final_front = [population[i] for i in p_idx]
    final_front.sort(key=lambda x: x['fit'][0])
    all_fits = np.array([p['fit'] for p in final_front])

    # 归一化用于指标计算
    f_min, f_max = all_fits.min(0), all_fits.max(0)
    den = f_max - f_min
    den[den == 0] = 1e-6
    norm_fits = (all_fits - f_min) / den

    # 计算 HV 和 IGD
    hv_val = calculate_hv(norm_fits)
    igd_val = calculate_igd(all_fits, true_pf) if true_pf is not None else 0.0

    # 计算加权综合得分
    weighted_sums = np.sum(norm_fits * WEIGHTS, axis=1)
    best_balanced_idx = np.argmin(weighted_sums)

    print("\n" + "=" * 110 + "\n📊 NSGA-III 综合性能评估报告")
    print(
        f"⏱️ 耗时: {end_time - start_time:.2f}s | 💾 内存峰值: {peak_mem / 1e6:.2f}MB | ⚙️ 评估次数: {GLOBAL_FITNESS_EVALS}")
    print(f"📐 HV (超体积): {hv_val:.4f} (参考点[1.1,1.1,1.1]) | 🎯 IGD (倒世代距离): {igd_val:.4f}")
    # print("-" * 110)
    # print(
    #     f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'重心X':<7} | {'重心Y':<7} | {'重心Z':<7} | {'12号承重'}")

    # detailed = []
    # for i, ind in enumerate(final_front):
    #     pd_l = ind['layout'].placed_devices;
    #     tw = sum(p.device.weight for p in pd_l)
    #     cx = sum(p.device.weight * p.get_center_3d()[0] for p in pd_l) / tw
    #     cy = sum(p.device.weight * p.get_center_3d()[1] for p in pd_l) / tw
    #     cz = sum(p.device.weight * p.get_center_3d()[2] for p in pd_l) / tw
    #     l12 = sum(pd.device.weight for pd in pd_l if _is_supported_by(pd, next(p for p in pd_l if p.device.id == 12)))
        # print(
        #     f"{i + 1:<6} | {ind['fit'][0]:<10.3f} | {ind['fit'][1]:<10.1f} | {ind['fit'][2]:<10.1f} | {cx:<8.1f} | {cy:<8.1f} | {cz:<8.1f} | {l12:<8.1f}")
        # detailed.append({'id': i + 1, 'fit': ind['fit'], 'cg': [cx, cy, cz], 'load12': l12, 'w_sum': weighted_sums[i]})

    print("-" * 110)
    print(f"⭐ 基于自定义权重 (惯量{WEIGHTS[0]} : 热力{WEIGHTS[1]} : 走线{WEIGHTS[2]}) 推荐的最佳加权和均衡解为:")
    print(
        f"   [方案 {best_balanced_idx + 1}] | 归一化空间加权得分 (Weighted Sum): {weighted_sums[best_balanced_idx]:.4f}")

    # # 可视化 (Plotly)
    # hts = [
    #     f"<b>方案 {d['id']}</b><br>F1: {d['fit'][0]:.3f}<br>F2: {d['fit'][1]:.1f}<br>F3: {d['fit'][2]:.1f}<br>加权得分: {d['w_sum']:.4f}"
    #     for d in detailed]
    # fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
    #                                    marker=dict(size=6, color=weighted_sums, colorscale='Viridis',
    #                                                colorbar=dict(title="Weighted Sum")), text=hts, hoverinfo='text')])
    # fig.add_trace(go.Scatter3d(x=[all_fits[best_balanced_idx, 0]], y=[all_fits[best_balanced_idx, 1]],
    #                            z=[all_fits[best_balanced_idx, 2]], mode='markers+text',
    #                            marker=dict(size=12, color='red', symbol='diamond'), text=["★ Best Solution"],
    #                            textposition="top center"))
    # fig.update_layout(title="NSGA-III Pareto Front 3D (HV & Weighted Sum Included)",
    #                   scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
    # fig.write_html("Pareto_Front_NSGA3_Final.html");
    # fig.show()

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

def run_moga(pop_size=50, max_gen=50, true_pf=None, sigma_share=0.5):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0

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

    # print(f"\n⚡ 开始 HD-MOGA 3目标进化迭代 (共 {max_gen} 代)...")
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
        # print(f"迭代 {gen + 1}/{max_gen} | 3目标帕累托前沿解数量: {rank0_count}")

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
    hv_value = calculate_hv(norm_fits)

    # 2. 计算 IGD
    if true_pf is not None:
        igd_value = calculate_igd(all_fits, true_pf)
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
    # print(f"🏆 HD-MOGA 优化完成！输出 {len(final_front)} 个帕累托最优解。")
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
            pd.device.weight for pd in pd_list if _is_supported_by(pd, next(p for p in pd_list if p.device.id == 12)))

        fit = ind['fit']
        info_dict = {'id': idx + 1, 'fit': fit, 'cg': [cg_x, cg_y, cg_z], 'load12': dev12_load}
        detailed_info.append(info_dict)

        # print(
        #     f"{idx + 1:<6} | {fit[0]:<10.3f} | {fit[1]:<10.1f} | {fit[2]:<10.1f} | {cg_x:<8.1f} | {cg_y:<8.1f} | {cg_z:<8.1f} | {dev12_load:<8.1f}")
        # save_detailed_csv(ind['layout'], f"moga_solution_{idx + 1}.csv", fit, [cg_x, cg_y, cg_z], dev12_load)

    # ================= 计算最佳加权和 (绝对均衡解) =================
    # 在归一化空间下计算各方案加权得分
    weighted_scores = np.sum(norm_fits * WEIGHTS, axis=1)
    best_balanced_idx = np.argmin(weighted_scores)
    best_balanced_score = weighted_scores[best_balanced_idx]

    print("-" * 110)
    print(f"⭐ 基于自定义权重 (惯量0.5 : 热力0.3 : 走线0.2) 推荐的加权绝对最优解为: [方案 {best_balanced_idx + 1}]")
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
    #     x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2],
    #     mode='markers',
    #     marker=dict(size=6, color=all_fits[:, 2], colorscale='Viridis', opacity=0.8,
    #                 line=dict(width=0.5, color='white')),
    #     text=hover_texts,
    #     hoverinfo='text',
    #     name='MOGA Pareto Solutions'
    # ))
    #
    # fig.add_trace(go.Scatter3d(
    #     x=[all_fits[best_balanced_idx, 0]],
    #     y=[all_fits[best_balanced_idx, 1]],
    #     z=[all_fits[best_balanced_idx, 2]],
    #     mode='markers+text',
    #     marker=dict(size=14, color='red', symbol='diamond', line=dict(color='yellow', width=2)),
    #     text=["★ Best Solution"],
    #     textposition="top center",
    #     textfont=dict(color='red', size=14, family='Arial Black'),
    #     hovertemplate="<b>★【加权最佳均衡解】★</b><br>" + hover_texts[best_balanced_idx] + "<extra></extra>",
    #     name='Best Weighted Solution'
    # ))
    #
    # fig.update_layout(
    #     title=dict(
    #         text="Interactive 3D Pareto Front (HD-MOGA) & Best Compromise Solution",
    #         x=0.5, y=0.95,
    #         xanchor='center', yanchor='top',
    #         font=dict(size=20, family='Arial')
    #     ),
    #     scene=dict(
    #         xaxis=dict(title='F1: Moment of Inertia', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
    #         yaxis=dict(title='F2: Thermal Loss', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
    #         zaxis=dict(title='F3: Routing Loss', backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
    #         camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    #     ),
    #     margin=dict(l=0, r=0, b=0, t=50),
    #     legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)')
    # )
    #
    # html_filename = "Pareto_Front_3D_HD_MOGA.html"
    # fig.write_html(html_filename)
    # print(f"\n🌐 3D 帕累托图已生成并保存为: {html_filename}，正在默认浏览器中打开...")
    # fig.show()

# ==========================================
# 6. SPEA2 核心算法模块
# ==========================================

def spea2_fitness_assignment(combined_pop):
    """
    SPEA2 适应度分配：Strength -> Raw Fitness -> Density -> Total Fitness
    """
    size = len(combined_pop)
    fits = np.array([p['fit'] for p in combined_pop])

    # 1. 计算 Strength S(i): 支配多少个个体
    strength = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if i != j and dominates(fits[i], fits[j]):
                strength[i] += 1

    # 2. 计算 Raw Fitness R(i): 被哪些个体支配的 Strength 之和
    raw_fitness = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if i != j and dominates(fits[j], fits[i]):
                raw_fitness[i] += strength[j]

    # 3. 密度估计 (Density Estimation): 第 k 近邻距离
    # k = sqrt(N + N_archive)
    k = int(math.sqrt(size))
    dist_matrix = np.zeros((size, size))
    # 目标空间归一化距离
    f_min, f_max = fits.min(axis=0), fits.max(axis=0)
    range_f = f_max - f_min
    range_f[range_f == 0] = 1e-6
    norm_fits = (fits - f_min) / range_f

    for i in range(size):
        dists = np.linalg.norm(norm_fits - norm_fits[i], axis=1)
        dist_matrix[i] = np.sort(dists)

    # SPEA2 Density: 1 / (d_k + 2)
    density = 1.0 / (dist_matrix[:, k] + 2.0)

    # 4. 总适应度: R(i) + D(i)
    for i in range(size):
        combined_pop[i]['spea2_fit'] = raw_fitness[i] + density[i]


def environmental_selection(combined_pop, archive_size):
    """
    环境选择：保留适应度 < 1 的非支配解，并进行存档截断/填充
    """
    # 筛选非支配解 (Raw Fitness < 1 的在 SPEA2 中即为 Pareto 解)
    next_archive = [p for p in combined_pop if p['spea2_fit'] < 1.0]

    if len(next_archive) < archive_size:
        # 如果不够，按适应度排序补充
        remaining = [p for p in combined_pop if p['spea2_fit'] >= 1.0]
        remaining.sort(key=lambda x: x['spea2_fit'])
        next_archive.extend(remaining[:(archive_size - len(next_archive))])
    elif len(next_archive) > archive_size:
        # 如果太多，进行截断 (Truncation Operator)
        while len(next_archive) > archive_size:
            # 重新计算归一化距离寻找最拥挤解
            fits = np.array([p['fit'] for p in next_archive])
            f_min, f_max = fits.min(axis=0), fits.max(axis=0)
            range_f = f_max - f_min;
            range_f[range_f == 0] = 1e-6
            n_fits = (fits - f_min) / range_f

            # 寻找具有最小距离的个体
            min_dist_idx = -1
            min_dists = []
            for i in range(len(next_archive)):
                dists = np.sort(np.linalg.norm(n_fits - n_fits[i], axis=1))
                min_dists.append(dists)

            # SPEA2 截断逻辑：比较第1, 2...k个最近邻距离
            # 此处简化为直接寻找最近邻距离最小的
            best_to_remove = 0
            for i in range(1, len(next_archive)):
                # 逐级比较距离
                for k in range(1, len(next_archive)):
                    if min_dists[i][k] < min_dists[best_to_remove][k]:
                        best_to_remove = i
                        break
                    elif min_dists[i][k] > min_dists[best_to_remove][k]:
                        break
            next_archive.pop(best_to_remove)

    return next_archive


def run_spea2(pop_size=50, max_gen=200, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start()
    start_time = time.time()

    population = []
    # print("🚀 初始化种群中...")
    while len(population) < pop_size:
        c = generate_random_chromosome()
        l = heuristic_decoder(c)
        if l.is_valid:
            f = calculate_fitness(l)
            if f[0] != float('inf'): population.append({'gene': c, 'fit': f, 'layout': l})

    archive = []

    # print(f"⚡ 开始 SPEA2 3目标优化 (共 {max_gen} 代)...")
    for gen in range(max_gen):
        # 1. 合并种群与存档
        combined = population + archive

        # 2. 适应度分配
        spea2_fitness_assignment(combined)

        # 3. 环境选择 (更新存档)
        archive = environmental_selection(combined, ARCHIVE_SIZE)

        # 4. 生成子代 (从存档中选择)
        offspring = []
        while len(offspring) < pop_size:
            # 锦标赛选择 (2元)
            p1 = random.choice(archive)
            p2 = random.choice(archive)
            parent = p1 if p1['spea2_fit'] < p2['spea2_fit'] else p2

            child_gene = copy.deepcopy(parent['gene'])
            if random.random() < CROSSOVER_RATE:
                i1, i2 = random.sample(range(1, len(child_gene)), 2)
                child_gene[i1], child_gene[i2] = child_gene[i2], child_gene[i1]
            if random.random() < MUTATION_RATE:
                mi = random.randint(0, len(child_gene) - 1)
                child_gene[mi]['rot'] = 90 if child_gene[mi]['rot'] == 0 else 0

            l = heuristic_decoder(child_gene)
            if l.is_valid:
                f = calculate_fitness(l)
                if f[0] != float('inf'): offspring.append({'gene': child_gene, 'fit': f, 'layout': l})

        population = offspring
        pf_count = len([p for p in archive if p['spea2_fit'] < 1.0])
        # if (gen + 1) % 10 == 0 or gen == 0:
        #     print(f"迭代 {gen + 1}/{max_gen} | 存档非支配解数量: {pf_count}")

    # 结果统计
    end_time = time.time()
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    final_front = [p for p in archive if p['spea2_fit'] < 1.0]
    if not final_front: final_front = archive
    final_front.sort(key=lambda x: x['fit'][0])

    all_fits = np.array([p['fit'] for p in final_front])
    f_min, f_max = all_fits.min(0), all_fits.max(0)
    den = f_max - f_min
    den[den == 0] = 1e-6
    norm_fits = (all_fits - f_min) / den
    hv = calculate_hv(norm_fits)
    igd = calculate_igd(all_fits, true_pf) if true_pf is not None else float('inf')

    print("\n" + "=" * 110)
    print("📊 SPEA2 综合性能评估报告")
    print(
        f"⏱️ 耗时: {end_time - start_time:.2f}s | 💾 内存峰值: {peak_mem / 1e6:.2f}MB | ⚙️ 评估次数: {GLOBAL_FITNESS_EVALS}")
    print(f"📐 HV: {hv:.4f} | 🎯 IGD: {igd:.4f}")
    # print("-" * 110)
    # print(
    #     f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'重心X':<7} | {'重心Y':<7} | {'重心Z':<7} | {'12号承重'}")

    # detailed = []
    # for i, ind in enumerate(final_front):
    #     pd_l = ind['layout'].placed_devices;
    #     tw = sum(p.device.weight for p in pd_l)
    #     cx = sum(p.device.weight * p.get_center_3d()[0] for p in pd_l) / tw
    #     cy = sum(p.device.weight * p.get_center_3d()[1] for p in pd_l) / tw
    #     cz = sum(p.device.weight * p.get_center_3d()[2] for p in pd_l) / tw
    #     l12 = sum(pd.device.weight for pd in pd_l if _is_supported_by(pd, next(p for p in pd_l if p.device.id == 12)))
        # print(
        #     f"{i + 1:<6} | {ind['fit'][0]:<10.3f} | {ind['fit'][1]:<10.1f} | {ind['fit'][2]:<10.1f} | {cx:<8.1f} | {cy:<8.1f} | {cz:<8.1f} | {l12:<8.1f}")
        # detailed.append({'id': i + 1, 'fit': ind['fit'], 'cg': [cx, cy, cz], 'load12': l12})
        # save_detailed_csv(ind['layout'], f"spea2_solution_{i + 1}.csv", ind['fit'], [cx, cy, cz], l12)

    w_scores = np.sum(norm_fits * WEIGHTS, axis=1)
    b_idx = np.argmin(w_scores)
    print("-" * 110)
    print(f"⭐ 推荐均衡解 (惯量0.5:热力0.3:走线0.2): [方案 {b_idx + 1}] Score: {w_scores[b_idx]:.4f}")

    # # 可视化 (Plotly)
    # hts = []
    # for i, info in enumerate(detailed):
    #     hts.append(
    #         f"<b>方案 {info['id']}</b><br>F1: {info['fit'][0]:.3f}<br>F2: {info['fit'][1]:.1f}<br>F3: {info['fit'][2]:.1f}<br>CG: ({info['cg'][0]:.1f},{info['cg'][1]:.1f},{info['cg'][2]:.1f})")
    # fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
    #                                    marker=dict(size=6, color=all_fits[:, 2], colorscale='Viridis'), text=hts,
    #                                    hoverinfo='text')])
    # fig.add_trace(
    #     go.Scatter3d(x=[all_fits[b_idx, 0]], y=[all_fits[b_idx, 1]], z=[all_fits[b_idx, 2]], mode='markers+text',
    #                  marker=dict(size=12, color='red', symbol='diamond'), text=["★ Best"], textposition="top center"))
    # fig.update_layout(title="SPEA2 Pareto Front 3D", scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
    # fig.write_html("Pareto_Front_SPEA2.html");
    # fig.show()


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

# ==========================================
# 5. DBO 离散行为算子 (滚球/跳舞/觅食/偷窃)
# ==========================================
def ox_crossover(p1, p2):
    """顺序交叉算子 (用于向目标逼近)"""
    n = len(p1)
    if n <= 2: return copy.deepcopy(p1)
    c = copy.deepcopy(p1)
    a, b = random.sample(range(1, n), 2)
    if a > b: a, b = b, a
    segment = p2[a:b+1]
    seg_ids = {g['id'] for g in segment}
    p1_idx = 1
    for i in range(1, n):
        if a <= i <= b: c[i] = copy.deepcopy(p2[i])
        else:
            while p1[p1_idx]['id'] in seg_ids: p1_idx += 1
            c[i] = copy.deepcopy(p1[p1_idx]); p1_idx += 1
    return c

def mutate_light(gene):
    """轻度变异 (用于局部觅食)"""
    c = copy.deepcopy(gene)
    if len(c) > 1:
        idx = random.randint(1, len(c)-1)
        c[idx]['rot'] = 90 if c[idx]['rot'] == 0 else 0
        if random.random() < 0.5:
            i, j = random.sample(range(1, len(c)), 2)
            c[i], c[j] = c[j], c[i]
    return c

def mutate_heavy(gene):
    """重度变异 (用于跳舞逃逸局部最优)"""
    c = copy.deepcopy(gene)
    for _ in range(3):
        if len(c) > 2:
            i, j = random.sample(range(1, len(c)), 2)
            c[i], c[j] = c[j], c[i]
            c[i]['rot'] = random.choice([0, 90])
    return c


def calculate_crowding_distance(fits, front):
    l = len(front)
    distances = [0.0] * l
    if l <= 2: return [float('inf')] * l

    for m in range(len(fits[0])):
        front_sorted = sorted(range(l), key=lambda i: fits[front[i]][m])
        distances[front_sorted[0]] = float('inf')
        distances[front_sorted[-1]] = float('inf')
        f_min = fits[front[front_sorted[0]]][m]
        f_max = fits[front[front_sorted[-1]]][m]
        if f_max == f_min: continue
        for i in range(1, l - 1):
            distances[front_sorted[i]] += (fits[front[front_sorted[i + 1]]][m] - fits[front[front_sorted[i - 1]]][
                m]) / (f_max - f_min)
    return distances

def run_ns_dbo(pop_size=50, max_gen=200, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start();
    start_time = time.time()

    population = []
    while len(population) < pop_size:
        c = generate_random_chromosome();
        l = heuristic_decoder(c)
        if l.is_valid:
            f = calculate_fitness(l)
            if f[0] != float('inf'): population.append({'gene': c, 'fit': f, 'layout': l})

    # print(f"⚡ 开始 HD-NS-DBO 3目标优化 (共 {max_gen} 代)...")
    for gen in range(max_gen):
        fits = np.array([p['fit'] for p in population])
        fronts = fast_non_dominated_sort(fits)

        # 提取 Elite (排名为 0 的个体)
        elite_indices = fronts[0]
        if not elite_indices: elite_indices = list(range(len(population)))

        # 寻找全局绝对最优 (通过拥挤距离)
        cd_elite = calculate_crowding_distance(fits, elite_indices)
        best_elite_idx = elite_indices[np.argmax(cd_elite)]

        offspring = []
        for i in range(pop_size):
            r = random.random()
            # 蜣螂的四种离散化行为策略映射
            if r < 0.2:
                # 滚球蜣螂 (向无支配精英靠拢 + 扰动)
                target = random.choice(elite_indices)
                c_gene = ox_crossover(population[i]['gene'], population[target]['gene'])
                c_gene = mutate_light(c_gene)
            elif r < 0.4:
                # 跳舞蜣螂 (大规模随机游走，跳出极值)
                c_gene = mutate_heavy(population[i]['gene'])
            elif r < 0.8:
                # 觅食蜣螂 (在精英解附近小范围局部搜索)
                target = random.choice(elite_indices)
                c_gene = mutate_light(population[target]['gene'])
            else:
                # 偷窃蜣螂 (严格向全局最佳精英靠拢)
                c_gene = ox_crossover(population[i]['gene'], population[best_elite_idx]['gene'])

            l = heuristic_decoder(c_gene)
            if l.is_valid:
                f = calculate_fitness(l)
                if f[0] != float('inf'): offspring.append({'gene': c_gene, 'fit': f, 'layout': l})

        # NSGA-II 样式的环境选择 (合并父代子代 -> 截断)
        combined_pop = population + offspring
        c_fits = np.array([p['fit'] for p in combined_pop])
        c_fronts = fast_non_dominated_sort(c_fits)

        next_pop = []
        for front in c_fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend([combined_pop[idx] for idx in front])
            else:
                cd = calculate_crowding_distance(c_fits, front)
                front_sorted = [front[idx] for idx in np.argsort(cd)[::-1]]
                remain = pop_size - len(next_pop)
                next_pop.extend([combined_pop[idx] for idx in front_sorted[:remain]])
                break
        population = next_pop

        # if (gen + 1) % 10 == 0 or gen == 0:
        #     fits_current = np.array([p['fit'] for p in population])
        #     pf_count = len(fast_non_dominated_sort(fits_current)[0])
        #     print(f"迭代 {gen + 1}/{max_gen} | 当前帕累托前沿解数量: {pf_count}")

    end_time = time.time();
    _, peak_mem = tracemalloc.get_traced_memory();
    tracemalloc.stop()

    # 提取最终帕累托前沿
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

    hv_val = calculate_hv(norm_fits)
    igd_val = calculate_igd(all_fits, true_pf) if true_pf is not None else 0.0

    weighted_sums = np.sum(norm_fits * WEIGHTS, axis=1)
    best_balanced_idx = np.argmin(weighted_sums)

    print("\n" + "=" * 110 + "\n📊 HD-NS-DBO 综合性能评估报告")
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
    fig.update_layout(title="HD-NS-DBO Pareto Front 3D (HV & Weighted Sum Included)",
                      scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
    fig.write_html("Pareto_Front_NS_DBO_Final.html");
    fig.show()
if __name__ == "__main__":
    PF_FILE = "True_PF_raw.npy"
    true_pf_raw = np.load(PF_FILE) if os.path.exists(PF_FILE) else None
    for i in range(1):
        # run_ieba(max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)
        # run_moead(max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)
        # run_nsga3(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)
        # run_moga(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw, sigma_share=0.5)
        # run_spea2(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)
        # run_nsga2(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)
        run_ns_dbo(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)