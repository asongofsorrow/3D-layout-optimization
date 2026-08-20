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
POP_SIZE = 100  # 种群数量
ARCHIVE_SIZE = 100  # SPEA2 外部存档数量 (通常与种群一致)
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
WEIGHTS = np.array([0.5, 0.3, 0.2])  # 权重：惯量、热力、走线

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
edges = [(1, 8, 1), (1, 9, 5), (1, 10, 5), (1, 11, 1), (1, 12, 5), (1, 15, 1), (1, 16, 5), (2, 11, 1), (2, 13, 1),
         (2, 16, 5), (3, 8, 1), (3, 11, 1), (3, 14, 1), (3, 15, 1), (3, 16, 5), (4, 17, 10), (5, 18, 10), (6, 17, 10),
         (7, 18, 10), (8, 16, 1), (8, 14, 1), (9, 15, 1), (10, 15, 1), (11, 12, 1), (11, 13, 1), (11, 14, 1),
         (11, 15, 1), (11, 16, 1), (11, 17, 1), (11, 18, 1), (12, 15, 1), (13, 17, 10), (13, 18, 10), (14, 15, 1),
         (15, 16, 1)]
for (u, v, w) in edges:
    W_MATRIX[u - 1][v - 1] = w
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

    def get_area(self): dims = self.get_effective_dims(); return dims[0] * dims[1]

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
# 4. 评估模块 (保持不变)
# ==========================================
def calculate_fitness(layout):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS += 1
    if not layout.is_valid: return [float('inf')] * 3
    total_w = sum(pd.device.weight for pd in layout.placed_devices)
    cg_x = sum(pd.device.weight * pd.get_center_3d()[0] for pd in layout.placed_devices) / total_w
    cg_y = sum(pd.device.weight * pd.get_center_3d()[1] for pd in layout.placed_devices) / total_w
    cg_z = sum(pd.device.weight * pd.get_center_3d()[2] for pd in layout.placed_devices) / total_w
    dev12 = next(p for p in layout.placed_devices if p.device.id == 12)
    dev12_load = sum(pd.device.weight for pd in layout.placed_devices if _is_supported_by(pd, dev12))

    penalty = 0
    if abs(cg_x) > LIMIT_COG_X: penalty += (abs(cg_x) - LIMIT_COG_X) * 1e7
    if abs(cg_y) > LIMIT_COG_Y: penalty += (abs(cg_y) - LIMIT_COG_Y) * 1e7
    if cg_z > LIMIT_COG_Z: penalty += (cg_z - LIMIT_COG_Z) * 1e7
    if dev12_load > LIMIT_DEV12_LOAD: penalty += (dev12_load - LIMIT_DEV12_LOAD) * 1e7

    I_xx, I_yy, I_zz = 0, 0, 0
    for pd in layout.placed_devices:
        L, W, H = pd.get_effective_dims();
        m = pd.device.weight;
        cx, cy, cz = pd.get_center_3d()
        I_xx += (m / 12.0) * (W ** 2 + H ** 2) + m * (cy ** 2 + cz ** 2)
        I_yy += (m / 12.0) * (L ** 2 + H ** 2) + m * (cx ** 2 + cz ** 2)
        I_zz += (m / 12.0) * (L ** 2 + W ** 2) + m * (cx ** 2 + cy ** 2)
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


# ==========================================
# 5. 布局微调与解码器 (保持不变)
# ==========================================
def get_all_supported_devices(layout, base_pd):
    res = []
    for pd in layout.placed_devices:
        if pd.support == base_pd: res.append(pd); res.extend(get_all_supported_devices(layout, pd))
    return res


def fine_tune_layout(layout):
    if not layout.is_valid: return

    # 步骤 1: 基础对齐 (Family Shift)
    for pd in layout.placed_devices:
        if pd.z > 0 and pd.support:
            dx, dy = pd.support.x - pd.x, pd.support.y - pd.y
            if dx == 0 and dy == 0: continue
            family = [pd] + get_all_supported_devices(layout, pd)
            old_pos = {dev: (dev.x, dev.y) for dev in family}
            for dev in family: dev.x += dx; dev.y += dy
            non_fam = [d for d in layout.placed_devices if d not in family]
            if any(check_collision_aabb(f, nf) for f in family for nf in non_fam) or \
                    not check_stacking_rules(pd, pd.support):
                for dev, (ox, oy) in old_pos.items(): dev.x, dev.y = ox, oy

    # 步骤 2: 深度微调 (4-Direction Greedy Search)
    current_score = sum(calculate_fitness(layout))  # 这里会增加评价次数
    for pd in layout.placed_devices:
        if pd.device.id == 12: continue
        family = [pd] + get_all_supported_devices(layout, pd)
        best_dx, best_dy = 0, 0

        # 尝试四个方向的微小位移
        for dx, dy in [(20, 0), (-20, 0), (0, 20), (0, -20)]:
            for dev in family: dev.x += dx; dev.y += dy
            # 边界与碰撞检测
            out = any((dev.z == 0 and (abs(dev.x) > RACK_LENGTH / 2 - dev.get_effective_dims()[0] / 2 or abs(dev.y) > RACK_WIDTH / 2 - dev.get_effective_dims()[1] / 2)) for dev in
                      family)
            non_fam = [d for d in layout.placed_devices if d not in family]
            collision = any(check_collision_aabb(f, nf) for f in family for nf in non_fam)

            if out or collision or (pd.support and not check_stacking_rules(pd, pd.support)):
                for dev in family: dev.x -= dx; dev.y -= dy
                continue

            new_fits = calculate_fitness(layout)  # 这里是评价次数剧增的来源
            new_score = sum(new_fits)
            if new_score < current_score:
                current_score, best_dx, best_dy = new_score, dx, dy

            for dev in family: dev.x -= dx; dev.y -= dy

        # 应用最优微调
        if best_dx != 0 or best_dy != 0:
            for dev in family: dev.x += best_dx; dev.y += best_dy


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
            layout.add(t)
            placed = True
            break
        if not placed: layout.is_valid = False; return layout
    fine_tune_layout(layout)
    return layout


def generate_random_chromosome():
    h = [{'id': d.id, 'rot': random.choice([0, 90])} for d in DEVICE_LIST if d.id != 12 and d.weight >= 15]
    l = [{'id': d.id, 'rot': random.choice([0, 90])} for d in DEVICE_LIST if d.id != 12 and d.weight < 15]
    random.shuffle(h)
    random.shuffle(l)
    return [{'id': 12, 'rot': random.choice([0, 90])}] + h + l


def dominates(f1, f2):
    cond = False
    for a, b in zip(f1, f2):
        if a > b: return False
        if a < b: cond = True
    return cond


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


# ==========================================
# 7. 性能评估与文件保存 (保持不变)
# ==========================================
def calculate_hypervolume_monte_carlo(pf_fits, ref=[1.1, 1.1, 1.1], samples=50000):
    if len(pf_fits) == 0: return 0.0
    s = np.random.uniform(0, ref[0], (samples, 3))
    count = 0
    for smp in s:
        for f in pf_fits:
            if all(f <= smp): count += 1; break
    return (count / samples) * np.prod(ref)


def calculate_igd_normalized(curr, true):
    mn, mx = true.min(0), true.max(0)
    den = mx - mn;
    den[den == 0] = 1e-6
    n_true, n_curr = (true - mn) / den, (curr - mn) / den
    dists = [np.min(np.linalg.norm(n_curr - p, axis=1)) for p in n_true]
    return np.mean(dists)


def save_detailed_csv(layout, filename, fits, cg, load12):
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['设备名称', 'X', 'Y', 'Z', '旋转'])
        for pd in layout.placed_devices: w.writerow([f"{pd.device.id}-{pd.device.name}", pd.x, pd.y, pd.z, pd.rotation])
        w.writerow([]);
        w.writerow(['--- 方案评估 ---'])
        w.writerow(['F1', 'F2', 'F3', 'CG_X', 'CG_Y', 'CG_Z', 'L12'])
        w.writerow(
            [round(fits[0], 3), round(fits[1], 2), round(fits[2], 2), round(cg[0], 2), round(cg[1], 2), round(cg[2], 2),
             round(load12, 2)])


# ==========================================
# 8. SPEA2 主运行逻辑
# ==========================================
def run_spea2(pop_size=50, max_gen=200, true_pf=None):
    global GLOBAL_FITNESS_EVALS
    GLOBAL_FITNESS_EVALS = 0
    tracemalloc.start();
    start_time = time.time()

    population = []
    print("🚀 初始化种群中...")
    while len(population) < pop_size:
        c = generate_random_chromosome();
        l = heuristic_decoder(c)
        if l.is_valid:
            f = calculate_fitness(l)
            if f[0] != float('inf'): population.append({'gene': c, 'fit': f, 'layout': l})

    archive = []

    print(f"⚡ 开始 SPEA2 3目标优化 (共 {max_gen} 代)...")
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
            p1 = random.choice(archive);
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
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"迭代 {gen + 1}/{max_gen} | 存档非支配解数量: {pf_count}")

    # 结果统计
    end_time = time.time();
    curr_mem, peak_mem = tracemalloc.get_traced_memory();
    tracemalloc.stop()
    final_front = [p for p in archive if p['spea2_fit'] < 1.0]
    if not final_front: final_front = archive
    final_front.sort(key=lambda x: x['fit'][0])

    all_fits = np.array([p['fit'] for p in final_front])
    f_min, f_max = all_fits.min(0), all_fits.max(0)
    den = f_max - f_min;
    den[den == 0] = 1e-6
    norm_fits = (all_fits - f_min) / den
    hv = calculate_hypervolume_monte_carlo(norm_fits)
    igd = calculate_igd_normalized(all_fits, true_pf) if true_pf is not None else float('inf')

    print("\n" + "=" * 110)
    print("📊 SPEA2 综合性能评估报告")
    print(
        f"⏱️ 耗时: {end_time - start_time:.2f}s | 💾 内存峰值: {peak_mem / 1e6:.2f}MB | ⚙️ 评估次数: {GLOBAL_FITNESS_EVALS}")
    print(f"📐 HV: {hv:.4f} | 🎯 IGD: {igd:.4f}")
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
        detailed.append({'id': i + 1, 'fit': ind['fit'], 'cg': [cx, cy, cz], 'load12': l12})
        save_detailed_csv(ind['layout'], f"spea2_solution_{i + 1}.csv", ind['fit'], [cx, cy, cz], l12)

    w_scores = np.sum(norm_fits * WEIGHTS, axis=1)
    b_idx = np.argmin(w_scores)
    print("-" * 110)
    print(f"⭐ 推荐均衡解 (惯量0.5:热力0.3:走线0.2): [方案 {b_idx + 1}] Score: {w_scores[b_idx]:.4f}")

    # 可视化 (Plotly)
    hts = []
    for i, info in enumerate(detailed):
        hts.append(
            f"<b>方案 {info['id']}</b><br>F1: {info['fit'][0]:.3f}<br>F2: {info['fit'][1]:.1f}<br>F3: {info['fit'][2]:.1f}<br>CG: ({info['cg'][0]:.1f},{info['cg'][1]:.1f},{info['cg'][2]:.1f})")
    fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
                                       marker=dict(size=6, color=all_fits[:, 2], colorscale='Viridis'), text=hts,
                                       hoverinfo='text')])
    fig.add_trace(
        go.Scatter3d(x=[all_fits[b_idx, 0]], y=[all_fits[b_idx, 1]], z=[all_fits[b_idx, 2]], mode='markers+text',
                     marker=dict(size=12, color='red', symbol='diamond'), text=["★ Best"], textposition="top center"))
    fig.update_layout(title="SPEA2 Pareto Front 3D", scene=dict(xaxis_title='F1', yaxis_title='F2', zaxis_title='F3'))
    fig.write_html("Pareto_Front_SPEA2.html");
    fig.show()


# ==========================================
# 9. True PF (模拟) 与入口
# ==========================================
def build_simple_pf():
    fits = []
    for _ in range(5):
        pop = [{'gene': generate_random_chromosome()} for _ in range(30)]
        for p in pop:
            l = heuristic_decoder(p['gene'])
            p['fit'] = calculate_fitness(l) if l.is_valid else [1e9] * 3
        fits.extend([p['fit'] for p in pop if p['fit'][0] < 1e8])
    f = np.array(fits)
    unique = np.unique(f, axis=0)
    res = []
    for i in range(len(unique)):
        dom = False
        for j in range(len(unique)):
            if i != j and dominates(unique[j], unique[i]): dom = True; break
        if not dom: res.append(unique[i])
    return np.array(res)


if __name__ == "__main__":
    PF_FILE = "True_PF_raw.npy"
    if not os.path.exists(PF_FILE):
        print("⚠️ 生成参考前沿...")
        tp = build_simple_pf();
        np.save(PF_FILE, tp)
    true_pf_raw = np.load(PF_FILE)
    run_spea2(pop_size=POP_SIZE, max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)