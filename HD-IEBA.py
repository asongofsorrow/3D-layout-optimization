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
# 1. 全局环境、物理限制与参数定义 (严格匹配原始版本)
# ==========================================
POP_SIZE = 100
NUM_ITERATIONS = 1000
RACK_LENGTH, RACK_WIDTH = 1520, 660

# --- IEBA 算法特定参数 ---
F_MIN, F_MAX = 0, 2.0  # 频率范围
A_0, R_0 = 0.9, 0.1  # 初始响度与脉冲发射率
ALPHA, GAMMA = 0.9, 0.9  # 衰减系数
ARCHIVE_SIZE = 50

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
    L1, W1, H1 = pd1.get_effective_dims();
    L2, W2, H2 = pd2.get_effective_dims()
    dx = max(0.0, abs(pd1.x - pd2.x) - (L1 + L2) / 2.0)
    dy = max(0.0, abs(pd1.y - pd2.y) - (W1 + W2) / 2.0)
    z1_min, z1_max, z2_min, z2_max = pd1.z, pd1.z + H1, pd2.z, pd2.z + H2
    dz_dir = (z1_min - z2_max) if z1_min >= z2_max else (z1_max - z2_min) if z2_min >= z1_max else 0.0
    return math.sqrt(dx ** 2 + dy ** 2 + dz_dir ** 2), dx, dy, dz_dir


def check_collision_aabb(pd1, pd2):
    L1, W1, H1 = pd1.get_effective_dims();
    L2, W2, H2 = pd2.get_effective_dims()
    cx1, cy1, cz1 = pd1.get_center_3d();
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
        L, W, H = pd.get_effective_dims();
        m = pd.device.weight;
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
        q_cache[pd] = res;
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

        random.shuffle(candidates)
        for pos in candidates:
            t = PlacedDevice(dev, pos['x'], pos['y'], pos['z'], g['rot'], pos['s'])
            if abs(pos['x']) > xl or abs(pos['y']) > yl: continue
            if any(check_collision_aabb(t, ex) for ex in layout.placed_devices): continue
            layout.add(t);
            placed = True;
            break
        if not placed: layout.is_valid = False; return layout
    return layout


def generate_random_chromosome():
    ids = [d.id for d in DEVICE_LIST if d.id != 12];
    random.shuffle(ids)
    return [{'id': 12, 'rot': random.choice([0, 90])}] + [{'id': i, 'rot': random.choice([0, 90])} for i in ids]


def calculate_igd(curr_pf, true_pf):
    if true_pf is None or len(curr_pf) == 0: return 0.0
    mn, mx = true_pf.min(0), true_pf.max(0);
    den = mx - mn;
    den[den == 0] = 1e-6
    n_true, n_curr = (true_pf - mn) / den, (curr_pf - mn) / den
    return np.mean([np.min(np.linalg.norm(n_curr - p, axis=1)) for p in n_true])


def calculate_hv(n_fits):
    if len(n_fits) == 0: return 0.0
    s = np.random.uniform(0, 1.1, (20000, 3))
    count = sum(1 for smp in s if any(all(f <= smp) for f in n_fits))
    return (count / 20000) * (1.1 ** 3)


# ==========================================
# 5. IEBA 算法核心与性能监控
# ==========================================
def dominates(f1, f2):
    return all(np.array(f1) <= np.array(f2)) and any(np.array(f1) < np.array(f2))


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

    print(f"⚡ 开始 IEBA 3目标布局优化 (共 {max_gen} 代)...")
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

        if (gen + 1) % 50 == 0:
            print(f"迭代 {gen + 1}/{max_gen} | 存档非支配解数量: {len(archive)}")

    # 停止监控并获取内存
    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 整理结果
    archive.sort(key=lambda x: x['fit'][0])
    all_fits = np.array([p['fit'] for p in archive])
    f_min, f_max = all_fits.min(0), all_fits.max(0);
    den = f_max - f_min;
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
    print("-" * 110)
    print(f"{'编号':<4} | {'惯量(F1)':<9} | {'热力(F2)':<9} | {'走线(F3)':<9} | {'综合得分'}")

    for i, ind in enumerate(archive):
        print(
            f"{i + 1:<6} | {ind['fit'][0]:<10.3f} | {ind['fit'][1]:<10.1f} | {ind['fit'][2]:<10.1f} | {weighted_sums[i]:.4f}")

    print("-" * 110)
    print(f"⭐ 推荐均衡解: [方案 {best_idx + 1}] | 加权得分: {weighted_sums[best_idx]:.4f}")

    # 3D 可视化
    fig = go.Figure(data=[go.Scatter3d(x=all_fits[:, 0], y=all_fits[:, 1], z=all_fits[:, 2], mode='markers',
                                       marker=dict(size=5, color=weighted_sums, colorscale='Viridis', showscale=True))])
    fig.add_trace(go.Scatter3d(x=[all_fits[best_idx, 0]], y=[all_fits[best_idx, 1]], z=[all_fits[best_idx, 2]],
                               mode='markers+text', marker=dict(size=12, color='red', symbol='diamond'),
                               text=["★ BEST"]))
    fig.update_layout(title="IEBA Pareto Front 3D Visualization",
                      scene=dict(xaxis_title='F1: Inertia', yaxis_title='F2: Thermal', zaxis_title='F3: Routing'))
    fig.show()


if __name__ == "__main__":
    PF_FILE = "True_PF_raw.npy"
    true_pf_raw = np.load(PF_FILE) if os.path.exists(PF_FILE) else None
    run_ieba(max_gen=NUM_ITERATIONS, true_pf=true_pf_raw)