import numpy as np
import matplotlib.pyplot as plt

# --- TẠO THAM SỐ HỆ THỐNG (THEO BÀI BÁO) ---
M = 2          # Số anten tại AP
P_T_dBm = 36   # Công suất phát tại AP (dBm)
P_T = 10**((P_T_dBm - 30) / 10) # Đổi sang Watt
sigma2_dBm = -94
sigma2 = 10**((sigma2_dBm - 30) / 10) # Công suất nhiễu (Watt)

# Tham số suy hao đường truyền (Path loss) tại 1m là -40 dB
PL_1m = 10**(-40 / 10)
alpha_AI = 2.2  # AP-IRS
alpha_IU = 2.8  # IRS-User
alpha_AU = 3.8  # AP-User

# Vị trí hình học
x_AP, y_AP = 0, 0
x_IRS, y_IRS = 500, 0

# ============================
# IRS Reflection Model (Circuit)
# ============================

f = 2.4e9                  # Hz
omega = 2 * np.pi * f
Z0 = 377                   # Free-space impedance

L1_MIN, L1_MAX = 1e-9, 6e-9
L2_MIN, L2_MAX = 0.2e-9, 3e-9
C_MIN, C_MAX = 0.1e-12, 10e-12
R_MIN, R_MAX = 0.1, 20

def get_path_loss(d, alpha):
    return PL_1m * (d ** (-alpha))

def generate_channels(N, d_user_x):
    y_user = 2
    d_AI = 500
    d_AU = np.sqrt(d_user_x**2 + y_user**2)
    d_IU = np.sqrt((500 - d_user_x)**2 + y_user**2)
    
    PL_AI = get_path_loss(d_AI, alpha_AI)
    PL_IU = get_path_loss(d_IU, alpha_IU)
    PL_AU = get_path_loss(d_AU, alpha_AU)
    
    G = np.sqrt(PL_AI / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))
    h_r = np.sqrt(PL_IU / 2) * (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
    h_d = np.sqrt(PL_AU / 2) * (np.random.randn(M, 1) + 1j * np.random.randn(M, 1))
    
    Phi = np.diag(h_r.conj().flatten()) @ G
    return Phi, h_d

def compute_reflection(x):
    L1 = x[0::4]
    L2 = x[1::4]
    C  = x[2::4]
    R  = x[3::4]

    j = 1j
    Z = (j*omega*L1 * (j*omega*L2 + 1/(j*omega*C) + R)) / \
        (j*omega*L1 + (j*omega*L2 + 1/(j*omega*C) + R))
    v = (Z - Z0)/(Z + Z0)
    return v

def compute_fitness(x, Phi, h_d):
    v = compute_reflection(x)
    combined_channel = v.conj().T @ Phi + h_d.conj().T
    norm_squared = np.sum(np.abs(combined_channel) ** 2)
    return np.log2(1 + (P_T * norm_squared) / sigma2)

def compute_without_irs(h_d):
    norm_squared = np.sum(np.abs(h_d) ** 2)
    return np.log2(1 + (P_T * norm_squared) / sigma2)

# =========================================================
# 1. THUẬT TOÁN ĐỀ XUẤT: PSO (Mô hình Mạch L, C, R)
# =========================================================
def adaptive_els_pso(N, Phi, h_d, num_particles=40, max_iter=100):
    phi1, phi2 = 2.05, 2.05
    phi = phi1 + phi2
    chi = 2 / np.abs(2 - phi - np.sqrt(phi**2 - 4*phi))
    
    DIM = 4*N
    X = np.zeros((num_particles, DIM))

    L1_REF, L2_REF, C_REF, R_REF = 2.5e-9, 0.7e-9, 1.5e-12, 2.0

    for i in range(num_particles):
        X[i,0::4] = np.clip(np.random.normal(L1_REF, 0.1*L1_REF, N), L1_MIN, L1_MAX)
        X[i,1::4] = np.clip(np.random.normal(L2_REF, 0.1*L2_REF, N), L2_MIN, L2_MAX)
        X[i,2::4] = np.clip(np.random.normal(C_REF, 0.2*C_REF, N), C_MIN, C_MAX)
        X[i,3::4] = np.clip(np.random.normal(R_REF, 0.2*R_REF, N), R_MIN, R_MAX)

    V = np.zeros((num_particles, DIM))
    V[:,0::4] = np.random.uniform(-0.1*(L1_MAX-L1_MIN), 0.1*(L1_MAX-L1_MIN), (num_particles, N))
    V[:,1::4] = np.random.uniform(-0.1*(L2_MAX-L2_MIN), 0.1*(L2_MAX-L2_MIN), (num_particles, N))
    V[:,2::4] = np.random.uniform(-0.1*(C_MAX-C_MIN), 0.1*(C_MAX-C_MIN), (num_particles, N))
    V[:,3::4] = np.random.uniform(-0.1*(R_MAX-R_MIN), 0.1*(R_MAX-R_MIN), (num_particles, N))
    
    pbest = X.copy()
    pbest_fitness = np.array([compute_fitness(X[i], Phi, h_d) for i in range(num_particles)])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]
    
    sigma_max, sigma_min = 1.0, 0.1
    
    for t in range(max_iter):
        w = 0.9 - 0.5 * (t / max_iter)
        for i in range(num_particles):
            r1, r2 = np.random.rand(DIM), np.random.rand(DIM)
            V[i] = chi * (w * V[i] + phi1 * r1 * (pbest[i] - X[i]) + phi2 * r2 * (gbest - X[i]))
            X[i] = X[i] + V[i]
            
            X[i,0::4] = np.clip(X[i,0::4], L1_MIN, L1_MAX)
            X[i,1::4] = np.clip(X[i,1::4], L2_MIN, L2_MAX)
            X[i,2::4] = np.clip(X[i,2::4], C_MIN, C_MAX)
            X[i,3::4] = np.clip(X[i,3::4], R_MIN, R_MAX)
            
            current_fit = compute_fitness(X[i], Phi, h_d)
            if current_fit > pbest_fitness[i]:
                pbest[i] = X[i].copy()
                pbest_fitness[i] = current_fit
                
        current_gbest_idx = np.argmax(pbest_fitness)
        if pbest_fitness[current_gbest_idx] > gbest_fitness:
            gbest = pbest[current_gbest_idx].copy()
            gbest_fitness = pbest_fitness[current_gbest_idx]
            
        sigma = sigma_max - (sigma_max - sigma_min) * (t / max_iter)
        els_gbest = gbest.copy()
        d = np.random.randint(0, DIM)
        
        if d % 4 == 0:
            els_gbest[d] = np.clip(els_gbest[d] + (L1_MAX-L1_MIN)*np.random.normal(0, sigma), L1_MIN, L1_MAX)
        elif d % 4 == 1:
            els_gbest[d] = np.clip(els_gbest[d] + (L2_MAX-L2_MIN)*np.random.normal(0, sigma), L2_MIN, L2_MAX)
        elif d % 4 == 2:
            els_gbest[d] = np.clip(els_gbest[d] + (C_MAX-C_MIN)*np.random.normal(0, sigma), C_MIN, C_MAX)
        else:
            els_gbest[d] = np.clip(els_gbest[d] + (R_MAX-R_MIN)*np.random.normal(0, sigma), R_MIN, R_MAX)
        
        els_fitness = compute_fitness(els_gbest, Phi, h_d)
        if els_fitness > gbest_fitness:
            gbest = els_gbest.copy()
            gbest_fitness = els_fitness
            
    return gbest

# =========================================================
# 2. THUẬT TOÁN ĐỐI CHỨNG: Ideal Upper Bound
# =========================================================
def compute_ideal_upper_bound(Phi, h_d):
    """Tính toán Rate lý tưởng (Coordinate Ascent) với |v| = 1"""
    N = Phi.shape[0]
    v_ideal = np.exp(1j * np.zeros(N))
    for _ in range(10): 
        for n in range(N):
            temp = v_ideal.conj() @ Phi + h_d.conj().T
            contrib_n = Phi[n, :] 
            temp_no_n = temp - v_ideal[n].conj() * contrib_n
            best_phase = np.angle(contrib_n @ temp_no_n.conj().flatten())
            v_ideal[n] = np.exp(1j * best_phase)
            
    combined = v_ideal.conj() @ Phi + h_d.conj().flatten()
    norm_squared = np.sum(np.abs(combined) ** 2)
    return np.log2(1 + (P_T * norm_squared) / sigma2)

# =========================================================
# 3. THUẬT TOÁN ĐỐI CHỨNG: Alternating Optimization (AO)
# =========================================================
# Tham số của mô hình Phase-Amplitude thực tế (Mô hình Toán học)
AO_BETA_MIN = 0.2
AO_PHI = -0.43 * np.pi
AO_K = 1.6

def compute_amplitude_ao(theta):
    return (1.0 - AO_BETA_MIN) * ((np.sin(theta - AO_PHI) + 1.0) / 2.0)**AO_K + AO_BETA_MIN

def build_reflection_vector_ao(theta):
    return compute_amplitude_ao(theta) * np.exp(1j * theta)

def compute_ao_rate(Phi, h_d):
    """Tính Rate dùng AO với mô hình Phase-Amplitude Beta(Theta)"""
    N = Phi.shape[0]
    theta = np.random.uniform(-np.pi, np.pi, N)
    v = build_reflection_vector_ao(theta)
    
    Psi = Phi @ Phi.conj().T        # Kích thước: (N, N)
    hd_hat = Phi @ h_d              # Kích thước: (N, 1)
    
    for iteration in range(15):
        for n in range(N):
            sum_val = 0.0
            for m in range(N):
                if m != n:
                    sum_val += Psi[n, m] * v[m]
            
            phi_n = sum_val + hd_hat[n, 0]
            
            # 1D Search tìm góc theta tối ưu
            theta_range = np.linspace(-np.pi, np.pi, 180)
            beta_range = compute_amplitude_ao(theta_range)
            f_val = (beta_range**2) * Psi[n, n].real + 2 * beta_range * np.abs(phi_n) * np.cos(np.angle(phi_n) - theta_range)
            
            best_idx = np.argmax(f_val)
            theta[n] = theta_range[best_idx]
            v[n] = build_reflection_vector_ao(np.array([theta[n]]))[0]
            
    combined = v.conj() @ Phi + h_d.conj().flatten()
    norm_squared = np.sum(np.abs(combined) ** 2)
    return np.log2(1 + (P_T * norm_squared) / sigma2)

# --- KỊCH BẢN MÔ PHỎNG VÀ VẼ ĐỒ THỊ ---
def run_simulations():
    np.random.seed(42) 
    num_realizations = 20 # Tăng lên 100-200 khi cần dữ liệu chuẩn
    
    plt.figure(figsize=(15, 6))

    # ====================================================
    # FIGURE 1: Rate vs Distance (N = 40)
    # ====================================================
    distances = np.arange(450, 501, 10) 
    N_fixed = 40
    
    print("=== Đang chạy mô phỏng Hình 1 (Rate vs Distance) ===")
    res_ub_1, res_ao_1, res_prac_1, res_wo_1 = [], [], [], []
    
    for d in distances:
        print(f" -> Tính toán cho khoảng cách d = {d}m...")
        r_ub, r_ao, r_prac, r_wo = [], [], [], []
        for _ in range(num_realizations):
            Phi, h_d = generate_channels(N_fixed, d)
            
            # 1. Ideal Upper Bound
            r_ub.append(compute_ideal_upper_bound(Phi, h_d))
            
            # 2. Alternating Optimization (Mô hình Toán)
            r_ao.append(compute_ao_rate(Phi, h_d))
            
            # 3. Thực tế đề xuất (Mô hình Mạch + PSO)
            gb_prac = adaptive_els_pso(N_fixed, Phi, h_d, num_particles=90, max_iter=300)
            r_prac.append(compute_fitness(gb_prac, Phi, h_d))
            
            # 4. Without IRS
            r_wo.append(compute_without_irs(h_d))

        res_ub_1.append(np.mean(r_ub))
        res_ao_1.append(np.mean(r_ao))
        res_prac_1.append(np.mean(r_prac))
        res_wo_1.append(np.mean(r_wo))

    plt.subplot(1, 2, 1)
    plt.plot(distances, res_ub_1, 'k-.', label='Ideal IRS')
    plt.plot(distances, res_ao_1, 'g^-', label='AO')
    plt.plot(distances, res_prac_1, 'ro-', label='PSO')
    plt.plot(distances, res_wo_1, 'bs:', label='Without IRS')
    plt.xlabel('AP-user horizontal distance, d (m)')
    plt.ylabel('Achievable Rate (bps/Hz)')
    plt.title(f'Fig 1: Rate vs Distance (N={N_fixed})')
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=9)

    # ====================================================
    # FIGURE 2: Rate vs N elements (d = 498m)
    # ====================================================
    elements_range = np.arange(10, 51, 10) 
    d_fixed = 498
    
    print("\n=== Đang chạy mô phỏng Hình 2 (Rate vs N Elements) ===")
    res_ub_2, res_ao_2, res_prac_2, res_wo_2 = [], [], [], []
    
    for N in elements_range:
        print(f" -> Tính toán cho số phần tử N = {N}...")
        r_ub, r_ao, r_prac, r_wo = [], [], [], []
        for _ in range(num_realizations):
            Phi, h_d = generate_channels(N, d_fixed)
            
            # 1. Ideal Upper Bound
            r_ub.append(compute_ideal_upper_bound(Phi, h_d))
            
            # 2. Alternating Optimization (Mô hình Toán)
            r_ao.append(compute_ao_rate(Phi, h_d))
            
            # 3. Thực tế đề xuất (Mô hình Mạch + PSO)
            gb_prac = adaptive_els_pso(N, Phi, h_d, num_particles=90, max_iter=300)
            r_prac.append(compute_fitness(gb_prac, Phi, h_d))
            
            # 4. Without IRS
            r_wo.append(compute_without_irs(h_d))

        res_ub_2.append(np.mean(r_ub))
        res_ao_2.append(np.mean(r_ao))
        res_prac_2.append(np.mean(r_prac))
        res_wo_2.append(np.mean(r_wo))

    plt.subplot(1, 2, 2)
    plt.plot(elements_range, res_ub_2, 'k-.', label='Ideal IRS')
    plt.plot(elements_range, res_ao_2, 'g^-', label='AO')
    plt.plot(elements_range, res_prac_2, 'ro-', label='PSO')
    plt.plot(elements_range, res_wo_2, 'bs:', label='Without IRS')
    plt.xlabel('Number of reflecting elements, N')
    plt.title(f'Fig 2: Rate vs N Elements (d={d_fixed}m)')
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulations()