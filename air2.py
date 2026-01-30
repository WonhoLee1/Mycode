import numpy as np
import matplotlib.pyplot as plt

# 1. 환경 설정 및 상수 (mm, tonne, s)
L, W = 2000.0, 1200.0           # mm
A = L * W                       # 면적 (mm^2)
P = 2 * (L + W)                 # 둘레 (mm)
mass = 0.04                     # tonne (40kg)
h0 = 300.0                      # mm
rho = 1.225e-12                 # tonne/mm^3
mu = 1.81e-11                   # tonne/(mm*s)
Cd = 1.05                       
g = 9810.0                      # mm/s^2
dt = 0.0001                     # s

def simulate_all_cases():
    # 데이터 저장 구조
    results = []
    modes = [0, 1, 2] # 0: 자유낙하, 1: 공기저항만, 2: 전체 효과
    
    for mode in modes:
        t, z, v, a = 0.0, h0, 0.0, -g
        res = {'t': [], 'z': [], 'v': [], 'a': [], 
               'f_drag': [], 'f_escape': [], 'f_visc': [], 'f_inert': []}
        
        while z > 0.5:
            h = max(z, 0.5)
            f_drag = 0.5 * rho * Cd * A * v**2 if mode >= 1 else 0.0
            
            f_escape, f_visc, f_inert = 0.0, 0.0, 0.0
            if mode == 2:
                # 동압 저항 (1/h^2) - 탈출 속도 기반
                v_out = (A * abs(v)) / (P * h)
                f_escape = 0.5 * rho * (v_out**2) * A * 2.5
                # 점성 저항 (1/h^3)
                f_visc = (mu * L * W**3 / h**3) * 0.42 * abs(v)
                # 관성 저항 (1/h)
                f_inert = (rho * L * W**3 / h) * 0.2 * abs(a)
            
            f_total = f_drag + f_escape + f_visc + f_inert
            a_new = (-mass * g + f_total) / mass
            
            v += a_new * dt
            z += v * dt
            t += dt
            a = a_new
            
            res['t'].append(t); res['z'].append(z); res['v'].append(v); res['a'].append(a)
            res['f_drag'].append(f_drag); res['f_escape'].append(f_escape)
            res['f_visc'].append(f_visc); res['f_inert'].append(f_inert)
            
            if v > 0 or t > 0.5: break
        results.append(res)
    return results

# 시뮬레이션 실행
case_data = simulate_all_cases()
full_case = case_data[2] # 모든 효과가 포함된 케이스

# 2. 그래프 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

labels = ['Free Fall', 'Air Drag Only', 'Full Squeeze Effect']
colors = ['gray', 'green', 'red']
styles = ['--', '--', '-']

# (1) 변위 비교
for i, data in enumerate(case_data):
    axes[0, 0].plot(data['t'], data['z'], color=colors[i], linestyle=styles[i], label=labels[i])
axes[0, 0].set_title('Position [mm]', fontsize=14)
axes[0, 0].set_ylabel('Height [mm]')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# (2) 속도 비교
for i, data in enumerate(case_data):
    axes[0, 1].plot(data['t'], data['v'], color=colors[i], linestyle=styles[i], label=labels[i])
axes[0, 1].axhline(-1700, color='blue', linestyle=':', label='Target Inversion (1700)')
axes[0, 1].set_title('Velocity [mm/s]', fontsize=14)
axes[0, 1].set_ylabel('Velocity [mm/s]')
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# (3) 가속도 비교
for i, data in enumerate(case_data):
    axes[1, 0].plot(data['t'], data['a'], color=colors[i], linestyle=styles[i], label=labels[i])
axes[1, 0].set_title('Acceleration [mm/s²]', fontsize=14)
axes[1, 0].set_ylabel('Accel [mm/s²]')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# (4) 저항 요소별 영향력 비교 (Log Scale)
ax_f = axes[1, 1]
ax_f.semilogy(full_case['t'], full_case['f_escape'], 'blue', label='Dynamic Pressure (1/h²)', linewidth=2.5)
ax_f.semilogy(full_case['t'], full_case['f_visc'], 'orange', label='Viscous Resistance (1/h³)', linewidth=2.5)
ax_f.semilogy(full_case['t'], full_case['f_inert'], 'purple', label='Inertial Resistance (1/h)', alpha=0.7)
ax_f.semilogy(full_case['t'], full_case['f_drag'], 'green', label='Standard Air Drag', alpha=0.5)
ax_f.axhline(mass*g, color='red', linestyle='--', label='Box Weight (mg)')

ax_f.set_title('Influence Comparison: Squeeze Force Components [N]', fontsize=14)
ax_f.set_ylabel('Force [N] (Log Scale)')
ax_f.set_xlabel('Time [s]')
ax_f.legend(loc='upper left'); ax_f.grid(True, which='both', alpha=0.2)

plt.tight_layout()
plt.show()
