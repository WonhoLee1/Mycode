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






import numpy as np
import matplotlib.pyplot as plt

# --- mm-tonne-s 단위계 ---
L, W = 2000.0, 1200.0
A = L * W
P = 2 * (L + W)
mass = 0.04
h0 = 300.0
rho = 1.225e-12
mu = 1.81e-11
g = 9810.0
dt = 0.0001
z_contact = 0.5  # 물리적 충돌로 간주하는 높이 (0.5mm)

def simulate_to_impact(mode):
    t, z, v, a = 0.0, h0, 0.0, -g
    res = {'t': [t], 'z': [z], 'v': [v], 'a': [a], 'f_res': [0.0]}
    
    while z > z_contact:
        h = max(z, z_contact) # Singularity 방지
        f_drag = 0.5 * rho * 1.05 * A * v**2 if mode >= 1 else 0.0
        
        f_sq = 0.0
        if mode == 2:
            f_esc = 0.5 * rho * ((A * abs(v)) / (P * h))**2 * A * 2.5
            f_vis = (mu * L * W**3 / h**3) * 0.42 * abs(v)
            f_ine = (rho * L * W**3 / h) * 0.2 * abs(a)
            f_sq = f_esc + f_vis + f_ine
            
        f_total = f_drag + f_sq
        a_new = (-mass * g + f_total) / mass
        
        v += a_new * dt
        z += v * dt
        t += dt
        a = a_new
        
        res['t'].append(t); res['z'].append(z); res['v'].append(v); res['a'].append(a)
        res['f_res'].append(f_total)
        
    return res

# 3가지 케이스 실행
c0 = simulate_to_impact(0) # 자유 낙하
c1 = simulate_to_impact(1) # 공기 저항
c2 = simulate_to_impact(2) # 풀 모델 (스퀴즈 이펙트)

# --- 결과 시각화 ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# (1) 속도 프로파일: 충돌 속도 비교
axes[0].plot(c0['t'], c0['v'], 'k--', label=f'Free Fall (Impact: {c0["v"][-1]:.1f} mm/s)')
axes[0].plot(c1['t'], c1['v'], 'g--', label=f'Air Drag Only (Impact: {c1["v"][-1]:.1f} mm/s)')
axes[0].plot(c2['t'], c2['v'], 'r-', linewidth=2, label=f'Full Model (Impact: {c2["v"][-1]:.1f} mm/s)')
axes[0].axhline(-1700, color='blue', linestyle=':', label='Velocity Inversion (1700)')
axes[0].set_title('Velocity Profile until Impact', fontsize=14)
axes[0].set_ylabel('Velocity [mm/s]'); axes[0].set_xlabel('Time [s]')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# (2) 가속도 비교: 저항력의 폭발적 증가 확인
axes[1].plot(c0['t'], c0['a'], 'k--', label='Free Fall')
axes[1].plot(c2['t'], c2['a'], 'r-', linewidth=2, label='Full Model')
axes[1].set_title('Acceleration Change (Cushioning Effect)', fontsize=14)
axes[1].set_ylabel('Accel [mm/s²]'); axes[1].set_xlabel('Time [s]')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"충돌 직전 속도 비교:")
print(f"- 자유 낙하: {abs(c0['v'][-1]):.2f} mm/s")
print(f"- 풀 모델 적용: {abs(c2['v'][-1]):.2f} mm/s")


import numpy as np
import matplotlib.pyplot as plt

# 1. 물리 상수 및 환경 설정 (mm, tonne, s)
L, W = 2000.0, 1200.0           # 상자 크기 (mm)
A = L * W                       # 바닥 면적 (mm^2)
P = 2 * (L + W)                 # 상자 둘레 (mm)
mass = 0.04                     # 질량 (40kg = 0.04 tonne)
h0 = 300.0                      # 초기 높이 (mm)
z_contact = 0.5                 # 충돌 기준 높이 (mm)

rho = 1.225e-12                 # 공기 밀도 (tonne/mm^3)
mu = 1.81e-11                   # 점성 계수 (tonne/mm*s)
Cd = 1.05                       # 항력 계수
g = 9810.0                      # 중력 가속도 (mm/s^2)
dt = 0.0001                     # 시간 간격 (s)

# 저항력 제한 설정 (100G 제한)
G_limit = 100.0
F_max_limit = mass * g * G_limit

def run_simulation(mode):
    """
    mode 0: 자유 낙하
    mode 1: 공기 저항만 추가
    mode 2: 통합 모델 (공기저항 + 스퀴즈 효과 + 탈출동압 + 저항력제한)
    """
    t, z, v, a = 0.0, h0, 0.0, -g
    res = {'t': [t], 'z': [z], 'v': [v], 'a': [a], 
           'f_drag': [0], 'f_escape': [0], 'f_visc': [0], 'f_total': [0]}
    
    while z > z_contact:
        # [안전장치 1] 수치 안정성을 위한 유효 높이 결정 (Overflow 방지)
        h_eff = max(z, z_contact)
        
        f_drag, f_escape, f_visc, f_inert = 0.0, 0.0, 0.0, 0.0
        
        if mode >= 1:
            f_drag = 0.5 * rho * Cd * A * v**2
            
        if mode == 2:
            # [수식 1] 탈출 동압 저항 (1/h^2) - 1700mm/s 역전의 핵심
            v_out = (A * abs(v)) / (P * h_eff)
            f_escape = 0.5 * rho * (v_out**2) * A * 2.5 # 손실계수 2.5
            
            # [수식 2] 점성 저항 (1/h^3) - 최종 연착륙 유도
            f_visc = (mu * L * W**3 / (h_eff**3)) * 0.42 * abs(v)
            
            # [수식 3] 부가질량 관성 저항 (1/h)
            f_inert = (rho * L * W**3 / h_eff) * 0.2 * abs(a)
            
        # [안전장치 2] 저항력 클램핑 (물리적 한계 및 수치 발산 방지)
        raw_f_total = f_drag + f_escape + f_visc + f_inert
        f_clamped = min(raw_f_total, F_max_limit)
        
        # 운동 방정식
        a_new = (-mass * g + f_clamped) / mass
        
        v += a_new * dt
        z += v * dt
        t += dt
        a = a_new
        
        res['t'].append(t); res['z'].append(z); res['v'].append(v); res['a'].append(a)
        res['f_drag'].append(f_drag); res['f_escape'].append(f_escape)
        res['f_visc'].append(f_visc); res['f_total'].append(f_clamped)
        
        if t > 0.5: break # 안전 종료
        
    return res

# 3가지 안 실행
cases = [run_simulation(i) for i in range(3)]
labels = ['Free Fall', 'Air Drag Only', 'Integrated Full Model']

# --- 결과 시각화 ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 변위 (Position)
for i, c in enumerate(cases):
    axes[0,0].plot(c['t'], c['z'], label=labels[i], linestyle='--' if i<2 else '-')
axes[0,0].set_title('Position [mm]', fontsize=14); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# 2. 속도 (Velocity) - 1700mm/s 역전 확인
for i, c in enumerate(cases):
    axes[0,1].plot(c['t'], c['v'], label=labels[i], linestyle='--' if i<2 else '-')
axes[0,1].axhline(-1700, color='blue', ls=':', label='Target Inversion (-1700)')
axes[0,1].set_title('Velocity [mm/s]', fontsize=14); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# 3. 가속도 (Acceleration) - 에어쿠션 피크 확인
for i, c in enumerate(cases):
    axes[1,0].plot(c['t'], c['a'], label=labels[i], linestyle='--' if i<2 else '-')
axes[1,0].set_title('Acceleration [mm/s²]', fontsize=14); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# 4. 저항력 구성 성분 분석 (Case 2 - Log Scale)
full = cases[2]
axes[1,1].semilogy(full['t'], full['f_escape'], label='Escape Press (1/h²)', lw=2)
axes[1,1].semilogy(full['t'], full['f_visc'], label='Viscous (1/h³)', lw=1.5, alpha=0.7)
axes[1,1].semilogy(full['t'], full['f_total'], 'r', label='Total (Clamped 100G)', lw=1.5)
axes[1,1].axhline(mass*g, color='k', ls='--', label='Weight (mg)')
axes[1,1].set_title('Force Analysis [N] (Log Scale)', fontsize=14); axes[1,1].legend(); axes[1,1].grid(True, which='both', alpha=0.2)

plt.tight_layout(); plt.show()
