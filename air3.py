import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 물리적 상수 설정 (mm, tonne, s 단위계)
# ==========================================
BOX_LENGTH_MM = 2000.0
BOX_WIDTH_MM = 1200.0
BOX_HEIGHT_MM = 250.0
BOX_MASS_TONNE = 0.04  # 40kg = 0.04 tonne

INITIAL_DROP_HEIGHT_MM = 300.0
GRAVITY_ACCEL_MM_S2 = 9810.0

AIR_DENSITY_TONNE_MM3 = 1.225e-12
AIR_VISCOSITY_TONNE_MM_S = 1.81e-11
STANDARD_DRAG_COEFFICIENT = 1.05

# ==========================================
# 2. 시험 결과 보정을 위한 스케일 계수 (Calibration Factors)
# ==========================================
# 이 값들을 조정하여 시험 그래프와 일치시킵니다.
AIR_DRAG_SCALE_FACTOR = 1.0           # 기본 공기 저항 보정
ESCAPE_VELOCITY_SCALE_FACTOR = 2.5    # 1700mm/s 역전 시점 보정 (1/h^2 항)
SQUEEZE_VISCOUS_SCALE_FACTOR = 0.42   # 지면 근처 점성 저항 보정 (1/h^3 항)
SQUEEZE_INERTIAL_SCALE_FACTOR = 0.2   # 공기 관성 저항 보정 (1/h 항)

# S-Curve 형성을 위한 핵심 변수 (누설 및 변형 고려)
# 이 값이 클수록 속도 곡선의 끝단이 S자로 부드럽게 꺾입니다.
EFFECTIVE_LEAKAGE_GAP_MM = 0.8        
MAX_ACCEL_LIMIT_G = 100.0             # 물리적 저항 한계 (100G)

# ==========================================
# 3. 시뮬레이션 함수 정의
# ==========================================
def run_calibration_simulation():
    # 시간 간격 및 초기 상태
    delta_time_seconds = 0.0001
    current_time_seconds = 0.0
    current_height_mm = INITIAL_DROP_HEIGHT_MM
    current_velocity_mm_s = 0.0
    current_acceleration_mm_s2 = -GRAVITY_ACCEL_MM_S2

    # 결과 저장용 리스트
    time_history = [0.0]
    height_history = [INITIAL_DROP_HEIGHT_MM]
    velocity_history = [0.0]
    accel_history = [-GRAVITY_ACCEL_MM_S2]
    
    box_area_mm2 = BOX_LENGTH_MM * BOX_WIDTH_MM
    box_perimeter_mm = 2 * (BOX_LENGTH_MM + BOX_WIDTH_MM)
    max_force_limit_newton = BOX_MASS_TONNE * GRAVITY_ACCEL_MM_S2 * MAX_ACCEL_LIMIT_G

    while current_height_mm > 0.01:
        # [S-Curve 로직] 실제 간극에 누설 간극을 더해 분모 발산을 방지하고 S자 감속 유도
        effective_h_mm = current_height_mm + EFFECTIVE_LEAKAGE_GAP_MM
        
        # (1) 일반 공기 저항력 계산
        force_air_drag = (0.5 * AIR_DENSITY_TONNE_MM3 * STANDARD_DRAG_COEFFICIENT * box_area_mm2 * (current_velocity_mm_s**2) * AIR_DRAG_SCALE_FACTOR)
        
        # (2) 탈출 속도에 의한 동압 저항 (1/h^2)
        air_escape_velocity = (box_area_mm2 * abs(current_velocity_mm_s)) / (box_perimeter_mm * effective_h_mm)
        force_escape_pressure = (0.5 * AIR_DENSITY_TONNE_MM3 * (air_escape_velocity**2) * box_area_mm2 * ESCAPE_VELOCITY_SCALE_FACTOR)
        
        # (3) 스퀴즈 필름 점성 저항 (1/h^3)
        force_squeeze_viscous = ((AIR_VISCOSITY_TONNE_MM_S * BOX_LENGTH_MM * (BOX_WIDTH_MM**3)) / 
                                 (effective_h_mm**3) * SQUEEZE_VISCOUS_SCALE_FACTOR * abs(current_velocity_mm_s))
        
        # (4) 공기 부가 질량 관성 저항 (1/h)
        force_squeeze_inertial = ((AIR_DENSITY_TONNE_MM3 * BOX_LENGTH_MM * (BOX_WIDTH_MM**3)) / 
                                  effective_h_mm * SQUEEZE_INERTIAL_SCALE_FACTOR * abs(current_acceleration_mm_s2))
        
        # 전체 저항력 합산 및 제한(Clamping) 적용
        total_resistance_force = force_air_drag + force_escape_pressure + force_squeeze_viscous + force_squeeze_inertial
        final_resistance_force = min(total_resistance_force, max_force_limit_newton)
        
        # 운동 방정식: m*a = -m*g + F_res
        new_acceleration = (-BOX_MASS_TONNE * GRAVITY_ACCEL_MM_S2 + final_resistance_force) / BOX_MASS_TONNE
        
        # 상태 업데이트 (Euler Integration)
        current_velocity_mm_s += new_acceleration * delta_time_seconds
        current_height_mm += current_velocity_mm_s * delta_time_seconds
        current_time_seconds += delta_time_seconds
        current_acceleration_mm_s2 = new_acceleration
        
        # 데이터 기록
        time_history.append(current_time_seconds)
        height_history.append(current_height_mm)
        velocity_history.append(current_velocity_mm_s)
        accel_history.append(new_acceleration)
        
        # [종료 조건] 속도가 0을 지나 양수가 되면 (반등 시점) 해석 중단
        if current_time_seconds > 0.05 and current_velocity_mm_s >= 0:
            break
            
    return time_history, height_history, velocity_history, accel_history

# 시뮬레이션 실행
t, z, v, a = run_calibration_simulation()

# ==========================================
# 4. 결과 시각화
# ==========================================
plt.figure(figsize=(15, 5))

# 속도 그래프 (S-Curve 확인용)
plt.subplot(1, 3, 1)
plt.plot(t, v, 'r-', linewidth=2, label='Simulated Velocity')
plt.axhline(-1700, color='b', linestyle=':', label='Exp. Inversion Point')
plt.title('Velocity Profile (S-Curve)', fontsize=12)
plt.ylabel('Velocity [mm/s]'); plt.xlabel('Time [s]'); plt.legend(); plt.grid(True, alpha=0.3)

# 변위 그래프
plt.subplot(1, 3, 2)
plt.plot(t, z, 'g-', linewidth=2)
plt.title('Drop Height [mm]', fontsize=12)
plt.ylabel('Height [mm]'); plt.xlabel('Time [s]'); plt.grid(True, alpha=0.3)

# 가속도 그래프
plt.subplot(1, 3, 3)
plt.plot(t, a, 'm-', linewidth=2)
plt.title('Deceleration (G-load)', fontsize=12)
plt.ylabel('Accel [mm/s²]'); plt.xlabel('Time [s]'); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"최종 안착 높이: {z[-1]:.4f} mm")
print(f"최대 낙하 속도: {min(v):.2f} mm/s")
