import os
# OMP 경고 및 충돌 해결을 위한 환경 변수 설정 (반드시 최상단)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import time

# --- [1. 가공 영역 확장 엔진 (Numba 가속)] ---
@njit
def compute_maximum_expandable_bounding_box(occupancy_status_grid, start_z_idx, start_y_idx, start_x_idx):
    grid_depth, grid_height, grid_width = occupancy_status_grid.shape
    z_min, z_max = start_z_idx, start_z_idx
    y_min, y_max = start_y_idx, start_y_idx
    x_min, x_max = start_x_idx, start_x_idx
    
    while True:
        expanded_flag = False
        if x_max + 1 < grid_width and np.all(occupancy_status_grid[z_min:z_max+1, y_min:y_max+1, x_max+1]):
            x_max += 1; expanded_flag = True
        if x_min - 1 >= 0 and np.all(occupancy_status_grid[z_min:z_max+1, y_min:y_max+1, x_min-1]):
            x_min -= 1; expanded_flag = True
        if y_max + 1 < grid_height and np.all(occupancy_status_grid[z_min:z_max+1, y_max+1, x_min:x_max+1]):
            y_max += 1; expanded_flag = True
        if y_min - 1 >= 0 and np.all(occupancy_status_grid[z_min:z_max+1, y_min-1, x_min:x_max+1]):
            y_min -= 1; expanded_flag = True
        if z_max + 1 < grid_depth and np.all(occupancy_status_grid[z_max+1, y_min:y_max+1, x_min:x_max+1]):
            z_max += 1; expanded_flag = True
        if z_min - 1 >= 0 and np.all(occupancy_status_grid[z_min-1, y_min:y_max+1, x_min:x_max+1]):
            z_min -= 1; expanded_flag = True
        if not expanded_flag: break
    return z_min, z_max, y_min, y_max, x_min, x_max

# --- [2. 최종 가공 시뮬레이션 엔진] ---
class MachiningSimulationEngine:
    def __init__(self, target_stl_file_path):
        # 예제 파일이 없으면 생성
        if not os.path.exists(target_stl_file_path):
            print(f"[알림] {target_stl_file_path} 파일이 없어 예제 박스를 생성합니다.")
            trimesh.creation.box(extents=[40, 40, 20]).export(target_stl_file_path)

        self.mesh_path = target_stl_file_path
        self.original_cad_mesh = trimesh.load(target_stl_file_path)
        if isinstance(self.original_cad_mesh, trimesh.Scene):
            self.original_cad_mesh = self.original_cad_mesh.dump(concatenate=True)
            
        self.original_min, self.original_max = self.original_cad_mesh.bounds
        self.dimensions = self.original_max - self.original_min
        self.cad_volume = self.original_cad_mesh.volume
        self.calculated_cutters = []

    def show_initial_mesh(self):
        """1단계: 원본 형상을 모달 창으로 확인"""
        print("\n[단계 1] 원본 형상을 확인하십시오. 창을 닫으면 시뮬레이션이 시작됩니다.")
        plotter = pv.Plotter(title="Step 1: Original CAD Preview")
        plotter.add_mesh(self.original_cad_mesh, color='white', show_edges=True)
        plotter.add_text("Close this window to start machining simulation", font_size=10)
        plotter.add_axes()
        plotter.show() # 모달 모드로 동작

    def set_parameters(self, resolution, min_side_mm, expansion, tolerance):
        """2단계: 파라미터 설정 및 연산 준비"""
        self.resolution = resolution
        self.min_side = min_side_mm
        self.stock_expansion_ratio = expansion
        self.tolerance = tolerance
        
        self.geometric_center = (self.original_min + self.original_max) / 2.0
        self.stock_boundary_min = self.geometric_center + (self.original_min - self.geometric_center) * expansion
        self.stock_boundary_max = self.geometric_center + (self.original_max - self.geometric_center) * expansion
        self.voxel_pitch = np.max(self.stock_boundary_max - self.stock_boundary_min) / self.resolution
        self.stock_volume = np.prod(self.stock_boundary_max - self.stock_boundary_min)

    def run_simulation(self):
        """3단계: 가공 연산 수행"""
        print("\n[단계 2] 가공 시뮬레이션 연산 중...")
        start_time = time.time()
        
        dims = np.ceil((self.stock_boundary_max - self.stock_boundary_min) / self.voxel_pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.voxel_pitch + self.stock_boundary_min
        is_inside = self.original_cad_mesh.contains(pts)
        occupancy_grid = (~is_inside).reshape(dims[::-1])
        
        target_voxels = np.sum(occupancy_grid)
        temp_grid = occupancy_grid.copy()
        mask = np.ones_like(temp_grid, dtype=bool)

        with tqdm(total=target_voxels, desc="Machining Progress") as pbar:
            while True:
                search_area = temp_grid & mask
                dist_map = distance_transform_edt(search_area)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search_area.shape)
                z1, z2, y1, y2, x1, x2 = compute_maximum_expandable_bounding_box(temp_grid, *seed)
                
                w_min = self.stock_boundary_min + np.array([x1, y1, z1]) * self.voxel_pitch
                w_max = self.stock_boundary_min + np.array([x2+1, y2+1, z2+1]) * self.voxel_pitch
                
                refined_b = self.refine_and_clamp_cutter([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                c_size = np.array([refined_b[1]-refined_b[0], refined_b[3]-refined_b[2], refined_b[5]-refined_b[4]])
                
                if np.all(c_size >= self.min_side):
                    self.calculated_cutters.append(refined_b)
                    prev_count = np.sum(temp_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(prev_count)
                else:
                    mask[seed] = False

        self.total_time = time.time() - start_time
        self.final_report(target_voxels, temp_grid)

    def final_report(self, total_voxels, remaining_grid):
        """4단계: 결과 리포트 출력 및 최종 시각화"""
        processed_voxels = total_voxels - np.sum(remaining_grid)
        accuracy = (processed_voxels / total_voxels) * 100
        total_cut_vol = sum([(b[1]-b[0])*(b[3]-b[2])*(b[5]-b[4]) for b in self.calculated_cutters])

        print("\n" + "="*60)
        print(" [시뮬레이션 가공 결과 리포트]")
        print(f" - 생성된 커팅 박스 개수: {len(self.calculated_cutters)} 개")
        print(f" - 가공 정밀도(Voxel): {accuracy:.2f} %")
        print(f" - 소요 시간: {self.total_time:.2f} 초")
        print(f" - CAD 원본 체적: {self.cad_volume:.2f} mm³")
        print(f" - 가공 후 남은 체적: {self.stock_volume - total_cut_vol:.2f} mm³")
        print(f" - 체적 오차율: {abs(self.cad_volume - (self.stock_volume - total_cut_vol))/self.cad_volume*100:.2f} %")
        print("="*60)

        # 최종 결과 시각화
        print("\n[단계 3] 최종 가공 결과를 표시합니다.")
        plotter = pv.Plotter(title="Step 3: Final Machining Result")
        # 가공된 형상 생성 (Boolean Subtract)
        final_mesh = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                    self.stock_boundary_min[1], self.stock_boundary_max[1],
                                    self.stock_boundary_min[2], self.stock_boundary_max[2]])
        for b in self.calculated_cutters:
            final_mesh = final_mesh.clip_box(bounds=b, invert=True)
            
        plotter.add_mesh(final_mesh, color='orange', label="Machined Part")
        # 커터 박스들도 시각화 (선택적)
        for i, b in enumerate(self.calculated_cutters):
            if i % 5 == 0: # 너무 많으면 무거우므로 일부만 표시
                plotter.add_mesh(pv.Box(bounds=b), color='cyan', opacity=0.1, style='wireframe')
        
        plotter.add_axes()
        plotter.show()

    def refine_and_clamp_cutter(self, b):
        refined = np.array(b).copy()
        for _ in range(2):
            for i in range(6):
                axis, dir_sign = i // 2, (-1 if i % 2 == 0 else 1)
                center_pt = [(refined[0]+refined[1])/2, (refined[2]+refined[3])/2, (refined[4]+refined[5])/2]
                center_pt[axis] = refined[i]
                dist = trimesh.proximity.signed_distance(self.original_cad_mesh, [center_pt])[0]
                refined[i] += -(dist + self.tolerance) * dir_sign
        refined[0] = max(refined[0], self.stock_boundary_min[0]); refined[1] = min(refined[1], self.stock_boundary_max[0])
        refined[2] = max(refined[2], self.stock_boundary_min[1]); refined[3] = min(refined[3], self.stock_boundary_max[1])
        refined[4] = max(refined[4], self.stock_boundary_min[2]); refined[5] = min(refined[5], self.stock_boundary_max[2])
        return refined

# --- [실행 제어 코드] ---
if __name__ == "__main__":
    # 엔진 초기화 (test_box.stl 파일 사용)
    engine = MachiningSimulationEngine(target_stl_file_path="test_box.stl")
    
    # 1. 처음 원본 형상을 모달 창으로 확인
    engine.show_initial_mesh()
    
    # 2. 가공 파라미터 명시적 설정
    engine.set_parameters(
        resolution=60,         # 그리드 분할 수 (정밀도)
        min_side_mm=2.0,       # 커터의 최소 실제 길이 (mm)
        expansion=1.1,         # 소재 확장 비율 (1.1 = 10%)
        tolerance=-0.05        # 가공 여유값 (mm)
    )
    
    # 3. 가공 진행 및 결과 보고/최종 창 출력
    engine.run_simulation()
