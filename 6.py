import numpy as np
import trimesh
import pyvista as pv
from pyvista import plotter
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os
import time

# --- [1. 가속 확장 로직 (Numba)] ---
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

# --- [2. 모달리스 시뮬레이션 엔진] ---
class ModelessMachiningEngine:
    def __init__(self, target_stl_file_path):
        """
        파일을 로드하고 즉시 시각화 창을 엽니다.
        """
        if not os.path.exists(target_stl_file_path):
            trimesh.creation.annulus(r_min=10, r_max=20, height=15).export(target_stl_file_path)

        self.original_cad_mesh = trimesh.load(target_stl_file_path)
        if isinstance(self.original_cad_mesh, trimesh.Scene):
            self.original_cad_mesh = self.original_cad_mesh.dump(concatenate=True)
            
        # 기하 정보 분석
        self.original_min, self.original_max = self.original_cad_mesh.bounds
        self.dimensions = self.original_max - self.original_min
        self.max_dim = self.dimensions.max()
        self.geometric_center = (self.original_min + self.original_max) / 2.0
        
        # 1/20 기본값 계산
        self.default_resolution = 20 
        self.default_min_side = self.max_dim / 20.0
        
        print("\n" + "="*65)
        print(" [STL 분석 및 모달리스 시각화 시작]")
        print(f" 모델 크기: {self.dimensions[0]:.2f} x {self.dimensions[1]:.2f} x {self.dimensions[2]:.2f}")
        print(f" 자동 제안: Res={self.default_resolution}, MinSide={self.default_min_side:.2f}")
        print("="*65)

        # PyVista Plotter 초기화 (모달리스처럼 동작시키기 위해 show(interactive_update=True) 활용 가능)
        self.plotter = pv.Plotter(title="Machining Preview (Modeless)")
        self.plotter.add_mesh(self.original_cad_mesh, color='white', opacity=0.5, label="Original CAD")
        self.plotter.add_axes()
        self.plotter.show(interactive_update=True) # 창을 띄우고 다음 코드로 넘어감

    def configure(self, resolution=None, min_side=None, expansion=1.1, tolerance=-0.1):
        self.resolution = resolution if resolution else self.default_resolution
        self.min_cutter_size = min_side if min_side else self.default_min_side
        self.stock_expansion_ratio = expansion
        self.tolerance = tolerance
        
        # 가공 범위 설정 및 시각화 업데이트
        self.stock_boundary_min = self.geometric_center + (self.original_min - self.geometric_center) * expansion
        self.stock_boundary_max = self.geometric_center + (self.original_max - self.geometric_center) * expansion
        self.voxel_pitch = np.max(self.stock_boundary_max - self.stock_boundary_min) / self.resolution
        
        # 가공 영역(Stock) 가이드라인 추가
        stock_box = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                   self.stock_boundary_min[1], self.stock_boundary_max[1],
                                   self.stock_boundary_min[2], self.stock_boundary_max[2]])
        self.plotter.add_mesh(stock_box, style='wireframe', color='yellow', label="Stock Boundary")
        self.plotter.update() # 시각화 갱신

    def run_simulation(self):
        # 1. 그리드 분석
        dims = np.ceil((self.stock_boundary_max - self.stock_boundary_min) / self.voxel_pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.voxel_pitch + self.stock_boundary_min
        is_inside = self.original_cad_mesh.contains(pts)
        occupancy_grid = (~is_inside).reshape(dims[::-1])
        
        temp_grid = occupancy_grid.copy()
        mask = np.ones_like(temp_grid, dtype=bool)
        self.calculated_cutter_list = []

        print("\n[작업] 커터 생성 및 실시간 시각화 중...")
        
        with tqdm(total=np.sum(occupancy_grid), desc="Cutter Planning") as pbar:
            while True:
                search_area = temp_grid & mask
                dist_map = distance_transform_edt(search_area)
                if dist_map.max() < 0.5: break
                seed = np.unravel_index(np.argmax(dist_map), search_area.shape)
                z1, z2, y1, y2, x1, x2 = compute_maximum_expandable_bounding_box(temp_grid, *seed)
                
                w_min = self.stock_boundary_min + np.array([x1, y1, z1]) * self.voxel_pitch
                w_max = self.stock_boundary_min + np.array([x2+1, y2+1, z2+1]) * self.voxel_pitch
                
                refined_b = self.refine_and_clamp_cutter([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                
                if np.all(np.array([refined_b[1]-refined_b[0], refined_b[3]-refined_b[2], refined_b[5]-refined_b[4]]) >= self.min_cutter_size):
                    self.calculated_cutter_list.append(refined_b)
                    
                    # 실시간으로 창에 커터 추가 (선택 사항, 너무 많으면 느려짐)
                    if len(self.calculated_cutter_list) % 10 == 0:
                        self.plotter.add_mesh(pv.Box(bounds=refined_b), color='cyan', opacity=0.2)
                        self.plotter.update()

                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(np.sum(occupancy_grid[z1:z2+1, y1:y2+1, x1:x2+1]))
                else:
                    mask[seed] = False

        # 2. 메쉬 커팅 연산 및 최종 결과물 업데이트
        print("\n[작업] 최종 솔리드 메쉬 생성 중...")
        current_mesh = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                      self.stock_boundary_min[1], self.stock_boundary_max[1],
                                      self.stock_boundary_min[2], self.stock_boundary_max[2]])
        
        for b in tqdm(self.calculated_cutter_list, desc="Final Boolean"):
            if current_mesh.n_points == 0: break
            clipped = current_mesh.clip_box(bounds=b, invert=True)
            if clipped.n_points > 0: current_mesh = clipped
            
        self.plotter.add_mesh(current_mesh, color='orange', label="Predicted Result")
        self.plotter.update()
        print("\n[완료] 시각화 창에서 결과를 확인하세요.")
        self.plotter.show() # 마지막에 창을 유지함

    def refine_and_clamp_cutter(self, initial_bounds):
        # (이전과 동일한 정밀 밀착 및 클램핑 로직)
        refined = np.array(initial_bounds).copy()
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

# --- [메인 프로세스] ---
if __name__ == "__main__":
    # 1. 엔진 시작과 동시에 STL 시각화 창이 열립니다 (모달리스 방식)
    engine = ModelessMachiningEngine(target_stl_file_path="my_model.stl")
    
    # 창이 떠 있는 상태에서 분석이 진행됩니다.
    time.sleep(2) # 사용자가 창을 볼 시간을 잠시 줌
    
    # 2. 비례 옵션 설정 및 가이드라인 표시
    engine.configure() 
    
    # 3. 가공 시뮬레이션 실행 (결과가 실시간으로 창에 반영됨)
    engine.run_simulation()
