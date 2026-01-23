import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os
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

# --- [2. 상세 리포팅 및 시각화 강화 엔진] ---
class DetailedMachiningEngine:
    def __init__(self, target_stl_file_path):
        if not os.path.exists(target_stl_file_path):
            trimesh.creation.annulus(r_min=10, r_max=20, height=15).export(target_stl_file_path)

        self.original_cad_mesh = trimesh.load(target_stl_file_path)
        if isinstance(self.original_cad_mesh, trimesh.Scene):
            self.original_cad_mesh = self.original_cad_mesh.dump(concatenate=True)
            
        self.original_min, self.original_max = self.original_cad_mesh.bounds
        self.dimensions = self.original_max - self.original_min
        self.cad_volume = self.original_cad_mesh.volume
        
        print("\n" + "="*60)
        print(f" [1. STL 로드 완료 및 분석]")
        print(f" - 경계상자: X({self.original_min[0]:.2f}~{self.original_max[0]:.2f}), "
              f"Y({self.original_min[1]:.2f}~{self.original_max[1]:.2f}), "
              f"Z({self.original_min[2]:.2f}~{self.original_max[2]:.2f})")
        print(f" - CAD 순수 체적: {self.cad_volume:.4f} mm³")
        print("="*60)

        # 모달리스 플로터 설정
        self.plotter = pv.Plotter(title="Real-time Machining Visualization")
        self.plotter.add_mesh(self.original_cad_mesh, color='white', opacity=0.2, label="Original CAD")
        self.plotter.add_axes()
        self.plotter.show(interactive_update=True)

    def set_parameters(self, resolution, min_side_mm, expansion=1.1, tolerance=-0.1):
        self.resolution = resolution
        self.min_side = min_side_mm # 실제 거리(mm)
        self.stock_expansion_ratio = expansion
        self.tolerance = tolerance
        
        self.geometric_center = (self.original_min + self.original_max) / 2.0
        self.stock_boundary_min = self.geometric_center + (self.original_min - self.geometric_center) * expansion
        self.stock_boundary_max = self.geometric_center + (self.original_max - self.geometric_center) * expansion
        self.voxel_pitch = np.max(self.stock_boundary_max - self.stock_boundary_min) / self.resolution
        
        # 가공 전 소재 체적 계산
        self.stock_volume = np.prod(self.stock_boundary_max - self.stock_boundary_min)
        
        stock_box = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                   self.stock_boundary_min[1], self.stock_boundary_max[1],
                                   self.stock_boundary_min[2], self.stock_boundary_max[2]])
        self.plotter.add_mesh(stock_box, style='wireframe', color='yellow', label="Stock")
        self.plotter.update()

    def run_simulation(self):
        # 그리드 초기화
        dims = np.ceil((self.stock_boundary_max - self.stock_boundary_min) / self.voxel_pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.voxel_pitch + self.stock_boundary_min
        is_inside = self.original_cad_mesh.contains(pts)
        occupancy_grid = (~is_inside).reshape(dims[::-1])
        
        target_machining_count = np.sum(occupancy_grid)
        temp_grid = occupancy_grid.copy()
        mask = np.ones_like(temp_grid, dtype=bool)
        self.calculated_cutters = []

        print("\n[2. 가공 시뮬레이션 시작...]")
        with tqdm(total=target_machining_count, desc="가공 진행율") as pbar:
            while True:
                search_area = temp_grid & mask
                dist_map = distance_transform_edt(search_area)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search_area.shape)
                z1, z2, y1, y2, x1, x2 = compute_maximum_expandable_bounding_box(temp_grid, *seed)
                
                w_min = self.stock_boundary_min + np.array([x1, y1, z1]) * self.voxel_pitch
                w_max = self.stock_boundary_min + np.array([x2+1, y2+1, z2+1]) * self.voxel_pitch
                
                refined_b = self.refine_and_clamp_cutter([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                
                # 최소 크기(mm) 검증
                c_size = np.array([refined_b[1]-refined_b[0], refined_b[3]-refined_b[2], refined_b[5]-refined_b[4]])
                if np.all(c_size >= self.min_side):
                    self.calculated_cutters.append(refined_b)
                    
                    # [시각화 강화] 커터를 화면에 즉시 추가 (하늘색 박스)
                    cutter_mesh = pv.Box(bounds=refined_b)
                    self.plotter.add_mesh(cutter_mesh, color='cyan', opacity=0.3, show_edges=True, name=f"c_{len(self.calculated_cutters)}")
                    self.plotter.update() # 실시간 렌더링 강제 업데이트

                    prev_count = np.sum(temp_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(prev_count)
                else:
                    mask[seed] = False

        self.generate_report(target_machining_count, temp_grid)

    def generate_report(self, total_voxels, remaining_grid):
        """가공 상세 리포트 출력"""
        processed_voxels = total_voxels - np.sum(remaining_grid)
        accuracy_percent = (processed_voxels / total_voxels) * 100
        
        # 실제 깎아낸 총 체적 근사치 (커터들의 합산 체적)
        total_cut_volume = 0
        for b in self.calculated_cutters:
            total_cut_volume += (b[1]-b[0]) * (b[3]-b[2]) * (b[5]-b[4])

        print("\n" + "="*60)
        print(" [3. 가공 결과 상세 리포트]")
        print(f" - 생성된 커팅 박스 총 개수: {len(self.calculated_cutters)} 개")
        print(f" - 목표 대비 가공 정밀도(Voxel): {accuracy_percent:.2f} %")
        print(f" - 소재 전체 체적: {self.stock_volume:.2f} mm³")
        print(f" - 제거된 총 체적(추정): {total_cut_volume:.2f} mm³")
        print(f" - 남은 예상 체적: {self.stock_volume - total_cut_volume:.2f} mm³")
        print(f" - CAD 원본 체적 대비 오차율: {abs(self.cad_volume - (self.stock_volume - total_cut_volume))/self.cad_volume*100:.2f} %")
        print("="*60)
        
        # 최종 메쉬 시각화
        self.finalize_visualization()

    def finalize_visualization(self):
        print("\n[4. 최종 불리언 메쉬 생성 중...]")
        current_mesh = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                      self.stock_boundary_min[1], self.stock_boundary_max[1],
                                      self.stock_boundary_min[2], self.stock_boundary_max[2]])
        for b in self.calculated_cutters:
            clipped = current_mesh.clip_box(bounds=b, invert=True)
            if clipped.n_points > 0: current_mesh = clipped
            
        self.plotter.add_mesh(current_mesh, color='orange', label="Final Part")
        self.plotter.update()
        print("[알림] 시각화 창에서 결과를 확인하세요.")
        self.plotter.show()

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

if __name__ == "__main__":
    engine = DetailedMachiningEngine("model.stl")
    # resolution을 높이고 min_side를 작게 줄수록 커터 개수가 늘어나며 정밀도가 상승합니다.
    engine.set_parameters(resolution=70, min_side_mm=1.0) 
    engine.run_simulation()
