import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os

# --- [1. 가속 가공 영역 확장 엔진 - 경계 검사 강화] ---
@njit
def compute_maximum_expandable_bounding_box(occupancy_status_grid, start_z_idx, start_y_idx, start_x_idx):
    grid_depth, grid_height, grid_width = occupancy_status_grid.shape
    z_min, z_max = start_z_idx, start_z_idx
    y_min, y_max = start_y_idx, start_y_idx
    x_min, x_max = start_x_idx, start_x_idx
    
    while True:
        expanded_flag = False
        # X축 확장 (경계 내 인덱스만 허용)
        if x_max + 1 < grid_width and np.all(occupancy_status_grid[z_min:z_max+1, y_min:y_max+1, x_max+1]):
            x_max += 1; expanded_flag = True
        if x_min - 1 >= 0 and np.all(occupancy_status_grid[z_min:z_max+1, y_min:y_max+1, x_min-1]):
            x_min -= 1; expanded_flag = True
        # Y축 확장
        if y_max + 1 < grid_height and np.all(occupancy_status_grid[z_min:z_max+1, y_max+1, x_min:x_max+1]):
            y_max += 1; expanded_flag = True
        if y_min - 1 >= 0 and np.all(occupancy_status_grid[z_min:z_max+1, y_min-1, x_min:x_max+1]):
            y_min -= 1; expanded_flag = True
        # Z축 확장
        if z_max + 1 < grid_depth and np.all(occupancy_status_grid[z_max+1, y_min:y_max+1, x_min:x_max+1]):
            z_max += 1; expanded_flag = True
        if z_min - 1 >= 0 and np.all(occupancy_status_grid[z_min-1, y_min:y_max+1, x_min:x_max+1]):
            z_min -= 1; expanded_flag = True
        if not expanded_flag: break
    return z_min, z_max, y_min, y_max, x_min, x_max

# --- [2. 가공 예측 및 경계 통제 엔진] ---
class MachiningProcessVisualPredictor:
    def __init__(self, target_stl_file_path, analytical_resolution=80, 
                 minimum_cutter_dimension=2.0, stock_expansion_ratio=1.1, 
                 cutting_penetration_depth=-0.1):
        
        self.original_cad_mesh = trimesh.load(target_stl_file_path)
        if isinstance(self.original_cad_mesh, trimesh.Scene):
            self.original_cad_mesh = self.original_cad_mesh.dump(concatenate=True)
            
        self.original_min_coords, self.original_max_coords = self.original_cad_mesh.bounds
        
        # 가공 소재(Stock) 범위 설정
        geometric_center = (self.original_min_coords + self.original_max_coords) / 2.0
        self.stock_boundary_min = geometric_center + (self.original_min_coords - geometric_center) * stock_expansion_ratio
        self.stock_boundary_max = geometric_center + (self.original_max_coords - geometric_center) * stock_expansion_ratio
        
        print(f"\n[통제된 가공 범위]")
        print(f"X: {self.stock_boundary_min[0]:.2f} ~ {self.stock_boundary_max[0]:.2f}")
        print(f"Y: {self.stock_boundary_min[1]:.2f} ~ {self.stock_boundary_max[1]:.2f}")
        print(f"Z: {self.stock_boundary_min[2]:.2f} ~ {self.stock_boundary_max[2]:.2f}\n")

        self.resolution = analytical_resolution
        self.min_cutter_size = minimum_cutter_dimension
        self.tolerance = cutting_penetration_depth
        self.voxel_pitch = np.max(self.stock_boundary_max - self.stock_boundary_min) / self.resolution
        
        self.calculated_cutter_list = []
        self.final_predicted_solid_mesh = None

    def refine_and_clamp_cutter(self, initial_bounds):
        """표면 밀착 후, 가공 범위를 절대 넘지 않도록 강제 제한(Clamping) 합니다."""
        refined = np.array(initial_bounds).copy()
        
        # 1. 표면 밀착 연산
        for _ in range(2):
            for i in range(6):
                axis, dir_sign = i // 2, (-1 if i % 2 == 0 else 1)
                center_pt = [(refined[0]+refined[1])/2, (refined[2]+refined[3])/2, (refined[4]+refined[5])/2]
                center_pt[axis] = refined[i]
                dist = trimesh.proximity.signed_distance(self.original_cad_mesh, [center_pt])[0]
                refined[i] += -(dist + self.tolerance) * dir_sign

        # 2. [핵심] 경계 클램핑 (Hard Constraints)
        # x_min, x_max, y_min, y_max, z_min, z_max 순서임에 주의
        refined[0] = max(refined[0], self.stock_boundary_min[0]) # x_min
        refined[1] = min(refined[1], self.stock_boundary_max[0]) # x_max
        refined[2] = max(refined[2], self.stock_boundary_min[1]) # y_min
        refined[3] = min(refined[3], self.stock_boundary_max[1]) # y_max
        refined[4] = max(refined[4], self.stock_boundary_min[2]) # z_min
        refined[5] = min(refined[5], self.stock_boundary_max[2]) # z_max
        
        return refined

    def execute_machining_analysis(self):
        dims = np.ceil((self.stock_boundary_max - self.stock_boundary_min) / self.voxel_pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.voxel_pitch + self.stock_boundary_min
        
        is_inside = self.original_cad_mesh.contains(pts)
        occupancy_grid = (~is_inside).reshape(dims[::-1])
        
        temp_grid = occupancy_grid.copy()
        mask = np.ones_like(temp_grid, dtype=bool)
        
        with tqdm(total=np.sum(occupancy_grid), desc="커터 생성 중") as pbar:
            while True:
                search_area = temp_grid & mask
                dist_map = distance_transform_edt(search_area)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search_area.shape)
                z1, z2, y1, y2, x1, x2 = compute_maximum_expandable_bounding_box(temp_grid, *seed)
                
                # 기초 월드 좌표
                w_min = self.stock_boundary_min + np.array([x1, y1, z1]) * self.voxel_pitch
                w_max = self.stock_boundary_min + np.array([x2+1, y2+1, z2+1]) * self.voxel_pitch
                
                # 개선된 클램핑 로직 적용
                refined_b = self.refine_and_clamp_cutter([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                
                # 유효 크기 확인
                if np.all(np.array([refined_b[1]-refined_b[0], refined_b[3]-refined_b[2], refined_b[5]-refined_b[4]]) >= self.min_cutter_size):
                    self.calculated_cutter_list.append(refined_b)
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(np.sum(occupancy_grid[z1:z2+1, y1:y2+1, x1:x2+1]))
                else:
                    mask[seed] = False

    def generate_final_prediction_mesh(self):
        print(f"[로그] 최종 예측 메쉬 생성 중...")
        # PyVista Box의 bounds 순서는 [x_min, x_max, y_min, y_max, z_min, z_max]
        current_mesh = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                      self.stock_boundary_min[1], self.stock_boundary_max[1],
                                      self.stock_boundary_min[2], self.stock_boundary_max[2]])
        
        for b in tqdm(self.calculated_cutter_list, desc="메쉬 커팅 연산"):
            if current_mesh.n_points == 0: break
            # clip_box는 지정된 박스 '밖'을 남깁니다 (invert=True)
            clipped = current_mesh.clip_box(bounds=b, invert=True)
            if clipped.n_points > 0:
                current_mesh = clipped
        
        self.final_predicted_solid_mesh = current_mesh

    def show_visualization_report(self):
        plotter = pv.Plotter(shape=(1, 2))
        
        # 가공 범위 경계 상자 시각화
        box_bounds = [self.stock_boundary_min[0], self.stock_boundary_max[0],
                      self.stock_boundary_min[1], self.stock_boundary_max[1],
                      self.stock_boundary_min[2], self.stock_boundary_max[2]]
        
        plotter.subplot(0, 0)
        plotter.add_mesh(pv.Box(bounds=box_bounds), style='wireframe', color='yellow')
        if self.calculated_cutter_list:
            cutter_mb = pv.MultiBlock([pv.Box(bounds=b) for b in self.calculated_cutter_list])
            plotter.add_mesh(cutter_mb, color='cyan', opacity=0.3, show_edges=True)
        
        plotter.subplot(0, 1)
        if self.final_predicted_solid_mesh:
            plotter.add_mesh(self.final_predicted_solid_mesh, color='orange', smooth_shading=True)
        plotter.add_mesh(self.original_cad_mesh, color='white', opacity=0.3)
        plotter.link_views()
        plotter.show()

if __name__ == "__main__":
    target_file = "sample.stl"
    if not os.path.exists(target_file):
        trimesh.creation.annulus(r_min=10, r_max=20, height=15).export(target_file)

    engine = MachiningProcessVisualPredictor(
        target_stl_file_path=target_file,
        analytical_resolution=60, 
        minimum_cutter_dimension=2.0,
        stock_expansion_ratio=1.1,
        cutting_penetration_depth=-0.2
    )
    
    engine.execute_machining_analysis()
    engine.generate_final_prediction_mesh()
    engine.show_visualization_report()
