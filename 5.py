import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os

# --- [1. 가속 가공 영역 확장 엔진] ---
@njit
def compute_maximum_expandable_bounding_box(occupancy_status_grid, start_z_idx, start_y_idx, start_x_idx):
    """
    격자 내에서 시작점으로부터 육면체를 확장하며 최대 가공 가능한 블록 크기를 계산합니다.
    """
    grid_depth, grid_height, grid_width = occupancy_status_grid.shape
    z_min, z_max = start_z_idx, start_z_idx
    y_min, y_max = start_y_idx, start_y_idx
    x_min, x_max = start_x_idx, start_x_idx
    
    while True:
        expanded_flag = False
        # X축 확장
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
            
        if not expanded_flag:
            break
    return z_min, z_max, y_min, y_max, x_min, x_max

# --- [2. 가공 예측 및 경계 좌표 출력 엔진] ---
class MachiningProcessVisualPredictor:
    def __init__(self, target_stl_file_path, analytical_resolution=80, 
                 minimum_cutter_dimension=2.0, stock_expansion_ratio=1.1, 
                 cutting_penetration_depth=-0.1):
        
        # 1. 원본 메쉬 로드
        self.original_cad_mesh = trimesh.load(target_stl_file_path)
        if isinstance(self.original_cad_mesh, trimesh.Scene):
            self.original_cad_mesh = self.original_cad_mesh.dump(concatenate=True)
            
        # ---------------------------------------------------------
        # [요청 기능] 원본 형상의 경계상자 좌표 출력
        # ---------------------------------------------------------
        self.original_min_coords, self.original_max_coords = self.original_cad_mesh.bounds
        print("\n" + "="*50)
        print("[원본 형상 경계상자 좌표 (Original Bounding Box)]")
        print(f"X-Axis: {self.original_min_coords[0]:.4f} to {self.original_max_coords[0]:.4f}")
        print(f"Y-Axis: {self.original_min_coords[1]:.4f} to {self.original_max_coords[1]:.4f}")
        print(f"Z-Axis: {self.original_min_coords[2]:.4f} to {self.original_max_coords[2]:.4f}")
        print("="*50 + "\n")

        # 2. 가공 소재(Stock) 범위 설정
        geometric_center = (self.original_min_coords + self.original_max_coords) / 2.0
        self.stock_boundary_min = geometric_center + (self.original_min_coords - geometric_center) * stock_expansion_ratio
        self.stock_boundary_max = geometric_center + (self.original_max_coords - geometric_center) * stock_expansion_ratio
        
        # 3. 속성 초기화
        self.resolution = analytical_resolution
        self.min_cutter_size = minimum_cutter_dimension
        self.tolerance = cutting_penetration_depth
        self.voxel_pitch = np.max(self.stock_boundary_max - self.stock_boundary_min) / self.resolution
        
        self.calculated_cutter_list = []
        self.final_predicted_solid_mesh = None

    def refine_cutter_to_surface(self, initial_bounds):
        """커터의 각 면을 원본 표면과 톨러런스에 맞춰 정밀 조정합니다."""
        refined = np.array(initial_bounds).copy()
        for _ in range(2):
            for i in range(6):
                axis = i // 2
                dir_sign = -1 if i % 2 == 0 else 1
                center_pt = [(refined[0]+refined[1])/2, (refined[2]+refined[3])/2, (refined[4]+refined[5])/2]
                center_pt[axis] = refined[i]
                dist = trimesh.proximity.signed_distance(self.original_cad_mesh, [center_pt])[0]
                # tolerance 반영 (음수 시 침투)
                refined[i] += -(dist + self.tolerance) * dir_sign
        return refined

    def execute_machining_analysis(self):
        """그리드 기반 체적 분석을 수행하여 커터 목록을 생성합니다."""
        dims = np.ceil((self.stock_boundary_max - self.stock_boundary_min) / self.voxel_pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.voxel_pitch + self.stock_boundary_min
        
        # 가공해야 할 영역(메쉬 외부) 식별
        is_inside = self.original_cad_mesh.contains(pts)
        occupancy_grid = (~is_inside).reshape(dims[::-1])
        
        temp_grid = occupancy_grid.copy()
        mask = np.ones_like(temp_grid, dtype=bool)
        
        print(f"[정보] {self.resolution} 해상도로 가공 경로를 계산 중입니다...")
        
        with tqdm(total=np.sum(occupancy_grid), desc="가공 체적 계산") as pbar:
            while True:
                search_area = temp_grid & mask
                dist_map = distance_transform_edt(search_area)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search_area.shape)
                z1, z2, y1, y2, x1, x2 = compute_maximum_expandable_bounding_box(temp_grid, *seed)
                
                w_min = self.stock_boundary_min + np.array([x1, y1, z1]) * self.voxel_pitch
                w_max = self.stock_boundary_min + np.array([x2+1, y2+1, z2+1]) * self.voxel_pitch
                
                refined_b = self.refine_cutter_to_surface([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                
                # 최소 크기 검증
                if np.all(np.array([refined_b[1]-refined_b[0], refined_b[3]-refined_b[2], refined_b[5]-refined_b[4]]) >= self.min_cutter_size):
                    self.calculated_cutter_list.append(refined_b)
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(np.sum(occupancy_grid[z1:z2+1, y1:y2+1, x1:x2+1]))
                else:
                    mask[seed] = False

    def generate_final_prediction_mesh(self):
        """
        [핵심] 소재 박스에서 커터들을 순차적으로 빼서 최종 형상을 예측합니다.
        PyVista의 clip_box(invert=True)는 메쉬 출력을 보장하는 가장 안정적인 방법입니다.
        """
        print(f"[로그] 최종 예측 메쉬 생성 중... (커터 수: {len(self.calculated_cutter_list)}개)")
        
        # 1. 기초 소재(Stock) 생성 (PyVista Box)
        stock_poly = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                    self.stock_boundary_min[1], self.stock_boundary_max[1],
                                    self.stock_boundary_min[2], self.stock_boundary_max[2]])
        
        # 2. 커터들을 사용하여 소재를 깎아냄 (Clipping)
        # 이 과정은 실제 솔리드 결과물(PolyData)을 반환합니다.
        current_mesh = stock_poly
        for b in tqdm(self.calculated_cutter_list, desc="메쉬 커팅 시뮬레이션"):
            current_mesh = current_mesh.clip_box(bounds=b, invert=True)
            
        self.final_predicted_solid_mesh = current_mesh

    def show_visualization_report(self):
        """결과 형상 메쉬를 포함하여 시각화합니다."""
        plotter = pv.Plotter(shape=(1, 2))
        
        # 왼쪽: 가공 소재와 배치된 커터들
        plotter.subplot(0, 0)
        plotter.add_text("Machining Process (Stock & Cutters)", font_size=10)
        plotter.add_mesh(pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                        self.stock_boundary_min[1], self.stock_boundary_max[1],
                                        self.stock_boundary_min[2], self.stock_boundary_max[2]]), 
                         style='wireframe', color='yellow')
        
        cutter_multi = pv.MultiBlock([pv.Box(bounds=b) for b in self.calculated_cutter_list])
        plotter.add_mesh(cutter_multi, color='cyan', opacity=0.3, show_edges=True)
        plotter.add_mesh(self.original_cad_mesh, color='white', opacity=0.1)

        # 오른쪽: 실제 결과 형상 메쉬 출력
        plotter.subplot(0, 1)
        plotter.add_text("Final Machined Prediction Solid", font_size=10)
        if self.final_predicted_solid_mesh:
            # 최종적으로 깎인 메쉬를 출력합니다.
            plotter.add_mesh(self.final_predicted_solid_mesh, color='orange', show_edges=False)
        
        plotter.add_mesh(self.original_cad_mesh, color='white', opacity=0.3)
        plotter.link_views()
        plotter.show()

# --- [메인 실행부] ---
if __name__ == "__main__":
    target_file = "sample_part.stl"
    if not os.path.exists(target_file):
        trimesh.creation.annulus(r_min=10, r_max=20, height=15).export(target_file)

    # 엔진 구동
    engine = MachiningProcessVisualPredictor(
        target_stl_file_path=target_file,
        analytical_resolution=60,       # 결과 메쉬 생성을 위해 적절한 값 설정
        minimum_cutter_dimension=2.5,
        stock_expansion_ratio=1.1,      # 110% 가공 범위
        cutting_penetration_depth=-0.2  # 0.2mm 침투 가공
    )
    
    engine.execute_machining_analysis()
    engine.generate_final_prediction_mesh()
    engine.show_visualization_report()
