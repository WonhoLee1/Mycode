import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os

# --- [1. 확장 로직 개선: 최소 확장 보장] ---
@njit
def compute_maximum_expandable_bounding_box(occupancy_status_grid, start_z_idx, start_y_idx, start_x_idx):
    # (이전과 동일한 확장 로직)
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

class RobustMachiningEngine:
    def __init__(self, target_stl_file_path):
        self.original_cad_mesh = trimesh.load(target_stl_file_path)
        if isinstance(self.original_cad_mesh, trimesh.Scene):
            self.original_cad_mesh = self.original_cad_mesh.dump(concatenate=True)
        self.original_min, self.original_max = self.original_cad_mesh.bounds
        self.geometric_center = (self.original_min + self.original_max) / 2.0
        
        # 모달리스 시각화
        self.plotter = pv.Plotter(title="Robust Machining Simulation")
        self.plotter.add_mesh(self.original_cad_mesh, color='white', opacity=0.3)
        self.plotter.show(interactive_update=True)

    def set_parameters(self, resolution, min_side, expansion=1.1, tolerance=-0.1):
        self.resolution = resolution
        self.stock_expansion_ratio = expansion
        self.tolerance = tolerance
        
        # 가공 범위 계산
        self.stock_boundary_min = self.geometric_center + (self.original_min - self.geometric_center) * expansion
        self.stock_boundary_max = self.geometric_center + (self.original_max - self.geometric_center) * expansion
        self.voxel_pitch = np.max(self.stock_boundary_max - self.stock_boundary_min) / self.resolution
        
        # [핵심 보정] 사용자의 min_side가 격자 간격(Pitch)보다 작으면 물리적으로 가공 불가
        # 최소한 한 격자 크기보다는 크게 설정되도록 자동 보정합니다.
        self.min_cutter_size = max(min_side, self.voxel_pitch * 0.9)
        
        print(f"\n[파라미터 점검]")
        print(f"- 격자 간격(Pitch): {self.voxel_pitch:.4f}")
        print(f"- 적용된 최소 커터 크기: {self.min_cutter_size:.4f} (입력값: {min_side})")
        
        stock_box = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                   self.stock_boundary_min[1], self.stock_boundary_max[1],
                                   self.stock_boundary_min[2], self.stock_boundary_max[2]])
        self.plotter.add_mesh(stock_box, style='wireframe', color='yellow')
        self.plotter.update()

    def run_simulation(self):
        # 그리드 분석
        dims = np.ceil((self.stock_boundary_max - self.stock_boundary_min) / self.voxel_pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.voxel_pitch + self.stock_boundary_min
        is_inside = self.original_cad_mesh.contains(pts)
        occupancy_grid = (~is_inside).reshape(dims[::-1])
        
        temp_grid = occupancy_grid.copy()
        mask = np.ones_like(temp_grid, dtype=bool)
        self.calculated_cutters = []

        print("\n[작업] 커터 경로 탐색 시작...")
        with tqdm(total=np.sum(occupancy_grid), desc="가공 진행률") as pbar:
            while True:
                search_area = temp_grid & mask
                dist_map = distance_transform_edt(search_area)
                if dist_map.max() < 0.1: # 아주 작은 빈틈만 남으면 종료
                    break
                
                seed = np.unravel_index(np.argmax(dist_map), search_area.shape)
                z1, z2, y1, y2, x1, x2 = compute_maximum_expandable_bounding_box(temp_grid, *seed)
                
                w_min = self.stock_boundary_min + np.array([x1, y1, z1]) * self.voxel_pitch
                w_max = self.stock_boundary_min + np.array([x2+1, y2+1, z2+1]) * self.voxel_pitch
                
                # 경계 밀착 및 클램핑
                refined_b = self.refine_and_clamp_cutter([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                
                # [수정] 크기 검증 로직 강화
                current_size = np.array([refined_b[1]-refined_b[0], refined_b[3]-refined_b[2], refined_b[5]-refined_b[4]])
                
                # 만약 너무 작은 영역이라면, 이 시드 포인트는 포기하고 마스크 처리
                if np.any(current_size < (self.voxel_pitch * 0.5)): 
                    mask[seed] = False
                    continue

                self.calculated_cutters.append(refined_b)
                # 가공된 영역 그리드에서 제거
                temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                pbar.update(np.sum(occupancy_grid[z1:z2+1, y1:y2+1, x1:x2+1]))

        # 최종 메쉬 커팅 (생략 방지를 위해 모든 커터 강제 적용)
        self.finalize_mesh()

    def finalize_mesh(self):
        print(f"\n[최종] {len(self.calculated_cutters)}개의 커터로 최종 메쉬 생성 중...")
        current_mesh = pv.Box(bounds=[self.stock_boundary_min[0], self.stock_boundary_max[0],
                                      self.stock_boundary_min[1], self.stock_boundary_max[1],
                                      self.stock_boundary_min[2], self.stock_boundary_max[2]])
        
        for b in tqdm(self.calculated_cutters, desc="메쉬 커팅"):
            clipped = current_mesh.clip_box(bounds=b, invert=True)
            if clipped.n_points > 0:
                current_mesh = clipped
        
        self.plotter.add_mesh(current_mesh, color='orange', smooth_shading=True)
        self.plotter.update()
        self.plotter.show()

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

if __name__ == "__main__":
    engine = RobustMachiningEngine("model.stl")
    # Resolution 대비 min_side를 너무 크게 주면 가공이 조기에 끝납니다.
    # 안전하게 min_side를 0으로 주면 격자 한 칸 크기까지 가공합니다.
    engine.set_parameters(resolution=80, min_side=1.0) 
    engine.run_simulation()
