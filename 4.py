import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os
import time

# --- 1. 가속 확장 로직 (Numba) ---
@njit
def grow_box_fast(grid, start_z, start_y, start_x):
    d, h, w = grid.shape
    z1, z2, y1, y2, x1, x2 = start_z, start_z, start_y, start_y, start_x, start_x
    while True:
        expanded = False
        if x2 + 1 < w and np.all(grid[z1:z2+1, y1:y2+1, x2+1]): x2 += 1; expanded = True
        if x1 - 1 >= 0 and np.all(grid[z1:z2+1, y1:y2+1, x1-1]): x1 -= 1; expanded = True
        if y2 + 1 < h and np.all(grid[z1:z2+1, y2+1, x1:x2+1]): y2 += 1; expanded = True
        if y1 - 1 >= 0 and np.all(grid[z1:z2+1, y1-1, x1:x2+1]): y1 -= 1; expanded = True
        if z2 + 1 < d and np.all(grid[z2+1, y1:y2+1, x1:x2+1]): z2 += 1; expanded = True
        if z1 - 1 >= 0 and np.all(grid[z1-1, y1:y2+1, x1:x2+1]): z1 -= 1; expanded = True
        if not expanded: break
    return z1, z2, y1, y2, x1, x2

# --- 2. 통합 제어 엔진 ---
class UltimateCutterEngine:
    def __init__(self, mesh_path, resolution=100, min_side=2.5, boundary_scale=1.1, tolerance=0.0):
        self.mesh = trimesh.load(mesh_path)
        self.mesh.fix_normals()
        
        # 가공 범위 설정 (110%)
        center = self.mesh.bounds.mean(axis=0)
        self.work_min = center + (self.mesh.bounds[0] - center) * boundary_scale
        self.work_max = center + (self.mesh.bounds[1] - center) * boundary_scale
        
        self.res = resolution
        self.min_side = min_side
        self.tol = tolerance  # 음수: 침투(더 깎음), 양수: 오프셋(살 남김)
        self.pitch = np.max(self.work_max - self.work_min) / self.res
        self.origin = self.work_min
        
        self.cutters = []
        self.final_grid = None

    def snap_to_surface(self, bounds):
        """[핵심] 박스의 6개 면을 표면으로부터 tolerance 만큼 정밀 이동"""
        refined = np.array(bounds).copy()
        # x_min, x_max, y_min, y_max, z_min, z_max 순서
        for _ in range(2): # 2회 반복으로 정밀도 향상
            for i in range(6):
                axis = i // 2
                direction = -1 if i % 2 == 0 else 1
                
                # 면의 중심 샘플링
                c = [(refined[0]+refined[1])/2, (refined[2]+refined[3])/2, (refined[4]+refined[5])/2]
                c[axis] = refined[i]
                
                # 원본과의 거리 측정
                dist = trimesh.proximity.signed_distance(self.mesh, [c])[0]
                
                # 톨러런스를 적용하여 면 이동
                # tol이 -0.1이면 실제 표면보다 0.1mm 더 안쪽으로 이동(침투)
                move = -(dist + self.tol) * direction
                refined[i] += move
        return refined

    def get_rms_error(self, current_grid):
        z, y, x = np.where(~current_grid)
        if len(x) == 0: return 0.0
        pts = np.column_stack([x, y, z]) * self.pitch + self.origin
        dists = trimesh.proximity.closest_point(self.mesh, pts)[1]
        return np.sqrt(np.mean(dists**2))

    def run(self):
        # 그리드 초기화
        dims = np.ceil((self.work_max - self.work_min) / self.pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.pitch + self.origin
        grid = (~self.mesh.contains(pts)).reshape(dims[::-1])
        
        temp_grid = grid.copy()
        skipping_mask = np.ones_like(temp_grid, dtype=bool)
        total_air = np.sum(grid)
        
        print(f"\n[작업 시작] 톨러런스: {self.tol}mm (음수:침투, 양수:오프셋)")
        print("-" * 70)

        with tqdm(total=total_air, desc="가공 진행률", unit="vx") as pbar:
            while True:
                search = temp_grid & skipping_mask
                dist_map = distance_transform_edt(search)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search.shape)
                z1, z2, y1, y2, x1, x2 = grow_box_fast(temp_grid, *seed)
                
                # 1. 초기 월드 좌표 바운즈 생성
                w_min = self.origin + np.array([x1, y1, z1]) * self.pitch
                w_max = self.origin + np.array([x2+1, y2+1, z2+1]) * self.pitch
                initial_bounds = [w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]]
                
                # 2. 표면 정밀 밀착 및 톨러런스 적용 (침투/오프셋)
                refined_bounds = self.snap_to_surface(initial_bounds)
                
                # 3. 크기 제약 검사
                size = np.array([refined_bounds[1]-refined_bounds[0], 
                                 refined_bounds[3]-refined_bounds[2], 
                                 refined_bounds[5]-refined_bounds[4]])
                
                if np.all(size >= self.min_side):
                    self.cutters.append(refined_bounds)
                    removed = np.sum(temp_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(removed)
                    
                    if len(self.cutters) % 50 == 0:
                        rms = self.get_rms_error(temp_grid)
                        tqdm.write(f" > Block #{len(self.cutters):<4} | 오차: {rms:.4f}mm")
                else:
                    skipping_mask[seed] = False

        self.final_grid = temp_grid

    def visualize(self, mesh_path):
        p = pv.Plotter(shape=(1, 2))
        p.subplot(0, 0)
        p.add_text(f"Cutters (Tol: {self.tol}mm)")
        boxes = pv.MultiBlock([pv.Box(bounds=b) for b in self.cutters])
        p.add_mesh(boxes, color='cyan', show_edges=True, opacity=0.5)
        p.add_mesh(self.mesh, color='white', opacity=0.2)

        p.subplot(0, 1)
        p.add_text("Final Machined Shape")
        z, y, x = np.where(~self.final_grid)
        pts = np.column_stack([x, y, z]) * self.pitch + self.origin
        if len(pts) > 0:
            glyphs = pv.PolyData(pts).glyph(geom=pv.Cube(x_length=self.pitch, y_length=self.pitch, z_length=self.pitch))
            p.add_mesh(glyphs, color='orange')
        p.add_mesh(self.mesh, color='white', opacity=0.3)
        p.link_views(); p.show()

# --- 실행부 ---
if __name__ == "__main__":
    f_path = 'model.stl'
    if not os.path.exists(f_path):
        trimesh.creation.annulus(r_min=10, r_max=25, height=15).export(f_path)

    # 설정 예시:
    # tolerance = -0.15  => 실제 면보다 0.15mm 더 파고듬 (침투)
    # tolerance = 0.5    => 실제 면보다 0.5mm 덜 깎음 (오프셋/살 남기기)
    engine = UltimateCutterEngine(mesh_path=f_path, resolution=80, min_side=2.0, tolerance=-0.1)
    engine.run()
    engine.visualize(f_path)
