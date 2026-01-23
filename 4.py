import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os

# --- 1. 가속 탐색: 박스 확장 ---
@njit
def grow_box_refined(grid, start_z, start_y, start_x):
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

# --- 2. 통합 엔진 ---
class AdvancedCutterEngine:
    def __init__(self, mesh_path, resolution=100, min_side=1.5, boundary_scale=1.1):
        self.mesh = trimesh.load(mesh_path)
        self.mesh.fix_normals()
        
        # [원본 경계 설정]
        self.b_min, self.b_max = self.mesh.bounds
        center = (self.b_min + self.b_max) / 2
        # 사용자가 요청한 110% 가공 영역 제한
        self.work_min = center + (self.b_min - center) * boundary_scale
        self.work_max = center + (self.b_max - center) * boundary_scale
        
        self.res = resolution
        self.min_side = min_side
        self.pitch = np.max(self.work_max - self.work_min) / self.res
        self.origin = self.work_min
        
        self.cutters = [] # [xmin, xmax, ymin, ymax, zmin, zmax] list

    def merge_redundant_cutters(self):
        """커터 내부에 완전히 포함된 작은 커터를 제거하여 최적화"""
        if not self.cutters: return
        
        keep = []
        # 크기순 정렬 (큰 것부터 비교)
        sorted_cutters = sorted(self.cutters, key=lambda b: (b[1]-b[0])*(b[3]-b[2])*(b[5]-b[4]), reverse=True)
        
        for i, c1 in enumerate(sorted_cutters):
            is_redundant = False
            for j, c2 in enumerate(keep):
                # c1이 c2에 완전히 포함되는지 체크
                if (c1[0] >= c2[0] and c1[1] <= c2[1] and 
                    c1[2] >= c2[2] and c1[3] <= c2[3] and 
                    c1[4] >= c2[4] and c1[5] <= c2[5]):
                    is_redundant = True
                    break
            if not is_redundant:
                keep.append(c1)
        self.cutters = keep

    def run(self):
        # 1. 그리드 생성 (110% 작업 영역 기준)
        dims = np.ceil((self.work_max - self.work_min) / self.pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.pitch + self.origin
        
        # 2. 볼륨 판정
        inside = self.mesh.contains(pts)
        grid = (~inside).reshape(dims[::-1])
        temp_grid = grid.copy()
        skipping_mask = np.ones_like(temp_grid, dtype=bool)
        
        print(f"\n[가공 범위 제한 모드] Scale: {1.1}, Resolution: {self.res}")
        with tqdm(total=np.sum(grid), desc="Generating Cutters") as pbar:
            while True:
                search = temp_grid & skipping_mask
                dist_map = distance_transform_edt(search)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search.shape)
                z1, z2, y1, y2, x1, x2 = grow_box_refined(temp_grid, *seed)
                
                # 월드 좌표 변환
                w_min = self.origin + np.array([x1, y1, z1]) * self.pitch
                w_max = self.origin + np.array([x2+1, y2+1, z2+1]) * self.pitch
                
                # 원본 표면 밀착 (Snapping)
                bounds = [w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]]
                # (생략: 이전의 snap_to_surface 로직 적용 가능)
                
                size = w_max - w_min
                if np.all(size >= self.min_side):
                    self.cutters.append(bounds)
                    removed = np.sum(temp_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(removed)
                else:
                    skipping_mask[seed] = False
        
        # 3. 커터 통합 로직 실행
        before_count = len(self.cutters)
        self.merge_redundant_cutters()
        print(f"통합 완료: {before_count} -> {len(self.cutters)} blocks")
        self.final_grid = temp_grid

    def visualize(self, mesh_path):
        p = pv.Plotter(shape=(1, 2))
        p.subplot(0, 0)
        p.add_text("Optimized Cutter Blocks", font_size=12)
        cutter_mb = pv.MultiBlock()
        for i, b in enumerate(self.cutters):
            box = pv.Box(bounds=b)
            box.cell_data["ID"] = np.full(box.n_cells, i)
            cutter_mb.append(box)
        p.add_mesh(cutter_mb, scalars="ID", cmap='turbo', show_edges=True)
        p.add_mesh(self.mesh, color='white', opacity=0.2)

        p.subplot(0, 1)
        p.add_text("Machining Result (Base - Cutters)", font_size=12)
        z, y, x = np.where(~self.final_grid)
        points = np.column_stack([x, y, z]) * self.pitch + self.origin
        if len(points) > 0:
            res_mesh = pv.PolyData(points).glyph(geom=pv.Cube(x_length=self.pitch, y_length=self.pitch, z_length=self.pitch))
            p.add_mesh(res_mesh, color='orange')
        
        p.link_views(); p.show()

# --- 실행 ---
if __name__ == "__main__":
    f_name = 'final_optimized.stl'
    if not os.path.exists(f_name):
        trimesh.creation.annulus(r_min=10, r_max=25, height=15).export(f_name)

    engine = AdvancedCutterEngine(mesh_path=f_name, resolution=100, min_side=2.0, boundary_scale=1.1)
    engine.run()
    engine.visualize(f_name)
