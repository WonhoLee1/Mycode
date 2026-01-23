import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os

# --- 1. 가속 탐색: 공기 영역 박스 확장 ---
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

# --- 2. 최적화 엔진 ---
class FinalMachiningEngine:
    def __init__(self, mesh_path, resolution=120, min_side=1.0, tolerance=0.01):
        self.mesh = trimesh.load(mesh_path)
        self.mesh.fix_normals()
        
        self.b_min, self.b_max = self.mesh.bounds
        self.base_size = self.b_max - self.b_min
        
        # [정밀도 튜닝 인자]
        self.res = resolution      # 높을수록 정밀 (100~150 권장)
        self.min_side = min_side    # 낮을수록 원본에 가까워짐 (1.0~2.0 권장)
        self.tol = tolerance        # 침투 오차 (0.01mm)
        
        self.pitch = np.max(self.base_size) / self.res
        self.origin = self.b_min
        self.cutters = []
        self.final_grid = None

    def snap_to_surface(self, bounds):
        """박스 면을 소수점 단위로 원본 표면에 밀착"""
        refined = np.array(bounds).copy()
        for _ in range(3):
            for i in range(6):
                axis, side = i // 2, (-1 if i % 2 == 0 else 1)
                c = [(refined[0]+refined[1])/2, (refined[2]+refined[3])/2, (refined[4]+refined[5])/2]
                c[axis] = refined[i]
                dist = trimesh.proximity.signed_distance(self.mesh, [c])[0]
                refined[i] += -(dist + self.tol) * side
        return refined

    def run(self):
        # 1. 그리드 생성
        dims = np.ceil(self.base_size / self.pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.pitch + self.origin
        
        # 2. 볼륨 분석 (Solid 내부 보호)
        inside = self.mesh.contains(pts)
        grid = (~inside).reshape(dims[::-1])
        temp_grid = grid.copy()
        skipping_mask = np.ones_like(temp_grid, dtype=bool)
        
        print(f"\n[시뮬레이션 가동] 해상도: {self.res}, 최소변: {self.min_side}mm")
        with tqdm(total=np.sum(grid), desc="Removing Volume") as pbar:
            while True:
                search = temp_grid & skipping_mask
                dist_map = distance_transform_edt(search)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search.shape)
                z1, z2, y1, y2, x1, x2 = grow_box_refined(temp_grid, *seed)
                
                # 좌표 변환 및 표면 피팅
                w_min = self.origin + np.array([x1, y1, z1]) * self.pitch
                w_max = self.origin + np.array([x2+1, y2+1, z2+1]) * self.pitch
                free_bounds = self.snap_to_surface([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                
                size = np.array([free_bounds[1]-free_bounds[0], free_bounds[3]-free_bounds[2], free_bounds[5]-free_bounds[4]])
                
                if np.all(size >= self.min_side):
                    self.cutters.append(free_bounds)
                    removed = np.sum(temp_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(removed)
                else:
                    skipping_mask[seed] = False
        
        self.final_grid = temp_grid

    def visualize(self, mesh_path):
        p = pv.Plotter(shape=(1, 3), title="Final Verification")
        
        # 뷰 1: 소재와 원본
        p.subplot(0, 0)
        p.add_text("1. Base Stock & Target", font_size=10)
        p.add_mesh(pv.Box(bounds=[self.b_min[0], self.b_max[0], self.b_min[1], self.b_max[1], self.b_min[2], self.b_max[2]]), style='wireframe')
        p.add_mesh(self.mesh, color='white', opacity=0.3)

        # 뷰 2: 커터 블록 (가공 경로)
        p.subplot(0, 1)
        p.add_text(f"2. {len(self.cutters)} Cutters", font_size=10)
        cutter_mb = pv.MultiBlock()
        for i, b in enumerate(self.cutters):
            box = pv.Box(bounds=b)
            box.cell_data["ID"] = np.full(box.n_cells, i)
            cutter_mb.append(box)
        p.add_mesh(cutter_mb, scalars="ID", cmap='turbo', show_edges=True, show_scalar_bar=False)

        # 뷰 3: 실제 가공 결과물 (매우 중요)
        p.subplot(0, 2)
        p.add_text("3. Machined Result (Overlap)", font_size=10)
        # 가공 후 남은 살점 (주황색)
        z, y, x = np.where(~self.final_grid)
        points = np.column_stack([x, y, z]) * self.pitch + self.origin
        if len(points) > 0:
            glyphs = pv.PolyData(points).glyph(geom=pv.Cube(x_length=self.pitch, y_length=self.pitch, z_length=self.pitch))
            p.add_mesh(glyphs, color='orange', label="Result")
        # 원본(반투명 흰색)을 겹쳐서 차이 확인
        p.add_mesh(self.mesh, color='white', opacity=0.4)
        
        p.link_views(); p.show()

# --- 실행 ---
if __name__ == "__main__":
    f_name = 'test_model.stl'
    if not os.path.exists(f_name):
        trimesh.creation.annulus(r_min=10, r_max=25, height=15).export(f_name)

    engine = FinalMachiningEngine(
        mesh_path=f_name,
        resolution=120,   # 세밀한 표현을 위해 120 이상 권장
        min_side=1.5,     # 원본에 밀착하려면 이 값을 작게(1.0~2.0) 설정하십시오
        tolerance=0.01
    )
    engine.run()
    engine.visualize(f_name)
