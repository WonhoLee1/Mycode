import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os

# --- 1. 가속 탐색: 최대한 큰 정수 단위 박스 찾기 ---
@njit
def grow_box_max(grid, start_z, start_y, start_x):
    d, h, w = grid.shape
    z1, z2, y1, y2, x1, x2 = start_z, start_z, start_y, start_y, start_x, start_x
    
    while True:
        expanded = False
        # 부피를 키우기 위해 각 축 방향으로 확장 시도
        # (X -> Y -> Z 순차 확장으로 큰 덩어리 유도)
        if x2 + 1 < w and np.all(grid[z1:z2+1, y1:y2+1, x2+1]): x2 += 1; expanded = True
        if x1 - 1 >= 0 and np.all(grid[z1:z2+1, y1:y2+1, x1-1]): x1 -= 1; expanded = True
        if y2 + 1 < h and np.all(grid[z1:z2+1, y2+1, x1:x2+1]): y2 += 1; expanded = True
        if y1 - 1 >= 0 and np.all(grid[z1:z2+1, y1-1, x1:x2+1]): y1 -= 1; expanded = True
        if z2 + 1 < d and np.all(grid[z2+1, y1:y2+1, x1:x2+1]): z2 += 1; expanded = True
        if z1 - 1 >= 0 and np.all(grid[z1-1, y1:y2+1, x1:x2+1]): z1 -= 1; expanded = True
        if not expanded: break
    return z1, z2, y1, y2, x1, x2

# --- 2. 하이엔드 최적화 엔진 ---
class MinCountPrecisionEngine:
    def __init__(self, mesh_path, resolution=100, min_side=3.0, tolerance=0.05):
        self.mesh = trimesh.load(mesh_path)
        self.mesh.fix_normals()
        
        self.b_min, self.b_max = self.mesh.bounds
        self.base_size = self.b_max - self.b_min
        
        # 설정 인자
        self.res = resolution
        self.min_side = min_side    # 작을수록 정밀하지만 블록 수가 늘어남
        self.tol = tolerance        # 0.05mm 수준의 정밀 밀착
        self.pitch = np.max(self.base_size) / self.res
        self.origin = self.b_min
        
        self.cutters = []
        self.final_grid = None

    def snap_to_surface(self, bounds):
        """육면체의 각 면을 원본 메쉬 표면에 실시간 피팅 (핵심 로직)"""
        refined = np.array(bounds).copy()
        # 6개 면을 독립적으로 이동 (xmin, xmax, ymin, ymax, zmin, zmax)
        for i in range(6):
            axis = i // 2
            direction = -1 if i % 2 == 0 else 1
            
            # 면의 중심점 샘플링
            c = [(refined[0]+refined[1])/2, (refined[2]+refined[3])/2, (refined[4]+refined[5])/2]
            c[axis] = refined[i]
            
            # 메쉬 표면까지의 거리 측정 (Signed Distance)
            # 양수: 외부(공기), 음수: 내부(살)
            dist = trimesh.proximity.signed_distance(self.mesh, [c])[0]
            
            # 면 이동: 표면과의 거리가 tolerance가 되도록 조정
            # 깎아야 할 공간(공기)을 최대한 확보하면서 살은 건드리지 않음
            move = -(dist + self.tol) * direction
            refined[i] += move
            
        return refined

    def run(self):
        # 초기 공간 분석
        dims = np.ceil(self.base_size / self.pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.pitch + self.origin
        
        inside = self.mesh.contains(pts)
        grid = (~inside).reshape(dims[::-1])
        temp_grid = grid.copy()
        skipping_mask = np.ones_like(temp_grid, dtype=bool)
        
        print(f"\n[최적화 가동] 블록 수 최소화 & 표면 밀착 모드")
        with tqdm(total=np.sum(grid), desc="Optimizing Toolpath") as pbar:
            while True:
                # 1. 남아있는 공기 영역 중 가장 큰 부피의 중심 찾기
                search = temp_grid & skipping_mask
                dist_map = distance_transform_edt(search)
                if dist_map.max() < 0.5: break
                
                seed = np.unravel_index(np.argmax(dist_map), search.shape)
                
                # 2. Greedy 확장 (정수 복셀 단위)
                z1, z2, y1, y2, x1, x2 = grow_box_max(temp_grid, *seed)
                
                # 3. 연속 공간 피팅 (소수점 단위 밀착)
                w_min = self.origin + np.array([x1, y1, z1]) * self.pitch
                w_max = self.origin + np.array([x2+1, y2+1, z2+1]) * self.pitch
                free_bounds = self.snap_to_surface([w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]])
                
                # 4. 크기 제약 확인
                size = np.array([free_bounds[1]-free_bounds[0], 
                                 free_bounds[3]-free_bounds[2], 
                                 free_bounds[5]-free_bounds[4]])
                
                if np.all(size >= self.min_side):
                    self.cutters.append(free_bounds)
                    # 처리된 영역 그리드에서 제거 (중복 방지)
                    removed = np.sum(temp_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(removed)
                else:
                    # 너무 작은 틈새는 Seed에서 제외하여 무한루프 방지
                    skipping_mask[seed] = False
        
        self.final_grid = temp_grid

    def visualize(self, mesh_path):
        p = pv.Plotter(shape=(1, 3), title="Final Manufacturing Report")
        
        # 1. 목표 형상 (CAD)
        p.subplot(0, 0)
        p.add_text("1. Target CAD", font_size=10)
        p.add_mesh(self.mesh, color='white', opacity=0.4)
        
        # 2. 최적화된 커터 (최소 개수)
        p.subplot(0, 1)
        p.add_text(f"2. Optimized Cutters: {len(self.cutters)} ea", font_size=10)
        cutter_mb = pv.MultiBlock()
        for i, b in enumerate(self.cutters):
            box = pv.Box(bounds=b)
            box.cell_data["ID"] = np.full(box.n_cells, i)
            cutter_mb.append(box)
        p.add_mesh(cutter_mb, scalars="ID", cmap='turbo', show_edges=True, show_scalar_bar=False)

        # 3. 실제 가공 결과 (Subtraction Simulation)
        p.subplot(0, 2)
        p.add_text("3. Simulated Machining Result", font_size=10)
        # 소재에서 커터를 빼고 남은 '살' 시각화
        z, y, x = np.where(~self.final_grid)
        points = np.column_stack([x, y, z]) * self.pitch + self.origin
        if len(points) > 0:
            glyphs = pv.PolyData(points).glyph(geom=pv.Cube(x_length=self.pitch, y_length=self.pitch, z_length=self.pitch))
            p.add_mesh(glyphs, color='gold', show_edges=False)
        
        p.link_views()
        p.show()

# --- 실행부 ---
if __name__ == "__main__":
    # 고리(Annulus) 형상으로 테스트
    f_name = 'precision_test.stl'
    if not os.path.exists(f_name):
        trimesh.creation.annulus(r_min=10, r_max=25, height=15).export(f_name)

    engine = MinCountPrecisionEngine(
        mesh_path=f_name,
        resolution=80,      # 높을수록 정밀하지만 계산량 증가
        min_side=5.0,       # 이 값을 키우면 블록 수가 획기적으로 줄어듭니다.
        tolerance=0.05      # 표면 밀착 허용 오차 (mm)
    )
    
    engine.run()
    engine.visualize(f_name)
