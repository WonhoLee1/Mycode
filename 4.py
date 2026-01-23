import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os
import time

# --- 1. 가속 확장 로직 (Numba 가속) ---
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
    def __init__(self, mesh_path, resolution=100, min_side=2.5, boundary_scale=1.1):
        self.mesh = trimesh.load(mesh_path)
        self.mesh.fix_normals()
        
        # 가공 범위 설정 (110%)
        center = self.mesh.bounds.mean(axis=0)
        self.work_min = center + (self.mesh.bounds[0] - center) * boundary_scale
        self.work_max = center + (self.mesh.bounds[1] - center) * boundary_scale
        
        self.res = resolution
        self.min_side = min_side
        self.pitch = np.max(self.work_max - self.work_min) / self.res
        self.origin = self.work_min
        
        self.cutters = []
        self.final_grid = None

    def get_rms_error(self, current_grid):
        """가공물과 원본 사이의 RMS 오차 산출"""
        z, y, x = np.where(~current_grid)
        if len(x) == 0: return 0.0
        pts = np.column_stack([x, y, z]) * self.pitch + self.origin
        dists = trimesh.proximity.closest_point(self.mesh, pts)[1]
        return np.sqrt(np.mean(dists**2))

    def merge_overlapping_cutters(self):
        """완전히 포함된 블록 제거 최적화"""
        if not self.cutters: return
        self.cutters.sort(key=lambda b: (b[1]-b[0])*(b[3]-b[2])*(b[5]-b[4]), reverse=True)
        refined = []
        for c in self.cutters:
            is_inside = any(c[0]>=r[0] and c[1]<=r[1] and c[2]>=r[2] and c[3]<=r[3] and c[4]>=r[4] and c[5]<=r[5] for r in refined)
            if not is_inside: refined.append(c)
        self.cutters = refined

    def run(self):
        # 그리드 초기화
        dims = np.ceil((self.work_max - self.work_min) / self.pitch).astype(int) + 2
        z, y, x = np.indices(dims[::-1])
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) * self.pitch + self.origin
        grid = (~self.mesh.contains(pts)).reshape(dims[::-1])
        
        temp_grid = grid.copy()
        skipping_mask = np.ones_like(temp_grid, dtype=bool)
        total_air = np.sum(grid)
        
        print(f"\n[작업 시작] 해상도:{self.res}, 최소변:{self.min_side}mm, 범위:110%")
        print(f"[정보] 분석 대상 복셀 수: {total_air:,}개")
        print("-" * 70)

        start_time = time.time()
        with tqdm(total=total_air, desc="가공 진행률", unit="vx") as pbar:
            while True:
                # 1. 잔여 공간 탐색
                search = temp_grid & skipping_mask
                dist_map = distance_transform_edt(search)
                max_d = dist_map.max()
                
                if max_d < 0.5:
                    tqdm.write("\n[이벤트] 잔여 공간이 해상도 미만입니다. 가공을 종료합니다.")
                    break
                
                seed = np.unravel_index(np.argmax(dist_map), search.shape)
                z1, z2, y1, y2, x1, x2 = grow_box_fast(temp_grid, *seed)
                
                # 2. 크기 제약 검사
                size = np.array([x2-x1+1, y2-y1+1, z2-z1+1]) * self.pitch
                
                if np.all(size >= self.min_side):
                    # 블록 확정
                    b = [self.origin[0]+x1*self.pitch, self.origin[0]+(x2+1)*self.pitch,
                         self.origin[1]+y1*self.pitch, self.origin[1]+(y2+1)*self.pitch,
                         self.origin[2]+z1*self.pitch, self.origin[2]+(z2+1)*self.pitch]
                    self.cutters.append(b)
                    
                    removed = np.sum(temp_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    pbar.update(removed)
                    
                    # 주기적 로그 (50개 블록 단위)
                    if len(self.cutters) % 50 == 0:
                        rms = self.get_rms_error(temp_grid)
                        tqdm.write(f" > Block #{len(self.cutters):<4} | 제거율: {pbar.n/total_air*100:4.1f}% | RMS 오차: {rms:.4f}mm")
                else:
                    # [정체 발생 지점] min_side 제약에 걸린 경우 mask 처리하여 다음 탐색에서 제외
                    skipping_mask[seed] = False
                    # 78% 정체 구간에서 어떤 일이 벌어지는지 알리기 위함
                    if np.random.rand() < 0.01: # 너무 자주 뜨지 않게 조절
                        tqdm.write(f" [분석] 좁은 틈새({size.max():.2f}mm) 발견 - 제약조건 미달로 우회 중...")

        # 3. 마무리 최적화
        print("-" * 70)
        print("[처리 중] 중복 커터 제거 및 데이터 통합...")
        before = len(self.cutters)
        self.merge_overlapping_cutters()
        
        self.final_grid = temp_grid
        elapsed = time.time() - start_time
        print(f"[완료] 소요시간: {elapsed:.1f}s | 최종 블록: {before}->{len(self.cutters)}개 | 최종 오차: {self.get_rms_error(temp_grid):.4f}mm")

    def visualize(self, mesh_path):
        p = pv.Plotter(shape=(1, 2), title="Machining Analysis Report")
        
        # 좌측: 커터 블록 (가공 경로)
        p.subplot(0, 0)
        p.add_text("1. Optimized Cutter Path", font_size=10)
        if self.cutters:
            boxes = pv.MultiBlock([pv.Box(bounds=b) for b in self.cutters])
            p.add_mesh(boxes, color='cyan', show_edges=True, opacity=0.6, label="Cutters")
        p.add_mesh(self.mesh, color='white', opacity=0.2)

        # 우측: 실제 최종 형상 (Base - Cutters)
        p.subplot(0, 1)
        p.add_text("2. Final Result vs Target", font_size=10)
        z, y, x = np.where(~self.final_grid)
        pts = np.column_stack([x, y, z]) * self.pitch + self.origin
        if len(pts) > 0:
            res_mesh = pv.PolyData(pts).glyph(geom=pv.Cube(x_length=self.pitch, y_length=self.pitch, z_length=self.pitch))
            p.add_mesh(res_mesh, color='orange', label="Actual Shape")
        p.add_mesh(self.mesh, color='white', opacity=0.3, label="Target CAD")
        
        p.link_views(); p.show()

# --- 실행부 ---
if __name__ == "__main__":
    f_path = 'precision_model.stl'
    if not os.path.exists(f_path):
        trimesh.creation.annulus(r_min=10, r_max=25, height=15).export(f_path)

    # 78% 정체가 심하다면 resolution을 약간 낮추거나 min_side를 높여보세요.
    engine = UltimateCutterEngine(mesh_path=f_path, resolution=80, min_side=2.0)
    engine.run()
    engine.visualize(f_path)
