import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import time

# --- 1. Numba 가속: 그리드 내 박스 확장 ---
@njit
def grow_box_conservative(grid, start_z, start_y, start_x):
    """공기(True) 영역만 채우는 최대 직육면체 확장"""
    d, h, w = grid.shape
    z1, z2 = start_z, start_z
    y1, y2 = start_y, start_y
    x1, x2 = start_x, start_x
    
    while True:
        expanded = False
        # X축
        if x2 + 1 < w and np.all(grid[z1:z2+1, y1:y2+1, x2+1]):
            x2 += 1; expanded = True
        if x1 - 1 >= 0 and np.all(grid[z1:z2+1, y1:y2+1, x1-1]):
            x1 -= 1; expanded = True
        # Y축
        if y2 + 1 < h and np.all(grid[z1:z2+1, y2+1, x1:x2+1]):
            y2 += 1; expanded = True
        if y1 - 1 >= 0 and np.all(grid[z1:z2+1, y1-1, x1:x2+1]):
            y1 -= 1; expanded = True
        # Z축
        if z2 + 1 < d and np.all(grid[z2+1, y1:y2+1, x1:x2+1]):
            z2 += 1; expanded = True
        if z1 - 1 >= 0 and np.all(grid[z1-1, y1:y2+1, x1:x2+1]):
            z1 -= 1; expanded = True
        if not expanded: break
    return z1, z2, y1, y2, x1, x2

# --- 2. 보정 로직: 침범 시 크기 축소 ---
def refine_cutter(mesh, center, size, steps=5):
    """커터가 메쉬 내부를 침범했는지 확인하고 미세 조정"""
    curr_size = np.array(size)
    for _ in range(steps):
        # 15개 샘플링 포인트 (중심, 8개 꼭짓점, 6개 면 중심)
        offsets = np.array([[0,0,0], [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
                            [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
                            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
        pts = center + (offsets * curr_size * 0.5 * 0.99) # 99% 지점 검사(경계 오차 방지)
        if np.any(mesh.contains(pts)):
            curr_size *= 0.95 # 침범 시 5%씩 축소
        else:
            break
    return curr_size

# --- 3. 메인 엔진 클래스 ---
class FinalCutterEngine:
    def __init__(self, mesh_path, resolution=100):
        self.mesh = trimesh.load(mesh_path)
        if not self.mesh.is_watertight:
            self.mesh.fill_holes()
            
        # [원본 바운딩 박스 추출]
        self.b_min, self.b_max = self.mesh.bounds
        self.base_center = (self.b_min + self.b_max) / 2
        self.base_size = self.b_max - self.b_min
        
        self.res = resolution
        self.pitch = np.max(self.base_size) / self.res
        self.origin = self.b_min
        self.cutters = []

    def process(self):
        # 그리드 초기화
        dims = np.ceil(self.base_size / self.pitch).astype(int)
        grid = np.ones(dims[::-1], dtype=bool)
        
        # 메쉬 복셀화하여 그리드에 마킹
        vox = self.mesh.voxelized(pitch=self.pitch)
        v_idx = ((vox.points - self.origin) / self.pitch).astype(int)
        # 안전 마진 (1복셀 두께로 살 보호)
        for dz in [-1,0,1]:
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    s_idx = v_idx + [dx, dy, dz]
                    vld = (s_idx[:,0]>=0) & (s_idx[:,0]<dims[0]) & \
                          (s_idx[:,1]>=0) & (s_idx[:,1]<dims[1]) & \
                          (s_idx[:,2]>=0) & (s_idx[:,2]<dims[2])
                    grid[s_idx[vld, 2], s_idx[vld, 1], s_idx[vld, 0]] = False

        temp_grid = grid.copy()
        print("최적 커터 생성 및 보정 중...")
        
        with tqdm() as pbar:
            while True:
                dist = distance_transform_edt(temp_grid)
                if dist.max() < 1.0: break
                
                seed = np.unravel_index(np.argmax(dist), temp_grid.shape)
                z1, z2, y1, y2, x1, x2 = grow_box_conservative(temp_grid, *seed)
                
                # 좌표 변환
                w_min = self.origin + np.array([x1, y1, z1]) * self.pitch
                w_max = self.origin + np.array([x2+1, y2+1, z2+1]) * self.pitch
                c = (w_min + w_max) / 2
                s = (w_max - w_min)
                
                # 정밀 보정
                safe_s = refine_cutter(self.mesh, c, s)
                self.cutters.append({'center': c, 'size': safe_s})
                
                temp_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                pbar.update(1)

    def visualize(self, original_path):
        p = pv.Plotter(shape=(1, 2), title="Cushion Subtractive Manufacturing Analysis")
        
        # 뷰 1: 원본 메쉬와 베이스 박스
        p.subplot(0, 0)
        p.add_text("1. Base Block & CAD Alignment", font_size=10)
        base_box = pv.Box(bounds=[self.b_min[0], self.b_max[0], self.b_min[1], self.b_max[1], self.b_min[2], self.b_max[2]])
        p.add_mesh(base_box, color='gray', style='wireframe', label="Base Block")
        p.add_mesh(pv.read(original_path), color='white', opacity=0.5, label="Original CAD")
        
        # 뷰 2: 불투명 커터 블록 (컬러맵 적용)
        p.subplot(0, 1)
        p.add_text("2. Non-Transparent Corrected Cutters", font_size=10)
        
        cutter_meshes = pv.MultiBlock()
        for d in self.cutters:
            box = pv.Box(center=d['center'], x_length=d['size'][0], y_length=d['size'][1], z_length=d['size'][2])
            cutter_meshes.append(box)
        
        # 컬러맵 적용: 각 블록에 ID를 부여하여 다른 색상 부여
        scalars = np.arange(len(cutter_meshes))
        p.add_mesh(cutter_meshes, scalars=scalars, cmap='turbo', opacity=1.0, show_edges=True, show_scalar_bar=False)
        
        p.link_views()
        p.camera_position = 'iso'
        p.show()

# --- 실행부 ---
if __name__ == "__main__":
    # 테스트 파일 생성 (복잡한 리브가 있는 형태 가정)
    test_file = 'cushion_sample.stl'
    mesh = trimesh.creation.annulus(r_min=8, r_max=20, height=10)
    mesh.export(test_file)

    engine = FinalCutterEngine(test_file, resolution=80)
    engine.process()
    
    print(f"\n[결과 리포트]")
    print(f"Base Center: {engine.base_center}")
    print(f"Base Size:   {engine.base_size}")
    print(f"커터 블록 수: {len(engine.cutters)}")
    
    engine.visualize(test_file)
