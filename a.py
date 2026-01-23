import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import time

# --- 1. Numba 가속: 상자 확장 로직 ---
@njit
def grow_box_in_grid(grid, start_z, start_y, start_x):
    """
    주어진 씨앗(Seed) 지점에서 시작하여
    False(장애물/이미 잘린 곳)를 만나기 전까지
    최대로 확장 가능한 직육면체(Box)의 범위를 찾습니다.
    """
    d, h, w = grid.shape
    
    # 1. Z축 방향 확장 (초기 두께 설정)
    z_min, z_max = start_z, start_z
    while z_min > 0 and grid[z_min - 1, start_y, start_x]:
        z_min -= 1
    while z_max < d - 1 and grid[z_max + 1, start_y, start_x]:
        z_max += 1
        
    # 2. Y축 방향 확장 (현재 Z 범위 내에서 가능한 최대 Y)
    y_min, y_max = start_y, start_y
    
    # Y- (아래로 확장)
    while y_min > 0:
        # 검사하려는 슬라이스(z_min ~ z_max)가 모두 True인지 확인
        valid = True
        for k in range(z_min, z_max + 1):
            if not grid[k, y_min - 1, start_x]:
                valid = False
                break
        if valid:
            y_min -= 1
        else:
            break
            
    # Y+ (위로 확장)
    while y_max < h - 1:
        valid = True
        for k in range(z_min, z_max + 1):
            if not grid[k, y_max + 1, start_x]:
                valid = False
                break
        if valid:
            y_max += 1
        else:
            break

    # 3. X축 방향 확장 (현재 Z, Y 범위 내에서 가능한 최대 X)
    x_min, x_max = start_x, start_x
    
    # X-
    while x_min > 0:
        valid = True
        for k in range(z_min, z_max + 1):
            for j in range(y_min, y_max + 1):
                if not grid[k, j, x_min - 1]:
                    valid = False
                    break
            if not valid: break
        if valid: x_min -= 1
        else: break
        
    # X+
    while x_max < w - 1:
        valid = True
        for k in range(z_min, z_max + 1):
            for j in range(y_min, y_max + 1):
                if not grid[k, j, x_max + 1]:
                    valid = False
                    break
            if not valid: break
        if valid: x_max += 1
        else: break
        
    return z_min, z_max, y_min, y_max, x_min, x_max

# --- 2. 메인 클래스: 자유 형상 커터 생성기 ---
class FreeformCutterGenerator:
    def __init__(self, mesh_path, resolution=100, min_cutter_volume_ratio=0.001):
        """
        Args:
            resolution (int): 긴 축 기준 복셀 해상도 (높을수록 정밀하고 자유도가 높음)
            min_cutter_volume_ratio (float): 전체 부피 대비 너무 작은 커터는 무시할 비율
        """
        self.mesh = trimesh.load(mesh_path)
        self.resolution = resolution
        self.cutters = [] # 결과 저장 [(center, size), ...]
        self.min_vol_ratio = min_cutter_volume_ratio
        
        # Bounding Box + Margin
        self.bounds = self.mesh.bounds
        extents = self.bounds[1] - self.bounds[0]
        max_dim = np.max(extents)
        self.pitch = max_dim / self.resolution # 복셀 하나의 실제 크기
        
        # 여유를 둔 Stock Bounds
        margin = self.pitch * 2
        self.stock_min = self.bounds[0] - margin
        self.stock_max = self.bounds[1] + margin
        self.stock_size = self.stock_max - self.stock_min

    def voxelize_space(self):
        """공간을 복셀화하고, '공기(Air)' 부분만 True로 된 3D 그리드 생성"""
        print("공간 복셀화 중 (정밀도 결정)...")
        
        # 1. Trimesh Voxelizer 사용 (Solid 부분 찾기)
        # pitch를 사용하여 전체 stock을 커버하는 그리드 생성
        # 주의: trimesh voxelize는 메쉬 내부를 채움
        voxelized = self.mesh.voxelized(pitch=self.pitch)
        
        # 2. 전체 Stock 크기에 맞는 Numpy Grid 생성
        # Stock의 크기를 pitch로 나누어 grid dimension 계산
        dims = np.ceil(self.stock_size / self.pitch).astype(int)
        
        # 전체를 True(Air)로 초기화
        self.air_grid = np.ones(dims, dtype=bool)
        
        # Mesh가 차지하는 부분을 False(Object)로 마킹
        # Voxel object의 index를 우리 grid의 offset에 맞춰 변환해야 함
        matrix = voxelized.transform
        # World coordinate -> Grid Index 변환 함수
        def world_to_grid(points):
            return ((points - self.stock_min) / self.pitch).astype(int)
            
        # Voxelized된 포인트들(Solid)을 가져옴
        solid_indices = world_to_grid(voxelized.points)
        
        # 인덱스 범위 체크 및 마킹
        valid_mask = (solid_indices[:,0] >= 0) & (solid_indices[:,0] < dims[0]) & \
                     (solid_indices[:,1] >= 0) & (solid_indices[:,1] < dims[1]) & \
                     (solid_indices[:,2] >= 0) & (solid_indices[:,2] < dims[2])
        solid_indices = solid_indices[valid_mask]
        
        self.air_grid[solid_indices[:,0], solid_indices[:,1], solid_indices[:,2]] = False
        
        print(f"Grid Size: {self.air_grid.shape}, Pitch: {self.pitch:.4f}")
        return self.air_grid

    def compute_cutters(self):
        """가장 큰 빈 공간부터 찾아내어 커터로 변환"""
        grid = self.air_grid.copy() # 원본 보존을 위해 복사
        
        total_voxels = grid.size
        min_voxels = total_voxels * self.min_vol_ratio
        
        print("최대 크기 커터 탐색 시작...")
        pbar = tqdm(total=np.sum(grid))
        
        while True:
            # 1. Distance Transform 수행 (가장 깊은 곳 찾기)
            # edt: 유클리드 거리 변환. 0(False)인 지점까지의 거리를 계산.
            # 즉, 값이 클수록 쿠션 표면에서 멀리 떨어진 허공의 중심임.
            dist_map = distance_transform_edt(grid)
            max_dist = dist_map.max()
            
            # 종료 조건: 남은 공간이 너무 좁거나 없을 때
            if max_dist <= 1.0: # 더 이상 유의미한 공간 없음
                break
                
            # 2. 최대 깊이 지점(Seed) 찾기
            # argmax는 1차원 인덱스를 반환하므로 unravel 필요
            seed_idx = np.unravel_index(np.argmax(dist_map), grid.shape)
            
            # 3. Seed로부터 최대 박스 확장 (Numba 가속)
            # 순서는 z, y, x
            z1, z2, y1, y2, x1, x2 = grow_box_in_grid(grid, seed_idx[0], seed_idx[1], seed_idx[2])
            
            # 4. 박스 크기 확인 및 종료 조건
            box_vol = (z2-z1+1) * (y2-y1+1) * (x2-x1+1)
            if box_vol < min_voxels and len(self.cutters) > 10: 
                # 너무 작은 조각만 남았으면 중단 (노이즈 방지)
                break
                
            # 5. Grid에서 해당 박스 영역 제거 (False로 변경)
            grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
            
            # 6. 결과 저장 (Grid Index -> World Coordinate 변환)
            # 인덱스는 [0, 0, 0]이 stock_min에 해당
            box_min_idx = np.array([x1, y1, z1])
            box_max_idx = np.array([x2+1, y2+1, z2+1]) # +1 to include the last voxel boundary
            
            world_min = self.stock_min + (box_min_idx * self.pitch)
            world_max = self.stock_min + (box_max_idx * self.pitch)
            
            center = (world_min + world_max) / 2.0
            size = world_max - world_min
            
            self.cutters.append({'center': center, 'size': size})
            
            # 진행률 업데이트
            removed_count = box_vol
            pbar.update(removed_count)
            
        pbar.close()
        print(f"생성 완료: {len(self.cutters)} 개의 커터 블록.")

    def visualize(self):
        p = pv.Plotter(title="Free-form Cutter Result")
        
        # 1. 원본
        mesh_pv = pv.read(self.mesh_path)
        p.add_mesh(mesh_pv, color='lightblue', opacity=0.6, label='Cushion')
        
        # 2. 커터 블록들
        blocks = pv.MultiBlock()
        # 색상을 랜덤하게 주어 블록 구분이 잘 되도록 함
        for info in self.cutters:
            c = info['center']
            s = info['size']
            bounds = [c[0]-s[0]/2, c[0]+s[0]/2, 
                      c[1]-s[1]/2, c[1]+s[1]/2, 
                      c[2]-s[2]/2, c[2]+s[2]/2]
            box = pv.Box(bounds=bounds)
            blocks.append(box)
            
        p.add_mesh(blocks, style='wireframe', color='red', line_width=2, label='Cutters')
        # 솔리드 뷰도 추가 (투명하게)
        p.add_mesh(blocks, cmap='jet', opacity=0.3, show_scalar_bar=False)

        p.add_legend()
        p.show_grid()
        p.show()

# --- 실행부 ---
if __name__ == "__main__":
    # 테스트 파일 생성
    input_file = 'complex_cushion.stl'
    try:
        # 꼬인 토러스 형상 (복잡한 곡면 예시)
        mesh = trimesh.creation.annulus(r_min=10, r_max=25, height=15)
        # 임의로 변형을 가함 (비정형성 추가)
        mesh.apply_scale([1.2, 0.8, 1.0]) 
        mesh.export(input_file)
    except:
        pass

    # 1. 생성기 초기화 (resolution을 높이면 커터 위치/크기가 더 자유로워짐)
    # resolution=100 정도면 상당히 정밀하며, 200 이상이면 매우 정밀함
    generator = FreeformCutterGenerator(input_file, resolution=120)
    
    # 2. 공간 분석
    generator.voxelize_space()
    
    # 3. 커터 계산 (Greedy Algorithm)
    start_time = time.time()
    generator.compute_cutters()
    print(f"Calculation Time: {time.time() - start_time:.2f} sec")
    
    # 4. 결과 데이터 출력 예시
    print("\n[Top 5 Largest Cutters]")
    # 부피순 정렬
    sorted_cutters = sorted(generator.cutters, key=lambda x: x['size'][0]*x['size'][1]*x['size'][2], reverse=True)
    for i, c in enumerate(sorted_cutters[:5]):
        print(f"#{i+1}: Center={np.round(c['center'], 2)}, Size={np.round(c['size'], 2)}")

    # 5. 시각화
    generator.visualize()
