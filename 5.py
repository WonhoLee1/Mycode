import numpy as np
import trimesh
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from numba import njit
from tqdm import tqdm
import os

# --- [1. Numba 가속 가공 공간 탐색 엔진] ---
@njit
def find_maximum_expandable_cutter_box(occupancy_grid, start_z_index, start_y_index, start_x_index):
    """
    그리드 내에서 지정된 시작점으로부터 사방으로 확장하며 가장 큰 빈 공간 육면체를 찾습니다.
    변수명 설명: 
    occupancy_grid: 공간 점유 상태 (True: 가공 가능 공간, False: 소재 영역)
    """
    depth_limit, height_limit, width_limit = occupancy_grid.shape
    z_min, z_max = start_z_index, start_z_index
    y_min, y_max = start_y_index, start_y_index
    x_min, x_max = start_x_index, start_x_index
    
    while True:
        has_expanded_in_any_direction = False
        # X축 양의 방향 확장 시도
        if x_max + 1 < width_limit and np.all(occupancy_grid[z_min:z_max+1, y_min:y_max+1, x_max+1]):
            x_max += 1
            has_expanded_in_any_direction = True
        # X축 음의 방향 확장 시도
        if x_min - 1 >= 0 and np.all(occupancy_grid[z_min:z_max+1, y_min:y_max+1, x_min-1]):
            x_min -= 1
            has_expanded_in_any_direction = True
        # Y축 양의 방향 확장 시도
        if y_max + 1 < height_limit and np.all(occupancy_grid[z_min:z_max+1, y_max+1, x_min:x_max+1]):
            y_max += 1
            has_expanded_in_any_direction = True
        # Y축 음의 방향 확장 시도
        if y_min - 1 >= 0 and np.all(occupancy_grid[z_min:z_max+1, y_min-1, x_min:x_max+1]):
            y_min -= 1
            has_expanded_in_any_direction = True
        # Z축 양의 방향 확장 시도
        if z_max + 1 < depth_limit and np.all(occupancy_grid[z_max+1, y_min:y_max+1, x_min:x_max+1]):
            z_max += 1
            has_expanded_in_any_direction = True
        # Z축 음의 방향 확장 시도
        if z_min - 1 >= 0 and np.all(occupancy_grid[z_min-1, y_min:y_max+1, x_min:x_max+1]):
            z_min -= 1
            has_expanded_in_any_direction = True
            
        if not has_expanded_in_any_direction:
            break
    return z_min, z_max, y_min, y_max, x_min, x_max

# --- [2. 가공 예측 및 메쉬 연산 엔진] ---
class HighPrecisionMachiningPredictor:
    def __init__(self, target_mesh_file_path, voxel_resolution_count=80, 
                 minimum_cutter_side_length=2.0, boundary_expansion_ratio=1.1, 
                 surface_penetration_tolerance=0.0):
        """
        상세 변수 설명:
        target_mesh_file_path: 원본 CAD STL 파일의 경로
        voxel_resolution_count: 가공 영역을 나눌 격자 해상도
        minimum_cutter_side_length: 허용 가능한 가장 작은 커터의 한 변 길이 (mm)
        boundary_expansion_ratio: 원본 대비 소재(Stock)의 크기 비율 (1.1 = 110%)
        surface_penetration_tolerance: 침투 깊이 (음수: 과삭/침투, 양수: 미삭/오프셋)
        """
        # 1. 메쉬 로드 및 정규화
        self.original_target_mesh = trimesh.load(target_mesh_file_path)
        if isinstance(self.original_target_mesh, trimesh.Scene):
            self.original_target_mesh = self.original_target_mesh.dump(concatenate=True)
        self.original_target_mesh.fix_normals()
        
        # 2. 가공 경계 영역(Boundary Region) 계산
        target_bounding_box_min, target_bounding_box_max = self.original_target_mesh.bounds
        target_geometric_center = (target_bounding_box_min + target_bounding_box_max) / 2.0
        
        self.boundary_region_min = target_geometric_center + (target_bounding_box_min - target_geometric_center) * boundary_expansion_ratio
        self.boundary_region_max = target_geometric_center + (target_bounding_box_max - target_geometric_center) * boundary_expansion_ratio
        
        # 3. 속성 저장
        self.voxel_resolution_count = voxel_resolution_count
        self.minimum_cutter_side_length = minimum_cutter_side_length
        self.surface_penetration_tolerance = surface_penetration_tolerance
        
        # 가공 소재(Stock) 메쉬 생성 - PyVista 및 Trimesh 양쪽 모두 활용
        self.raw_stock_material_mesh = trimesh.creation.box(bounds=[self.boundary_region_min, self.boundary_region_max])
        
        # 격자 간격 계산
        self.voxel_pitch_distance = np.max(self.boundary_region_max - self.boundary_region_min) / self.voxel_resolution_count
        
        self.generated_cutter_boxes_list = []
        self.final_machined_prediction_mesh = None

    def refine_cutter_face_to_surface(self, initial_bounds):
        """커터의 6개 면을 원본 표면에 맞게 미세 조정(Snapping) 합니다."""
        refined_bounds = np.array(initial_bounds).copy()
        for _ in range(2): # 수렴도를 높이기 위한 반복
            for i in range(6):
                axis_index = i // 2
                direction_multiplier = -1 if i % 2 == 0 else 1
                
                face_center_point = [(refined_bounds[0]+refined_bounds[1])/2, 
                                     (refined_bounds[2]+refined_bounds[3])/2, 
                                     (refined_bounds[4]+refined_bounds[5])/2]
                face_center_point[axis_index] = refined_bounds[i]
                
                # 원본과의 거리 측정 (음수: 내부, 양수: 외부)
                signed_distance_to_surface = trimesh.proximity.signed_distance(self.original_target_mesh, [face_center_point])[0]
                
                # 침투 오차(tolerance)를 반영하여 면 이동
                adjustment_move_distance = -(signed_distance_to_surface + self.surface_penetration_tolerance) * direction_multiplier
                refined_bounds[i] += adjustment_move_distance
        return refined_bounds

    def perform_cutter_path_generation(self):
        """그리드 기반으로 가공 영역을 분석하고 최적의 커터 리스트를 생성합니다."""
        grid_dimensions = np.ceil((self.boundary_region_max - self.boundary_region_min) / self.voxel_pitch_distance).astype(int) + 2
        z_indices, y_indices, x_indices = np.indices(grid_dimensions[::-1])
        
        # 격자 점들의 월드 좌표 계산
        sampling_points = np.stack([x_indices.ravel(), y_indices.ravel(), z_indices.ravel()], axis=1) * self.voxel_pitch_distance + self.boundary_region_min
        
        # 메쉬 내부 여부 판정 (살려야 할 살 영역 확인)
        is_point_inside_mesh = self.original_target_mesh.contains(sampling_points)
        air_occupancy_grid = (~is_point_inside_mesh).reshape(grid_dimensions[::-1])
        
        working_occupancy_grid = air_occupancy_grid.copy()
        unprocessible_seed_mask = np.ones_like(working_occupancy_grid, dtype=bool)
        
        print(f"\n[로그] 커터 경로 최적화 시작 (해상도: {self.voxel_resolution_count})")
        
        with tqdm(total=np.sum(air_occupancy_grid), desc="가공 체적 분석 중") as progress_bar:
            while True:
                available_search_space = working_occupancy_grid & unprocessible_seed_mask
                distance_field_map = distance_transform_edt(available_search_space)
                max_internal_distance = distance_field_map.max()
                
                if max_internal_distance < 0.5:
                    break # 더 이상 큰 공간이 없음
                
                seed_index = np.unravel_index(np.argmax(distance_field_map), available_search_space.shape)
                z1, z2, y1, y2, x1, x2 = find_maximum_expandable_cutter_box(working_occupancy_grid, *seed_index)
                
                # 월드 좌표 바운더리 생성
                world_min_coord = self.boundary_region_min + np.array([x1, y1, z1]) * self.voxel_pitch_distance
                world_max_coord = self.boundary_region_min + np.array([x2+1, y2+1, z2+1]) * self.voxel_pitch_distance
                
                raw_bounds = [world_min_coord[0], world_max_coord[0], world_min_coord[1], world_max_coord[1], world_min_coord[2], world_max_coord[2]]
                
                # 정밀 밀착 및 침투 적용
                final_precise_bounds = self.refine_cutter_face_to_surface(raw_bounds)
                
                # 최소 크기 제약 확인
                side_lengths = np.array([final_precise_bounds[1]-final_precise_bounds[0], 
                                         final_precise_bounds[3]-final_precise_bounds[2], 
                                         final_precise_bounds[5]-final_precise_bounds[4]])
                
                if np.all(side_lengths >= self.minimum_cutter_side_length):
                    self.generated_cutter_boxes_list.append(final_precise_bounds)
                    removed_voxels_count = np.sum(working_occupancy_grid[z1:z2+1, y1:y2+1, x1:x2+1])
                    working_occupancy_grid[z1:z2+1, y1:y2+1, x1:x2+1] = False
                    progress_bar.update(removed_voxels_count)
                else:
                    unprocessible_seed_mask[seed_index] = False

    def generate_final_machined_mesh_prediction(self, use_pyvista_boolean=True):
        """
        소재 솔리드에서 커터들을 제거한 최종 형상을 생성합니다.
        OpenSCAD 대신 PyVista(VTK)의 불리언 연산을 기본으로 사용합니다.
        """
        print(f"\n[로그] 최종 형상 솔리드 연산 시작 (커터 수: {len(self.generated_cutter_boxes_list)}개)")
        
        # 1. 소재를 PyVista PolyData로 변환
        stock_polydata = pv.wrap(self.raw_stock_material_mesh)
        
        if use_pyvista_boolean:
            # PyVista의 'clip_box' 기능을 활용하여 소재에서 커터 부위를 순차적으로 도려냅니다.
            # 이 방법은 실제 Boolean보다 빠르고 안정적입니다.
            current_prediction_mesh = stock_polydata
            for cutter_bounds in tqdm(self.generated_cutter_boxes_list, desc="솔리드 컷팅 연산 중"):
                # 각 커터 박스 영역을 '제거'합니다. (invert=True)
                current_prediction_mesh = current_prediction_mesh.clip_box(bounds=cutter_bounds, invert=True)
            
            self.final_machined_prediction_mesh = current_prediction_mesh
        else:
            # 대안: Trimesh 불리언 (더 정밀하지만 실패 가능성이 높음)
            merged_cutters_mesh = trimesh.util.concatenate([trimesh.creation.box(bounds=[[b[0],b[2],b[4]],[b[1],b[3],b[5]]]) 
                                                           for b in self.generated_cutter_boxes_list])
            self.final_machined_prediction_mesh = self.raw_stock_material_mesh.difference(merged_cutters_mesh)

    def visualize_machining_results(self):
        """결과를 시각화합니다."""
        plotter_window = pv.Plotter(shape=(1, 2), title="가공 형상 최종 예측 리포트")
        
        # 뷰 1: 가공 소재와 커터들의 배열
        plotter_window.subplot(0, 0)
        plotter_window.add_text("1. 가공 공정 (소재 및 커터 경로)", font_size=12)
        plotter_window.add_mesh(self.raw_stock_material_mesh, color='gray', opacity=0.3, style='wireframe')
        
        cutter_multi_block = pv.MultiBlock([pv.Box(bounds=b) for b in self.generated_cutter_boxes_list])
        plotter_window.add_mesh(cutter_multi_block, color='cyan', opacity=0.5, show_edges=True)
        plotter_window.add_mesh(self.original_target_mesh, color='white', opacity=0.2)

        # 뷰 2: 최종 가공 예측 솔리드
        plotter_window.subplot(0, 1)
        plotter_window.add_text("2. 최종 예측 형상 (Boundary Solid - Cutters)", font_size=12)
        if self.final_machined_prediction_mesh is not None:
            plotter_window.add_mesh(self.final_machined_prediction_mesh, color='orange', smooth_shading=True)
        
        # 비교를 위해 원본을 반투명하게 겹침
        plotter_window.add_mesh(self.original_target_mesh, color='white', opacity=0.4)
        
        plotter_window.link_views()
        plotter_window.show()

# --- [3. 메인 실행 흐름] ---
if __name__ == "__main__":
    test_file_name = "machining_target_part.stl"
    
    # 테스트용 파일 생성 (고리 모양)
    if not os.path.exists(test_file_name):
        trimesh.creation.annulus(r_min=15, r_max=25, height=20).export(test_file_name)

    # 엔진 초기화
    simulator_engine = HighPrecisionMachiningPredictor(
        target_mesh_file_path=test_file_name,
        voxel_resolution_count=100,             # 공간 분석 정밀도
        minimum_cutter_side_length=1.5,         # 커터 최소 크기 (작을수록 정밀)
        boundary_expansion_ratio=1.1,           # 110% 영역 소재
        surface_penetration_tolerance=-0.1      # 0.1mm 침투 가공
    )
    
    # 가공 계획 생성
    simulator_engine.perform_cutter_path_generation()
    
    # 최종 형상 솔리드 예측 계산
    # PyVista의 clip_box 기반 엔진을 사용하여 OpenSCAD 없이도 작동합니다.
    simulator_engine.generate_final_machined_mesh_prediction(use_pyvista_boolean=True)
    
    # 시각화 리포트 출력
    simulator_engine.visualize_machining_results()
