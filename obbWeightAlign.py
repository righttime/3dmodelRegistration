import numpy as np
import pyvista as pv
import sys
import os

# 시스템 정보 출력
print(f"[시스템 정보] Python 버전: {sys.version}")
print(f"[시스템 정보] PyVista 버전: {pv.__version__}")
print(f"[시스템 정보] 작업 경로: {os.getcwd()}")

# PyVista 에러 처리 및 로깅 설정
try:
    pv.set_error_output_file('pyvista_errors.log')
    print("[시스템 정보] 오류 로그 설정 완료")
except Exception as e:
    print(f"[시스템 정보] 오류 로그 설정 실패: {e}")

# 렌더러 설정 확인 - 오류가 발생하는 부분 수정
try:
    # 안전하게 접근
    renderer_info = getattr(pv.global_theme, 'renderer', 'unknown')
    print(f"[시스템 정보] 기본 렌더러: {renderer_info}")
except Exception as e:
    print(f"[시스템 정보] 렌더러 정보 접근 오류: {e}")

def add_meshs(plotter, meshs, title, subplot_index, color=None):
    try:
        print(f"[로그] add_meshs 시작: {title}, subplot_index={subplot_index}")
        for mesh in meshs:
            try:
                plotter.subplot(subplot_index//2, subplot_index%2)
                print(f"[로그] subplot 설정 완료: {subplot_index//2}, {subplot_index%2}")
                
                plotter.add_text(title, font_size=14)
                print(f"[로그] 텍스트 추가 완료: {title}")
                
                if color is not None:
                    plotter.add_mesh(mesh, color=color)
                    print(f"[로그] 메쉬 추가 완료 (색상 지정): {color}")
                else:
                    plotter.add_mesh(mesh)
                    print(f"[로그] 메쉬 추가 완료 (기본 색상)")
                
                plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z',
                         line_width=3,
                         x_color='red',
                         y_color='green',
                         z_color='blue')
                print(f"[로그] 축 추가 완료")
            except Exception as e:
                print(f"[오류] add_meshs 내부 반복 중 오류 발생: {e}")
        print(f"[로그] add_meshs 완료: {title}")
    except Exception as e:
        print(f"[오류] add_meshs 함수에서 오류 발생: {e}")

def get_obb(mesh):
    """
    PyVista 메쉬의 OBB(Oriented Bounding Box)를 계산합니다.
    PCA 기반 방식으로 메쉬의 주축을 찾아 OBB를 계산합니다.
    
    Args:
        mesh: PyVista 메쉬
        
    Returns:
        pv.PolyData: OBB를 나타내는 PyVista 메쉬
    """
    vertices = mesh.points
    
    # 점들의 평균 계산 (OBB 중심)
    mean_pt = np.mean(vertices, axis=0)
    
    # 평균을 중심으로 점들을 이동
    centered_pts = vertices - mean_pt
    
    # 공분산 행렬 계산
    cov = np.cov(centered_pts, rowvar=False)
    
    # 고유값과 고유벡터 계산
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # 고유값이 큰 순서대로 정렬 (주축 순서대로)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # OBB의 주축 (각 열이 하나의 축)
    obb_axes = eigvecs
    
    # 주축 방향으로 점들 투영
    projections = np.dot(centered_pts, obb_axes)
    
    # 각 축에 대한 최소/최대값 계산
    mins = np.min(projections, axis=0)
    maxs = np.max(projections, axis=0)
    
    # OBB의 크기 계산
    obb_extents = maxs - mins
    
    # 8개의 코너 좌표 계산
    half_sizes = obb_extents / 2
    corners_local = np.array([
        [-half_sizes[0], -half_sizes[1], -half_sizes[2]],
        [half_sizes[0], -half_sizes[1], -half_sizes[2]],
        [half_sizes[0], half_sizes[1], -half_sizes[2]],
        [-half_sizes[0], half_sizes[1], -half_sizes[2]],
        [-half_sizes[0], -half_sizes[1], half_sizes[2]],
        [half_sizes[0], -half_sizes[1], half_sizes[2]],
        [half_sizes[0], half_sizes[1], half_sizes[2]],
        [-half_sizes[0], half_sizes[1], half_sizes[2]]
    ])
    
    # 로컬 좌표를 월드 좌표로 변환
    corners = np.empty((8, 3))
    for i in range(8):
        corners[i] = mean_pt + np.dot(obb_axes, corners_local[i])
    
    # OBB 모서리 정의 (각 모서리는 두 코너 인덱스로 구성)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])
    
    # PyVista 메쉬 생성
    obb_mesh = pv.PolyData(corners)
    
    # 모서리 생성
    lines = np.hstack([[2, edge[0], edge[1]] for edge in edges])
    obb_mesh.lines = lines
    
    return obb_mesh

def align_with_obb(mesh):
    """
    OBB 축을 기준으로 메쉬를 정렬합니다. 
    가장 짧은 축은 z축, 가장 긴 축은 x축, 중간 길이 축은 y축으로 정렬됩니다.
    
    Args:
        mesh: PyVista 메쉬
        
    Returns:
        aligned_mesh: 정렬된 메쉬
    """
    try:
        print("[로그] align_with_obb 함수 시작")
        # 메쉬 복사
        aligned_mesh = mesh.copy()
        
        # 메쉬 정점
        vertices = mesh.points
        print(f"[로그] 정점 개수: {len(vertices)}")
        
        # OBB 계산에 필요한 값 추출
        # 점들의 평균 계산 (OBB 중심)
        mean_pt = np.mean(vertices, axis=0)
        print(f"[로그] OBB 중심 좌표: {mean_pt}")
        
        # 평균을 중심으로 점들을 이동
        centered_pts = vertices - mean_pt
        print(f"[로그] 센터링 완료")
        
        # 공분산 행렬 계산
        cov = np.cov(centered_pts, rowvar=False)
        print(f"[로그] 공분산 행렬 계산 완료")
        
        # 고유값과 고유벡터 계산
        try:
            print(f"[로그] 고유값 계산 시작")
            eigvals, eigvecs = np.linalg.eigh(cov)
            print(f"[로그] 고유값: {eigvals}")
            print(f"[로그] 고유벡터 행렬 형태: {eigvecs.shape}")
        except Exception as e:
            print(f"[오류] 고유값 계산 중 오류 발생: {e}")
            raise
        
        # 고유값이 큰 순서대로 정렬 (주축 순서대로)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        print(f"[로그] 정렬된 고유값: {eigvals}")
        
        # 고유값에 따라 축 정렬: 가장 큰 고유값 -> x축, 중간 -> y축, 가장 작은 -> z축
        rotation_matrix = eigvecs
        print(f"[로그] 회전 행렬 계산 완료")
        
        # 회전 변환 적용 (기존 좌표계에서 표준 좌표계로 변환)
        transformed_vertices = np.dot(vertices - mean_pt, rotation_matrix)
        print(f"[로그] 정점 변환 완료")
        
        # 변환된 정점 적용
        aligned_mesh.points = transformed_vertices
        print(f"[로그] 변환된 메쉬 생성 완료")
        
        print(f"[로그] align_with_obb 함수 완료")
        return aligned_mesh, mean_pt, rotation_matrix
    except Exception as e:
        print(f"[오류] align_with_obb 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 기본값 반환
        return mesh.copy(), np.zeros(3), np.eye(3)

def find_ray_mesh_intersection_approximate(mesh, ray_origin, ray_direction, max_distance_from_ray=0.01):
    """
    레이캐스팅 - 레이 위에 있는 점들 중에서 가장 먼 점을 찾음
    
    Args:
        mesh: PyVista 메쉬
        ray_origin: 레이 시작점
        ray_direction: 레이 방향 벡터
        max_distance_from_ray: 레이에서 고려할 최대 거리 (매우 작은 값 사용)
        
    Returns:
        edge_point: 레이 방향으로 가장 먼 외곽점
    """
    try:
        print(f"[로그] 레이캐스팅 함수 시작 (방향: {ray_direction})")
        vertices = mesh.points
        
        # 레이 방향 벡터 정규화
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        print(f"[로그] 정규화된 레이 방향: {ray_direction}")
        
        # 각 정점에서 레이 원점까지의 벡터
        vec_to_points = vertices - ray_origin
        
        # 레이 방향으로의 투영 거리 계산
        projections = np.dot(vec_to_points, ray_direction)
        print(f"[로그] 투영 거리 계산 완료: 최소={np.min(projections)}, 최대={np.max(projections)}")
        
        # 투영 값이 양수인 점만 선택 (레이 방향에 있는 점)
        valid_indices = np.where(projections > 0)[0]
        print(f"[로그] 유효한 점 개수: {len(valid_indices)}")
        
        if len(valid_indices) > 0:
            # 유효한 정점들
            valid_vertices = vertices[valid_indices]
            valid_projections = projections[valid_indices]
            
            # 레이에서 각 정점까지의 최단 거리 계산
            # 공식: ||(p - o) - ((p - o)·d)d|| 여기서 p는 정점, o는 레이 원점, d는 레이 방향 벡터
            projection_points = ray_origin + ray_direction.reshape(1, 3) * valid_projections.reshape(-1, 1)
            distances_from_ray = np.linalg.norm(valid_vertices - projection_points, axis=1)
            
            # 레이에 가까운 점만 고려 (정확히 레이 위에 있는 점을 찾기 위해 매우 작은 허용 오차 사용)
            near_ray_indices = np.where(distances_from_ray <= max_distance_from_ray)[0]
            
            if len(near_ray_indices) > 0:
                print(f"[로그] 레이 위/근처의 점 개수: {len(near_ray_indices)}")
                
                # 레이에 가까우면서 레이 방향으로 가장 먼 점 찾기
                filtered_indices = valid_indices[near_ray_indices]
                
                # 레이 방향으로 가장 먼 점의 인덱스
                max_proj_idx = filtered_indices[np.argmax(projections[filtered_indices])]
                edge_point = vertices[max_proj_idx]
                
                # 외곽점과 레이 원점 사이의 거리 계산
                distance_to_edge = np.linalg.norm(edge_point - ray_origin)
                print(f"[로그] 찾은 외곽점 좌표: {edge_point}")
                print(f"[로그] 원점에서 외곽점까지 거리: {distance_to_edge}")
                
                return edge_point
            else:
                print(f"[로그] 레이 위에 점이 없어 max_distance_from_ray를 증가시킵니다.")
                # 점이 없으면 거리 제한을 점진적으로 늘려가며 다시 시도
                for multiplier in [2, 5, 10, 20, 50]:
                    new_max_distance = max_distance_from_ray * multiplier
                    near_ray_indices = np.where(distances_from_ray <= new_max_distance)[0]
                    
                    if len(near_ray_indices) > 0:
                        print(f"[로그] 거리 {new_max_distance}로 {len(near_ray_indices)}개 점 발견")
                        filtered_indices = valid_indices[near_ray_indices]
                        max_proj_idx = filtered_indices[np.argmax(projections[filtered_indices])]
                        edge_point = vertices[max_proj_idx]
                        
                        # 외곽점과 레이의 거리 출력
                        closest_point_on_ray = ray_origin + ray_direction * np.dot(edge_point - ray_origin, ray_direction)
                        deviation = np.linalg.norm(edge_point - closest_point_on_ray)
                        print(f"[로그] 찾은 외곽점 좌표: {edge_point}")
                        print(f"[로그] 레이에서 외곽점까지 편차: {deviation}")
                        
                        return edge_point
                
                # 여전히 못찾으면 투영 거리가 가장 큰 점 선택
                print(f"[로그] 레이 근처에 점이 없어 투영 거리가 가장 큰 점 선택")
                max_proj_idx = valid_indices[np.argmax(projections[valid_indices])]
                edge_point = vertices[max_proj_idx]
                print(f"[로그] 투영 거리만 고려한 외곽점 좌표: {edge_point}")
                return edge_point
        
        print("[로그] 외곽점을 찾지 못했습니다.")
        return None
    except Exception as e:
        print(f"[오류] 레이캐스팅 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def align_to_edge_point(mesh):
    """
    +Y축 방향으로 레이캐스팅하여 외곽점을 찾습니다.
    메쉬는 이동시키지 않고 외곽점만 반환합니다.
    
    Args:
        mesh: PyVista 메쉬
        
    Returns:
        mesh: 원본 메쉬 (변경 없음)
        edge_point: 찾은 외곽점 좌표
    """
    try:
        print("[로그] align_to_edge_point 함수 시작")
        # 메쉬 복사 (이제 이동하지 않으므로 복사 불필요하지만 일관성을 위해 유지)
        mesh_copy = mesh.copy()
        
        # 무게중심 계산
        weight_center = np.mean(mesh.points, axis=0)
        print(f"[로그] 무게중심 좌표: {weight_center}")
        
        # Y축 방향 레이 설정
        ray_origin = weight_center
        ray_direction = np.array([0, 1, 0])  # +Y 방향
        print(f"[로그] 레이 원점: {ray_origin}, 방향: {ray_direction}")
        
        # 레이캐스팅으로 외곽점 찾기
        # 모델 크기에 비례하여 max_distance_from_ray 설정 (매우 작은 값으로 시작)
        model_size = np.ptp(mesh.points, axis=0).max()
        max_distance = model_size * 0.001  # 모델 최대 크기의 0.1% (매우 작은 값)
        print(f"[로그] 모델 크기: {model_size}, 초기 최대 거리: {max_distance}")
        
        edge_point = find_ray_mesh_intersection_approximate(mesh, ray_origin, ray_direction, max_distance)
        
        if edge_point is not None:
            print(f"[로그] 외곽점을 찾았습니다: {edge_point}")
            return mesh_copy, edge_point
        else:
            print("[로그] 외곽점을 찾지 못했습니다. 원본 메쉬 반환")
            return mesh_copy, weight_center  # 외곽점 대신 무게중심 반환
            
    except Exception as e:
        print(f"[오류] align_to_edge_point 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에러 발생 시 원본 메쉬 반환
        return mesh.copy(), np.zeros(3)

def align_y_direction(mesh):
    """
    무게중심이 OBB 중심의 +Y 방향에 오도록 메쉬를 회전시킵니다.
    
    표준 좌표계에서 Y축 양의 방향은 위쪽(Green 화살표)입니다.
    이 함수는 무게중심이 OBB 중심에서 볼 때 +Y 방향(Green 화살표 방향)에 
    위치하도록 메쉬를 회전시킵니다.
    
    Args:
        mesh: PyVista 메쉬
        
    Returns:
        rotated_mesh: Y축 방향으로 정렬된 메쉬
        need_y_rotation: Y축 회전이 적용되었는지 여부
        center_diff: 정렬 전 OBB 중심과 무게중심의 차이
        rotated_center_diff: 정렬 후 OBB 중심과 무게중심의 차이
    """
    try:
        print("[로그] align_y_direction 함수 시작")
        # 메쉬 복사
        mesh_copy = mesh.copy()
        
        # OBB와 무게중심 계산
        print("[로그] OBB와 무게중심 계산 시작")
        obb = get_obb(mesh_copy)
        weight_center = np.mean(mesh_copy.points, axis=0)
        obb_center = np.mean(obb.points, axis=0)
        
        # 무게중심과 OBB 중심의 차이 (올바른 방식으로 계산)
        center_diff = np.abs(weight_center) - np.abs(obb_center)
        print(f"[로그] 무게중심: {weight_center}")
        print(f"[로그] OBB 중심: {obb_center}")
        print(f"[로그] 중심 차이 벡터: {center_diff}")
        
        # Y축 기준으로 회전 여부 결정
        # OBB 중심에서 무게중심까지의 Y 방향 성분이 음수이면 회전 필요
        need_y_rotation = center_diff[1] < 0
        print(f"[로그] Y 방향 값: {center_diff[1]}, 회전 필요: {need_y_rotation}")
        
        # 회전 행렬 계산 및 적용
        if need_y_rotation:
            print("[로그] Z축 기준 180도 회전 적용 시작")
            # Z축 기준 180도 회전 행렬
            z_rotation = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
            
            # 회전 행렬 적용
            mesh_copy.points = np.dot(mesh_copy.points, z_rotation)
            print("[로그] 회전 행렬 적용 완료")
            
            # 회전 후 중심 차이 계산
            print("[로그] 회전 후 중심 차이 계산 시작")
            rotated_obb = get_obb(mesh_copy)
            rotated_weight_center = np.mean(mesh_copy.points, axis=0)
            rotated_obb_center = np.mean(rotated_obb.points, axis=0)
            rotated_center_diff = rotated_weight_center - rotated_obb_center
            print(f"[로그] 회전 후 무게중심: {rotated_weight_center}")
            print(f"[로그] 회전 후 OBB 중심: {rotated_obb_center}")
            print(f"[로그] 회전 후 중심 차이 벡터: {rotated_center_diff}")
        else:
            print("[로그] 회전이 필요하지 않습니다.")
            rotated_center_diff = center_diff
        
        print("[로그] align_y_direction 함수 완료")
        return mesh_copy, need_y_rotation, center_diff, rotated_center_diff
    except Exception as e:
        print(f"[오류] align_y_direction 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 기본값 반환
        return mesh.copy(), False, np.zeros(3), np.zeros(3)

def transform_mesh(mesh, verbose=True):
    """
    메쉬를 변환하는 총괄 함수:
    1. OBB 축에 정렬
    2. 무게중심이 +Y 방향에 오도록 회전
    3. +Y 방향 외곽점을 찾지만 메쉬는 이동시키지 않음
    
    Args:
        mesh: PyVista 메쉬
        verbose: 변환 과정 출력 여부
        
    Returns:
        transformed_mesh: OBB 축 정렬 및 Y축 방향 정렬된 메쉬 (외곽점 기준 이동은 하지 않음)
        transformation_info: 변환 정보를 담은 딕셔너리
    """
    try:
        print("[로그] transform_mesh 함수 시작")
        transformation_info = {}
        
        # 단계 1: OBB 축에 정렬
        print("[로그] 통합 변환 - OBB 축 정렬 시작")
        aligned_mesh, obb_center, rotation_matrix = align_with_obb(mesh)
        transformation_info['obb_center'] = obb_center
        transformation_info['rotation_matrix'] = rotation_matrix
        print("[로그] 통합 변환 - OBB 축 정렬 완료")
        
        if verbose:
            print("=== OBB 축 정렬 완료 ===")
        
        # 단계 2: 무게중심이 +Y 방향에 오도록 회전
        print("[로그] 통합 변환 - Y축 방향 정렬 시작")
        y_aligned_mesh, need_y_rotation, center_diff, rotated_center_diff = align_y_direction(aligned_mesh)
        transformation_info['need_y_rotation'] = need_y_rotation
        transformation_info['center_diff_before_y_rotation'] = center_diff
        transformation_info['center_diff_after_y_rotation'] = rotated_center_diff
        print("[로그] 통합 변환 - Y축 방향 정렬 완료")
        
        if verbose:
            if need_y_rotation:
                print("=== 무게중심이 +Y 방향에 오도록 회전 적용 ===")
                print(f"회전 전 중심 차이: {center_diff}")
                print(f"회전 후 중심 차이: {rotated_center_diff}")
            else:
                print("=== 무게중심이 이미 +Y 방향에 있어 회전 생략 ===")
                print(f"중심 차이: {center_diff}")
        
        # 단계 3: +Y 방향 외곽점을 찾기만 하고 메쉬는 이동시키지 않음
        print("[로그] 통합 변환 - +Y 방향 외곽점 찾기 시작")
        _, edge_point = align_to_edge_point(y_aligned_mesh)  # 메쉬는 변경하지 않음, 외곽점만 구함
        transformation_info['edge_point'] = edge_point
        print("[로그] 통합 변환 - 외곽점 찾기 완료")
        
        if verbose:
            print(f"=== +Y 방향 외곽점 찾기 완료 ===")
            print(f"외곽점 좌표: {edge_point}")
        
        # 변환 정보 요약
        if verbose:
            print("\n=== 변환 정보 요약 ===")
            print(f"1. OBB 정렬: 주축을 기준으로 메쉬 정렬")
            if need_y_rotation:
                print(f"2. Y축 정렬: 무게중심이 +Y 방향에 오도록 회전")
            else:
                print(f"2. Y축 정렬: 이미 적절히 정렬됨")
            print(f"3. +Y 방향 외곽점 파악: {edge_point}")
        
        print("[로그] transform_mesh 함수 완료")
        # 이동하지 않은 y_aligned_mesh를 반환
        return y_aligned_mesh, transformation_info
    except Exception as e:
        print(f"[오류] transform_mesh 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 기본 값 반환
        return mesh.copy(), {'error': str(e)}

def select_region_by_angle(mesh, angle_range_x=(-25, 25), angle_range_z=(-5, 5)):
    """
    무게중심에서 Y축 방향 기준으로 특정 각도 범위 내의 영역을 선택합니다.
    
    Args:
        mesh: PyVista 메쉬
        angle_range_x: X축 기준 각도 범위 (도 단위)
        angle_range_z: Z축 기준 각도 범위 (도 단위)
        
    Returns:
        region_mask: 선택된 영역을 나타내는 불리언 마스크
    """
    try:
        print(f"[로그] 각도 범위 기반 영역 선택 시작: X {angle_range_x}도, Z {angle_range_z}도")
        
        # 메쉬 정점 가져오기
        vertices = mesh.points
        
        # 무게중심 계산
        weight_center = np.mean(vertices, axis=0)
        print(f"[로그] 무게중심 좌표: {weight_center}")
        
        # 각 정점에서 무게중심까지의 벡터
        vectors = vertices - weight_center
        
        # 각 벡터 정규화
        norms = np.linalg.norm(vectors, axis=1)
        # 0으로 나누기 방지
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms.reshape(-1, 1)
        
        # 기준 축 설정
        y_axis = np.array([0, 1, 0])  # Y축 방향 (+Y)
        x_axis = np.array([1, 0, 0])
        z_axis = np.array([0, 0, 1])
        
        # 로깅을 위한 메쉬 크기 계산
        mesh_size = np.ptp(vertices, axis=0).max()
        print(f"[로그] 메쉬 크기: {mesh_size}")
        
        # 각 점의 방향 벡터를 Y축과의 각도로 변환
        # Y축 방향 성분 추출
        y_components = normalized_vectors[:, 1]  # Y 성분
        
        # 양의 Y 방향에 있는 점만 선택 (1차 필터링)
        positive_y_indices = np.where(y_components > 0)[0]
        
        if len(positive_y_indices) == 0:
            print("[로그] 양의 Y 방향에 점이 없습니다.")
            return np.zeros(len(vertices), dtype=bool)
        
        # 단순화된 각도 계산 방식: YZ 평면과 XY 평면에 투영 후 각도 계산
        # 변수 초기화
        in_range = np.zeros(len(vertices), dtype=bool)
        
        for idx in positive_y_indices:
            # 현재 정점의 정규화된 방향 벡터
            direction = normalized_vectors[idx]
            
            # Y축 성분이 충분히 강한지 확인 (수직에 가까운 점 필터링)
            if direction[1] < 0.2:  # Y 성분이 작으면 너무 수평임
                continue
                
            # XY 평면에 투영 - X축 대비 각도 계산
            xy_proj = np.array([direction[0], direction[1], 0])
            xy_norm = np.linalg.norm(xy_proj)
            
            if xy_norm > 1e-6:  # 0 벡터 방지
                xy_proj_norm = xy_proj / xy_norm
                # X 각도 계산 (-π ~ π)
                angle_x = np.degrees(np.arctan2(xy_proj_norm[1], xy_proj_norm[0]) - np.pi/2)
                # -90도로 조정 (Y축이 기준)
                if angle_x < -90:
                    angle_x += 360
                
                # 각도 절대값 변환 (좌우 대칭)
                angle_x = abs(angle_x)
                
                # YZ 평면에 투영 - Z축 대비 각도 계산
                yz_proj = np.array([0, direction[1], direction[2]])
                yz_norm = np.linalg.norm(yz_proj)
                
                if yz_norm > 1e-6:  # 0 벡터 방지
                    yz_proj_norm = yz_proj / yz_norm
                    # Z 각도 계산
                    angle_z = np.degrees(np.arctan2(yz_proj_norm[1], yz_proj_norm[2]) - np.pi/2)
                    # -90도로 조정 (Y축이 기준)
                    if angle_z < -90:
                        angle_z += 360
                    
                    # 각도 절대값 변환 (상하 대칭)
                    angle_z = abs(angle_z)
                    
                    # 단순히 각도 범위 내에 있는지 확인 (절대값 사용)
                    x_in_range = angle_x <= max(abs(angle_range_x[0]), abs(angle_range_x[1]))
                    z_in_range = angle_z <= max(abs(angle_range_z[0]), abs(angle_range_z[1]))
                    
                    # 둘 다 범위 내에 있으면 선택
                    if x_in_range and z_in_range:
                        in_range[idx] = True
        
        # 더 직관적인 방법: 원뿔 방정식 활용
        cone_mask = np.zeros(len(vertices), dtype=bool)
        cone_height = 1.0  # 정규화된 벡터이므로 높이는 1
        
        # X, Z 각도 최대값
        max_angle_x = max(abs(angle_range_x[0]), abs(angle_range_x[1]))
        max_angle_z = max(abs(angle_range_z[0]), abs(angle_range_z[1]))
        
        # 원뿔 형태를 현실적으로 조정
        cone_factor = 2.0  # 원뿔 각도를 약간 넓혀줌
        
        # 양의 Y 방향에 있는 정점만 처리 (성능 최적화)
        for idx in positive_y_indices:
            # 현재 정점의 정규화된 방향 벡터
            v = normalized_vectors[idx]
            
            # Y 성분이 충분히 크면 (기본 필터링)
            if v[1] > 0.1:  # Y 방향 임계값
                # 원뿔 각도 검사: Y축 기준 X, Z 방향 각도
                angle_from_y = np.degrees(np.arccos(np.clip(v[1], -1.0, 1.0)))
                
                # X와 Z 성분으로 각도 계산
                # X 방향 각도
                if abs(v[0]) > 1e-6:
                    ratio_x = abs(v[0] / v[1])  # Y에 대한 X의 비율
                    angle_x = np.degrees(np.arctan(ratio_x))
                else:
                    angle_x = 0.0
                    
                # Z 방향 각도
                if abs(v[2]) > 1e-6:
                    ratio_z = abs(v[2] / v[1])  # Y에 대한 Z의 비율
                    angle_z = np.degrees(np.arctan(ratio_z))
                else:
                    angle_z = 0.0
                
                # 각도 범위 내에 있는지 확인 (여유있게 적용)
                x_in_range = angle_x <= (max_angle_x * cone_factor)
                z_in_range = angle_z <= (max_angle_z * cone_factor)
                
                if x_in_range and z_in_range:
                    cone_mask[idx] = True
        
        # 두 방법 중 하나라도 True이면 최종 선택
        region_mask = np.logical_or(in_range, cone_mask)
        
        # 결과가 없는 경우 원뿔 각도를 확장하여 다시 시도
        if np.sum(region_mask) == 0:
            print("[로그] 첫 시도에서 선택된 정점이 없어 원뿔 각도를 확장하여 재시도합니다.")
            # 원뿔 각도를 크게 확장
            expanded_cone_mask = np.zeros(len(vertices), dtype=bool)
            enlarged_factor = 3.0  # 원뿔 각도를 크게 확장
            
            for idx in positive_y_indices:
                v = normalized_vectors[idx]
                if v[1] > 0.05:  # 더 관대한 Y 방향 임계값
                    # 원뿔 각도 확인 (매우 관대하게)
                    angle_from_y = np.degrees(np.arccos(np.clip(v[1], -1.0, 1.0)))
                    
                    # 확장된 원뿔 각도 범위 내에 있는지 확인
                    if angle_from_y <= max(max_angle_x, max_angle_z) * enlarged_factor:
                        expanded_cone_mask[idx] = True
            
            region_mask = expanded_cone_mask
            print(f"[로그] 확장된 원뿔 각도로 {np.sum(region_mask)}개 정점 선택")
        
        n_selected = np.sum(region_mask)
        print(f"[로그] 원뿔 내부 점 개수: {n_selected}")
        
        # 너무 많은 점이 선택된 경우 필터링 (선택적)
        if n_selected > 10000:
            print(f"[로그] 선택된 점이 너무 많습니다. Y 방향성이 강한 점만 유지합니다.")
            # Y 성분이 더 강한 점만 유지
            selected_indices = np.where(region_mask)[0]
            y_strengths = normalized_vectors[selected_indices, 1]  # Y 성분 강도
            # 상위 50%만 유지
            threshold = np.percentile(y_strengths, 50)
            for i, idx in enumerate(selected_indices):
                if y_strengths[i] < threshold:
                    region_mask[idx] = False
            
            print(f"[로그] 필터링 후 정점 개수: {np.sum(region_mask)}")
        
        print(f"[로그] 최종 선택된 영역 정점 개수: {np.sum(region_mask)}")
        return region_mask
        
    except Exception as e:
        print(f"[오류] 각도 범위 영역 선택 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(mesh.points), dtype=bool)

def visualize_region(mesh, region_mask, color=[1, 0, 0]):
    """
    선택된 영역을 시각화합니다.
    
    Args:
        mesh: PyVista 메쉬
        region_mask: 선택된 영역을 나타내는 불리언 마스크
        color: 영역 색상 [r, g, b] (0-1 범위)
        
    Returns:
        colored_mesh: 색상이 적용된 메쉬
    """
    try:
        # 메쉬 복사
        colored_mesh = mesh.copy()
        
        # 색상 배열 생성
        colors = np.zeros((len(mesh.points), 3))
        # 기본 색상 (회색)
        colors[:] = [0.8, 0.8, 0.8]
        # 선택된 영역 색상 설정
        colors[region_mask] = color
        
        # 메쉬에 색상 적용
        colored_mesh['colors'] = colors
        
        return colored_mesh
        
    except Exception as e:
        print(f"[오류] 영역 시각화 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return mesh

def find_boundary_points(mesh, region_mask, n_neighbors=10):
    """
    선택된 영역(region_mask)에서 외곽 경계점들을 찾습니다.
    
    Args:
        mesh: PyVista 메쉬
        region_mask: 선택된 영역을 나타내는 불리언 마스크
        n_neighbors: 이웃 검색에 사용할 포인트 개수
        
    Returns:
        boundary_indices: 경계점 인덱스 목록
    """
    try:
        print("[로그] 경계점 추출 시작")
        # 영역에 해당하는 정점 인덱스
        region_indices = np.where(region_mask)[0]
        
        if len(region_indices) == 0:
            print("[로그] 선택된 영역에 정점이 없습니다.")
            return []
        
        vertices = mesh.points
        region_vertices = vertices[region_indices]
        
        # KDTree를 사용하여 각 정점의 주변 정점 찾기
        from scipy.spatial import KDTree
        tree = KDTree(vertices)
        
        # 경계점 목록
        boundary_indices = []
        
        for idx in region_indices:
            # 현재 정점 좌표
            point = vertices[idx]
            
            # 주변 정점 찾기
            distances, neighbors = tree.query(point, k=n_neighbors)
            
            # 주변 정점 중 영역 외부에 있는 정점이 있는지 확인
            has_outside_neighbor = False
            for neighbor_idx in neighbors:
                if neighbor_idx < len(region_mask) and not region_mask[neighbor_idx]:
                    has_outside_neighbor = True
                    break
            
            # 영역 외부에 있는 정점이 있으면 경계점으로 간주
            if has_outside_neighbor:
                boundary_indices.append(idx)
        
        print(f"[로그] 경계점 추출 완료: {len(boundary_indices)}개 발견")
        return boundary_indices
    
    except Exception as e:
        print(f"[오류] 경계점 추출 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return []

def region_growing(mesh, seed_indices, max_angle_diff=35.0, max_distance=None):
    """
    시드 포인트에서 시작하여 유사한 법선 방향을 가진 인접 정점으로 영역을 확장합니다.
    
    Args:
        mesh: PyVista 메쉬
        seed_indices: 시작점 인덱스 목록
        max_angle_diff: 법선 벡터 간 최대 허용 각도 차이 (도 단위)
        max_distance: 최대 거리 제한 (None이면 제한 없음)
        
    Returns:
        grown_region: 확장된 영역을 나타내는 불리언 마스크
    """
    try:
        print("[로그] 영역 확장(Region Growing) 시작")
        vertices = mesh.points
        
        # 법선 벡터 계산
        # 항상 안전하게 법선 계산하는 방식으로 수정
        print("[로그] 법선 벡터 계산 시작")
        try:
            # 기존 법선 벡터 확인 (point_data 사용)
            if hasattr(mesh, 'point_data') and 'Normals' in mesh.point_data:
                normals = mesh.point_data['Normals']
                print("[로그] 메쉬에서 법선 벡터 정보 추출")
            else:
                # 법선 벡터 계산
                mesh_copy = mesh.copy()  # 원본 메쉬 보존을 위해 복사
                mesh_copy.compute_normals(point_normals=True, cell_normals=False, inplace=True)
                if hasattr(mesh_copy, 'point_data') and 'Normals' in mesh_copy.point_data:
                    normals = mesh_copy.point_data['Normals']
                else:
                    # 직접 계산 시도
                    print("[로그] 법선 직접 계산 시도")
                    # 임시 방편으로 단순한 법선 계산 (매우 기본적인 방식)
                    normals = np.zeros_like(vertices)
                    for i in range(len(vertices)):
                        # 기본 법선값으로 외곽 방향 사용
                        normal = vertices[i] - np.mean(vertices, axis=0)
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            normals[i] = normal / norm
                        else:
                            normals[i] = np.array([0, 1, 0])  # 기본값
                
                print("[로그] 법선 벡터 계산 완료")
        except Exception as e:
            print(f"[오류] 법선 벡터 계산 중 오류: {e}")
            # 오류 발생 시 기본 법선 벡터 생성
            normals = np.zeros_like(vertices)
            for i in range(len(vertices)):
                normal = vertices[i] - np.mean(vertices, axis=0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normals[i] = normal / norm
                else:
                    normals[i] = np.array([0, 1, 0])  # 기본값
            print("[로그] 기본 법선 벡터 생성 완료")
        
        # 법선 벡터 정규화
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1.0  # 0으로 나누기 방지
        normals = normals / norms.reshape(-1, 1)
        
        # 메쉬 크기 기반 거리 기준 설정 (명시적 값이 없는 경우)
        if max_distance is None:
            max_distance = np.ptp(vertices, axis=0).max() * 0.02  # 모델 크기의 2%
            print(f"[로그] 최대 거리 기준 자동 설정: {max_distance}")
        
        # KDTree를 사용하여 각 정점의 주변 정점 찾기
        from scipy.spatial import KDTree
        tree = KDTree(vertices)
        
        # 성장된 영역을 저장할 불리언 마스크
        grown_region = np.zeros(len(vertices), dtype=bool)
        
        # 시드 포인트가 없으면 빈 결과 반환
        if len(seed_indices) == 0:
            print("[로그] 시드 포인트가 없습니다. 빈 결과 반환.")
            return grown_region
        
        # 처리할 정점 목록 (seed_indices에서 시작)
        queue = list(seed_indices)
        # 처리할 정점에 대한 마킹 (중복 방지)
        in_queue = np.zeros(len(vertices), dtype=bool)
        in_queue[seed_indices] = True
        
        # 시드 포인트는 이미 영역에 포함
        grown_region[seed_indices] = True
        
        # 시드 포인트들의 평균 법선 벡터
        avg_normal = np.mean(normals[seed_indices], axis=0)
        norm_avg = np.linalg.norm(avg_normal)
        if norm_avg > 0:
            avg_normal = avg_normal / norm_avg
        else:
            # 평균 법선이 0이면 기본값 사용
            avg_normal = np.array([0, 1, 0])
        
        # 영역 확장
        processed_count = 0
        while queue:
            current_idx = queue.pop(0)
            processed_count += 1
            
            # 주변 정점 찾기
            distances, neighbors = tree.query(vertices[current_idx], k=min(20, len(vertices)))
            
            for i, neighbor_idx in enumerate(neighbors):
                # 이미 처리한 정점은 건너뜀
                if neighbor_idx >= len(vertices) or grown_region[neighbor_idx] or in_queue[neighbor_idx]:
                    continue
                
                # 거리 체크
                if distances[i] > max_distance:
                    continue
                
                # 법선 벡터 유사도 체크
                neighbor_normal = normals[neighbor_idx]
                # 코사인 유사도 (두 벡터의 내적)
                similarity = np.dot(avg_normal, neighbor_normal)
                # 아크코사인을 사용하여 각도로 변환 (라디안)
                angle_diff = np.degrees(np.arccos(np.clip(similarity, -1.0, 1.0)))
                
                # 최대 각도 차이 이내인 경우 영역에 포함
                if angle_diff <= max_angle_diff:
                    grown_region[neighbor_idx] = True
                    queue.append(neighbor_idx)
                    in_queue[neighbor_idx] = True
            
            # 로깅 (진행 상황)
            if processed_count % 100 == 0:
                print(f"[로그] 영역 확장 진행 중: {processed_count}개 정점 처리, 현재 영역 크기: {np.sum(grown_region)}")
        
        print(f"[로그] 영역 확장 완료: 총 {np.sum(grown_region)}개 정점 포함")
        return grown_region
        
    except Exception as e:
        print(f"[오류] 영역 확장(Region Growing) 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(mesh.points), dtype=bool)

def main():
    try:
        print("[로그] 메인 함수 실행 시작")
        # 메인 함수 구현
        model_path = "assets/data/transformed/transformed_model_20250408_091623.stl"
        
        print(f"[로그] 모델 경로: {model_path}")
        print("[로그] PyVista Plotter 초기화 시작")
        
        # PyVista 설정 변경 - 오프스크린 렌더링 비활성화
        try:
            print("[로그] 플롯 테마 설정 시작")
            pv.set_plot_theme('document')
            print("[로그] 플롯 테마 설정 완료")
        except Exception as e:
            print(f"[오류] 플롯 테마 설정 중 오류: {e}")
        
        # 창이 닫혀도 스크립트가 종료되지 않도록 설정
        try:
            # 렌더러 변경 시도
            print("[로그] Plotter 생성 시작")
            # 인자 단순화 - 문제가 발생할 수 있는 인자 제거
            p = pv.Plotter(window_size=[1024, 768], shape=(2, 2))
            print("[로그] PyVista Plotter 생성 완료")
        except Exception as e:
            print(f"[오류] Plotter 생성 중 오류: {e}")
            # 더 단순한 설정으로 다시 시도
            p = pv.Plotter()
            print("[로그] 기본 설정으로 Plotter 생성")
        
        print("[로그] PyVista Plotter 초기화 완료")
        
        # 모델 로드
        print("[로그] 모델 로드 시작")
        try:
            mesh = pv.read(model_path)    
            print(f"[로그] 모델 로드 완료: 점 {mesh.n_points}개, 폴리곤 {mesh.n_cells}개")
        except Exception as e:
            print(f"[오류] 모델 로드 중 오류 발생: {e}")
            # 테스트용 간단한 메쉬로 대체
            mesh = pv.Sphere()
            print("[로그] 테스트용 구체 메쉬로 대체")
        
        print("[로그] OBB 계산 시작")
        obb = get_obb(mesh)
        print("[로그] OBB 계산 완료")
        
        # 원본 메쉬와 OBB 표시
        print("[로그] 원본 메쉬와 OBB 시각화 시작")
        add_meshs(p, [mesh, obb], "Original", 0)
        print("[로그] 원본 메쉬와 OBB 시각화 완료")
        
        # 단계별 변환 수행
        # 1. OBB 축에 정렬
        print("[로그] OBB 축 정렬 시작")
        aligned_mesh, obb_center, rotation_matrix = align_with_obb(mesh)
        aligned_obb = get_obb(aligned_mesh)
        print("[로그] OBB 축 정렬 완료")
        
        # 2. Y축 방향 정렬
        print("[로그] Y축 방향 정렬 시작")
        y_aligned_mesh, need_y_rotation, center_diff, rotated_center_diff = align_y_direction(aligned_mesh)
        y_aligned_obb = get_obb(y_aligned_mesh)
        print("[로그] Y축 방향 정렬된 메쉬 시각화 시작")
        add_meshs(p, [y_aligned_mesh, y_aligned_obb], "Y-Aligned Mesh", 1, color='lightblue')
        print("[로그] Y축 방향 정렬된 메쉬 시각화 완료")
        
        # 3. +Y 방향 외곽점을 찾기
        print("[로그] +Y 방향 외곽점 찾기 시작")
        _, edge_point = align_to_edge_point(y_aligned_mesh)  # 메쉬는 변경하지 않음, 외곽점만 구함
        print("[로그] +Y 방향 외곽점 찾기 완료")
        
        # 4. Y축 방향 기준으로 각도 범위 내의 영역 선택
        print("[로그] Y축 방향 기준 각도 범위 영역 선택 시작")
        angle_range_x = (-25, 25)   # X축 기준 각도 범위 (±25도)
        angle_range_z = (-5, 5)     # Z축 기준 각도 범위 (±5도)
        region_mask = select_region_by_angle(y_aligned_mesh, angle_range_x, angle_range_z)
        print("[로그] 각도 범위 영역 선택 완료")
        
        # 5. 선택된 영역의 경계점 찾기
        print("[로그] 선택된 영역의 경계점 찾기 시작")
        boundary_indices = find_boundary_points(y_aligned_mesh, region_mask, n_neighbors=15)
        print("[로그] 경계점 찾기 완료")
        
        # 6. 경계점에서 region growing 시작
        print("[로그] 경계점에서 Region Growing 시작")
        grown_region = region_growing(y_aligned_mesh, boundary_indices)
        print("[로그] Region Growing 완료")
        
        # 영역 시각화
        print("[로그] 선택된 영역 시각화 시작")
        colored_mesh = visualize_region(y_aligned_mesh, region_mask, color=[0, 1, 1])  # 청록색으로 영역 표시
        
        # 영역이 선택된 메쉬 표시 (서브플롯 2)
        p.subplot(1, 0)
        p.add_mesh(colored_mesh, scalars='colors', rgb=True)
        p.add_text("Selected Region", font_size=14)
        print("[로그] 선택된 영역 시각화 완료")
        
        # Region Growing 결과 시각화
        # print("[로그] Region Growing 결과 시각화 시작")
        # grown_mesh = visualize_region(y_aligned_mesh, grown_region, color=[1, 0.5, 0])  # 주황색으로 영역 표시
        # p.subplot(0, 1)
        # p.add_mesh(grown_mesh, scalars='colors', rgb=True)
        # print("[로그] Region Growing 결과 시각화 완료")
        
        # 경계점 시각화
        if boundary_indices:
            boundary_points = y_aligned_mesh.points[boundary_indices]
            boundary_cloud = pv.PolyData(boundary_points)
            p.subplot(0, 1)
            p.add_mesh(boundary_cloud, color='red', point_size=8, render_points_as_spheres=True)
            print("[로그] 경계점 시각화 완료")
        
        # 외곽점 찾은 결과를 시각화
        print("[로그] 외곽점 시각화 시작")
        
        # 시각화를 위한 외곽점 마커 추가
        try:
            # +Y 방향 레이 표시
            weight_center = np.mean(y_aligned_mesh.points, axis=0)
            ray_origin = weight_center
            ray_direction = np.array([0, 1, 0])  # +Y 방향
            ray_length = np.ptp(y_aligned_mesh.points, axis=0).max() * 0.5
            ray_end = ray_origin + ray_direction * ray_length
            
            # 레이 표시
            ray_line = pv.Line(ray_origin, ray_end)
            p.subplot(1, 0)
            p.add_mesh(ray_line, color='yellow', line_width=3)
            
            # 각도 범위 시각화 (원뿔 또는 사각뿔)
            try:
                print(f"[로그] 각도 범위 시각화 시작: X {angle_range_x}도, Z {angle_range_z}도")
                
                # 원뿔 높이
                cone_height = ray_length
                
                # 무게중심에 작은 구 추가 (무게중심 시각화)
                center_sphere = pv.Sphere(center=weight_center, radius=ray_length/40)
                p.subplot(1, 0)
                p.add_mesh(center_sphere, color='green')
                print("[로그] 무게중심 시각화 추가")
                
                # 원뿔 생성 (PyVista를 사용한 방법)
                # y 방향으로 정렬된 원뿔 생성
                max_angle = max(max(abs(angle_range_x[0]), abs(angle_range_x[1])), 
                               max(abs(angle_range_z[0]), abs(angle_range_z[1])))
                # 각도에서 원뿔 반지름 계산 (삼각함수 이용)
                cone_radius = cone_height * np.tan(np.radians(max_angle))
                
                # 원뿔 생성
                cone = pv.Cone(center=weight_center + ray_direction * (cone_height/2),
                              direction=ray_direction,
                              height=cone_height,
                              radius=cone_radius,
                              resolution=30)
                
                # 원뿔을 정확한 각도로 절단하여 사용할 경우를 위한 준비
                # 타원형 원뿔 생성 (X, Z 각도가 다를 경우)
                if abs(angle_range_x[0]) != abs(angle_range_z[0]) or abs(angle_range_x[1]) != abs(angle_range_z[1]):
                    # 보다 복잡한 형태가 필요한 경우 수동으로 각 포인트 생성
                    cone_points = []
                    # 원점
                    cone_points.append(weight_center)
                    
                    # 바닥면 포인트 (타원형)
                    n_points = 36  # 가장자리 포인트 개수
                    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
                    
                    # X, Z 방향 반지름 (각각의 각도 범위에 따라 다름)
                    x_radius = cone_height * np.tan(np.radians(max(abs(angle_range_x[0]), abs(angle_range_x[1]))))
                    z_radius = cone_height * np.tan(np.radians(max(abs(angle_range_z[0]), abs(angle_range_z[1]))))
                    
                    for angle in angles:
                        # 타원형 바닥면 좌표 계산
                        x = x_radius * np.cos(angle)
                        z = z_radius * np.sin(angle)
                        # 원뿔 높이만큼 y 방향으로 이동
                        point = weight_center + np.array([x, cone_height, z])
                        cone_points.append(point)
                    
                    # 포인트 클라우드로 변환
                    cone_cloud = pv.PolyData(np.array(cone_points))
                    
                    # 볼록 껍질 생성
                    custom_cone = cone_cloud.delaunay_3d().extract_surface()
                    
                    # 반투명 원뿔 추가
                    p.subplot(1, 0)
                    p.add_mesh(custom_cone, color='yellow', opacity=0.2)
                    print("[로그] 커스텀 타원형 원뿔 시각화 완료")
                else:
                    # 반투명 원뿔 추가
                    p.subplot(1, 0)
                    p.add_mesh(cone, color='yellow', opacity=0.2)
                    print("[로그] 원형 원뿔 시각화 완료")
                
                print("[로그] 각도 범위 시각화 완료")
            except Exception as e:
                print(f"[오류] 각도 범위 시각화 중 오류: {e}")
                import traceback
                traceback.print_exc()
            
            # 외곽점 표시
            if edge_point is not None and not np.array_equal(edge_point, np.zeros(3)):
                edge_sphere = pv.Sphere(center=edge_point, radius=ray_length/30)
                p.add_mesh(edge_sphere, color='red')
                
                # 무게중심에서 외곽점으로 연결선 추가
                connection_line = pv.Line(weight_center, edge_point)
                p.add_mesh(connection_line, color='yellow', line_width=2, line_style='dashed')
                
                print("[로그] 레이와 외곽점 시각화 추가 완료")
        except Exception as e:
            print(f"[오류] 레이 시각화 중 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # 통합 함수로 한번에 변환
        print("[로그] 통합 변환 함수 실행 시작")
        transformed_mesh, transformation_info = transform_mesh(mesh)
        transformed_obb = get_obb(transformed_mesh)
        print("[로그] 최종 변환된 메쉬 시각화 시작")
        
        # 최종 변환된 메쉬에도 모든 작업 적용 및 결합 결과 표시
        region_mask_final = select_region_by_angle(transformed_mesh, angle_range_x, angle_range_z)
        boundary_indices_final = find_boundary_points(transformed_mesh, region_mask_final, n_neighbors=15)
        grown_region_final = region_growing(transformed_mesh, boundary_indices_final)
        
        # 최종 결과 시각화 (선택 영역 + Region Growing 결과)
        final_mask = np.logical_or(region_mask_final, grown_region_final)
        final_colored_mesh = visualize_region(transformed_mesh, final_mask, color=[0, 0.8, 0.8])
        
        p.subplot(1, 1)
        p.add_mesh(final_colored_mesh, scalars='colors', rgb=True)
        p.add_text("Final Result with Region Growing", font_size=14)
        print("[로그] 최종 변환된 메쉬 시각화 완료")
        
        # 변환 정보 출력
        print("\n=== 변환 정보 요약 ===")
        print(f"통합 함수 Y축 회전 필요: {transformation_info['need_y_rotation']}")
        if 'edge_point' in transformation_info:
            print(f"외곽점 좌표: {transformation_info['edge_point']}")
        print(f"선택된 영역 정점 개수: {np.sum(region_mask)}")
        print(f"영역 확장 후 정점 개수: {np.sum(grown_region)}")
        print(f"최종 영역 정점 개수: {np.sum(final_mask)}")
        
        # 시각화 부분 수정
        print("[로그] 시각화 표시 시작")
        try:
            # 시각화 실행
            print("[로그] 렌더링 시작 직전")
            # 단순 방식으로 실행
            p.show(full_screen=False)
            print("[로그] 시각화 표시 완료 - 정상 종료")
        except Exception as e:
            print(f"[오류] 기본 시각화 실패: {e}")
            
            # 방법 2: 이미지로 저장
            try:
                print("[로그] 이미지 저장 시도")
                p.save_graphic("obb_visualization.png")
                print(f"[로그] 'obb_visualization.png' 파일로 결과가 저장되었습니다.")
            except Exception as e2:
                print(f"[오류] 이미지 저장 실패: {e2}")
            
            # 방법 3: 스크린샷으로 저장
            try:
                print("[로그] 스크린샷 저장 시도")
                p.screenshot("obb_screenshot.png")
                print(f"[로그] 'obb_screenshot.png' 파일로 스크린샷이 저장되었습니다.")
            except Exception as e3:
                print(f"[오류] 스크린샷 저장 실패: {e3}")
    
    except Exception as e:
        print(f"[치명적 오류] 메인 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        print("[로그] 프로그램 시작")
        main()
        print("[로그] 프로그램 정상 종료")
    except Exception as e:
        print(f"[치명적 오류] 프로그램 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
