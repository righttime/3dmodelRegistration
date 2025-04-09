import os
import numpy as np
import pyvista as pv

def align_with_obb(mesh, debug=False):
    """
    OBB 축을 기준으로 메쉬를 정렬합니다. 
    가장 짧은 축은 z축, 가장 긴 축은 x축, 중간 길이 축은 y축으로 정렬됩니다.
    
    구체적인 변환 과정:
    1. 메쉬 중심점(OBB 중심)을 원점으로 이동
    2. PCA를 통해 주축을 찾아 회전 행렬 계산
    3. 회전 행렬을 적용하여 메쉬 정렬
    
    Args:
        mesh: PyVista 메쉬
        debug: 디버깅 모드 (중간 과정 시각화)
        
    Returns:
        aligned_mesh: 정렬된 메쉬
        obb_center: OBB 중심점
        rotation_matrix: 회전 행렬
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
        # 주의: centered_pts는 이미 중심이 원점으로 이동된 점들임
        transformed_vertices = np.dot(centered_pts, rotation_matrix)
        print(f"[로그] 정점 변환 완료")
        
        # 변환된 정점 적용
        aligned_mesh.points = transformed_vertices
        print(f"[로그] 변환된 메쉬 생성 완료")
        
        # 디버깅 모드: 변환 과정 시각화
        if debug:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # 원본 메쉬 점 시각화
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.', alpha=0.1)
            ax1.set_title('원본 메쉬')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 중심이 원점으로 이동된 점 시각화
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(centered_pts[:, 0], centered_pts[:, 1], centered_pts[:, 2], c='g', marker='.', alpha=0.1)
            ax2.set_title('중심이 원점으로 이동된 메쉬')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 회전 변환 후 점 시각화
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2], c='r', marker='.', alpha=0.1)
            ax3.set_title('회전 변환 후 메쉬')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            plt.tight_layout()
            plt.show()
        
        print(f"[로그] align_with_obb 함수 완료")
        return aligned_mesh, mean_pt, rotation_matrix
    except Exception as e:
        print(f"[오류] align_with_obb 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 기본값 반환
        return mesh.copy(), np.zeros(3), np.eye(3)

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
    

def transform_mesh(mesh):
    """
    메쉬를 변환하는 총괄 함수:
    1. OBB 축에 정렬
    2. 무게중심이 +Y 방향에 오도록 회전
    3. +Y 방향 외곽점을 찾지만 메쉬는 이동시키지 않음
    
    Args:
        mesh: PyVista 메쉬
        
    Returns:
        transformed_mesh: OBB 축 정렬 및 Y축 방향 정렬된 메쉬 (외곽점 기준 이동은 하지 않음)
        transformation_info: 변환 정보를 담은 딕셔너리 (최종 변환 행렬 포함)
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

        # 단계 2: 무게중심이 +Y 방향에 오도록 회전
        print("[로그] 통합 변환 - Y축 방향 정렬 시작")
        y_aligned_mesh, need_y_rotation, center_diff, rotated_center_diff = align_y_direction(aligned_mesh)
        transformation_info['need_y_rotation'] = need_y_rotation
        transformation_info['center_diff_before_y_rotation'] = center_diff
        transformation_info['center_diff_after_y_rotation'] = rotated_center_diff
        print("[로그] 통합 변환 - Y축 방향 정렬 완료")

        # 단계 3: +Y 방향 외곽점을 찾기만 하고 메쉬는 이동시키지 않음
        print("[로그] 통합 변환 - +Y 방향 외곽점 찾기 시작")
        _, edge_point = align_to_edge_point(y_aligned_mesh)  # 메쉬는 변경하지 않음, 외곽점만 구함
        transformation_info['edge_point'] = edge_point
        print("[로그] 통합 변환 - 외곽점 찾기 완료")

        # -------------------------------------------------------------------------
        # 완전히 새로운 접근법: 원본 메쉬와 변환된 메쉬의 정점을 직접 이용해 변환 행렬 계산
        # -------------------------------------------------------------------------
        print("[로그] 새로운 방식으로 변환 행렬 계산 시작")
        try:
            # 원본 메쉬와 변환된 메쉬의 정점
            src_points = mesh.points
            dst_points = y_aligned_mesh.points
            
            # 최소 필요한 포인트 수 (여러 점 사용해서 정확도 향상)
            min_points = min(1000, len(src_points))
            
            # 포인트 샘플링 (모든 점을 다 사용하면 느려질 수 있음)
            step = max(1, len(src_points) // min_points)
            src_sample = src_points[::step]
            dst_sample = dst_points[::step]
            
            # 실제 사용한 샘플 수
            n_samples = len(src_sample)
            print(f"[로그] 변환 행렬 계산에 {n_samples}개 정점 사용")
            
            # 호모지니어스 좌표로 변환
            src_homogeneous = np.ones((n_samples, 4))
            src_homogeneous[:, :3] = src_sample
            
            # 최소 제곱법으로 변환 행렬 계산 (Ax = b 형태로 풀기)
            # 각 점 (x,y,z)에 대해 변환 행렬의 각 행을 계산
            
            # 준비: 큰 행렬 A와 b 구성
            A = np.zeros((n_samples * 3, 12))
            b = np.zeros(n_samples * 3)
            
            for i in range(n_samples):
                # 원본 점
                px, py, pz = src_sample[i]
                # 대상 점
                qx, qy, qz = dst_sample[i]
                
                # x' = m11*x + m12*y + m13*z + m14
                A[i*3, 0] = px
                A[i*3, 1] = py
                A[i*3, 2] = pz
                A[i*3, 3] = 1
                b[i*3] = qx
                
                # y' = m21*x + m22*y + m23*z + m24
                A[i*3+1, 4] = px
                A[i*3+1, 5] = py
                A[i*3+1, 6] = pz
                A[i*3+1, 7] = 1
                b[i*3+1] = qy
                
                # z' = m31*x + m32*y + m33*z + m34
                A[i*3+2, 8] = px
                A[i*3+2, 9] = py
                A[i*3+2, 10] = pz
                A[i*3+2, 11] = 1
                b[i*3+2] = qz
            
            # 최소 제곱법으로 풀기
            # x = (A^T A)^(-1) A^T b
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # 계산된 변환 행렬의 맞춤 오차 출력
            if len(residuals) > 0:
                avg_error = np.sqrt(np.sum(residuals) / n_samples)
                print(f"[로그] 변환 행렬 맞춤 평균 오차: {avg_error:.9f}")
            
            # 변환 행렬로 재구성
            transform_matrix = np.eye(4)
            transform_matrix[0, :] = x[0:4]
            transform_matrix[1, :] = x[4:8]
            transform_matrix[2, :] = x[8:12]
            
            # 변환 행렬 저장
            transformation_info['transform_matrix'] = transform_matrix
            
            # 검증: 샘플 포인트를 변환해서 실제 변환과 비교
            test_src = src_sample[:10]  # 처음 10개 포인트만 테스트
            test_dst = dst_sample[:10]
            
            # 샘플 포인트에 변환 행렬 적용
            test_homogeneous = np.ones((len(test_src), 4))
            test_homogeneous[:, :3] = test_src
            transformed = np.dot(test_homogeneous, transform_matrix.T)
            transformed = transformed[:, :3]  # 다시 3D 좌표로 변환
            
            # 실제 변환된 포인트와 비교
            errors = np.linalg.norm(transformed - test_dst, axis=1)
            max_error = np.max(errors)
            avg_error = np.mean(errors)
            print(f"[로그] 변환 검증: 평균 오차 = {avg_error:.9f}, 최대 오차 = {max_error:.9f}")
            
            print("[로그] 새로운 방식으로 변환 행렬 계산 완료")
            
        except Exception as e:
            print(f"[오류] 새로운 변환 행렬 계산 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 발생 시 기존 방식으로 계산된 변환 행렬 사용
            print("[로그] 기존 방식으로 변환 행렬 계산")
            # OBB 정렬 변환 행렬
            T1 = np.eye(4)
            T1[:3, 3] = -obb_center
            
            R = np.eye(4)
            R[:3, :3] = rotation_matrix
            
            obb_transform = np.dot(R, T1)
            
            # Y축 방향 정렬 변환 행렬
            y_transform = np.eye(4)
            if need_y_rotation:
                y_transform[:3, :3] = np.array([
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]
                ])
            
            final_transform = np.dot(y_transform, obb_transform)
            transformation_info['transform_matrix'] = final_transform
            print("[로그] 기존 방식으로 변환 행렬 계산 완료")

        return y_aligned_mesh, transformation_info

    except Exception as e:
        print(f"[오류] transform_mesh 함수에서 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 기본값 반환
        return mesh.copy(), {'transform_matrix': np.eye(4)}

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
    
def get_transform_matrix(mesh):
    """
    메쉬의 변환 행렬만 계산하여 반환합니다.
    
    Args:
        mesh: PyVista 메쉬
        
    Returns:
        transform_matrix: 원본 메쉬에서 변환된 메쉬로의 변환 행렬
    """
    try:
        print("[로그] 변환 행렬 계산 시작")
        _, transformation_info = transform_mesh(mesh)
        transform_matrix = transformation_info.get('transform_matrix', np.eye(4))
        print("[로그] 변환 행렬 계산 완료")
        return transform_matrix
    except Exception as e:
        print(f"[오류] 변환 행렬 계산 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return np.eye(4)

def apply_transform(mesh, transform_matrix):
    """
    메쉬에 변환 행렬을 적용합니다.
    
    Args:
        mesh: PyVista 메쉬
        transform_matrix: 4x4 변환 행렬
        
    Returns:
        transformed_mesh: 변환된 메쉬
    """
    try:
        print("[로그] 변환 행렬 적용 시작")
        
        # 메쉬 복사
        transformed_mesh = mesh.copy()
        
        # 메쉬 정점 얻기
        vertices = mesh.points
        
        # 호모지니어스 좌표로 변환 (nx3 -> nx4)
        n = vertices.shape[0]
        homogeneous_vertices = np.ones((n, 4))
        homogeneous_vertices[:, :3] = vertices
        
        # 변환 행렬 적용
        transformed_homogeneous = np.dot(homogeneous_vertices, transform_matrix.T)
        
        # 호모지니어스 좌표에서 다시 3D 좌표로 변환
        transformed_vertices = transformed_homogeneous[:, :3]
        
        # 변환된 정점 적용
        transformed_mesh.points = transformed_vertices
        
        print("[로그] 변환 행렬 적용 완료")
        return transformed_mesh
    except Exception as e:
        print(f"[오류] 변환 행렬 적용 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return mesh.copy()

def validate_transformation(mesh, debug=False):
    """
    변환 결과를 검증합니다.
    표준 파이프라인과 단일 변환 행렬 적용 결과를 비교합니다.
    
    Args:
        mesh: 원본 메쉬
        debug: 디버깅 정보 출력 여부
        
    Returns:
        mean_error: 두 방법 간의 평균 오차 (mm)
        max_error: 두 방법 간의 최대 오차 (mm)
        transformed_standard: 표준 파이프라인으로 변환된 메쉬
        transformed_direct: 단일 변환 행렬로 변환된 메쉬
    """
    try:
        print("[로그] 변환 검증 시작")
        
        # 1. 표준 파이프라인으로 변환
        transformed_standard, info = transform_mesh(mesh)
        standard_points = transformed_standard.points
        
        # 2. 단일 변환 행렬로 변환
        transform_matrix = info['transform_matrix']
        transformed_direct = apply_transform(mesh, transform_matrix)
        direct_points = transformed_direct.points
        
        # 3. 두 결과 비교
        if len(standard_points) != len(direct_points):
            print("[오류] 두 메쉬의 정점 수가 다릅니다!")
            return float('inf'), float('inf'), transformed_standard, transformed_direct
        
        # 각 정점 간의 거리 계산
        distances = np.linalg.norm(standard_points - direct_points, axis=1)
        
        # 평균 및 최대 오차
        mean_error = np.mean(distances)
        max_error = np.max(distances)
        
        print(f"[로그] 변환 오차: 평균 = {mean_error:.9f}mm, 최대 = {max_error:.9f}mm")
        
        # 디버깅 정보 출력
        if debug:
            # 가장 큰 차이가 나는 지점 찾기
            if max_error > 1e-6:  # 의미 있는 차이가 있는 경우만
                max_error_idx = np.argmax(distances)
                print(f"[디버그] 최대 오차 위치 (인덱스 {max_error_idx}):")
                print(f"  표준 파이프라인: {standard_points[max_error_idx]}")
                print(f"  변환 행렬 적용: {direct_points[max_error_idx]}")
                
                # 변환 행렬 정보 출력
                print(f"[디버그] 변환 행렬:")
                print(transform_matrix)
                
                # 오차 분포 출력
                percentiles = [50, 75, 90, 95, 99, 99.9]
                for p in percentiles:
                    p_value = np.percentile(distances, p)
                    print(f"[디버그] {p}th 백분위수: {p_value:.9f}mm")
        
        return mean_error, max_error, transformed_standard, transformed_direct
    except Exception as e:
        print(f"[오류] 변환 검증 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), float('inf'), None, None

def SlieArchIosAlignAndSegmentation(path, return_transform_only=False, visualize=True):
    """
    IOS 스캔 데이터를 정렬하고 분할합니다.
    
    Args:
        path: 메쉬 파일 경로
        return_transform_only: True인 경우 변환 행렬만 반환
        visualize: 시각화 여부
        
    Returns:
        if return_transform_only:
            transform_matrix: 변환 행렬만 반환
        else:
            final_mask_mesh: 분할된 메쉬
            transform_matrix: 변환 행렬
            transformation_info: 변환 정보 (선택적)
    """
    if path == "":
        raise ValueError("path is empty")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file or directory {path} does not exist.")
            
    mesh = pv.read(path)
    
    # 변환 행렬만 반환하는 경우
    if return_transform_only:
        transform_matrix = get_transform_matrix(mesh)
        return transform_matrix

    # 메시 정렬
    transformed_mesh, transformation_info = transform_mesh(mesh)
    transformed_obb = get_obb(transformed_mesh)
    
    # 최종 변환된 메쉬에도 모든 작업 적용 및 결합 결과 표시
    region_mask_final = select_region_by_angle(transformed_mesh)
    boundary_indices_final = find_boundary_points(transformed_mesh, region_mask_final, n_neighbors=15)
    grown_region_final = region_growing(transformed_mesh, boundary_indices_final)

    # 최종 결과 시각화 (선택 영역 + Region Growing 결과)
    final_mask = np.logical_or(region_mask_final, grown_region_final)
    final_colored_mesh = visualize_region(transformed_mesh, final_mask, color=[0, 0.8, 0.8])
    final_mask_mesh = transformed_mesh.extract_points(final_mask)

    # seg_output_path = os.path.join(output_dir, f"segmented_arch_{timestamp}.stl")
    # segmented_mesh.save(seg_output_path, binary=False)
    # print(f"[메인] 세그먼테이션 결과 ASCII STL로 저장 완료: {seg_output_path}")
    
    # 변환 행렬 추출
    transform_matrix = transformation_info.get('transform_matrix', np.eye(4))

    # 시각화가 필요한 경우
    if visualize:
        p = pv.Plotter()
        # p.add_mesh(transformed_mesh, color='#555555', opacity=0.5)
        p.add_mesh(transformed_obb, color='red', line_width=2)
        p.add_mesh(final_mask_mesh, opacity=0.5)
        # p.add_mesh(boundary_indices_final, color='yellow', opacity=0.5)
        # p.add_mesh(grown_region_final, color='red', opacity=0.5)
        p.show()

    # 최종 결과 반환: 분할된 메쉬와 변환 행렬 모두 반환
    return final_mask_mesh, transform_matrix, transformation_info

if __name__ == "__main__":
    model_path = "assets/data/transformed/transformed_model_20250408_091623.stl"
    
    # 1. 전체 처리 실행: 세그멘테이션 결과와 변환 행렬 얻기
    print("[메인] IOS 아치 정렬 및 세그멘테이션 시작")
    segmented_mesh, transform_matrix, info = SlieArchIosAlignAndSegmentation(model_path, visualize=True)
    
    # 2. 결과 출력
    print("[메인] 처리 완료")
    print(f"세그먼테이션 결과: {segmented_mesh.n_points}개 정점, {segmented_mesh.n_cells}개 셀")
    print("변환 행렬:")
    print(transform_matrix)
    
    # 3. STL 파일로 저장 (선택적)
    try:
        # 결과 디렉토리 확인 및 생성
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일명 생성 (현재 시간 기반)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 세그먼테이션 결과 저장
        seg_output_path = os.path.join(output_dir, f"segmented_arch_{timestamp}.stl")
        # UnstructuredGrid를 PolyData로 변환
        surface_mesh = segmented_mesh.extract_surface()
        surface_mesh.save(seg_output_path, binary=False)  # ASCII 형식으로 저장
        print(f"[메인] 세그먼테이션 결과 ASCII STL로 저장 완료: {seg_output_path}")
        