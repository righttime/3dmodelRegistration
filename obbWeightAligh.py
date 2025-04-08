import numpy as np
import pyvista as pv
import copy
import os
import time
from datetime import datetime
from scipy.spatial import KDTree

def ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2, epsilon=1e-6):
    """
    Möller–Trumbore 알고리즘을 사용한 레이-삼각형 교차 계산
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    
    # 레이와 삼각형이 거의 평행하면 교차 없음
    if abs(a) < epsilon:
        return False, None
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    # 삼각형 외부 교차
    if u < 0.0 or u > 1.0:
        return False, None
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    
    # 삼각형 외부 교차
    if v < 0.0 or u + v > 1.0:
        return False, None
    
    # t값 계산 (ray_origin + t * ray_direction = 교차점)
    t = f * np.dot(edge2, q)
    
    # 교차점이 레이 뒤에 있음
    if t < epsilon:
        return False, None
    
    # 교차점 계산
    intersection_point = ray_origin + t * ray_direction
    return True, (intersection_point, t)

def find_ray_mesh_intersection(mesh, ray_origin, ray_direction):
    """
    메쉬와 레이의 모든 교차점 찾기 (정확하지만 느림)
    """
    vertices = mesh.points
    triangles = mesh.faces.reshape(-1, 4)[:, 1:4]  # PyVista 메쉬에서 삼각형 추출
    
    intersections = []
    
    for triangle_idx in range(len(triangles)):
        v0 = vertices[triangles[triangle_idx][0]]
        v1 = vertices[triangles[triangle_idx][1]]
        v2 = vertices[triangles[triangle_idx][2]]
        
        hit, result = ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2)
        if hit:
            intersection_point, t = result
            intersections.append((intersection_point, t))
    
    # 거리에 따라 정렬
    if intersections:
        intersections.sort(key=lambda x: x[1])  # t값(거리)으로 정렬
        return [point for point, _ in intersections]
    
    return []

def find_ray_mesh_intersection_approximate(mesh, ray_origin, ray_direction, max_distance_from_ray=5.0):
    """
    근사적인 레이캐스팅 (빠르지만 정확도 개선)
    실제 레이 경로 상 또는 가까이에 있는 정점 중 가장 먼 점을 찾음
    max_distance_from_ray: 레이에서 이 거리 이내에 있는 점들만 고려 (단위: 모델 스케일)
    """
    vertices = mesh.points
    
    # 레이 방향 벡터 정규화
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    # 각 정점에서 레이 원점까지의 벡터
    vec_to_points = vertices - ray_origin
    
    # 레이 방향으로의 투영 거리 계산
    projections = np.dot(vec_to_points, ray_direction)
    
    # 투영 값이 양수인 점만 선택 (레이 방향에 있는 점)
    valid_indices = np.where(projections > 0)[0]
    
    if len(valid_indices) > 0:
        # 유효한 정점들
        valid_vertices = vertices[valid_indices]
        valid_projections = projections[valid_indices]
        
        # 레이에서 각 정점까지의 최단 거리 계산
        # 공식: ||(p - o) - ((p - o)·d)d|| 여기서 p는 정점, o는 레이 원점, d는 레이 방향 벡터
        projection_points = ray_origin + ray_direction.reshape(1, 3) * valid_projections.reshape(-1, 1)
        distances_from_ray = np.linalg.norm(valid_vertices - projection_points, axis=1)
        
        # 레이에서 max_distance_from_ray 이내에 있는 점들만 필터링
        near_ray_indices = np.where(distances_from_ray <= max_distance_from_ray)[0]
        
        if len(near_ray_indices) > 0:
            # 레이에 가까우면서 레이 방향으로 가장 먼 점 찾기
            filtered_indices = valid_indices[near_ray_indices]
            max_proj_idx = filtered_indices[np.argmax(projections[filtered_indices])]
            return [vertices[max_proj_idx]]
    
    return []

def compute_vertex_normals(vertices, triangles):
    """
    NumPy를 사용하여 정점 법선 벡터를 계산합니다.
    
    Args:
        vertices: 정점 좌표 배열 (N, 3)
        triangles: 삼각형 인덱스 배열 (M, 3)
        
    Returns:
        vertex_normals: 정점 법선 벡터 배열 (N, 3)
    """
    # 초기화
    vertex_normals = np.zeros_like(vertices, dtype=np.float32)
    
    # 삼각형 법선 계산
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # 삼각형 변 벡터
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # 삼각형 법선 계산 (정규화 안 함)
    triangle_normals = np.cross(edge1, edge2)
    
    # 각 삼각형의 법선을 해당 정점에 누적
    for i in range(len(triangles)):
        for j in range(3):
            vertex_normals[triangles[i, j]] += triangle_normals[i]
    
    # 법선 정규화 (안전하게 처리)
    norms = np.linalg.norm(vertex_normals, axis=1)
    
    # 0이 아닌 값들만 정규화
    for i in range(len(vertex_normals)):
        if norms[i] > 1e-10:  # 작은 숫자로 0 체크
            vertex_normals[i] = vertex_normals[i] / norms[i]
    
    return vertex_normals

def region_growing(mesh, seed_point, max_angle_diff=90.0, max_distance=150.0):
    """
    순수 법선 벡터 유사성 기반 리전 그로잉 (고립된 삼각형 문제 해결 버전)
    시드 포인트에서 시작하여 인접한 정점들 중 법선 벡터가 비슷한 정점들로 확장
    
    seed_point: 시작점 (외곽점)
    max_angle_diff: 법선 벡터 사이의 최대 허용 각도 차이 (도 단위)
    max_distance: 시드 포인트에서의 최대 허용 거리
    
    반환: 리전에 포함된 정점 인덱스 목록
    """
    vertices = mesh.points
    
    # PyVista 메쉬에서 삼각형 추출
    triangles = mesh.faces.reshape(-1, 4)[:, 1:4]
    
    # 정점 법선 계산
    vertex_normals = compute_vertex_normals(vertices, triangles)
    
    # 시드 포인트에 가장 가까운 정점들 찾기 (복수의 시드 사용)
    distances = np.linalg.norm(vertices - seed_point, axis=1)
    # 거리 기준 상위 10개 정점을 시드로 사용
    close_vertex_indices = np.argsort(distances)[:10]
    
    print(f"시드 포인트 근처의 정점 10개: {close_vertex_indices}")
    
    seed_idx = close_vertex_indices[0]  # 가장 가까운 정점
    seed_normal = vertex_normals[seed_idx]
    
    print(f"주 시드 정점: {seed_idx}, 법선 벡터: {seed_normal}")
    
    # 전체 메쉬의 연결성 검사
    # 각 삼각형에 포함된 정점 기록 및 각 정점이 어떤 삼각형들에 속하는지 기록
    vertex_to_triangles = [[] for _ in range(len(vertices))]
    for i in range(len(triangles)):
        for vertex_idx in triangles[i]:
            vertex_to_triangles[vertex_idx].append(i)
    
    # 연결성 통계 (디버깅용)
    connection_counts = np.array([len(tris) for tris in vertex_to_triangles])
    print(f"정점별 연결된 삼각형 수 통계:")
    print(f"  최소: {np.min(connection_counts)}, 최대: {np.max(connection_counts)}")
    print(f"  평균: {np.mean(connection_counts):.2f}, 중앙값: {np.median(connection_counts)}")
    
    # 메쉬 정점 간 인접성 그래프 구축 (직접적으로 연결된 정점들)
    adjacency = [set() for _ in range(len(vertices))]
    for tri in triangles:
        adjacency[tri[0]].add(tri[1])
        adjacency[tri[0]].add(tri[2])
        adjacency[tri[1]].add(tri[0])
        adjacency[tri[1]].add(tri[2])
        adjacency[tri[2]].add(tri[0])
        adjacency[tri[2]].add(tri[1])
    
    # 복수의 시드 정점을 사용한 리전 그로잉 초기화
    region = set()
    visited = set()
    queue = []
    
    # 여러 시드 추가 (고립된 삼각형 문제 방지)
    for idx in close_vertex_indices:
        if idx not in visited and len(adjacency[idx]) > 0:
            visited.add(idx)
            region.add(idx)
            queue.append(idx)
            
    if not queue:
        # 시드 포인트 주변에 적절한 시드 정점이 없는 경우, 전체 메쉬에서 추가 시드 선택
        # 연결성이 높은 정점들을 시드로 추가
        for i in range(min(50, len(vertices))):
            random_idx = np.random.randint(0, len(vertices))
            if random_idx not in visited and len(adjacency[random_idx]) > 5:
                visited.add(random_idx)
                region.add(random_idx)
                queue.append(random_idx)
                if len(queue) >= 5:  # 최대 5개의 추가 시드
                    break
                
    print(f"초기 시드 정점 수: {len(queue)}")
    
    cos_threshold = np.cos(np.radians(max_angle_diff))
    print(f"리전 그로잉 시작: 최대 각도 차이 {max_angle_diff}도 (cos {cos_threshold:.4f}), 최대 거리 {max_distance}")
    
    # BFS 기반 리전 그로잉
    iteration = 0
    last_region_size = len(region)
    
    while queue:
        current_idx = queue.pop(0)  # FIFO 방식으로 pop
        current_normal = vertex_normals[current_idx]
        
        # 현재 정점과 직접 연결된 모든 정점 탐색
        for neighbor_idx in adjacency[current_idx]:
            if neighbor_idx in visited:
                continue
            
            # 거리 조건 확인 (선택적)
            if max_distance > 0:
                vertex_distance = np.linalg.norm(vertices[neighbor_idx] - seed_point)
                if vertex_distance > max_distance:
                    continue
            
            # 이웃 정점의 법선
            neighbor_normal = vertex_normals[neighbor_idx]
            
            # 법선 각도 계산 (절대값을 사용하여 방향성 무시)
            cos_angle = abs(np.dot(current_normal, neighbor_normal))
            
            # 각도 조건 확인
            if cos_angle >= cos_threshold:
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)
                region.add(neighbor_idx)
        
        # 로깅: 리전 크기가 크게 변하면 출력
        iteration += 1
        if iteration % 1000 == 0 or len(region) - last_region_size >= 100:
            print(f"반복 {iteration}, 현재 리전 크기: {len(region)}개 정점 ({len(region) - last_region_size}개 추가)")
            last_region_size = len(region)
    
    print(f"리전 그로잉 완료: {len(region)}개 정점 포함")
    
    # 리전이 너무 작으면 거리 기준으로 정점 추가
    if len(region) < 100:
        print(f"리전이 너무 작습니다. 가장 가까운 정점 100개를 추가합니다.")
        close_vertices = np.argsort(distances)[:100]
        region.update(close_vertices)
        print(f"확장 후 리전 크기: {len(region)}개 정점")
    
    return list(region)

def visualize_region(mesh, region_vertices, color=[0, 1, 1]):
    """
    메쉬에서 선택된 리전을 시각화 (색상으로 구분)
    """
    # 원본 메쉬 복사
    region_mesh = mesh.copy()
    
    # 색상 배열 생성
    scalars = np.ones(len(region_mesh.points)) * 0.7  # 밝은 회색
    
    # 리전 정점들을 지정된 색상 인덱스로 설정
    for idx in region_vertices:
        scalars[idx] = 2.0  # 청록색 표시용 값
    
    # 색상 저장
    region_mesh['region_scalars'] = scalars
    
    return region_mesh

def main():
    # Transformed 3D 모델 파일 경로
    model_path = "assets/data/transformed/transformed_model_20250408_091623.stl"
    
    # 레이캐스팅 방식 선택
    # 0: 정확한 방식(느림), 1: 근사적 방식(빠름)
    raycasting_method = 1
    
    # 리전 그로잉 파라미터
    use_region_growing = True
    max_angle_diff = 10.8    # 법선 벡터 최대 각도 차이 (도)
    max_distance = 10.0      # 시드 포인트에서 최대 거리 (0으로 설정하면 거리 제한 없음)
    
    # 메쉬 로드
    print(f"=== 메쉬 로딩 시작 ===")
    mesh = pv.read(model_path)
    
    # 메쉬 정보 출력
    print(f"=== 메쉬 정보 ===")
    print(f"점 개수: {mesh.n_points}")
    print(f"삼각형 개수: {mesh.n_cells}")
    
    # 포인트 클라우드 생성
    print(f"=== 포인트 클라우드 생성 ===")
    vertices = mesh.points
    
    # OBB (Oriented Bounding Box) 계산
    print(f"=== OBB 계산 ===")
    # PyVista는 OBB를 직접 지원하지 않으므로 PCA를 사용하여 계산
    mean_pt = np.mean(vertices, axis=0)
    pts = vertices - mean_pt
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # 크기 순으로 정렬 (큰 것부터)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # OBB 회전 행렬 (각 열은 OBB의 주축)
    obb_rotation = eigvecs
    
    # OBB 중심은 포인트 클라우드의 중심
    obb_center = mean_pt
    
    # OBB 크기 계산
    # 각 축에 대한 투영 거리의 최대-최소
    projections = np.dot(pts, obb_rotation)
    obb_mins = np.min(projections, axis=0)
    obb_maxs = np.max(projections, axis=0)
    obb_extent = obb_maxs - obb_mins
    
    # 무게중심 계산 (정점의 평균)
    center_of_mass = np.mean(vertices, axis=0)
    
    # OBB 정보 출력
    print(f"OBB 중심: {obb_center}")
    print(f"무게중심: {center_of_mass}")
    print(f"OBB 크기(변의 길이): {obb_extent}")
    print(f"OBB 회전 행렬:\n{obb_rotation}")
    
    # ---------- 새로운 좌표계 설정 ----------
    
    # 1. OBB 변의 길이에 따라 축 설정
    # 가장 짧은 변 -> z축, 가장 긴 변 -> x축, 중간 길이 변 -> y축
    extent_indices = np.argsort(obb_extent)
    z_axis_idx = extent_indices[0]  # 가장 짧은 변
    y_axis_idx = extent_indices[1]  # 중간 길이 변
    x_axis_idx = extent_indices[2]  # 가장 긴 변
    
    # OBB 회전 행렬에서 축 벡터 추출
    x_axis = obb_rotation[:, x_axis_idx]
    y_axis = obb_rotation[:, y_axis_idx]
    z_axis = obb_rotation[:, z_axis_idx]
    
    # 2. 방향 조정: z축은 무게중심이 OBB 중심보다 아래에 오도록
    com_to_obb = center_of_mass - obb_center
    if np.dot(z_axis, com_to_obb) > 0:
        z_axis = -z_axis  # 방향 반전
    
    # 3. 직교성 유지를 위해 축 재계산
    # x축과 z축의 외적으로 y축 재계산
    y_axis = np.cross(z_axis, x_axis)
    # y축과 z축의 외적으로 x축 재계산
    x_axis = np.cross(y_axis, z_axis)
    
    # 단위 벡터로 정규화
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 새로운 좌표계 행렬 생성
    new_rotation = np.column_stack((x_axis, y_axis, z_axis))
    
    # 5. 무게중심에서 y축 방향으로 레이캐스팅하여 마지막에 맞은 외곽점 찾기
    print(f"=== 레이캐스팅 시작 ===")
    
    # 레이캐스팅 방식에 따른 설명
    method_names = ["정확한 방식 (느림)", "근사적 방식 (빠름)"]
    print(f"레이캐스팅 방식: {method_names[raycasting_method]}")
    
    # OBB 중심과 무게중심의 y축 관계에 따라 초기 레이 방향 결정
    # OBB 중심의 y 좌표가 무게중심의 y 좌표보다 크면 -y 방향으로, 아니면 +y 방향으로 쏨
    obb_center_projection = np.dot(obb_center - center_of_mass, y_axis)
    
    if obb_center_projection > 0:
        initial_ray_direction = -y_axis  # -y축 방향
        print("OBB 중심이 무게중심보다 +y축 방향에 있어 -y축 방향으로 레이캐스팅을 시작합니다.")
    else:
        initial_ray_direction = y_axis  # +y축 방향
        print("OBB 중심이 무게중심보다 -y축 방향에 있어 +y축 방향으로 레이캐스팅을 시작합니다.")
    
    start_time = time.time()
    ray_origin = center_of_mass
    ray_direction = initial_ray_direction.copy()
    
    # 레이캐스팅 방식에 따라 외곽점 계산
    if raycasting_method == 0:  # 정확한 방식
        intersections = find_ray_mesh_intersection(mesh, ray_origin, ray_direction)
        
        if intersections:
            edge_point = intersections[-1]  # 마지막 교차점
        else:
            # 반대 방향으로 시도
            print(f"첫 번째 방향으로 교차점이 없습니다. 반대 방향으로 시도합니다.")
            ray_direction = -ray_direction
            intersections = find_ray_mesh_intersection(mesh, ray_origin, ray_direction)
            
            if intersections:
                edge_point = intersections[-1]
            else:
                # 투영 방식 사용
                print("레이캐스팅으로 외곽점을 찾지 못했습니다. 투영 방식을 사용합니다.")
                y_projections = np.dot(vertices - center_of_mass, y_axis)
                max_idx = np.argmax(y_projections)
                edge_point = vertices[max_idx]
                ray_direction = initial_ray_direction  # 원래 방향으로 복원
                
    else:  # 근사적 방식 (방법 1)
        intersections = find_ray_mesh_intersection_approximate(mesh, ray_origin, ray_direction)
        
        if intersections:
            edge_point = intersections[-1]
        else:
            # 반대 방향으로 시도
            print(f"첫 번째 방향으로 교차점이 없습니다. 반대 방향으로 시도합니다.")
            ray_direction = -ray_direction
            intersections = find_ray_mesh_intersection_approximate(mesh, ray_origin, ray_direction)
            
            if intersections:
                edge_point = intersections[-1]
            else:
                # 투영 방식 사용
                print("레이캐스팅으로 외곽점을 찾지 못했습니다. 투영 방식을 사용합니다.")
                y_projections = np.dot(vertices - center_of_mass, y_axis)
                max_idx = np.argmax(y_projections)
                edge_point = vertices[max_idx]
                ray_direction = initial_ray_direction  # 원래 방향으로 복원
    
    end_time = time.time()
    print(f"y축 방향 외곽점: {edge_point}")
    print(f"레이캐스팅 시간: {end_time - start_time:.4f}초")
    
    # 외곽점에서 리전 그로잉 수행
    region_vertices = []
    region_center = edge_point.copy()  # 기본값은 외곽점
    
    if use_region_growing:
        print(f"=== 리전 그로잉 시작 ===")
        start_time = time.time()
        region_vertices = region_growing(mesh, edge_point, 
                                         max_angle_diff=max_angle_diff, 
                                         max_distance=max_distance)
        end_time = time.time()
        print(f"리전 그로잉 결과: {len(region_vertices)}개 정점 포함")
        print(f"리전 그로잉 시간: {end_time - start_time:.4f}초")
        
        # 리전 정점들의 중심점 계산
        if len(region_vertices) > 0:
            region_vertices_coords = vertices[region_vertices]
            region_center = np.mean(region_vertices_coords, axis=0)
            print(f"리전 중심점: {region_center}")
        
        # 리전을 시각화한 메쉬 생성
        region_mesh = visualize_region(mesh, region_vertices, color=[0, 1, 1])
    
    # ====== 커스텀 좌표계를 기본 좌표계에 맞추는 변환 ======
    
    # 1. 변환 행렬 생성
    # 회전 행렬: 커스텀 좌표계 기저벡터를 기본 좌표계 기저벡터([1,0,0], [0,1,0], [0,0,1])로 변환
    rotation_to_standard = np.linalg.inv(new_rotation)
    
    # 이동 벡터: 기준점을 원점으로 이동 (리전 중심점 또는 외곽점)
    translation_to_origin = -region_center
    
    # 2. 메쉬를 기본 좌표계에 맞게 변환
    transformed_mesh = mesh.copy()
    
    # 변환 적용
    transformed_points = vertices + translation_to_origin  # 이동
    transformed_points = np.dot(transformed_points, rotation_to_standard)  # 회전
    
    # 변환된 포인트 적용
    transformed_mesh.points = transformed_points
    
    # 변환된 메쉬의 무게중심 계산
    transformed_com = np.mean(transformed_points, axis=0)
    
    # ====== 결과 시각화 ======
    
    # PyVista 플로터 생성
    p = pv.Plotter(window_size=[1024, 768], shape=(2, 2))
    
    # 원본 메쉬 시각화
    p.subplot(0, 0)
    p.add_text("3D Model with OBB", font_size=14)
    p.add_mesh(mesh, color='white', opacity=0.7)
    
    # OBB 시각화
    corners = np.empty((8, 3))
    for i in range(8):
        # 각 코너의 좌표
        corner = np.array([
            obb_mins[0] if (i & 1) == 0 else obb_maxs[0],
            obb_mins[1] if (i & 2) == 0 else obb_maxs[1],
            obb_mins[2] if (i & 4) == 0 else obb_maxs[2]
        ])
        # 회전 및 이동 적용
        corners[i] = np.dot(obb_rotation, corner) + obb_center
    
    # OBB 모서리 생성
    edges = np.array([
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])
    
    lines = pv.PolyData()
    lines.points = corners
    lines.lines = np.hstack([[2, e[0], e[1]] for e in edges])
    p.add_mesh(lines, color='red', line_width=2)
    
    # 좌표계 추가
    origin = center_of_mass
    length = np.max(obb_extent) * 0.2
    
    # X축 (빨강)
    x_endpoint = origin + x_axis * length
    x_line = pv.Line(origin, x_endpoint)
    p.add_mesh(x_line, color='red', line_width=3)
    
    # Y축 (초록)
    y_endpoint = origin + y_axis * length
    y_line = pv.Line(origin, y_endpoint)
    p.add_mesh(y_line, color='green', line_width=3)
    
    # Z축 (파랑)
    z_endpoint = origin + z_axis * length
    z_line = pv.Line(origin, z_endpoint)
    p.add_mesh(z_line, color='blue', line_width=3)
    
    # 레이와 포인트 추가
    ray_end = ray_origin + ray_direction * length * 1.5
    ray_line = pv.Line(ray_origin, ray_end)
    p.add_mesh(ray_line, color='yellow', line_width=3)
    
    # 중요 포인트 추가
    p.add_mesh(pv.Sphere(center=center_of_mass, radius=length/20), color='green')
    p.add_mesh(pv.Sphere(center=obb_center, radius=length/20), color='red')
    p.add_mesh(pv.Sphere(center=edge_point, radius=length/20), color='blue')
    
    # 리전 그로잉 결과 시각화
    if use_region_growing and region_vertices:
        p.subplot(0, 1)
        p.add_text("Region Growing Result", font_size=14)
        region_mesh.set_active_scalars('region_scalars')
        
        # 간단한 사용자 정의 색상 매핑
        # 회색(0.7)과 청록색(2.0)의 두 가지 색상만 사용
        p.add_mesh(region_mesh, scalars='region_scalars',
                  clim=[0.7, 2.0],  # 색상 범위
                  cmap=['lightgray', 'cyan'])  # 범위의 시작과 끝에 사용할 색상
        
        # 리전 중심 추가
        if len(region_vertices) > 3:
            p.add_mesh(pv.Sphere(center=region_center, radius=length/20), color='cyan')
    
    # 변환된 메쉬 시각화
    p.subplot(1, 0)
    p.add_text("Transformed Mesh", font_size=14)
    p.add_mesh(transformed_mesh, color='white')
    
    # 표준 좌표계 추가
    origin = [0, 0, 0]
    
    # X축 (빨강)
    x_endpoint = [length, 0, 0]
    x_line = pv.Line(origin, x_endpoint)
    p.add_mesh(x_line, color='red', line_width=3)
    
    # Y축 (초록)
    y_endpoint = [0, length, 0]
    y_line = pv.Line(origin, y_endpoint)
    p.add_mesh(y_line, color='green', line_width=3)
    
    # Z축 (파랑)
    z_endpoint = [0, 0, length]
    z_line = pv.Line(origin, z_endpoint)
    p.add_mesh(z_line, color='blue', line_width=3)
    
    # 변환된 무게중심 표시
    p.add_mesh(pv.Sphere(center=transformed_com, radius=length/20), color='green')
    
    # 전체 요약 보기
    p.subplot(1, 1)
    p.add_text("Combined View", font_size=14)
    p.add_mesh(mesh, color='white', opacity=0.3)
    p.add_mesh(lines, color='red', line_width=2)
    
    # 좌표축 추가
    # X축 (빨강)
    p.add_mesh(x_line, color='red', line_width=3)
    # Y축 (초록)
    p.add_mesh(y_line, color='green', line_width=3)
    # Z축 (파랑)
    p.add_mesh(z_line, color='blue', line_width=3)
    
    # 중요 포인트 추가
    p.add_mesh(pv.Sphere(center=center_of_mass, radius=length/20), color='green')
    p.add_mesh(pv.Sphere(center=obb_center, radius=length/20), color='red')
    p.add_mesh(pv.Sphere(center=edge_point, radius=length/20), color='blue')
    
    if use_region_growing and len(region_vertices) > 3:
        # 리전 중심 표시
        p.add_mesh(pv.Sphere(center=region_center, radius=length/20), color='cyan')
        
        # 리전 정점 표시
        subset_points = vertices[region_vertices]
        point_cloud = pv.PolyData(subset_points)
        p.add_mesh(point_cloud, color='cyan', point_size=5)
    
    # 시각화 창 표시
    p.link_views()
    p.show()
    
    # 변환된 메쉬 저장
    output_dir = "assets/data/aligned"
    os.makedirs(output_dir, exist_ok=True)
    
    # 현재 시간과 레이캐스팅 방식 및 리전 그로잉 여부를 파일 이름에 포함
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_suffix = ["accurate", "approx"][raycasting_method]
    
    if use_region_growing:
        output_path = os.path.join(output_dir, f"aligned_model_{method_suffix}_region_{timestamp}.stl")
    else:
        output_path = os.path.join(output_dir, f"aligned_model_{method_suffix}_{timestamp}.stl")
    
    # STL 파일로 저장
    print(f"=== 변환된 메쉬 저장 ===")
    transformed_mesh.save(output_path)
    print(f"변환된 메쉬가 다음 경로에 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()