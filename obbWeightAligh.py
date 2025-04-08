import open3d as o3d
import numpy as np
import copy
import os
import time
from datetime import datetime

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
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
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
    vertices = np.asarray(mesh.vertices)
    
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

def main():
    # Transformed 3D 모델 파일 경로
    model_path = "assets/data/transformed/transformed_model_20250408_091623.stl"
    
    # 레이캐스팅 방식 선택
    # 0: 정확한 방식(느림), 1: 근사적 방식(빠름)
    raycasting_method = 1
    
    # 메쉬 로드
    print(f"=== 메쉬 로딩 시작 ===")
    mesh = o3d.io.read_triangle_mesh(model_path)
    
    # 메쉬 정보 출력
    print(f"=== 메쉬 정보 ===")
    print(f"점 개수: {len(mesh.vertices)}")
    print(f"삼각형 개수: {len(mesh.triangles)}")
    
    # 메쉬 법선 계산 (시각화에 필요)
    mesh.compute_vertex_normals()
    
    # 포인트 클라우드 생성
    print(f"=== 포인트 클라우드 생성 ===")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    
    # OBB (Oriented Bounding Box) 계산
    print(f"=== OBB 계산 ===")
    obb = pcd.get_oriented_bounding_box()
    obb.color = np.array([1, 0, 0])  # 빨간색 (RGB만 지원됨)
    
    # 무게중심 계산 (정점의 평균)
    vertices = np.asarray(mesh.vertices)
    center_of_mass = np.mean(vertices, axis=0)
    
    # OBB 정보 출력
    obb_center = obb.center
    obb_extent = obb.extent  # [x, y, z] 크기
    obb_rotation = obb.R     # 회전 행렬
    
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
    
    # 레이 시각화를 위한 선분 생성
    # 무게중심에서 외곽점으로 직접 선을 그림 (ray_direction과 무관하게)
    ray_points = np.vstack((ray_origin, edge_point))
    ray_lines = np.array([[0, 1]])  # 0번과 1번 점 연결
    
    ray_line_set = o3d.geometry.LineSet()
    ray_line_set.points = o3d.utility.Vector3dVector(ray_points)
    ray_line_set.lines = o3d.utility.Vector2iVector(ray_lines)
    ray_line_set.colors = o3d.utility.Vector3dVector([np.array([1, 1, 0])])  # 노란색
    
    # 원본 메쉬 및 원본 좌표계 저장
    original_mesh = copy.deepcopy(mesh)
    
    # 5. 새로운 좌표계 시각화
    # 원점에 좌표계 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=30, origin=[0, 0, 0])
    
    # 외곽점에 새로운 좌표계 생성
    custom_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=30, origin=edge_point)
    
    # 새로운 좌표계 방향으로 회전
    custom_frame.rotate(new_rotation, center=edge_point)
    
    # 무게중심 표시를 위한 구 생성
    com_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    com_sphere.paint_uniform_color([0, 1, 0])  # 초록색
    com_sphere.compute_vertex_normals()
    com_sphere.translate(center_of_mass)
    
    # OBB 중심 표시를 위한 구 생성
    obb_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    obb_center_sphere.paint_uniform_color([1, 0, 0])  # 빨간색
    obb_center_sphere.compute_vertex_normals()
    obb_center_sphere.translate(obb_center)
    
    # 외곽점 표시를 위한 구 생성
    edge_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    edge_sphere.paint_uniform_color([0, 0, 1])  # 파란색
    edge_sphere.compute_vertex_normals()
    edge_sphere.translate(edge_point)
    
    # ====== 커스텀 좌표계를 기본 좌표계에 맞추는 변환 ======
    
    # 1. 변환 행렬 생성
    # 회전 행렬: 커스텀 좌표계 기저벡터를 기본 좌표계 기저벡터([1,0,0], [0,1,0], [0,0,1])로 변환
    rotation_to_standard = np.linalg.inv(new_rotation)
    
    # 이동 벡터: 외곽점을 원점으로 이동
    translation_to_origin = -edge_point
    
    # 2. 메쉬를 기본 좌표계에 맞게 변환
    transformed_mesh = copy.deepcopy(original_mesh)
    
    # 외곽점을 원점으로 이동
    transformed_mesh.translate(translation_to_origin)
    
    # 커스텀 좌표축을 기본 좌표축과 일치하도록 회전
    transformed_mesh.rotate(rotation_to_standard, center=[0, 0, 0])
    
    # 법선 재계산
    transformed_mesh.compute_vertex_normals()
    
    # 변환된 메쉬의 OBB 계산
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = transformed_mesh.vertices
    transformed_obb = transformed_pcd.get_oriented_bounding_box()
    transformed_obb.color = np.array([0, 1, 0])  # 초록색
    
    # 기본 좌표계 생성
    standard_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=30, origin=[0, 0, 0])
    
    # 변환된 메쉬의 무게중심 계산
    transformed_vertices = np.asarray(transformed_mesh.vertices)
    transformed_com = np.mean(transformed_vertices, axis=0)
    
    # 변환된 무게중심 표시를 위한 구 생성
    transformed_com_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    transformed_com_sphere.paint_uniform_color([0, 1, 0])  # 초록색
    transformed_com_sphere.compute_vertex_normals()
    transformed_com_sphere.translate(transformed_com)
    
    # 메쉬, OBB, 좌표계, 레이 시각화
    print(f"=== 메쉬, OBB, 좌표계, 레이 시각화 ===")
    o3d.visualization.draw_geometries(
        [original_mesh, obb, coordinate_frame, custom_frame, ray_line_set],
        window_name="3D Model with OBB, Custom Coordinate System and Ray",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )
    
    # 모든 요소 함께 시각화
    print(f"=== 모든 요소 시각화 ===")
    o3d.visualization.draw_geometries(
        [original_mesh, obb, custom_frame, com_sphere, obb_center_sphere, edge_sphere, ray_line_set],
        window_name="3D Model with All Elements",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )
    
    # 변환된 메쉬와 기본 좌표계 시각화
    print(f"=== 변환된 메쉬와 기본 좌표계 시각화 ===")
    o3d.visualization.draw_geometries(
        [transformed_mesh, standard_frame, transformed_obb],
        window_name="Transformed 3D Model with Standard Coordinate System",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )
    
    # 변환된 메쉬와 무게중심 시각화
    print(f"=== 변환된 메쉬와 무게중심 시각화 ===")
    o3d.visualization.draw_geometries(
        [transformed_mesh, standard_frame, transformed_com_sphere, transformed_obb],
        window_name="Transformed 3D Model with Center of Mass",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )
    
    # 변환된 메쉬 저장
    output_dir = "assets/data/aligned"
    os.makedirs(output_dir, exist_ok=True)
    
    # 현재 시간과 레이캐스팅 방식을 파일 이름에 포함
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_suffix = ["accurate", "approx"][raycasting_method]
    output_path = os.path.join(output_dir, f"aligned_model_{method_suffix}_{timestamp}.stl")
    
    # STL 파일로 저장
    print(f"=== 변환된 메쉬 저장 ===")
    o3d.io.write_triangle_mesh(output_path, transformed_mesh)
    print(f"변환된 메쉬가 다음 경로에 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()