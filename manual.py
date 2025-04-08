import open3d as o3d
import numpy as np
import copy

def main():
    # 변환된 3D 모델 파일 경로
    model_path = "assets/data/transformed/transformed_model_20250408_091623.stl"
    
    # 메쉬 로드
    print(f"=== 메쉬 로딩 시작 ===")
    original_mesh = o3d.io.read_triangle_mesh(model_path)
    mesh = copy.deepcopy(original_mesh)  # 원본 복사본 생성
    
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
    
    # 무게중심 계산 (꼭지점 평균)
    vertices = np.asarray(mesh.vertices)
    center_of_mass = np.mean(vertices, axis=0)
    
    # OBB의 중심과 크기(변의 길이) 가져오기
    obb_center = obb.center
    obb_extent = obb.extent  # [x, y, z] 크기
    obb_rotation = obb.R  # 회전 행렬
    
    print(f"OBB 중심: {obb_center}")
    print(f"무게중심: {center_of_mass}")
    print(f"OBB 크기(변의 길이): {obb_extent}")
    
    # 1. 축 결정: OBB의 변 길이에 따라
    # OBB의 축과 길이를 정렬
    extents_with_axes = [(obb_extent[i], i) for i in range(3)]
    extents_with_axes.sort()  # 길이에 따라 정렬
    
    # 가장 짧은 변 -> z축, 가장 긴 변 -> x축, 중간 길이 변 -> y축
    z_axis_idx = extents_with_axes[0][1]  # 가장 짧은 변
    y_axis_idx = extents_with_axes[1][1]  # 중간 길이 변
    x_axis_idx = extents_with_axes[2][1]  # 가장 긴 변
    
    # OBB 축 벡터 추출
    x_axis = obb_rotation[:, x_axis_idx]
    y_axis = obb_rotation[:, y_axis_idx]
    z_axis = obb_rotation[:, z_axis_idx]
    
    # 4. z축 방향 결정 (무게중심이 OBB 중심보다 아래에 있도록)
    z_direction = center_of_mass - obb_center
    # z축과 z_direction 사이의 각도의 코사인 값 계산
    cos_angle = np.dot(z_axis, z_direction) / (np.linalg.norm(z_axis) * np.linalg.norm(z_direction))
    
    if cos_angle > 0:  # 양수면 같은 방향
        z_axis = -z_axis  # 방향 반전
    
    # 직교성 유지를 위해 y축 재계산
    y_axis = np.cross(z_axis, x_axis)
    # x축도 다시 계산
    x_axis = np.cross(y_axis, z_axis)
    
    # 단위 벡터로 정규화
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 새로운 좌표계 회전 행렬 (각 열이 좌표축)
    new_rotation = np.column_stack((x_axis, y_axis, z_axis))
    
    # 정렬된 메쉬 생성 (변환된 메쉬)
    aligned_mesh = copy.deepcopy(original_mesh)
    
    # 5. 좌표계 원점 설정을 위한 레이캐스팅
    # Open3D에서는 직접적인 레이캐스팅 기능이 제한적이므로 간단한 방법 사용
    
    # 무게중심에서 y축 방향의 가장 먼 점 찾기
    vertices = np.asarray(aligned_mesh.vertices)
    
    # y축 방향 투영 값 계산
    y_projections = np.dot(vertices - center_of_mass, y_axis)
    
    # 가장 큰 투영 값을 가진 점 (y축 방향으로 가장 먼 점)
    max_idx = np.argmax(y_projections)
    edge_point = vertices[max_idx]
    
    print(f"y축 방향 외곽점: {edge_point}")
    
    # 이 외곽점을 원점으로 설정
    translation_vector = -edge_point
    
    # 메쉬 변환 적용 (원점 이동 후 회전)
    aligned_mesh.translate(translation_vector)
    
    # 회전 중심을 원점으로 설정하고 새로운 회전 적용
    # O3D에서는 회전 행렬의 역행렬을 적용하여 물체를 정렬
    R_inv = np.linalg.inv(new_rotation)
    aligned_mesh.rotate(R_inv, center=[0, 0, 0])
    
    # 법선 재계산
    aligned_mesh.compute_vertex_normals()
    
    # 정렬된 좌표계 생성
    aligned_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
    
    # 정렬된 메쉬의 OBB 계산
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = aligned_mesh.vertices
    aligned_obb = aligned_pcd.get_oriented_bounding_box()
    aligned_obb.color = np.array([0, 1, 0, 0.5])  # 초록색으로 표시
    
    # 원본 메쉬와 정렬된 메쉬 시각화
    print(f"=== 원본 메쉬와 OBB 시각화 ===")
    o3d.visualization.draw_geometries(
        [original_mesh, obb],
        window_name="Original 3D Model with OBB",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )
    
    print(f"=== 정렬된 메쉬와 좌표계 시각화 ===")
    o3d.visualization.draw_geometries(
        [aligned_mesh, aligned_frame, aligned_obb],
        window_name="Aligned 3D Model with Coordinate System",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )
    
    # 정렬된 메쉬 저장
    output_path = "assets/data/transformed/aligned_model.stl"
    o3d.io.write_triangle_mesh(output_path, aligned_mesh)
    print(f"정렬된 메쉬가 다음 경로에 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()