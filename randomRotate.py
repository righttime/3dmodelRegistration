import open3d as o3d
import numpy as np
import os
from datetime import datetime

def random_rotation_matrix():
    """랜덤한 회전 행렬 생성"""
    # 랜덤한 3차원 회전 (오일러 각 방식)
    angles = np.random.uniform(0, 2 * np.pi, 3)  # 세 축에 대한 랜덤 각도
    
    # 각 축별 회전 행렬
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    # 세 축 회전을 합성
    R = Rz @ Ry @ Rx
    return R

def apply_random_transformation(mesh):
    """메쉬에 랜덤한 회전과 이동만 적용"""
    # 메쉬 중심점 계산
    center = mesh.get_center()
    
    # 랜덤 회전 행렬 생성
    R = random_rotation_matrix()
    
    # 랜덤 이동 벡터 생성 (-50에서 50 사이)
    t = np.random.uniform(-50, 50, 3)
    
    # 4x4 변환 행렬 생성
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    # 메쉬 중심으로 이동하고, 회전 후, 다시 원래 위치로
    mesh.translate(-center)
    mesh.transform(transformation)
    
    return mesh

def main():
    # 3D 모델 파일 경로
    model_path = "assets/data/ios_with_smilearch.stl"
    
    # 메쉬 로드
    print(f"=== 메쉬 로딩 시작 ===")
    mesh = o3d.io.read_triangle_mesh(model_path)
    
    # 메쉬 정보 출력
    print(f"=== 메쉬 정보 ===")
    print(f"점 개수: {len(mesh.vertices)}")
    print(f"삼각형 개수: {len(mesh.triangles)}")
    
    # 메쉬 법선 계산 (시각화에 필요)
    mesh.compute_vertex_normals()
    
    # 랜덤 회전 및 이동 적용
    print(f"=== 랜덤 회전 및 이동 적용 ===")
    mesh = apply_random_transformation(mesh)
    
    # 변환된 메쉬 저장
    output_dir = "assets/data/transformed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 현재 시간을 사용하여 고유한 파일 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"transformed_model_{timestamp}.stl")
    
    # STL 파일로 저장
    print(f"=== 변환된 메쉬 저장 ===")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"변환된 메쉬가 다음 경로에 저장되었습니다: {output_path}")
    
    # 좌표축 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20, origin=[0, 0, 0])
    
    # 메쉬 시각화
    print(f"=== 메쉬 시각화 ===")
    o3d.visualization.draw_geometries(
        [mesh, coordinate_frame],
        window_name="3D Model with Random Rotation and Translation",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()