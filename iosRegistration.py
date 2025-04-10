import open3d as o3d
import numpy as np
import copy

# Y=0 평면 정렬
def align_to_occlusal_plane(pcd):
    points = np.asarray(pcd.points)
    y_max = np.max(points[:, 1])
    pcd.translate([0, -y_max, 0])
    return pcd

def try_icp_registration(source, target, voxel_size, init_transform):
    threshold = voxel_size
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return reg_p2p

def iosSmileArchRegistration(targetMeshPath, sourceMeshPath):
    # 메쉬 로드 및 포인트 클라우드 변환    
    target = o3d.io.read_triangle_mesh(targetMeshPath)
    source = o3d.io.read_triangle_mesh(sourceMeshPath)
    
    target_pcd = target.sample_points_uniformly(number_of_points=20000)
    source_pcd = source.sample_points_uniformly(number_of_points=20000)
    
    target_pcd = align_to_occlusal_plane(target_pcd)
    source_pcd = align_to_occlusal_plane(source_pcd)

    # 3. Voxel 다운샘플링
    voxel_size = 0.3    
    target_down = target_pcd.voxel_down_sample(voxel_size=voxel_size)
    source_down = source_pcd.voxel_down_sample(voxel_size=voxel_size)

    # 4. 법선 벡터 계산    
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    # X축 이동 범위 설정 (-5mm에서 +5mm까지)
    x_translations = np.linspace(-10, 10, 5)  # 5개의 X축 위치
    y_rotations = np.linspace(0, 2*np.pi, 8)  # 8개의 Y축 회전

    best_result = None
    best_fitness = 0

    print("Trying different positions and rotations...")
    for x_trans in x_translations:
        for angle in y_rotations:
            # Y축 회전 + X축 이동 변환 행렬
            init_transform = np.array([
                [np.cos(angle), 0, np.sin(angle), x_trans],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]
            ])
            
            result = try_icp_registration(source_down, target_down, init_transform)
            #print(f"X translation: {x_trans:.1f}, Angle: {angle:.2f}, Fitness: {result.fitness:.4f}")
            
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_result = result

    print("\nBest registration result:")
    print(best_result)
    print(f"Best fitness: {best_fitness}")

if __name__ == "__main__":
    targetMeshPath = "assets/library/smile_arch_origin.stl"
    sourceMeshPath = "results/partial_smile_arch.stl"