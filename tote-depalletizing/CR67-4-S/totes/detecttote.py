import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def are_point_clouds_within_distance(pcd1, pcd2, max_distance):
    # Compute distance from each point in pcd1 to the nearest point in pcd2
    distances = np.asarray(pcd1.compute_point_cloud_distance(pcd2))

    # Return True if the closest pair of points is within max_distance
    return np.min(distances) <= max_distance

if __name__ == "__main__":
    # Read point cloud from a .ply file
    pcd = o3d.io.read_point_cloud("ensenso_capture_4.ply")
    X = np.asarray(pcd.points)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )

    Xtote = X[(X[:,2]>600) & (X[:,2]<800) & (X[:,0]>60) & (X[:,1]>-230) & (X[:,1]<80)]
    Xtote=Xtote-np.mean(Xtote,axis=0)

    pcdtote = o3d.geometry.PointCloud()
    pcdtote.points = o3d.utility.Vector3dVector(Xtote)

    # Optional: Save to PLY file
    o3d.io.write_point_cloud("tote_front.ply", pcdtote)

    # o3d.visualization.draw_geometries([coordinate_frame,pcdtote])

    pcd = o3d.io.read_point_cloud("ensenso_capture_2.ply")
    
    
    X = np.asarray(pcd.points)

    X=X-np.mean(X,axis=0)
    X[:,2]=X[:,2]-np.min(X[:,2])
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30)) 

    xmin = np.min(X,axis=0)
    xmax = np.max(X,axis=0)

    print("Apply point-to-plane ICP")
    

    # target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30)) 
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-6,
    relative_rmse=1e-4,
    max_iteration=100  # Set your desired maximum iterations
    )

    threshold=500

    for i3 in np.arange(100,1000,100):
        # for i0 in np.arange(xmin[0]-100,xmax[0]+100,300):
        for i0 in np.arange(-500,+500,100):
            # for i1 in np.arange(xmin[1]-100,xmax[1]+100,300):
            for i1 in np.arange(-500,500,100):
                trans_init = np.asarray([[1, 0, 0, i0],
                                    [0, 1, 0, i1],
                                    [0, 0, 1, i3], [0.0, 0.0, 0.0, 1.0]])

                source = pcdtote.voxel_down_sample(voxel_size=10)
                target = pcd.voxel_down_sample(voxel_size=5)
                
                srctrans_pcd = source.transform(trans_init)
                if not are_point_clouds_within_distance(srctrans_pcd,target,threshold):
                    print("Not in distance ",[i0,i1,i3])
                    continue
                
                reg_p2l = o3d.pipelines.registration.registration_generalized_icp(
                    source, target, threshold, trans_init,estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    criteria=criteria
                    )
                # print(reg_p2l)
                # print("Transformation is:")
                # print(reg_p2l.transformation)
                if reg_p2l.inlier_rmse < 7 and (reg_p2l.inlier_rmse>0 and reg_p2l.fitness>0):
                    print("Inlier RMSE:",[i0,i1,i3], reg_p2l.inlier_rmse)
                    print("Fitness:", [i0,i1,i3],reg_p2l.fitness)
                    
                    draw_registration_result(source, target, reg_p2l.transformation)

                
                # print("Matched points:", len(reg_p2l.correspondence_set))