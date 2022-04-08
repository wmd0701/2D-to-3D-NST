import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import sample_points_from_meshes

def plot_pointcloud(mesh, n_sample_points=5000, title=""):
    """
    Sample point clouds from mesh and plot point clouds
    Reference: https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
    Arguments:
        mesh: mesh obj read by PyTorch3D
        n_sample_points: number of point clouds to sample
        title: title of plot
    """
    points = sample_points_from_meshes(mesh, n_sample_points)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def mesh_normalization(mesh):
    """
    Scale mesh such that its center locates at (0,0,0) and fits into unit sphere
    Reference: https://pytorch3d.org/tutorials/fit_textured_mesh
    Arguments:
        mesh: mesh obj read by PyTorch3D
    Returns:
        center: center of mesh, can be used to normalized back
        scale: scale of mesh, can be used to normalized back
    """
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    # print("Vertices Shape: ", verts.shape)

    return center, scale

