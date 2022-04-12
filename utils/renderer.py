import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        FoVOrthographicCameras,
        RasterizationSettings, 
        MeshRenderer, 
        MeshRasterizer,  
        SoftSilhouetteShader,
        SoftPhongShader,
        PointLights, 
    )
except:
    print("PyTorch3D not installed! Ignore this message if running 2D NST.")
from utils.poisson_disc_sampling import Poisson_disc_sampling
from utils.plot import image_grid
from utils.mesh_preprocess import mesh_normalization

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_renderer(rendering_size = 512, faces_per_pixel = 50, sil_shader = False):
    """
    Get a differentiable renderer from PyTorch3D.
    Arguments:
        rendering_size: image size of rendering, can be int or (int, int)
        faces_per_pixel: number of faces per pixel to track along depth axis
        sil_shader: whether to use silhouette shader or soft phong shader, boolean
    Returns:
        PyTorch3D renderer object
    """    

    # Rasterization settings for differentiable rendering, where the blur_radius initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning', ICCV 2019
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(image_size = rendering_size, 
                                                       blur_radius = np.log(1. / 1e-4 - 1.)*sigma, 
                                                       faces_per_pixel = faces_per_pixel)   # perspective_correct=False # fix a bug that silhouettes cannot be plotted correctly, see https://www.youtube.com/watch?v=v3hTD9m2tM8&ab_channel=CODEMENTAL at time 32:50        
        
    renderer = MeshRenderer(rasterizer = MeshRasterizer(raster_settings = raster_settings_silhouette),
                            shader = SoftSilhouetteShader() if sil_shader else SoftPhongShader(device = device))
    
    return renderer

def get_lights(diffuse = 1.0, specular = 0.2, ambient = 0.5):
    """
    Get lightings.
    Arguments:
        diffuse: amplitude of diffuse light
        specular: amplitude of specular light
        ambient: amplitude of ambient light
    Returns:
        PyTorch3D light object
    """
    diffuse_color = ((diffuse, diffuse, diffuse),)
    specular_color = ((specular, specular, specular),)
    ambient_color = ((ambient, ambient, ambient),)

    lights = PointLights(device=device, diffuse_color = diffuse_color, specular_color = specular_color, ambient_color = ambient_color)
    return lights

def get_cameras(sampling_cameras = True, elevs = torch.tensor([0]), azims = torch.tensor([0]), perspective_camera = True, camera_dist = 2.7):
    """
    Get cameras at different positions.
    Arguments:
        sampling_cameras: whether to apply Poisson disc sampling for camera positions, boolean
        elevs: elevations in polar coordinate, ignored when sampling_cameras is True
        azims: azimuths in polar coordinate, ignored when sampling_cameras is True
        perspective_camera: whether to use perspective camera or orthographic camera, boolean
        camera_dist: distance of camera to mesh center (0,0,0)
    Returns:
        list of PyTorch3D camera objects
    """
    # Poisson disc sampling for azim, elev in polar coordinate system
    if sampling_cameras:
        camera_angles = Poisson_disc_sampling()
        azims = camera_angles[:, 0] - 180
        elevs = camera_angles[:, 1] - 90
    
    Rs, Ts = look_at_view_transform(dist=camera_dist, elev=elevs, azim=azims)

    if perspective_camera:
        cameras = [FoVPerspectiveCameras(device=device, R=R[None, ...], T=T[None, ...]) for R, T in zip(Rs, Ts)]
    else:
        cameras = [FoVOrthographicCameras(device=device, R=R[None, ...], T=T[None, ...]) for R, T in zip(Rs, Ts)]

    return cameras

# get a fixed camera that is used for visualization
def get_visual_camera(perspective_camera = True, camera_dist = 2.7, azim = 0, elev = 0, pos = None):
    """
    Get a single camera at some fixed position.
    Arguments:
        perspective_camera: whether to use perspective camera or orthographic camera, boolean
        camera_dist: distance of camera to mesh center (0,0,0)
        elev: elevation in polar coordinate
        azim: azimuth in polar coordinate
        pos: camera postion in 3D cartesian space, will override polar coordinate
    Returns:
        PyTorch3D camera object
    """
    # two ways to define camera positions:
    # 1. polar coordinate through azim, elev and dist
    # 2. cartesian coordinate through pos, will override polar coordinate if not None
    R, T = look_at_view_transform(dist=camera_dist, elev = elev, azim = azim, eye = pos)
    
    camera = FoVPerspectiveCameras(device=device, R=R, T=T) if perspective_camera else FoVOrthographicCameras(device=device, R=R, T=T)   
    return camera

def get_rgba_rendering(mesh, renderer, camera, lights):
    """
    Get rendering of mesh.
    Arguments:
        mesh: 3D mesh object to be rendered
        renderer: PyTorch3D differentiable renderer
        camera: PyTorch3D camera
        lights: PyTorch3D lights
    Returns:
        rendering tensor of shape (M,N,4), where the first 3 channels are RGB rendering and the last channel is silhouette
    """
    # lights at camera position
    lights.location = camera.get_camera_center()

    rgba_rendering = renderer(mesh, cameras=camera, lights = lights)[0]
    return rgba_rendering

def grid_plot(mesh, 
              perspective_camera = True,
              camera_dist = 2.7,
              elevs = torch.tensor([0, 0, 0, 0, 90, -90]), 
              azims = torch.tensor([-180, -90, 0, 90, 0, 0]), 
              n_rows = 2, 
              n_cols = 3, 
              rendering_size = 512, 
              faces_per_pixel = 50,
              sil_shader = True,
              rgb = False):
    """
    Get multiple renderings of mesh from different view points, and plot renderings in grid.
    Arguments:
        mesh: 3D mesh object to be rendered
        perspective_camera: whether to use perspective camera or orthographic camera, boolean
        camera_dist: distance of camera to mesh center (0,0,0)
        elevs: elevations in polar coordinate
        azims: azimuths in polar coordinate
        n_rows: number of rows in plot grid
        n_cols: number of columns in plot grid
        rendering_size: image size of rendering
        faces_per_pixel: number of faces per pixel to track along depth axis
        sil_shader: whether to use silhouette renderer or soft phong renderer, boolean
        rgb: whether to plot RGB rendering or silhouette, boolean
    """

    # normalize mesh
    _, _ = mesh_normalization(mesh)

    Rs, Ts = look_at_view_transform(dist=camera_dist, elev=elevs, azim=azims)

    if perspective_camera:
        cameras = [FoVPerspectiveCameras(device=device, R=R[None, ...], T=T[None, ...]) for R, T in zip(Rs, Ts)]
    else:
        cameras = [FoVOrthographicCameras(device=device, R=R[None, ...], T=T[None, ...]) for R, T in zip(Rs, Ts)]

    lights = get_lights()

    renderer = get_renderer(rendering_size = rendering_size, faces_per_pixel = faces_per_pixel, sil_shader = sil_shader)

    rendering_rgbas = torch.stack([get_rgba_rendering(mesh, renderer, camera, lights) for camera in cameras])

    image_grid(rendering_rgbas.detach().cpu().numpy(), rows=n_rows, cols=n_cols, rgb=rgb)
    plt.show()

def single_plot(mesh, 
                perspective_camera = True,
                camera_dist = 2.7,
                elev = 0, 
                azim = 0,
                rendering_size = 512, 
                faces_per_pixel = 50,
                sil_shader = True,
                rgb = False):
    """
    Plot a single rendering of mesh from some view point.
    Arguments:
        mesh: 3D mesh object to be rendered
        perspective_camera: whether to use perspective camera or orthographic camera, boolean
        camera_dist: distance of camera to mesh center (0,0,0)
        elev: elevation in polar coordinate
        azim: azimuth in polar coordinate
        rendering_size: image size of rendering
        faces_per_pixel: number of faces per pixel to track along depth axis
        sil_shader: whether to use silhouette renderer or soft phong renderer, boolean
        rgb: whether to plot RGB rendering or silhouette, boolean
    """

    # normalize mesh
    _, _ = mesh_normalization(mesh)

    R, T = look_at_view_transform(dist=camera_dist, elev=elev, azim=azim)
    camera = FoVPerspectiveCameras(device=device, R=R, T=T) if perspective_camera else FoVOrthographicCameras(device=device, R=R, T=T)   

    lights = get_lights()

    renderer = get_renderer(rendering_size = rendering_size, faces_per_pixel = faces_per_pixel, sil_shader = sil_shader)

    rendering_rgba = get_rgba_rendering(mesh, renderer, camera, lights).detach().cpu()
    
    plt.axis('off')
    if rgb:
        plt.imshow(rendering_rgba[..., :3])
    else:
        plt.imshow(rendering_rgba[..., 3])