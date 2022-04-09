
import torch
import numpy as np
from tqdm.notebook import tqdm
from pytorch3d.io import save_obj
from pytorch3d.renderer import TexturesVertex
from utils.data_loader import tensor_loader
from utils.plot import visualize_prediction
from utils.model import get_models_2D_NST, get_models_3D_NST, get_models_OpsOnBNST
from utils.NoStdStreams import NoStdStreams
from utils.mesh_preprocess import mesh_normalization
from utils.renderer import get_renderer, get_lights, get_cameras, get_rgba_rendering, get_visual_camera

style_layers_default  = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
style_weights_default = [1e6/n**2 for n in [64,128,256,512]]

content_layers_default = ['conv4_2']
content_weights_default = [1]

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pipeline_2D_NST(style_img,
                    content_img,  
                    input_img, 
                    mask_img,
                    n_iters = 200,
                    style_weights = style_weights_default,
                    content_weights = content_weights_default,
                    style_layers = style_layers_default, 
                    content_layers = content_layers_default,
                    style_loss_types = {'gram': 1}, 
                    learning_rate = 1,
                    need_content = False,
                    masking = False,
                    model_pooling = 'max',
                    mask_pooling = 'avg',
                    silent = True,
                    fft_level = 0,
                    freq_lower = None,
                    freq_upper = None):
    """
    Pipleline for running 2D neural style transfer.
    Arguments:
        style_img: style image tensor of shape (1,3,h,w)
        content_img: content image tensor of shape (1,3,h,w)
        input_img: input image tensor, whose reshape depends on whether to consider content loss and types of style losses
        mask_img: mask image tensor whose shape has be to equal to input_img
        n_iters: number of iterations
        style_weights: list of weights attached to each style layer
        content_weights: list of weights attached to each content layer
        style_layers: list of style layer names, e.g. ['conv1_1', 'conv3_1']
        content_layers: list of content layer names, e.g. ['conv4_2']
        style_loss_types: dictionary with style loss type as key and its weight as value, e.g. {'gram':1}
        learning_rate: learning rate for LBFGS optimizer
        need_content: whether to consider content loss, boolean
        masking: whether to apply mask, boolean
        model_pooling: type of pooling layer in style/content model, can be 'max' or 'avg'
        mask_pooling: type of pooling layer in mask model, can be 'max' or 'avg'
        silent: whether to print less to console, boolean
        fft_level: apply FFT filter on which feature level
        freq_lower: FFT high pass filter threshold
        freq_upper: FFT low pass filter threshold
    Returns:
        input_img: tensor for stylized input image
        loss_history: dictionary stores weight and value of each loss
        style_losses: list of style loss layers

    """

    if not silent:
        print('Building the style transfer model..')
        print()

    # sanity check on image sizes
    if 'morest' in style_loss_types and style_img.shape[-2:] != input_img.shape[-2:]:
        raise RuntimeError("Style image " + str(style_img.shape[-2:]) + " and input image " + str(input_img.shape[-2:]) + " not of same size!")
    if masking and input_img.shape[-2:] != mask_img.shape[-2:]:
        raise RuntimeError("Input image " + str(input_img.shape[-2:]) + " and mask image " + str(mask_img.shape[-2:]) + " not of same size!")

    # get style and content models        
    model_style, model_content, model_mask, style_losses, content_losses = get_models_2D_NST(   style_img,
                                                                                                content_img,
                                                                                                style_layers = style_layers,
                                                                                                content_layers = content_layers,
                                                                                                style_loss_types = style_loss_types,
                                                                                                need_content = need_content,
                                                                                                masking = masking,
                                                                                                model_pooling = model_pooling,
                                                                                                mask_pooling = mask_pooling,
                                                                                                silent = silent,
                                                                                                fft_level = fft_level,
                                                                                                freq_lower = freq_lower,
                                                                                                freq_upper = freq_upper)
    
    # optimize only the input image and not the model parameters, so set all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model_style.requires_grad_(False)
    model_content.requires_grad_(False)

    # optimizer
    optimizer = torch.optim.LBFGS([input_img], lr=learning_rate) #LBFGS([input_img], lr=lr)

    # loss history
    loss_history = {name: {'weight':weight, 'values':[]} for name, weight in style_loss_types.items()}
    if need_content:
        loss_history['content'] = {'weight': content_weights[0], 'values':[]}

    if not silent:
        print()
        print('Optimizing..')
    run = [0]
    while run[0] <= n_iters:

        def closure():
            # with torch.no_grad():
                # input_img.clamp_(eps, one)
            
            optimizer.zero_grad()

            # forward pass
            if masking:     # first, update mask
                model_mask(mask_img)
            model_style(input_img)
            
            # a dictionary saves style losses at this iteration
            loss_current_iter = {name: 0 for name in style_loss_types}
            
            # compute style loss
            for sl, sl_weight in zip(style_losses, style_weights):
                for name in style_loss_types:
                    loss_current_iter[name] += sl.losses[name] * sl_weight * style_loss_types[name]
            
            # total style loss
            style_score = 0
            for v in loss_current_iter.values():
                style_score += v
            
            # compute content loss
            if need_content:
                model_content(input_img)
                loss_current_iter['content'] = 0
                for cl, cl_weight in zip(content_losses, content_weights):
                    loss_current_iter['content'] += cl.loss * cl_weight
            
            # total loss (style + content)
            all_score = style_score + loss_current_iter['content'] if need_content else style_score
            
            # add current iter loss to history
            for name, value in loss_current_iter.items():
                loss_history[name]['values'].append(value.detach().cpu())
            
            # run KeOps kernels silently
            with NoStdStreams():
                all_score.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                print("run {}:".format(run))
                if not silent:
                    if need_content:
                        print('Style Loss : {:.4f} Content Loss: {:.4f}'.format(style_score.item(), loss_current_iter['content'].item()))
                    else:
                        print('Style Loss : {:.4f}'.format(style_score.item()))
                    print()

            if need_content:
                return all_score
            else:
                return style_score

        optimizer.step(closure)


    # with torch.no_grad():
        # input_img.clamp_(0, 1)

    return input_img, loss_history, style_losses

def pipeline_2D_NST_OpsOnBNST(  style_img,
                                input_img, 
                                n_iters = 200,
                                style_weights = style_weights_default,
                                style_layers = style_layers_default, 
                                learning_rate = 1,
                                model_pooling = 'max',
                                silent = True,
                                indices = None,
                                mean_coef = 1, mean_bias = 0, mean_freq_lower = None, mean_freq_upper = None,
                                std_coef = 1, std_bias = 0, std_freq_lower = None, std_freq_upper = None):
    """
    Pipleline for running 2D neural style transfer with detailed operations on batch normalization statistics.
    Arguments:
        style_img: style image tensor of shape (1,3,h,w)
        input_img: input image tensor, whose reshape depends on whether to consider content loss and types of style losses
        n_iters: number of iterations
        style_weights: list of weights attached to each style layer
        style_layers: list of style layer names, e.g. ['conv1_1', 'conv3_1']
        learning_rate: learning rate for LBFGS optimizer
        model_pooling: type of pooling layer in style/content model, can be 'max' or 'avg'
        silent: whether to print less to console, boolean
        indices: a subset of channels where BNST loss is computed
        *coef and *bias: params for affine transformation, e.g. x --> x * x_coef + x_bias
        *freq_lower: FFT high pass filter threshold
        *freq_upper: FFT low pass filter threshold
    Returns:
        input_img: tensor for stylized input image
        mean_loss_history: list stores BN mean loss
        std_loss_history: list stores BN std loss
        style_losses: list of style loss layers

    """

    if not silent:
        print('Building the style transfer model..')
        print()


    # get style and content models        
    model_style, style_losses = get_models_OpsOnBNST(   style_img,
                                                        style_layers = style_layers,
                                                        model_pooling = model_pooling,
                                                        silent = silent,
                                                        indices = indices,
                                                        mean_coef = mean_coef, mean_bias = mean_bias, mean_freq_lower=mean_freq_lower, mean_freq_upper=mean_freq_upper,
                                                        std_coef = std_coef, std_bias = std_bias, std_freq_lower=std_freq_lower, std_freq_upper=std_freq_upper
                                                        )
    
    # optimize only the input image and not the model parameters, so set all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model_style.requires_grad_(False)
    
    # optimizer
    optimizer = torch.optim.LBFGS([input_img], lr=learning_rate) #LBFGS([input_img], lr=lr)

    # loss history
    mean_loss_history = []
    std_loss_history = []

    if not silent:
        print()
        print('Optimizing..')
    run = [0]
    while run[0] <= n_iters:

        def closure():
            # with torch.no_grad():
                # input_img.clamp_(eps, one)
            
            optimizer.zero_grad()

            # forward pass
            model_style(input_img)
            
            # loss in current optimization iteration
            mean_loss = 0
            std_loss = 0

            # compute style loss
            for sl, sl_weight in zip(style_losses, style_weights):
                mean_loss += sl.mean_loss * sl_weight
                std_loss += sl.std_loss * sl_weight

            # add current iter loss to history
            mean_loss_history.append(mean_loss.detach().cpu())
            std_loss_history.append(std_loss.detach().cpu())

            # backward
            mean_and_std_loss = mean_loss + std_loss
            mean_and_std_loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                print("run {}:".format(run))
                if not silent:
                    print('mean loss : {:.4f}'.format(mean_loss.item()) + '   std loss : {:.4f}'.format(std_loss.item()))
                    print()

            return mean_and_std_loss

        optimizer.step(closure)


    # with torch.no_grad():
        # input_img.clamp_(0, 1)

    return input_img, mean_loss_history, std_loss_history, style_losses


def pipeline_3D_NST(org_mesh,
                    style_img,
                    optim_type = 'reshaping',
                    optim_init = None,
                    rendering_size = (512,512),
                    style_layers = style_layers_default,
                    style_weights = style_weights_default,
                    n_views_per_iter = 1,
                    cameras = None,
                    sampling_cameras = True,
                    elevs = torch.tensor([0]),
                    azims = torch.tensor([0]),
                    perspective_camera = True,
                    camera_dist = 2.7,
                    faces_per_pixel = 50,
                    n_iterations = 500, 
                    learning_rate = 1e-5,
                    plot_period = [10, 50, 100, 200, 500],
                    style_loss_types = {'gram': 1}, 
                    masking = False,
                    model_pooling = 'max',
                    mask_pooling = 'avg',
                    clamping = False,
                    reshaping_rgb = False):
    """
    Pipeline for running 3D neural style transfer, either reshaping or texturing.
    Arguments:
        org_mesh: 3D mesh object
        style_img: tensor of style image of shape (1,3,h,w)
        optim_type: 'reshaping' or 'texturing', string
        optim_init: whether to have some initial value for per-vertex color during texturing
        rendering_size: image size of rendering
        style_layers: list of style layer names, e.g. ['conv1_1', 'conv3_1']
        style_weights: list of weights attached to each style layer
        n_views_per_iter: how many camera views are used for optimization per iteration, int
        cameras: list of PyTorch3D cameras, will be generated if not given
        sampling_cameras: whether to Poisson disc sampling for camera positions, boolean
        elevs: camera elevations, will be ignored if sampling_cameras is True
        azims: camera azimuths, will be ignored if sampling_cameras is True
        perspective_camera: whether to use perspective camera or orthographical camera, boolean
        camera_dist: distance of camera to mesh center (0,0,0)
        faces_per_pixel: number of faces per pixel track along depth axis
        n_iterations: number of optimization iterations
        learning_rate: optimizer learning rate
        plot_period: at which iteration to plot rendering and save mesh object
        style_loss_types: dictionary with style loss type as key and its weight as value, e.g. {'gram':1}
        masking: whether to use silhouette as mask, boolean
        model_pooling: type of pooling layer in style model, can be 'max' or 'avg'
        mask_pooling: type of pooling layer in mask model, can be 'max' or 'avg'
        clamping: whether to clamp per-vertex color in range [0,1] in case of texturing
        reshaping_rgb: whether to conduct reshaping with colorful renderings instead of silhouettes, boolean
    Returns:
        what_to_optimize: per-vertex position offset or per-vertex color depending on task type
        cameras: generated camera, may be reused in case of sequential reshaping and texturing
        loss_history: dictionary of loss history
        rendering_at_iter: list stores renderings at plot_period

    """
  
    # normalize mesh
    center, scale = mesh_normalization(org_mesh)

    # optimizer and the tensor to be optimized
    # what_to_optimize: per-vertex position offset in case of reshaping and per-vertex color in case of texturing 
    verts_shape = org_mesh.verts_packed().shape
    if optim_type == 'reshaping':
        if optim_init is None:
            what_to_optimize = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
        else:
            what_to_optimize = optim_init.view(verts_shape[0], 3)
            what_to_optimize /= 1e2 # offsets should be smaller
            what_to_optimize = what_to_optimize.detach()
            what_to_optimize.requires_grad = True
    elif optim_type == 'texturing':
        if optim_init is None:
            what_to_optimize = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)
            # what_to_optimize = torch.rand(1, verts_shape[0], 3, device=device, requires_grad=True)
        else:
            what_to_optimize = optim_init.view(1, verts_shape[0], 3)
            what_to_optimize = (what_to_optimize - what_to_optimize.min()) / (what_to_optimize.max() - what_to_optimize.min()) # normalize to [0,1]
            what_to_optimize = what_to_optimize.detach() # make leaf tensor
            what_to_optimize.requires_grad = True
        org_mesh.textures = TexturesVertex(verts_features = what_to_optimize)
    else:
        raise RuntimeError("Either reshaping or texturing! But now it is: " + optim_type)
    optimizer = torch.optim.Adam([what_to_optimize], lr=learning_rate)

    # in case of reshaping using colorful rendering
    if optim_type == 'reshaping' and reshaping_rgb:
        init_color = torch.full([1, verts_shape[0], 3], 0.5, device=device)
        org_mesh.textures = TexturesVertex(verts_features = init_color)
    
    # get renderer, cameras and lights
    rendering_size = style_img.shape[-2:] if 'morest' in style_loss_types else rendering_size # 'morest' loss requires style image and rendering of same size
    renderer = get_renderer(rendering_size = rendering_size, faces_per_pixel = faces_per_pixel, sil_shader = optim_type == 'reshaping')
    lights = get_lights()
    if cameras is None: # cameras for reshaping should be re-used for texturing
        print("no cameras are given, and sampling_cameras is", sampling_cameras)
        cameras = get_cameras(sampling_cameras = sampling_cameras, elevs = elevs, azims = azims, perspective_camera = perspective_camera, camera_dist = camera_dist)
    else:
        print("cameras are already given, reusing the given cameras")
    camera_visual = get_visual_camera(perspective_camera = perspective_camera, camera_dist = camera_dist)
    n_views_per_iter = min(n_views_per_iter, len(cameras))
    print("number of cameras:", len(cameras))
    # print("n_views_per_iter is:", n_views_per_iter)
    
    # get some initial renderings
    org_rendering_rgba = get_rgba_rendering(org_mesh, renderer, camera_visual, lights).detach().cpu()
    
    # VGG with style loss layers
    model_style, model_mask, style_losses = get_models_3D_NST(style_img, style_layers, style_loss_types, masking, model_pooling = model_pooling, mask_pooling = mask_pooling)
    
    # loss history
    loss_history = {name: {'weight':weight, 'values':[]} for name, weight in style_loss_types.items()}
        
    # return value, which keeps renderings at specified iterations
    rendering_at_iter = {}
    
    # optimization loop
    loop = tqdm(range(1, n_iterations + 1))
    for i in loop:
        if clamping and optim_type == 'texturing':
            with torch.no_grad():
                what_to_optimize.clamp_(0., 1.)
        
        optimizer.zero_grad()
        
        # update mesh
        if optim_type == 'reshaping':
            new_mesh = org_mesh.offset_verts(what_to_optimize)
        else: # optim_type == 'texturing'
            new_mesh = org_mesh
            new_mesh.textures = TexturesVertex(verts_features = what_to_optimize) 

        # a dictionary saves style losses at this iteration
        loss_current_iter = {name:0 for name in style_loss_types}
        
        for j in np.random.permutation(len(cameras)).tolist()[:n_views_per_iter]:
            # get data tensors
            rendering_rgba = get_rgba_rendering(new_mesh, renderer, cameras[j], lights) 
            rendering = rendering_rgba[..., 3] if optim_type == 'reshaping' and not reshaping_rgb else rendering_rgba[..., :3]
            rendering_tensor = tensor_loader(rendering)
            mask_tensor = tensor_loader(rendering_rgba[...,3], mask = True)
            
            # forward pass
            if masking:
                model_mask(mask_tensor)
            model_style(rendering_tensor)

            # sum up style losses over all style loss layers
            for sl, sl_weight in zip(style_losses, style_weights):
                for name in style_loss_types:
                    loss_current_iter[name] += sl.losses[name] * sl_weight

        # average loss among n_views_per_iter    
        for name in style_loss_types:
            loss_current_iter[name] /= n_views_per_iter
        
        # weighted sum of the losses
        # !!! careful !!!
        # style_loss_types: {style loss name : style loss weight}
        # loss_history: {style/smoothing/regularization loss name : {'weight' : weight, values : [...] }}
        # loss_current_iter: {style/smoothing/regularization loss name : value}
        sum_loss = 0
        for loss_name, loss_value in loss_current_iter.items():
            weighted_loss = loss_value * loss_history[loss_name]['weight']
            sum_loss += weighted_loss
            loss_history[loss_name]['values'].append(weighted_loss.detach().cpu())
        
        # print losses
        loop.set_description("total_loss = %.5f" % sum_loss)

        # Optimization step
        with NoStdStreams():        # run KeOps kernels silently
            sum_loss.backward()
        optimizer.step()
        
        # plot renderings
        if i in plot_period:
            with torch.no_grad():
                new_rendering_rgba = get_rgba_rendering(new_mesh, renderer, camera_visual, lights).detach().cpu()
            rgb = optim_type == 'texturing' or reshaping_rgb
            sil = optim_type == 'reshaping'
            visualize_prediction(new_rendering_rgba = new_rendering_rgba, org_rendering_rgba = org_rendering_rgba, rgb = rgb, sil = sil, title="iter: %d" % i)
                 
            # save mesh
            final_verts, final_faces = new_mesh.get_mesh_verts_faces(0)
            final_verts = final_verts * scale + center
            final_obj = "./runtime_objs/obj_iter" + str(i) + ".obj"
            save_obj(final_obj, final_verts, final_faces)

            # add rendering to dictionary
            rendering_at_iter["rendering_iter" + str(i)] = new_rendering_rgba[..., 3] if optim_type == 'reshaping' else new_rendering_rgba[..., :3]

    return what_to_optimize, cameras, loss_history, rendering_at_iter