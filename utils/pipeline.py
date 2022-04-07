
import torch
from utils.model import get_models_losses_masks_2D_NST
from utils.NoStdStreams import NoStdStreams

style_layers_default  = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
style_weights_default = [1e6/n**2 for n in [64,128,256,512]]

content_layers_default = ['conv4_2']
content_weights_default = [1]

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
                    silent = True):
    """
    Pipleline for running 2D neural style transfer
    Arguments:
        style_img: style image tensor of shape (1,3,M,N)
        content_img: content image tensor of shape (1,3,M,N)
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
    Returns:
        input_img: tensor for stylized input image
        loss_history: dictionary stores weight and value of each loss

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
    model_style, model_content, model_mask, style_losses, content_losses = get_models_losses_masks_2D_NST(  style_img,
                                                                                                            content_img,
                                                                                                            style_layers = style_layers,
                                                                                                            content_layers = content_layers,
                                                                                                            style_loss_types = style_loss_types,
                                                                                                            need_content = need_content,
                                                                                                            masking = masking,
                                                                                                            model_pooling = model_pooling,
                                                                                                            mask_pooling = mask_pooling,
                                                                                                            silent = silent)
    
    # optimize the input and not the model parameters, so update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model_style.requires_grad_(False)
    model_content.requires_grad_(False)

    # optimizer
    optimizer = torch.optim.LBFGS([input_img], lr=learning_rate) #LBFGS([input_img], lr=lr)

    # loss history
    loss_history = {name: {'weight':weight, 'values':[]} for name, weight in style_loss_types.items()}
    if need_content:
        loss_history['content'] = {'weight': 1., 'values':[]}

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

    return input_img, loss_history