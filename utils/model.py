import torch
from copy import deepcopy
import torchvision.models as models
from utils.layer import GetMask, ContentLoss, StyleLoss, StyleLossOpsOnBNST

style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
content_layers_default = ['conv4_2']

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download vgg
cnn = models.vgg19(pretrained=True).features.to(device).eval()

#####################################################################################
# Accoring to the Gatys' raw implementation, the actual desired layers are not convs,
# rather activated convs, i.e. output from relu layers.
#####################################################################################
# print("The desired content layers are: ", content_layers_default)
# print("But the truely needed are     : ", [x.replace('conv', 'relu') for  x in content_layers_default])
# print()
# print("The desired style layers are  : ", style_layers_default)
# print("But the truely needed are     : ", [x.replace('conv', 'relu') for  x in style_layers_default])

def get_models_2D_NST(  style_img,
                        content_img,
                        style_layers = style_layers_default,
                        content_layers = content_layers_default,
                        style_loss_types = {'gram':1},
                        need_content = False,
                        masking = False,
                        model_pooling = 'max',
                        mask_pooling = 'avg',
                        silent = True,
                        fft_level = 0,
                        freq_lower = None,
                        freq_upper = None):
    """
    Get style model, content model, mask model, style loss layers and content loss layer for 2D neural style transfer
    Arguments:
        style_img: style image tensor of shape (1,3,M,N)
        content_img: content image tensor of shape (1,3,M,N)
        style_layers: list of style layer names, such as ['conv1_1', 'conv3_1']
        content_layers: list of content layer names, such as ['conv4_2']
        style_loss_types: dictionary with style loss name as key and its weight as value, e.g. {'gram':1}
        need_content: whether content loss is considered or not, boolean
        masking: whether to apply masking or not, boolean
        model_pooling: type of pooling layer in style/content model, can be 'avg' or 'max'
        mask_pooling: type of pooling layer in mask model, can be 'avg' or 'max'
        silent: whether to print less to console, boolean
        fft_level: apply FFT filter on which feature level
        freq_lower: FFT high pass filter threshold
        freq_upper: FFT low pass filter threshold
    Returns:
        model_style: style model
        model_content: content model
        model_mask: mask model
        style_losses: list of style loss layers
        content_losses: list of content loss layers
    """

    # model for masks
    model_mask = torch.nn.Sequential()
    mask_layers = []
    first_layer = GetMask()
    model_mask.add_module('mask_1', first_layer)
    mask_layers.append(first_layer)
    for i in range(4):
        if mask_pooling == 'max':
            layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        elif mask_pooling =='avg':
            layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise RuntimeError("Pooling must be either max or avg! But now it is:" + mask_pooling)
        
        # add pooling layer
        model_mask.add_module(mask_pooling + '_pooling_' + str(i+1), layer)

        # add mask layer
        get_input_layer = GetMask()
        model_mask.add_module('mask_' + str(i+2), get_input_layer)
        mask_layers.append(get_input_layer)
    
    # model for style
    style_losses = []
    style_layers   = [x.replace('conv', 'relu') for  x in style_layers]
    model_style   = torch.nn.Sequential()

    # model for content
    # benefits of having independent models for computing content loss and style loss: content image and style image can be in different sizes
    content_losses = []
    content_layers = [x.replace('conv', 'relu') for  x in content_layers]
    model_content = torch.nn.Sequential()
    

    #####################################################################
    # For the following part, the pytorch tutorial makes great mistakes.
    # Reference https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    # Currently already corrected by Mengdi
    #####################################################################
    conv_i = 1
    relu_i = 1
    pool_i = 1    # pool_i is important since conv blocks are separated by pooling layers
    bn_i = 1
    
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            name = 'conv{}_{}'.format(pool_i, conv_i)
            conv_i += 1
        elif isinstance(layer, torch.nn.ReLU):
            name = 'relu{}_{}'.format(pool_i, relu_i)
            layer = torch.nn.ReLU(inplace=False) # in-place ReLU doesnt work well
            relu_i += 1
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = 'bn{}_{}'.format(pool_i, bn_i)
            bn_i += 1
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = 'pool{}'.format(pool_i)
            if model_pooling == 'max':
                layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            elif model_pooling == 'avg':
                layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                raise RuntimeError("Pooling must be either max or avg! But now it is:" + model_pooling)
            pool_i += 1
            conv_i = 1
            relu_i = 1
            bn_i = 1  
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))


        model_style.add_module(name, layer)
        model_content.add_module(name, deepcopy(layer))


        # add style layers
        if name in style_layers:
            target_style = model_style(style_img).detach()
            style_loss = StyleLoss(target_style, style_loss_types, mask_layers[pool_i-1], False, fft_level=fft_level, freq_lower=freq_lower, freq_upper=freq_upper)
            model_style.add_module("style_loss_{}_{}".format(pool_i, relu_i-1), style_loss)
            style_losses.append(style_loss)
        
        # add content layers
        if name in content_layers:
            target_content = model_content(content_img).detach()
            content_loss = ContentLoss(target_content)
            model_content.add_module("content_loss_{}_{}".format(pool_i, relu_i-1), content_loss)
            content_losses.append(content_loss)


    # trim off the layers after the last content and style losses
    for i in range(len(model_style) - 1, -1, -1):
        if isinstance(model_style[i], StyleLoss): # or isinstance(model_content[i], ContentLoss):
            break
    
    for j in range(len(model_content) - 1, -1, -1):
        if isinstance(model_content[j], ContentLoss): # or isinstance(model_style[j], StyleLoss):
            break
        
    model_style   = model_style  [:(i + 1)]
    model_content = model_content[:(j + 1)]
    
    # set self.masking to True if masking. This has to be done afterwards, otherwise model build 
    # may fail because style image and mask image are not necessarily of same size
    for l in style_losses:
        l.masking = masking

    if not silent:
        print()
        print("Style model is as follow:")
        print(model_style)
        print()
        
        if need_content:
            print("Content model is as follow:")
            print(model_content)
        
    return model_style, model_content, model_mask, style_losses, content_losses


def get_models_2D_NST_OpsOnBNST(style_img,
                                style_layers = style_layers_default,
                                model_pooling = 'max',
                                silent = True,
                                indices = None,
                                mean_coef = 1, mean_bias = 0, mean_freq_lower = None, mean_freq_upper = None,
                                std_coef = 1, std_bias = 0, std_freq_lower = None, std_freq_upper = None
                                ):
    """
    Get style model and style loss layers, specialized for StyleLossOpsOnBNST
    Arguments:
        style_img: style image tensor of shape (1,3,M,N)
        style_layers: list of style layer names, such as ['conv1_1', 'conv3_1']
        model_pooling: type of pooling layer in style model, can be 'avg' or 'max'
        silent: whether to print less to console, boolean
        indices: a subset of channels where BNST loss is computed
        *coef and *bias: params for affine transformation, e.g. x --> x * x_coef + x_bias
        *freq_lower: FFT high pass filter threshold
        *freq_upper: FFT low pass filter threshold
    Returns:
        model_style: style model
        style_losses: list of style loss layers
    """

    # model for style
    style_losses = []
    style_layers   = [x.replace('conv', 'relu') for  x in style_layers]
    model_style   = torch.nn.Sequential()

    # single element to list
    def element_to_list(element, length):
        return [element] * length if not isinstance(element, list) else element
    indices = element_to_list(indices, len(style_layers))
    mean_coef = element_to_list(mean_coef, len(style_layers))
    mean_bias = element_to_list(mean_bias, len(style_layers))
    mean_freq_lower = element_to_list(mean_freq_lower, len(style_layers))
    mean_freq_upper = element_to_list(mean_freq_upper, len(style_layers))
    std_coef = element_to_list(std_coef, len(style_layers))
    std_bias = element_to_list(std_bias, len(style_layers))
    std_freq_lower = element_to_list(std_freq_lower, len(style_layers))
    std_freq_upper = element_to_list(std_freq_upper, len(style_layers))
    
    conv_i = 1
    relu_i = 1
    pool_i = 1    # pool_i is important since conv blocks are separated by pooling layers
    bn_i = 1
    
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            name = 'conv{}_{}'.format(pool_i, conv_i)
            conv_i += 1
        elif isinstance(layer, torch.nn.ReLU):
            name = 'relu{}_{}'.format(pool_i, relu_i)
            layer = torch.nn.ReLU(inplace=False) # in-place ReLU doesnt work well
            relu_i += 1
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = 'bn{}_{}'.format(pool_i, bn_i)
            bn_i += 1
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = 'pool{}'.format(pool_i)
            if model_pooling == 'max':
                layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            elif model_pooling == 'avg':
                layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                raise RuntimeError("Pooling must be either max or avg! But now it is:" + model_pooling)
            pool_i += 1
            conv_i = 1
            relu_i = 1
            bn_i = 1  
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model_style.add_module(name, layer)
        
        # add style layers
        if name in style_layers:
            target_style = model_style(style_img).detach()
            idx = style_layers.index(name)
            style_loss = StyleLossOpsOnBNST( target_style, indices=indices[idx], 
                                    mean_coef=mean_coef[idx], mean_bias=mean_bias[idx], mean_freq_lower=mean_freq_lower[idx], mean_freq_upper=mean_freq_upper[idx],
                                    std_coef=std_coef[idx], std_bias=std_bias[idx], std_freq_lower=std_freq_lower[idx], std_freq_upper=std_freq_upper[idx])
            model_style.add_module("style_loss_{}_{}".format(pool_i, relu_i-1), style_loss)
            style_losses.append(style_loss)


    # trim off the layers after the last content and style losses
    for i in range(len(model_style) - 1, -1, -1):
        if isinstance(model_style[i], StyleLoss): # or isinstance(model_content[i], ContentLoss):
            break
        
    model_style   = model_style  [:(i + 1)]
    
    
    if not silent:
        print()
        print("Style model is as follow:")
        print(model_style)
        print()
        
    return model_style, style_losses