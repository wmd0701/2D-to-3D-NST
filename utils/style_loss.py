import torch
from torchinterp1d import Interp1d
from utils.NoStdStreams import NoStdStreams
from pykeops.torch import Genred

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gram_matrix(input):
    """
    Compute gram matrix
    Arguments:
        input: flattened feature map of shape (c, h*w)
    Returns:
        gram matrix of shape (c,c)
    """
    hw = input.shape[1]
    G = torch.mm(input, input.transpose(0,1))  # compute the gram product 
    G.div_(hw)

    return G

def BN_mean_and_std(input):
    """
    Compute batch normalization means and stds
    Arguments:
        input: flattened feature map of shape (c, h*w)
    Returns:
        mean: vector of length c
        std: vector of length c
    """
    mean = input.mean(1)
    std  = input.std(1)
    return mean, std

def histogram_loss(input_feature_normalized, style_feature_normalized_ordered, style_quantiles):    
    """
    Compute histogram matching loss, which can improve stability for neural style transfer
    Arguments:
        input_feature_normalized: normalized input feature
        style_feature_normalized_order: normalized and ordered style feature
        style_quantiles: histogram quantiles of style feature
    Returns:
        channel-wise avg histogram matching loss
    """
    c = input_feature_normalized.shape[0]   # number of channels
    histo_losses = 0
    for feature, s_ordered, s_quantiles in zip(input_feature_normalized, style_feature_normalized_ordered, style_quantiles):
        _, bin_idx, i_counts = feature.unique(return_inverse=True, return_counts=True, sorted=True)
        i_quantiles = torch.cumsum(i_counts, dim=0, dtype=torch.float32)
        i_quantiles = i_quantiles / i_quantiles[-1]
        
        # histogram match
        # in case there is only one unique value in feature
        if len(i_counts) == 1 or len(s_quantiles) == 1:
            matched_feature = torch.full_like(bin_idx, s_ordered.mean(), dtype=torch.float32) 
        else:
            matched_feature = Interp1d()(s_quantiles, s_ordered, i_quantiles)[0][bin_idx]
            
        # histogram loss
        histo_loss = torch.nn.functional.mse_loss(feature, matched_feature)
        histo_losses += histo_loss

    return histo_losses / c

# kernel formulas for KeOps
formulas = {'linear': '(x | y)',                # or '(x | y) + p' 
            'poly'  : 'Square((x | y))',        # or 'Pow((x | y) + p, 2)' , which is slower than Square    
            'rbf'   : 'Exp(-p * SqDist(x , y))',
            'dist'  : 'SqDist(x , y)'
            }

# formula parameters
# !!! VGG has only 4 different #channels: 64, 128, 256, 512
formula_paras  = {64:  ['p = Pm(1)', 'x = Vi(64)',  'y = Vj(64)'],
                  128: ['p = Pm(1)', 'x = Vi(128)', 'y = Vj(128)'],
                  256: ['p = Pm(1)', 'x = Vi(256)', 'y = Vj(256)'],
                  512: ['p = Pm(1)', 'x = Vi(512)', 'y = Vj(512)']
                  }

# A dictionary contains all kernel functions. It has keys 'linear', 'poly' and 'rbf',
# and the value to a key is also a dictionary, with number of channel as key (64, 128, 256, 512)
kernels = {}
for formula in formulas:
    kernels[formula] = {}
    for n_channel in formula_paras:
        kernels[formula][n_channel] = Genred(formulas[formula],
                                             formula_paras[n_channel],
                                             reduction_op = 'Sum',
                                             # axis = 0 or 1
                                             )

# Problem: KeOps requires gradients to be contiguous
# Solution: https://github.com/getkeops/keops/issues/30#issuecomment-540009963
# Function to ensure contiguous gradient in backward pass. To be applied after KeOps reduction.
class ContiguousBackward(torch.autograd.Function):
    """
    Compel gradient to be contiguous in backward pass for KeOps functions
    Usage:
        tensor = ContiguousBackward().apply(tensor)
    """
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()

# self-defined mean reduction of kernel
def kernel_mean(kernel, p, x, y):
    """
    Compute normalized reduction (sum) of kernel matrix
    Arguments:
        kernel: a kernel from the dictionary "kernels"
        p: constant argument in the kernel
        x: 1st kernel variable
        y: 2nd kernel variable
    Returns:
        normalized reduction of k(y,y) - 2*k(x,y)
    """
    # !!! mean_xx actually not important, as it has nothing to do with input image
    with NoStdStreams():
        # mean_xx = ContiguousBackward().apply(kernel(p, x, x)).sum()/(x.shape[0]**2)
        mean_yy = ContiguousBackward().apply(kernel(p, y, y)).sum()/(y.shape[0]**2)
        mean_xy = ContiguousBackward().apply(kernel(p, x, y)).sum()/(x.shape[0] * y.shape[0]) 
    # return mean_xx + mean_yy - 2 * mean_xy
    return mean_yy - 2 * mean_xy

# self-defined mean distance
def mean_square_distance(c, x, y):
    """
    Compute normalized reduction (sum) of kernel matrix for the kernel L2-distance
    Arguments:
        c: number of channels
        x: 1st kernel variable
        y: 2nd kernel variable
    Returns:
        mean of L2Dist(x,y)
    """
    kernel = kernels['dist'][c]
    with NoStdStreams():
        dist = kernel(torch.tensor([0.]).to(device), x, y).sum()
    return (dist/(x.shape[0]*y.shape[0])).detach()