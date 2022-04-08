# reference https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/

import torch

# 2D FFT filter
# the function can work on tensor of shape (M,N), (c,M,N), (1,c,M,N)
def fft_filter_2D(input, freq_lower = None, freq_upper = None):
    """
    2D FFT filter
    Arguments:
        input: tensor to be filtered, , can have shape (M,N), (c,M,N), (1,c,M,N)
        freq_lower: FFT high pass filter threshold
        freq_upper: FFT low pass filter threshold
    Returns:
        filtered tensor
    """
    
    if freq_lower is None and freq_upper is None: 
        return input
    
    freq1 = torch.abs(torch.fft.rfftfreq(input.shape[-1]))
    freq2 = torch.abs(torch.fft.fftfreq(input.shape[-2]))

    if freq_lower is None:
        pass1 = freq1 <= freq_upper
        pass2 = freq2 <= freq_upper
    elif freq_upper is None:
        pass1 = freq1 >= freq_lower
        pass2 = freq2 >= freq_lower
    else:
        pass1 = (freq1 <= freq_upper) * (freq1 >= freq_lower)
        pass2 = (freq2 <= freq_upper) * (freq2 >= freq_lower)
    
    kernel = torch.outer(pass2, pass1).to(input.device)

    fft_input = torch.fft.rfftn(input)
    
    return torch.fft.irfftn(fft_input * kernel)
    
# 1D FFT filter
def fft_filter_1D(input, freq_lower = None, freq_upper = None):
    """
    1D FFT filter
    Arguments:
        input: tensor to be filtered, must be 1D vector
        freq_lower: FFT high pass filter threshold
        freq_upper: FFT low pass filter threshold
    Return:
        filtered tensor
    """
    if freq_lower is None and freq_upper is None: 
        return input
    
    freq = torch.abs(torch.fft.rfftfreq(len(input)))
    
    if freq_lower is None:
        passs = freq <= freq_upper
    elif freq_upper is None:
        passs = freq >= freq_lower
    else:
        passs = (freq <= freq_upper) * (freq >= freq_lower)
        
    kernel = passs.to(input.device)

    fft_input = torch.fft.rfft(input)
    
    return torch.fft.irfft(fft_input * kernel)