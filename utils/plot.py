import torch
import matplotlib.pyplot as plt

# Plot losses
def plot_loss(losses, save_plot=True, loss_name = None, start = 0, end = None):
    """
    Plot loss vs iteration.
    Arguments:
        losses: dictionary of losses
        save_plot: whether save loss plot as png image, boolean
        loss_name: name of style loss type which will be plotted while other types not
        start: plot start iteration
        end: plot end iteration 
    Returns:
        no return
    """

    if end is None:
        end = len(losses.values[0]['values'])
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    # when loss_name is None, plot all loses
    if loss_name is None:
        for k, l in losses.items():
            ax.plot(l['values'][start:end], label= k + " loss, weight " + str(l['weight']))
    else:
        k = loss_name
        l = losses[k]
        ax.plot(l['values'][start:end], label= k + " loss, weight " + str(l['weight']))
    ax.legend(fontsize="16")
    # ax.set_xlim([start, end])
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Weighted Loss", fontsize="16")
    ax.set_title("Weighted Loss vs Iterations", fontsize="16")
    if save_plot:
        plt.savefig("loss_plot.png")

def plot_statistics(statistics, config_idx = None, style_layer = 'conv1_1', save_plot=False, title='mean'):
    """
    Plot and compare statistics.
    Arguments:
        statistics: dictionary of statistics
        config_idx: plot statistics only for this configuration
        style_layer: statistics from which layer to plot
        save_plot: whether save plot as png image, boolean
        title: title of plot
    Returns:
        no return
    """

    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    if config_idx is None:
        for k, l in statistics.items():
            ax.plot(l[style_layer], label = "config" + str(k))
    else:
        k = config_idx
        l = statistics[k]
        ax.plot(l[style_layer], label = "config" + str(k))

    ax.legend(fontsize="16")
    ax.set_xlabel("channel", fontsize="16")
    ax.set_ylabel(title, fontsize="16")
    ax.set_title(style_layer + ' ' + title + " in each channel", fontsize="16")
    if save_plot:
        plt.savefig("statistics_plot.png")

def plot_statistics_difference(statistics, config_idx1=0, config_idx2=1, style_layer='conv1_1', percent = True, save_plot=False, title='mean'):
    """
    Plot difference between two statistics.
    Arguments:
        statistics: dictionary of statistics
        config_idx1: index of 1st configuration
        config_idx2: index of 2nd configuration
        style_layer: statistics from which layer to compare
        percent: compute absolute difference or relative difference, boolean
        save_plot: whether save plot as png image, boolean
        title: title of plot
    Returns:
        no return
    """

    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    value1 = statistics[config_idx1]
    value2 = statistics[config_idx2]

    eps = 1e-3
    percentage = (value1[style_layer] - value2[style_layer]) / (value1[style_layer].abs() + value2[style_layer].abs() + eps)
    
    label = "config" + str(config_idx1) + " - config" + str(config_idx2)
    if percent:
        ax.plot(percentage.abs(), label = label)
    else:
        ax.plot(value1[style_layer] - value2[style_layer], label = label)
    
    ax.legend(fontsize="16")
    ax.set_xlabel("channel", fontsize="16")
    ax.set_ylabel("relative difference" if percent else "absolute difference", fontsize="16")
    ax.set_title(style_layer + ' ' + title + " different in each channel", fontsize="16")
    if save_plot:
        plt.savefig("statistics_difference_plot.png")

def plot_gram_matrix(grams, global_normalizing = False):
    """
    Plot gram matrix as grayscale image.
    Arguments:
        grams: dictionary of gram matrices
        global_normalizing: whether gram matrices of different configs are normalized in same way, boolean
    Returns:
        no return
    """
    
    first_value = list(grams.values())[0]
    n_cols = len(first_value)
    
    # global normalizing
    maxs = {k : 0.0 for k in list(first_value.keys())}
    for k, l in grams.items():
        for k2, l2 in l.items():
            maxs[k2] = max(maxs[k2], l2.max())
    
    for k, l in grams.items():
        plt.figure(figsize=(4 * n_cols, 4))    
        
        i = 1
        for k2, l2 in l.items():
            plt.subplot(1, n_cols, i)
            plt.imshow(l2 if not global_normalizing else l2/maxs[k2])
            plt.title(k2 + ", mean={:.2f}, max={:.2f}".format(l2.mean(), l2.max()))
            i += 1
        plt.suptitle("config" + str(k))

def flexible_plot(data_list, config_idx = None, save_plot=False, x_title = 'channel', y_title = 'std', title='no title'):
    """
    A more flexible plot function, mainly used when controlling and comparing BN statistics.
    Arguments:
        data_list: list of data
        config_idx: plot only data of this config
        save_plot: whether save plot as png image, boolean
        x_title: label on x axis
        y_title: label on y axis
        title: title of plot
    Returns:
        no return
    """
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    if config_idx is None:
        i = 0
        for data in data_list:
            ax.plot(data, label = "config" + str(i))
            i += 1
    else:
        ax.plot(data[config_idx], label = "config" + str(config_idx))

    ax.legend(fontsize="16")
    ax.set_xlabel(x_title, fontsize="16")
    ax.set_ylabel(y_title, fontsize="16")
    ax.set_title(title, fontsize="16")
    if save_plot:
        plt.savefig("flexible_plot.png")

def plot_spectrum(statistics, config_idx = None, style_layer = 'conv1_1', save_plot=False, title='std'):
    """
    Compute, plot and compare spectrums of statistics.
    Arguments:
        statistics: dictionary of statistics
        config_idx: plot statistics only for this configuration
        style_layer: statistics from which layer
        save_plot: whether save plot as png image, boolean
        title: title of plot
    Returns:
        no return
    """

    # compute spectrum frequencies    
    N = len(statistics[0][style_layer])
    xf = torch.fft.rfftfreq(N)
    
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    if config_idx is None:
        for k, l in statistics.items():
            # compute spectrum amplitude
            yf = torch.abs(torch.fft.rfft(l[style_layer]))
            ax.plot(xf, yf, label = "config" + str(k))
    else:
        k = config_idx
        l = statistics[k]
        # compute spectrum amplitude
        yf = torch.abs(torch.fft.rfft(l[style_layer]))
        ax.plot(xf, yf, label = "config" + str(k))

    ax.legend(fontsize="16")
    ax.set_xlabel("frequency", fontsize="16")
    ax.set_ylabel("amplitude", fontsize="16")
    ax.set_title(style_layer + ' ' + title + " spectrum", fontsize="16")
    if save_plot:
        plt.savefig("spectrum_plot.png")

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.
    Reference: https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/docs/tutorials/utils/plot_image_grid.py
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

def visualize_prediction(new_rendering_rgba, org_rendering_rgba, rgb = False, title=''):
    """
    Plot the latest rendering vs original rendering. This function may be provoked many times
    during optimization iterations.
    Arguments:
        new_rendering_rgba: new rendering tensor of shape (h,w,4)
        org_rendering_rgba: old rendering tensor of shapr (h,w,4) 
        rgb: whether to plot RGB channels or silhouette channel, boolean
        titile: title of plot
    Returns:
        no return
    """
    if rgb:
        plt.figure(figsize=(16, 4))    
        plt.subplot(1, 4, 1)
        plt.imshow(new_rendering_rgba[..., :3])
        plt.title(title)
        plt.subplot(1, 4, 2)
        plt.imshow(new_rendering_rgba[..., 3])
        plt.axis("off")
        plt.subplot(1, 4, 3)
        plt.imshow(org_rendering_rgba[..., :3])
        plt.axis("off")
        plt.subplot(1, 4, 4)
        plt.imshow(org_rendering_rgba[..., 3])
        plt.axis("off")
    else:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(new_rendering_rgba[..., 3])
        plt.title(title)
        plt.subplot(1, 2, 2)
        plt.imshow(org_rendering_rgba[..., 3])
        plt.axis("off")
