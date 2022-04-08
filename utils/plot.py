import matplotlib.pyplot as plt

# Plot losses
def plot_losses_2D_NST(losses, save_plot=True, loss_name = None, start = 0, end = None):
    """
    Plot loss vs iteration
    Arguments:
        losses: dictionary of losses
        save_plot: whether save loss plot as png image, boolean
        loss_name: name of style loss type which will be plotted while other types not
        start: plot start iteration
        end: plot end iteration 
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
    Plot and compare statistics
    Arguments:
        statistics: dictionary of statistics
        config_idx: plot statistics only for this configuration
        style_layer: statistics from which layer to plot
        save_plot: whether save plot as png image, boolean
        title: title of plot
    """

    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    if config_idx is None:
        for k, l in statistics.items():
            ax.plot(l[style_layer], label = "config" + str(k))
    else:
        k = list(statistics.keys())[config_idx]
        l = list(statistics.values())[config_idx]
        ax.plot(l[style_layer], label = "config" + str(k))

    ax.legend(fontsize="16")
    ax.set_xlabel("channel", fontsize="16")
    ax.set_ylabel(title, fontsize="16")
    ax.set_title(style_layer + ' ' + title + " in each channel", fontsize="16")
    if save_plot:
        plt.savefig("statistics_plot.png")

def plot_statistics_difference(statistics, config_idx1=0, config_idx2=1, style_layer='conv1_1', percent = True, save_plot=False, title='mean'):
    """
    Plot difference between two statistics
    Arguments:
        statistics: dictionary of statistics
        config_idx1: index of 1st configuration
        config_idx2: index of 2nd configuration
        style_layer: statistics from which layer to compare
        percent: compute absolute difference or relative difference, boolean
        save_plot: whether save plot as png image, boolean
        title: title of plot
    """

    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    value1 = list(statistics.values())[config_idx1]
    value2 = list(statistics.values())[config_idx2]

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
    Plot gram matrix as grayscale image
    Arguments:
        grams: dictionary of gram matrices
        global_normalizing: whether gram matrices of different configs are normalized in same way, boolean
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