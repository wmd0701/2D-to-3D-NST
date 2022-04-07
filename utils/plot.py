import matplotlib.pyplot as plt

# Plot losses
def plot_losses_2D_NST(losses, save_plot=True, loss_name = None, start = 0, end = None):
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