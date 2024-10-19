import torch
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_durations(score_attaque, score_defense, show_result=False):
    plt.figure(1)
    sc_atq = torch.tensor(score_attaque, dtype=torch.float)
    sc_dfs = torch.tensor(score_defense, dtype=torch.float)
    sc_diff = sc_atq - sc_dfs

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(sc_atq.numpy(), label='hero')
    plt.plot(sc_dfs.numpy(), label='adversaire')
    plt.plot(sc_diff.numpy(), label='difference')
    # Take 100 episode averages and plot them too
    if len(sc_atq) >= 100:
        means_atq = sc_atq.unfold(0, 100, 1).mean(1).view(-1)
        means_atq = torch.cat((torch.zeros(99), means_atq))
        plt.plot(means_atq.numpy())

        means_dfs = sc_dfs.unfold(0, 100, 1).mean(1).view(-1)
        means_dfs = torch.cat((torch.zeros(99), means_dfs))
        plt.plot(means_dfs.numpy())

        means_diff = sc_diff.unfold(0, 100, 1).mean(1).view(-1)
        means_diff = torch.cat((torch.zeros(99), means_diff))
        plt.plot(means_diff.numpy())

    plt.legend(loc='lower left')

    plt.pause(0.002)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())