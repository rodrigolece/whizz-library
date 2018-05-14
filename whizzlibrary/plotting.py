
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # to use the inset in subplot




def plotSingInfo(mat, nb_vecs, topic_names):
    nb_topics = len(topic_names)

    u, s, vt = np.linalg.svd(mat, full_matrices=False)         # full_matrices False means that v is m by n

    fig, axes = plt.subplots(figsize=(15,8), nrows=2, ncols=2)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # plt.setp(axes, xticks=range(nb_topics), xticklabels=topic_names) not that useful anymore after rearangement

    plt.sca(axes[0,0])
    plt.imshow(mat, aspect='auto', vmin=0, vmax=1400)

    plt.sca(axes[0,1])
    plt.semilogy(s)
    plt.title('Singular values')
    plt.grid()

    width, height = 2.4, 1.2     # inches, could also use "30%"
    inset_ax = inset_axes(axes[0,1], width, height)
    inset_ax.plot(s)
    inset_ax.grid()

    plt.sca(axes[1,0])
    plt.plot(u[:,:nb_vecs])
    plt.title('Top left singular vectors')
    plt.legend(range(nb_vecs))
    plt.axhline(0, color='k', linestyle='--', alpha=0.6)
    plt.xticks(range(nb_topics), topic_names)
    plt.grid()

    plt.sca(axes[1,1])
    axes[1,1].set_prop_cycle('color', colors[nb_vecs:])
    plt.plot(u[:,-nb_vecs:])
    plt.title('Bottom left singular vectors')
    plt.legend(range(nb_topics-nb_vecs, nb_topics), loc=1)
    plt.axhline(0, color='k', linestyle='--', alpha=0.6)
    plt.xticks(range(nb_topics), topic_names)
    plt.grid()

    # plt.tight_layout() # incompatible with inset_ax
