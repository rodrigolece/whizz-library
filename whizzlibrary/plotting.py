
import numpy as np
import scipy.stats as st # for pearsonr, has to be imported explicitly

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # to use the inset in subplot
from mpl_toolkits.axes_grid1 import make_axes_locatable         # to scale the colorbar



def plotSingInfo(mat, topic_names, nb_vecs=4):
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


def correlationMat(mat):
    nb_topics, _ = mat.shape
    correl_mat = np.zeros((nb_topics, nb_topics))

    for i in range(nb_topics):
        for j in range(i,nb_topics):
            correl_mat[i,j], _ = st.pearsonr(mat[i,:], mat[j,:])

    correl_mat[correl_mat == 0.0] = np.nan

    return correl_mat


def plotCorrelations(mat, topic_names):
    correl_mat = correlationMat(mat)
    nb_topics = len(topic_names)

    triu = np.triu(correl_mat, k=1)
    triu = triu[np.nonzero(triu)]
    m, M = np.min(triu), np.max(triu)

    topics_min = np.concatenate(np.where(correl_mat == m))
    topics_max = np.concatenate(np.where(correl_mat == M))

    _, axes = plt.subplots(figsize=(15, 4), ncols=3) #figsize=(nb_topics+1,nb_topics+1)
    plt.sca(axes[0])
    plt.imshow(correl_mat, vmin=0)

    axes[0].xaxis.tick_top()
    axes[0].yaxis.tick_right()
    plt.xticks(range(nb_topics), topic_names)
    plt.yticks(range(nb_topics), topic_names)

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.5)
    plt.colorbar(cax=cax)

    plt.sca(axes[1])
    plt.plot(mat[topics_min[0],:], mat[topics_min[1],:], 'o')
    plt.axis('equal')
    plt.xlabel(topic_names[topics_min[0]])
    plt.ylabel(topic_names[topics_min[1]])
    plt.title('%s:   rho = %.3f' % ('-'.join(topic_names[topics_min]), m ))

    plt.sca(axes[2])
    plt.plot(mat[topics_max[0],:], mat[topics_max[1],:], 'o')
    plt.axis('equal')
    plt.xlabel(topic_names[topics_max[0]])
    plt.ylabel(topic_names[topics_max[1]])
    plt.title('%s:   rho = %.3f' % ('-'.join(topic_names[topics_max]), M ))

    plt.tight_layout()
