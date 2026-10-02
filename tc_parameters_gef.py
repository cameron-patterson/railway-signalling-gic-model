import numpy as np
from matplotlib.gridspec import GridSpec
from models import model_no_crossbonds
import matplotlib.pyplot as plt


def block_length_ordered():
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])

    outputs = model_no_crossbonds(section_name='glasgow_edinburgh_falkirk', axle_pos_a=[], axle_pos_b=[], ex_uniform=np.array([0]), ey_uniform=np.array([0]), y_trac=1.6, y_sig=0.1)
    ia = outputs['i_relays_a']

    data = np.load('data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_distances_bearings.npz')
    block_lengths = data['distances']
    order1 = np.argsort(block_lengths)
    ia = ia[order1]

    ax0.plot(block_lengths[order1], ia, '.')
    ax0.set_xlabel('Track circuit number')
    ax0.set_ylabel('Relay current (A)')

    plt.show()


def sig_parallel_admittance():
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])

    sig_leak = np.array([0.01, 0.1, 1])
    for i in range(0, len(sig_leak)):
        outputs = model_no_crossbonds(section_name='glasgow_edinburgh_falkirk', axle_pos_a=[], axle_pos_b=[],
                                      ex_uniform=np.array([0]), ey_uniform=np.array([5]), y_trac=1.6, y_sig=sig_leak[i])
        ia = outputs['i_relays_a']
        ax0.plot(ia, '.', label=f'Signal rail parallel admittance = {sig_leak[i]} S/km')

    ax0.set_xlabel('Track circuit number')
    ax0.set_ylabel('Relay current (A)')
    ax0.legend()
    ax0.axhline(0.055, color='tomato', linestyle='--')
    ax0.axhline(-0.055, color='tomato', linestyle='--')

    plt.show()


def sig_parallel_admittance_one():
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(3, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    e_fields = np.array([-10, 0, 10])
    leaks = np.array([0.01, 0.1, 1])
    axs = [ax0, ax1, ax2]
    for j in range(0, len(leaks)):
        for i in range(0, len(e_fields)):
            outputs = model_no_crossbonds(section_name='glasgow_edinburgh_falkirk', axle_pos_a=[], axle_pos_b=[],
                                          ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[i]]), y_trac=1.6, y_sig=leaks[j])
            ia = outputs['i_relays_a']
            axs[j].plot(ia, '.', label=f'E field = {e_fields[i]} S/km')
        axs[j].legend()
        axs[j].axhline(0.055, color='tomato', linestyle='--')
        axs[j].axhline(-0.055, color='tomato', linestyle='--')
    axs[2].set_xlabel('Track circuit number')
    axs[1].set_ylabel('Relay current (A)')

    plt.show()


def trac_parallel_admittance_one():
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(4, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    e_fields = np.array([-10, 0, 10])
    leaks = np.array([1.6, 5, 9])
    axs = [ax0, ax1, ax2]
    for j in range(0, len(leaks)):
        for i in range(0, len(e_fields)):
            outputs = model_no_crossbonds(section_name='glasgow_edinburgh_falkirk', axle_pos_a=[], axle_pos_b=[],
                                          ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[i]]), y_trac=leaks[j], y_sig=0.1)
            ia = outputs['i_relays_a']
            axs[j].plot(ia, '.', label=f'E field = {e_fields[i]} S/km')
        axs[j].legend()
        axs[j].axhline(0.055, color='tomato', linestyle='--')
        axs[j].axhline(-0.055, color='tomato', linestyle='--')
    axs[2].set_xlabel('Track circuit number')
    axs[1].set_ylabel('Relay current (A)')

    plt.show()


def trac_parallel_admittance_merged():
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    e_fields = np.array([-10, 0, 10])
    leaks = np.array([1.6, 5, 9])
    for j in range(0, len(leaks)):
        for i in range(0, len(e_fields)):
            outputs = model_no_crossbonds(section_name='glasgow_edinburgh_falkirk', axle_pos_a=[], axle_pos_b=[],
                                          ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[i]]), y_trac=leaks[j], y_sig=0.1)
            ia = outputs['i_relays_a']
            ax0.plot(ia, '.')
    ax0.axhline(0.055, color='tomato', linestyle='--')
    ax0.axhline(-0.055, color='tomato', linestyle='--')
    ax0.set_xlabel('Track circuit number')
    ax0.set_ylabel('Relay current (A)')

    data = np.load(f'data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_distances_bearings.npz')
    bearings = data['bearings']
    ax1.plot(np.rad2deg(bearings), 'x')

    plt.show()


#block_length_ordered()
#sig_parallel_admittance()
#sig_parallel_admittance_one()
#trac_parallel_admittance_one()
#trac_parallel_admittance_merged()