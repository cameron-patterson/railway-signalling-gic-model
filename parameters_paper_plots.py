import numpy as np
import matplotlib.pyplot as plt
from models import model, test_model
from matplotlib.gridspec import GridSpec


def feed_polarity_staggering(sec):
    ey_values = np.linspace(-10, 10, 3)
    ex_values = np.zeros(np.shape(ey_values))
    output_no_stagger = model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6, electrical_staggering=False)
    i_relays_no_stagger = output_no_stagger["i_relays_a"]
    output_stagger = model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6, electrical_staggering=True)
    i_relays_stagger = output_stagger["i_relays_a"]

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, wspace=0.01, left=0.08, bottom=0.09, hspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    markersize = 10

    ax0.scatter(range(0, len(i_relays_no_stagger[:, 0])), i_relays_no_stagger[:, 0], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='tomato', label='Ey = 10 V/km')
    ax1.scatter(range(0, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[::2, 0], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='tomato', label='Ey = 10 V/km')
    ax1.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 0], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='tomato')
    ax2.scatter(range(0, len(i_relays_no_stagger[:, 2])), i_relays_no_stagger[:, 2], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='cornflowerblue', label='Ey = -10 V/km')
    ax3.scatter(range(0, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[::2, 2], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='cornflowerblue', label='Ey = -10 V/km')
    ax3.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 2], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='cornflowerblue')

    ax0.scatter(range(0, len(i_relays_no_stagger[:, 0])), i_relays_no_stagger[:, 1], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='white', label='No E')
    ax1.scatter(range(0, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[::2, 1], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='white', label='No E')
    ax1.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 1], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='white')
    ax2.scatter(range(0, len(i_relays_no_stagger[:, 2])), i_relays_no_stagger[:, 1], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='white', label='No E')
    ax3.scatter(range(0, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[::2, 1], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='white', label='No E')
    ax3.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 1], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='white')

    ax0.set_ylim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax3.set_ylim(-1, 1)

    ax0.set_title('No feed polarity staggering')
    ax1.set_title('Feed polarity staggering')

    ax0.set_xticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax3.set_yticks([])

    ax0.axhline(0, color='black', alpha=0.2, linewidth=1, linestyle='--')
    ax1.axhline(0, color='black', alpha=0.2, linewidth=1, linestyle='--')
    ax2.axhline(0, color='black', alpha=0.2, linewidth=1, linestyle='--')
    ax3.axhline(0, color='black', alpha=0.2, linewidth=1, linestyle='--')

    ax0.legend(loc='lower center', ncol=2, fontsize=10)
    ax1.legend(loc='lower center', ncol=2, fontsize=10)
    ax2.legend(loc='lower center', ncol=2, fontsize=10)
    ax3.legend(loc='lower center', ncol=2, fontsize=10)

    fig.supxlabel('Block Number')
    fig.supylabel('Relay Current (A)')

    plt.show()


def feed_polarity_staggering_merged(sec):
    ey_values = np.linspace(-10, 10, 3)
    ex_values = np.zeros(np.shape(ey_values))
    output_no_stagger = model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6, electrical_staggering=False)
    i_relays_no_stagger = output_no_stagger["i_relays_a"]
    output_stagger = model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6, electrical_staggering=True)
    i_relays_stagger = output_stagger["i_relays_a"]

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, wspace=0.01, left=0.1, bottom=0.09, hspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    markersize = 30

    ax1.scatter(range(0, len(i_relays_no_stagger[:, 0])), i_relays_no_stagger[:, 0], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='tomato', label='Ey = 10 V/km')
    ax2.scatter(range(0, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[::2, 0], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='tomato', label='Ey = 10 V/km')
    ax2.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 0], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='tomato')
    ax1.scatter(range(0, len(i_relays_no_stagger[:, 2])), i_relays_no_stagger[:, 2], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='cornflowerblue', label='Ey = -10 V/km')
    ax2.scatter(range(0, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[::2, 2], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='cornflowerblue', label='Ey = -10 V/km')
    ax2.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 2], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='cornflowerblue')

    ax2.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 1], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='white')
    ax1.scatter(range(0, len(i_relays_no_stagger[:, 2])), i_relays_no_stagger[:, 1], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='white', label='No E')
    ax2.scatter(range(0, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[::2, 1], s=markersize, linewidths=0.25, marker='^', edgecolor='black', facecolor='white', label='No E')
    ax2.scatter(range(1, len(i_relays_stagger[:, 0]), 2), i_relays_stagger[1::2, 1], s=markersize, linewidths=0.25, marker='v', edgecolor='black', facecolor='white')

    ax1.set_ylim(-1, 1)
    ax2.set_ylim(-1, 1)

    ax1.set_title('No feed polarity staggering')
    ax2.set_title('Feed polarity staggering')

    ax1.set_xticks([])

    ax1.axhline(0, color='black', alpha=0.2, linewidth=1, linestyle='--', zorder=-10)
    ax2.axhline(0, color='black', alpha=0.2, linewidth=1, linestyle='--', zorder=-10)
    ax1.axhline(0.055, color='red', alpha=0.2, linewidth=1, linestyle='-', zorder=-10)
    ax1.axhline(-0.055, color='red', alpha=0.2, linewidth=1, linestyle='-', zorder=-10)
    ax2.axhline(0.055, color='red', alpha=0.2, linewidth=1, linestyle='-', zorder=-10)
    ax2.axhline(-0.055, color='red', alpha=0.2, linewidth=1, linestyle='-', zorder=-10)

    ax1.legend(loc='lower center', ncol=2, fontsize=10)
    ax2.legend(loc='lower center', ncol=2, fontsize=10)

    fig.supxlabel('Track Circuit Number')
    fig.supylabel('Relay Current (A)')

    plt.savefig('polarity_staggering.pdf')
    plt.show()


def block_bearing(sec):
    lon_lats = np.load(f'data/rail_data/{sec}/{sec}_block_lons_lats.npz')
    lons = lon_lats['lons']
    lats = lon_lats['lats']
    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_lengths = data['distances']
    block_sum = np.insert(np.cumsum(block_lengths), 0, 0)

    ey_values = np.linspace(-10, 10, 5)
    ex_values = np.zeros(np.shape(ey_values))
    output_90 = model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6)
    i_relays_90 = output_90["i_relays_a"]
    output_test_90 = test_model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6)
    i_relays_test_90 = output_test_90["i_relays_a"]

    ex_values = np.linspace(-(10/np.sqrt(2)), (10/np.sqrt(2)), 5)
    ey_values = np.linspace(-(10/np.sqrt(2)), (10/np.sqrt(2)), 5)
    output_45 = model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6)
    i_relays_45 = output_45["i_relays_a"]
    output_test_45 = test_model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6)
    i_relays_test_45 = output_test_45["i_relays_a"]

    ex_values = np.linspace(-10, 10, 5)
    ey_values = np.zeros(np.shape(ex_values))
    output_0 = model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6)
    i_relays_0 = output_0["i_relays_a"]
    output_test_0 = test_model(sec, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6)
    i_relays_test_0 = output_test_0["i_relays_a"]

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 9, hspace=0.2, wspace=0.01, left=0.08, bottom=0.09)
    ax_map1 = fig.add_subplot(gs[0, :4])
    ax_map2 = fig.add_subplot(gs[0, 4:8])
    ax1 = fig.add_subplot(gs[1, :4])
    ax2 = fig.add_subplot(gs[1, 4:8])
    ax3 = fig.add_subplot(gs[2, :4])
    ax4 = fig.add_subplot(gs[2, 4:8])
    ax5 = fig.add_subplot(gs[3, :4])
    ax6 = fig.add_subplot(gs[3, 4:8])
    ax7 = fig.add_subplot(gs[1, 8])
    ax8 = fig.add_subplot(gs[2, 8])
    ax9 = fig.add_subplot(gs[3, 8])

    markersize = 20
    alpha = 0.2

    ax_map1.scatter(lons, lats, s=6, marker='o', facecolor='white', edgecolor='black')
    ax_map2.scatter(block_sum, np.zeros(len(block_sum)), s=6, marker='o', facecolor='white', edgecolor='black')
    ax_map1.plot(lons, lats, zorder=-1, linewidth=1)
    ax_map2.plot(block_sum, np.zeros(len(block_sum)), zorder=-1, linewidth=1)

    ax1.scatter(range(0, len(i_relays_90[:, 0]), 2), i_relays_90[::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -10 V/km', zorder=1)
    ax1.scatter(range(0, len(i_relays_90[:, 0]), 2), i_relays_90[::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -5 V/km', zorder=1)
    ax1.scatter(range(0, len(i_relays_90[:, 0]), 2), i_relays_90[::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', label='No geoelectric field', zorder=1)
    ax1.scatter(range(0, len(i_relays_90[:, 0]), 2), i_relays_90[::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 5 V/km', zorder=1)
    ax1.scatter(range(0, len(i_relays_90[:, 0]), 2), i_relays_90[::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 10 V/km', zorder=1)
    ax1.scatter(range(1, len(i_relays_90[:, 0]), 2), i_relays_90[1::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax1.scatter(range(1, len(i_relays_90[:, 0]), 2), i_relays_90[1::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax1.scatter(range(1, len(i_relays_90[:, 0]), 2), i_relays_90[1::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', alpha=alpha, zorder=1)
    ax1.scatter(range(1, len(i_relays_90[:, 0]), 2), i_relays_90[1::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax1.scatter(range(1, len(i_relays_90[:, 0]), 2), i_relays_90[1::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)

    ax2.scatter(range(0, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -10 V/km', zorder=1)
    ax2.scatter(range(0, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -5 V/km', zorder=1)
    ax2.scatter(range(0, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', label='No geoelectric field', zorder=1)
    ax2.scatter(range(0, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 5 V/km', zorder=1)
    ax2.scatter(range(0, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 10 V/km', zorder=1)
    ax2.scatter(range(1, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[1::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax2.scatter(range(1, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[1::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax2.scatter(range(1, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[1::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', alpha=alpha, zorder=1)
    ax2.scatter(range(1, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[1::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax2.scatter(range(1, len(i_relays_test_90[:, 0]), 2), i_relays_test_90[1::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)

    ax3.scatter(range(0, len(i_relays_45[:, 0]), 2), i_relays_45[::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -10 V/km', zorder=1)
    ax3.scatter(range(0, len(i_relays_45[:, 0]), 2), i_relays_45[::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -5 V/km', zorder=1)
    ax3.scatter(range(0, len(i_relays_45[:, 0]), 2), i_relays_45[::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', label='No geoelectric field', zorder=1)
    ax3.scatter(range(0, len(i_relays_45[:, 0]), 2), i_relays_45[::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 5 V/km', zorder=1)
    ax3.scatter(range(0, len(i_relays_45[:, 0]), 2), i_relays_45[::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 10 V/km', zorder=1)
    ax3.scatter(range(1, len(i_relays_45[:, 0]), 2), i_relays_45[1::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax3.scatter(range(1, len(i_relays_45[:, 0]), 2), i_relays_45[1::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax3.scatter(range(1, len(i_relays_45[:, 0]), 2), i_relays_45[1::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', alpha=alpha, zorder=1)
    ax3.scatter(range(1, len(i_relays_45[:, 0]), 2), i_relays_45[1::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax3.scatter(range(1, len(i_relays_45[:, 0]), 2), i_relays_45[1::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)

    ax4.scatter(range(0, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -10 V/km', zorder=1)
    ax4.scatter(range(0, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -5 V/km', zorder=1)
    ax4.scatter(range(0, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', label='No geoelectric field', zorder=1)
    ax4.scatter(range(0, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 5 V/km', zorder=1)
    ax4.scatter(range(0, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 10 V/km', zorder=1)
    ax4.scatter(range(1, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[1::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax4.scatter(range(1, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[1::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax4.scatter(range(1, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[1::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', alpha=alpha, zorder=1)
    ax4.scatter(range(1, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[1::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax4.scatter(range(1, len(i_relays_test_45[:, 0]), 2), i_relays_test_45[1::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)

    ax5.scatter(range(0, len(i_relays_0[:, 0]), 2), i_relays_0[::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -10 V/km', zorder=1)
    ax5.scatter(range(0, len(i_relays_0[:, 0]), 2), i_relays_0[::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -5 V/km', zorder=1)
    ax5.scatter(range(0, len(i_relays_0[:, 0]), 2), i_relays_0[::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', label='No geoelectric field', zorder=1)
    ax5.scatter(range(0, len(i_relays_0[:, 0]), 2), i_relays_0[::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 5 V/km', zorder=1)
    ax5.scatter(range(0, len(i_relays_0[:, 0]), 2), i_relays_0[::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 10 V/km', zorder=1)
    ax5.scatter(range(1, len(i_relays_0[:, 0]), 2), i_relays_0[1::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax5.scatter(range(1, len(i_relays_0[:, 0]), 2), i_relays_0[1::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax5.scatter(range(1, len(i_relays_0[:, 0]), 2), i_relays_0[1::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', alpha=alpha, zorder=1)
    ax5.scatter(range(1, len(i_relays_0[:, 0]), 2), i_relays_0[1::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax5.scatter(range(1, len(i_relays_0[:, 0]), 2), i_relays_0[1::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)

    ax6.scatter(range(0, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -10 V/km', zorder=1)
    ax6.scatter(range(0, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = -5 V/km', zorder=1)
    ax6.scatter(range(0, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', label='No geoelectric field', zorder=1)
    ax6.scatter(range(0, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 5 V/km', zorder=1)
    ax6.scatter(range(0, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', label='E = 10 V/km', zorder=1)
    ax6.scatter(range(1, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[1::2, 0], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax6.scatter(range(1, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[1::2, 1], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax6.scatter(range(1, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[1::2, 2], s=markersize, linewidths=0.25, marker='o', facecolor='white', edgecolor='black', alpha=alpha, zorder=1)
    ax6.scatter(range(1, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[1::2, 3], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)
    ax6.scatter(range(1, len(i_relays_test_0[:, 0]), 2), i_relays_test_0[1::2, 4], s=markersize, linewidths=0.25, marker='o', edgecolor='black', alpha=alpha, zorder=1)

    ax7.annotate("", xytext=(0, 0), xy=(1, 0), arrowprops=dict(arrowstyle="->"))
    ax7.text(0.5, 1.25, r'$90\degree$', ha='center')
    ax8.annotate("", xytext=(0, 0), xy=(np.sqrt(0.5), np.sqrt(0.5)), arrowprops=dict(arrowstyle="->"))
    ax8.text(0.5, 1.25, r'$45\degree$', ha='center')
    ax9.annotate("", xytext=(0, 0), xy=(0, 1), arrowprops=dict(arrowstyle="->"))
    ax9.text(0.5, 1.25, r'$0\degree$', ha='center')

    ylim_min = -1.5
    ylim_max = 1
    ax1.set_ylim(ylim_min, ylim_max)
    ax2.set_ylim(ylim_min, ylim_max)
    ax3.set_ylim(ylim_min, ylim_max)
    ax4.set_ylim(ylim_min, ylim_max)
    ax5.set_ylim(ylim_min, ylim_max)
    ax6.set_ylim(ylim_min, ylim_max)

    ax7.set_xlim(-0.5, 1.5)
    ax8.set_xlim(-0.5, 1.5)
    ax9.set_xlim(-0.5, 1.5)
    ax7.set_ylim(-0.5, 1.5)
    ax8.set_ylim(-0.5, 1.5)
    ax9.set_ylim(-0.5, 1.5)

    ax_map1.set_title('Realistic Line Geometry')
    ax_map2.set_title(r'Straight Line Geometry ($90\degree$)')
    ax7.set_title('E-field Bearing')

    ax_map1.set_xticks([])
    ax_map2.set_xticks([])
    # ax1.set_xticks([])
    # ax2.set_xticks([])
    # ax3.set_xticks([])
    # ax4.set_xticks([])
    ax7.set_xticks([])
    ax8.set_xticks([])
    ax9.set_xticks([])

    ax_map1.set_yticks([])
    ax_map2.set_yticks([])
    ax2.set_yticks([])
    ax4.set_yticks([])
    ax6.set_yticks([])
    ax7.set_yticks([])
    ax8.set_yticks([])
    ax9.set_yticks([])

    fig.supxlabel('Track Circuit Number')
    fig.supylabel('Relay Current (A)')

    ax1.legend(loc='lower center', ncol=3, fontsize=10)
    ax2.legend(loc='lower center', ncol=3, fontsize=10)
    ax3.legend(loc='lower center', ncol=3, fontsize=10)
    ax4.legend(loc='lower center', ncol=3, fontsize=10)
    ax5.legend(loc='lower center', ncol=3, fontsize=10)
    ax6.legend(loc='lower center', ncol=3, fontsize=10)

    ax1.grid(axis='x', zorder=-2)
    ax2.grid(axis='x', zorder=-2)
    ax3.grid(axis='x', zorder=-2)
    ax4.grid(axis='x', zorder=-2)
    ax5.grid(axis='x', zorder=-2)
    ax6.grid(axis='x', zorder=-2)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)
    ax5.set_axisbelow(True)
    ax6.set_axisbelow(True)

    plt.savefig('block_bearings.pdf')
    plt.show()


def block_bearing_thresholds_rs(sec):
    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    threshold = 0.055
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])

        output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6)
        currents_all_e[i] = output['i_relays_a']

    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e.shape[1])]
    valid_sections = np.isfinite(min_strength_idx)
    misoperation_bearings = np.full(len(first_misoperation_bearing), np.nan)
    misoperation_bearings[valid_sections] = bearings[first_misoperation_bearing[valid_sections]]
    misoperation_bearings = np.rad2deg(misoperation_bearings)

    # Plot results
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(np.arange(0, len(block_bearings), 2), block_bearings[::2], s=markersize, label='Bearing of Block (Relative to Feed Polarity)', facecolor='white', edgecolor='black', linewidths=0.5)
    block_bearings[5] = block_bearings[5] - 360  # Tweaking block 5 as it is > 180 degrees
    ax0.scatter(np.arange(1, len(block_bearings), 2), block_bearings[1::2] + 180, s=markersize, facecolor='white', edgecolor='black', linewidths=0.5)
    ax0.scatter(np.arange(0, len(block_bearings)), misoperation_bearings, s=markersize, label='Bearing of Minimum RSF Misoperation Electric Field', facecolor='orangered', edgecolor='black', marker='X', linewidths=0.5)
    ax0.set_ylim(0, 395)
    ax0.set_xlabel('Track Circuit Number')
    ax0.set_ylabel(r'Bearing ($\degree$)')
    ax0.legend(loc='upper center', ncols=2)
    plt.savefig('block_bearing_thresholds_rs.pdf')
    plt.show()


def block_bearing_thresholds_ws(sec):
    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e_midpoint = np.load(f'data/parameters_paper_data/{sec}_currents_all_e_ws_midpoint.npy')
    currents_all_e_exiting = np.load(f'data/parameters_paper_data/{sec}_currents_all_e_ws_exiting.npy')

    threshold = 0.081
    misoperations_mask = currents_all_e_midpoint < -threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_midpoint.shape[1])]
    valid_sections = np.isfinite(min_strength_idx)
    misoperation_bearings = np.full(len(first_misoperation_bearing), np.nan)
    misoperation_bearings[valid_sections] = bearings[first_misoperation_bearing[valid_sections]]
    misoperation_bearings_neg_midpoint = np.rad2deg(misoperation_bearings)

    misoperations_mask = currents_all_e_exiting < -threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_exiting.shape[1])]
    valid_sections = np.isfinite(min_strength_idx)
    misoperation_bearings = np.full(len(first_misoperation_bearing), np.nan)
    misoperation_bearings[valid_sections] = bearings[first_misoperation_bearing[valid_sections]]
    misoperation_bearings_neg_exiting = np.rad2deg(misoperation_bearings)

    misoperations_mask = currents_all_e_midpoint > threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_midpoint.shape[1])]
    valid_sections = np.isfinite(min_strength_idx)
    misoperation_bearings = np.full(len(first_misoperation_bearing), np.nan)
    misoperation_bearings[valid_sections] = bearings[first_misoperation_bearing[valid_sections]]
    misoperation_bearings_pos_midpoint = np.rad2deg(misoperation_bearings)

    misoperations_mask = currents_all_e_exiting > threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_exiting.shape[1])]
    valid_sections = np.isfinite(min_strength_idx)
    misoperation_bearings = np.full(len(first_misoperation_bearing), np.nan)
    misoperation_bearings[valid_sections] = bearings[first_misoperation_bearing[valid_sections]]
    misoperation_bearings_pos_exiting = np.rad2deg(misoperation_bearings)

    # Plot results
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax1.scatter(np.arange(0, len(block_bearings)), block_bearings, s=markersize, label='Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='v')
    ax1.scatter(np.arange(0, len(block_bearings)), misoperation_bearings_neg_midpoint, s=markersize, label='Bearing of Minimum WSF Misoperation Electric Field', facecolor='limegreen', edgecolor='black', marker='X', linewidths=0.5)
    ax2.scatter(np.arange(0, len(block_bearings)), block_bearings, s=markersize, label='Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='v')
    ax2.scatter(np.arange(0, len(block_bearings)), misoperation_bearings_neg_exiting, s=markersize, label='Bearing of Minimum WSF Misoperation Electric Field', facecolor='limegreen', edgecolor='black', marker='X', linewidths=0.5)
    block_bearings[5] = block_bearings[5] - 180  # Tweaking block 5 as it is > 180 degrees
    ax3.scatter(np.arange(0, len(block_bearings)), block_bearings + 180, s=markersize, label='Antiparallel Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='^')
    ax3.scatter(np.arange(0, len(block_bearings)), misoperation_bearings_pos_midpoint, s=markersize, label='Bearing of Minimum WSF Misoperation Electric Field', facecolor='limegreen', edgecolor='black', marker='X', linewidths=0.5)
    ax4.scatter(np.arange(0, len(block_bearings)), block_bearings + 180, s=markersize, label='Antiparallel Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='^')
    ax4.scatter(np.arange(0, len(block_bearings)), misoperation_bearings_pos_exiting, s=markersize, label='Bearing of Minimum WSF Misoperation Electric Field', facecolor='limegreen', edgecolor='black', marker='X', linewidths=0.5)

    ax1.set_ylim(0, 395)
    ax2.set_ylim(0, 395)
    ax3.set_ylim(0, 395)
    ax4.set_ylim(0, 395)

    ax1.set_xticks([])
    ax3.set_xticks([])

    ax3.set_yticks([])
    ax4.set_yticks([])

    ax1.set_xlabel('Track Circuit Number')
    ax1.set_ylabel(r'Bearing ($\degree$)')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')
    ax3.legend(loc='lower center')
    ax4.legend(loc='lower center')
    plt.show()


def block_bearing_thresholds_ws_merged(sec):
    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    currents_all_e_midpoint = np.load(f'data/parameters_paper_data/{sec}_currents_all_e_ws_midpoint.npy')
    currents_all_e_exiting = np.load(f'data/parameters_paper_data/{sec}_currents_all_e_ws_exiting.npy')

    threshold = 0.081
    misoperations_mask = (currents_all_e_midpoint < -threshold) | (currents_all_e_midpoint > threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_midpoint.shape[1])]
    valid_sections = np.isfinite(min_strength_idx)
    misoperation_bearings = np.full(len(first_misoperation_bearing), np.nan)
    misoperation_bearings[valid_sections] = bearings[first_misoperation_bearing[valid_sections]]
    misoperation_bearings_midpoint = np.rad2deg(misoperation_bearings)

    misoperations_mask = (currents_all_e_midpoint < -threshold) | (currents_all_e_midpoint > threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_exiting.shape[1])]
    valid_sections = np.isfinite(min_strength_idx)
    misoperation_bearings = np.full(len(first_misoperation_bearing), np.nan)
    misoperation_bearings[valid_sections] = bearings[first_misoperation_bearing[valid_sections]]
    misoperation_bearings_exiting = np.rad2deg(misoperation_bearings)

    # Plot results
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.scatter(np.arange(0, len(block_bearings)), block_bearings, s=markersize, label='Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='v')
    ax2.scatter(np.arange(0, len(block_bearings)), block_bearings, s=markersize, label='Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='v')
    block_bearings[5] = block_bearings[5] - 360  # Tweaking block 5 as it is > 180 degrees
    ax1.scatter(np.arange(0, len(block_bearings)), block_bearings + 180, s=markersize, label='Antiparallel Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='^')
    ax2.scatter(np.arange(0, len(block_bearings)), block_bearings + 180, s=markersize, label='Antiparallel Bearing of Block', facecolor='white', edgecolor='black', linewidths=0.5, marker='^')
    ax1.scatter(np.arange(0, len(block_bearings)), misoperation_bearings_midpoint, s=markersize, label='Bearing of Minimum WSF Misoperation Electric Field', facecolor='limegreen', edgecolor='black', marker='X', linewidths=0.5)
    ax2.scatter(np.arange(0, len(block_bearings)), misoperation_bearings_exiting, s=markersize, label='Bearing of Minimum WSF Misoperation Electric Field', facecolor='limegreen', edgecolor='black', marker='X', linewidths=0.5)

    ax1.set_ylim(0, 395)
    ax2.set_ylim(0, 395)

    ax1.set_xticks([])

    ax2.set_xlabel('Track Circuit Number')
    ax1.set_ylabel(r'Bearing ($\degree$)')
    ax2.set_ylabel(r'Bearing ($\degree$)')
    ax1.legend(loc='upper center', ncols=3)
    ax2.legend(loc='upper center', ncols=3)

    plt.savefig('block_bearing_thresholds_ws.pdf')
    plt.show()


def block_length_sorted_rs(sec):
    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_lengths = data['distances']
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    threshold = 0.055
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])

        output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6)
        currents_all_e[i] = output['i_relays_a']

    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e.shape[1])]
    min_strength_e_field = np.full(currents_all_e.shape[1], np.nan)
    valid_sections = np.isfinite(min_strength_idx)
    min_strength_e_field[valid_sections] = e_values[
        min_strength_idx[valid_sections].astype(int)
    ]

    # Sort by length
    sort_idx = np.argsort(block_lengths)
    block_lengths_sorted = block_lengths[sort_idx]
    min_strength_e_field_sorted = min_strength_e_field[sort_idx]

    # Plot Results
    x_range = np.arange(0, len(block_lengths_sorted), 1)
    # Fitting line
    # Clean NaN values
    mask = ~np.isnan(x_range) & ~np.isnan(min_strength_e_field_sorted)
    x_range_clean = x_range[mask]
    min_strength_e_field_clean = min_strength_e_field_sorted[mask]
    coefficients = np.polyfit(x_range_clean, min_strength_e_field_clean, 1)
    p = np.poly1d(coefficients)
    xp = np.linspace(min(x_range), max(x_range), 100)
    # Plotting
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(xp, p(xp), color='black', linestyle='--', zorder=-1)
    ax0.scatter(x_range, min_strength_e_field_sorted, s=markersize, facecolor='orangered', edgecolor='black', marker='X', linewidths=0.5)
    ax0.set_ylim(0, 20)
    ax0.set_xlabel('Track Circuits Sorted by Increasing Length')
    ax0.set_ylabel('Minimum RSF Misoperation Electric Field Strength (V/km)')
    plt.savefig('blocks_length_sorted_rs.pdf')
    plt.show()


def block_length_sorted_ws(sec):
    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_lengths = data['distances']
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    currents_all_e_midpoint = np.load(f'data/parameters_paper_data/{sec}_currents_all_e_ws_midpoint.npy')
    currents_all_e_exiting = np.load(f'data/parameters_paper_data/{sec}_currents_all_e_ws_exiting.npy')
    threshold = 0.081

    misoperations_mask = (currents_all_e_midpoint > threshold) | (currents_all_e_midpoint < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_midpoint.shape[1])]
    min_strength_e_field = np.full(currents_all_e_midpoint.shape[1], np.nan)
    valid_sections = np.isfinite(min_strength_idx)
    min_strength_e_field[valid_sections] = e_values[
        min_strength_idx[valid_sections].astype(int)
    ]

    # Sort by length
    sort_idx = np.argsort(block_lengths)
    block_lengths_sorted = block_lengths[sort_idx]
    min_strength_e_field_sorted = min_strength_e_field[sort_idx]

    # Plot Results
    x_range = np.arange(0, len(block_lengths_sorted), 1)
    # Fitting line
    # Clean NaN values
    mask = ~np.isnan(x_range) & ~np.isnan(min_strength_e_field_sorted)
    x_range_clean = x_range[mask]
    min_strength_e_field_clean = min_strength_e_field_sorted[mask]
    coefficients = np.polyfit(x_range_clean, min_strength_e_field_clean, 1)
    p = np.poly1d(coefficients)
    xp = np.linspace(min(x_range), max(x_range), 100)
    # Plotting
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(xp, p(xp), color='black', linestyle='--', zorder=-1)
    ax0.scatter(x_range, min_strength_e_field_sorted, s=markersize, facecolor='limegreen', edgecolor='black', marker='X', linewidths=0.5)
    ax0.set_ylim(0, 14)
    ax0.set_xlabel('Track Circuits Sorted by Increasing Length')
    ax0.set_ylabel('Minimum RSF Misoperation Electric Field Strength (V/km)')
    plt.savefig('blocks_length_sorted_ws.pdf')
    plt.show()


def rail_impedance_rs(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    threshold = 0.055
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6)
        currents_all_e[i] = output['i_relays_a']
    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='v', edgecolor='black', facecolor='orangered', linewidths=0.5, label='0.0289 ohm/km')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    threshold = 0.055
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6, z_trac=0.25, z_sig=0.25)
        currents_all_e[i] = output['i_relays_a']
    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='^', edgecolor='black', facecolor='mistyrose', linewidths=0.5, label='0.25 ohm/km')

    ax0.legend(loc='upper center')
    ax0.set_xlabel('Track Circuit Number')
    ax0.set_ylabel('Minimum RSF Misoperation Electric Field Strength (V/km)')

    plt.savefig('rail_impedance_rs.pdf')
    plt.show()


def rail_impedance_ws(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    threshold = 0.081
    axles = np.load(f'data/axle_positions/glasgow_edinburgh_falkirk_train_end_axles_midpoint_a.npy')
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6, axle_pos_a=ax)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='v', edgecolor='black', facecolor='limegreen', linewidths=0.5, label='0.0289 ohm/km')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    threshold = 0.081
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6, axle_pos_a=ax, z_trac=0.25, z_sig=0.25)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='^', edgecolor='black', facecolor='honeydew', linewidths=0.5, label='0.25 ohm/km')

    ax0.legend(loc='upper center')
    ax0.set_xlabel('Track Circuit Number')
    ax0.set_ylabel('Minimum WSF Misoperation Electric Field Strength (V/km)')

    plt.savefig('rail_impedance_ws.pdf')
    plt.show()


def traction_rail_leakage_rs(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    threshold = 0.055

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=0.53)
        currents_all_e[i] = output['i_relays_a']
    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='v', edgecolor='black', facecolor='mistyrose', linewidths=0.5, label='0.530 S/km')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6)
        currents_all_e[i] = output['i_relays_a']
    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='o', edgecolor='black', facecolor='orangered', linewidths=0.5, label='1.6 S/km')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=4.14)
        currents_all_e[i] = output['i_relays_a']
    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='^', edgecolor='black', facecolor='tomato', linewidths=0.5, label='4.14 S/km')

    ax0.legend(loc='upper center')
    ax0.set_xlabel('Track Circuit Number')
    ax0.set_ylabel('Minimum RSF Misoperation Electric Field Strength (V/km)')

    plt.savefig('traction_rail_leakage_rs.pdf')
    plt.show()


def traction_rail_leakage_ws(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    threshold = 0.081
    axles = np.load(f'data/axle_positions/glasgow_edinburgh_falkirk_train_end_axles_midpoint_a.npy')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=0.53, axle_pos_a=ax)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='v', edgecolor='black', facecolor='honeydew', linewidths=0.5, label='0.53 S/km')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6, axle_pos_a=ax)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='o', edgecolor='black', facecolor='limegreen', linewidths=0.5, label='1.6 S/km')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=4.14, axle_pos_a=ax)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='^', edgecolor='black', facecolor='palegreen', linewidths=0.5, label='4.14 S/km')

    ax0.legend(loc='upper center')
    ax0.set_xlabel('Track Circuit Number')
    ax0.set_ylabel('Minimum WSF Misoperation Electric Field Strength (V/km)')

    plt.savefig('traction_rail_leakage_ws.pdf')
    plt.show()

# feed_polarity_staggering('glasgow_edinburgh_falkirk')
# feed_polarity_staggering_merged('glasgow_edinburgh_falkirk')
# block_bearing('glasgow_edinburgh_falkirk')
# block_bearing_thresholds_rs('glasgow_edinburgh_falkirk')
# block_bearing_thresholds_ws('glasgow_edinburgh_falkirk')
# block_bearing_thresholds_ws_merged('glasgow_edinburgh_falkirk')
# block_length_sorted_rs('glasgow_edinburgh_falkirk')
# block_length_sorted_ws('glasgow_edinburgh_falkirk')
# rail_impedance_rs('glasgow_edinburgh_falkirk')
# rail_impedance_ws('glasgow_edinburgh_falkirk')
# traction_rail_leakage_rs('glasgow_edinburgh_falkirk')
traction_rail_leakage_ws('glasgow_edinburgh_falkirk')
