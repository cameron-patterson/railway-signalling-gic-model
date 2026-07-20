from matplotlib.gridspec import GridSpec
from models import model
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cartopy as cartopy
import matplotlib.image as mpimg


def mast_resistivity_merged():
    mast_res_ge = np.load(f'data/resistivity_data/glasgow_edinburgh_falkirk_mast_resistivities.npz')
    mast_res_a20_ge = mast_res_ge['a20']
    mast_res_a50_ge = mast_res_ge['a50']
    mast_res_a80_ge = mast_res_ge['a80']
    mast_res_wcml = np.load(f'data/resistivity_data/west_coast_main_line_mast_resistivities.npz')
    mast_res_a20_wcml = mast_res_wcml['a20']
    mast_res_a50_wcml = mast_res_wcml['a50']
    mast_res_a80_wcml = mast_res_wcml['a80']
    mast_res_ecml = np.load(f'data/resistivity_data/east_coast_main_line_mast_resistivities.npz')
    mast_res_a20_ecml = mast_res_ecml['a20']
    mast_res_a50_ecml = mast_res_ecml['a50']
    mast_res_a80_ecml = mast_res_ecml['a80']

    markersize = 1
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    ax0.scatter(np.arange(0, len(mast_res_a50_ge), 1), mast_res_a50_ge, s=markersize, marker='.', color='darkorchid', zorder=1, label='Glasgow to Edinburgh via Falkirk')
    ax1.scatter(np.arange(0, len(mast_res_a50_ecml), 1), mast_res_a50_ecml, s=markersize, marker='.', color='darkorchid', zorder=1, label='East Coast Main Line')
    ax2.scatter(np.arange(0, len(mast_res_a50_wcml), 1), mast_res_a50_wcml, s=markersize, marker='.', color='darkorchid', zorder=1, label='West Coast Main Line')

    linewidth = 0.8
    for i in range(0, len(mast_res_a20_ge)):
        ax0.plot((i, i), (mast_res_a50_ge[i], mast_res_a80_ge[i]), color='plum', zorder=-1, linewidth=linewidth)
        ax0.plot((i, i), (mast_res_a20_ge[i], mast_res_a50_ge[i]), color='plum', zorder=-1, linewidth=linewidth)
    for i in range(0, len(mast_res_a20_ecml)):
        ax1.plot((i, i), (mast_res_a50_ecml[i], mast_res_a80_ecml[i]), color='plum', zorder=-1, linewidth=linewidth)
        ax1.plot((i, i), (mast_res_a20_ecml[i], mast_res_a50_ecml[i]), color='plum', zorder=-1, linewidth=linewidth)
    for i in range(0, len(mast_res_a20_wcml)):
        ax2.plot((i, i), (mast_res_a50_wcml[i], mast_res_a80_wcml[i]), color='plum', zorder=-1, linewidth=linewidth)
        ax2.plot((i, i), (mast_res_a20_wcml[i], mast_res_a50_wcml[i]), color='plum', zorder=-1, linewidth=linewidth)

    ax0.set_ylim(5, 40000)
    ax1.set_ylim(5, 40000)
    ax2.set_ylim(5, 40000)
    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Mast number')
    ax1.set_ylabel('Resistivity at mast location (\u03A9 m)')
    ax0.legend(loc='upper center')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')

    plt.savefig(f'plots/1. mast_res_merged.pdf')
    #plt.show()


def block_leakage_merged():
    block_leak_ge = np.load(f'data/resistivity_data/glasgow_edinburgh_falkirk_block_leakage.npz')
    block_leak_a20_ge = block_leak_ge['a20']
    block_leak_a50_ge = block_leak_ge['a50']
    block_leak_a80_ge = block_leak_ge['a80']
    block_leak_wcml = np.load(f'data/resistivity_data/west_coast_main_line_block_leakage.npz')
    block_leak_a20_wcml = block_leak_wcml['a20']
    block_leak_a50_wcml = block_leak_wcml['a50']
    block_leak_a80_wcml = block_leak_wcml['a80']
    block_leak_ecml = np.load(f'data/resistivity_data/east_coast_main_line_block_leakage.npz')
    block_leak_a20_ecml = block_leak_ecml['a20']
    block_leak_a50_ecml = block_leak_ecml['a50']
    block_leak_a80_ecml = block_leak_ecml['a80']

    markersize = 1
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    ax0.scatter(np.arange(0, len(block_leak_a50_ge), 1), block_leak_a50_ge, s=markersize, marker='o', color='royalblue', zorder=1, label='Glasgow to Edinburgh via Falkirk')
    ax1.scatter(np.arange(0, len(block_leak_a50_ecml), 1), block_leak_a50_ecml, s=markersize, marker='o', color='royalblue', zorder=1, label='East Coast Main Line')
    ax2.scatter(np.arange(0, len(block_leak_a50_wcml), 1), block_leak_a50_wcml, s=markersize, marker='o', color='royalblue', zorder=1, label='West Coast Main Line')

    for i in range(0, len(block_leak_a20_ge)):
        ax0.plot((i, i), (block_leak_a50_ge[i], block_leak_a80_ge[i]), color='lightsteelblue', zorder=-2, linewidth=1)
        ax0.plot((i, i), (block_leak_a20_ge[i], block_leak_a50_ge[i]), color='lightsteelblue', zorder=-2, linewidth=1)
    for i in range(0, len(block_leak_a20_ecml)):
        ax1.plot((i, i), (block_leak_a50_ecml[i], block_leak_a80_ecml[i]), color='lightsteelblue', zorder=-2, linewidth=1)
        ax1.plot((i, i), (block_leak_a20_ecml[i], block_leak_a50_ecml[i]), color='lightsteelblue', zorder=-2, linewidth=1)
    for i in range(0, len(block_leak_a20_wcml)):
        ax2.plot((i, i), (block_leak_a50_wcml[i], block_leak_a80_wcml[i]), color='lightsteelblue', zorder=-2, linewidth=1)
        ax2.plot((i, i), (block_leak_a20_wcml[i], block_leak_a50_wcml[i]), color='lightsteelblue', zorder=-2, linewidth=1)

    ax0.set_ylim(0, 20)
    ax1.set_ylim(0, 20)
    ax2.set_ylim(0, 20)

    # ax1.axvline(265, color='black', alpha=0.5)
    # ax1.axvline(313, color='black', alpha=0.5)
    # ax1.axvline(459, color='black', alpha=0.5)
    # ax1.axvline(541, color='black', alpha=0.5)

    ax2.set_xlabel('Track circuit number')
    ax1.set_ylabel('Parallel admittance (S $\mathregular{km^{-1}}$)')
    ax0.legend(loc='upper center')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')

    plt.savefig(f'plots/3. block_leak_merged.pdf')
    #plt.show()


def block_leakage_total_merged():
    block_lengths_ge = np.load(f'data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_distances_bearings.npz')['distances']
    block_lengths_wcml = np.load(f'data/rail_data/west_coast_main_line/west_coast_main_line_distances_bearings.npz')['distances']
    block_lengths_ecml = np.load(f'data/rail_data/east_coast_main_line/east_coast_main_line_distances_bearings.npz')['distances']

    block_leak_ge = np.load(f'data/resistivity_data/glasgow_edinburgh_falkirk_block_leakage.npz')
    block_leak_a20_ge = np.multiply(block_leak_ge['a20'], block_lengths_ge)
    block_leak_a50_ge = np.multiply(block_leak_ge['a50'], block_lengths_ge)
    block_leak_a80_ge = np.multiply(block_leak_ge['a80'], block_lengths_ge)
    block_leak_wcml = np.load(f'data/resistivity_data/west_coast_main_line_block_leakage.npz')
    block_leak_a20_wcml = np.multiply(block_leak_wcml['a20'], block_lengths_wcml)
    block_leak_a50_wcml = np.multiply(block_leak_wcml['a50'], block_lengths_wcml)
    block_leak_a80_wcml = np.multiply(block_leak_wcml['a80'], block_lengths_wcml)
    block_leak_ecml = np.load(f'data/resistivity_data/east_coast_main_line_block_leakage.npz')
    block_leak_a20_ecml = np.multiply(block_leak_ecml['a20'], block_lengths_ecml)
    block_leak_a50_ecml = np.multiply(block_leak_ecml['a50'], block_lengths_ecml)
    block_leak_a80_ecml = np.multiply(block_leak_ecml['a80'], block_lengths_ecml)

    markersize = 1
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    ax0.scatter(np.arange(0, len(block_leak_a50_ge), 1), block_leak_a50_ge, s=markersize, marker='o', color='royalblue', zorder=1, label='Glasgow to Edinburgh via Falkirk')
    ax1.scatter(np.arange(0, len(block_leak_a50_ecml), 1), block_leak_a50_ecml, s=markersize, marker='o', color='royalblue', zorder=1, label='East Coast Main Line')
    ax2.scatter(np.arange(0, len(block_leak_a50_wcml), 1), block_leak_a50_wcml, s=markersize, marker='o', color='royalblue', zorder=1, label='West Coast Main Line')

    ax0.scatter(np.arange(0, len(block_lengths_ge)), block_lengths_ge * 1.6, s=markersize, marker='o', color='grey', zorder=1)
    ax1.scatter(np.arange(0, len(block_lengths_ecml)), block_lengths_ecml * 1.6, s=markersize, marker='o', color='grey', zorder=1)
    ax2.scatter(np.arange(0, len(block_lengths_wcml)), block_lengths_wcml * 1.6, s=markersize, marker='o', color='grey', zorder=1)

    for i in range(0, len(block_leak_a20_ge)):
        ax0.plot((i, i), (block_leak_a50_ge[i], block_leak_a80_ge[i]), color='lightsteelblue', zorder=-2, linewidth=1)
        ax0.plot((i, i), (block_leak_a20_ge[i], block_leak_a50_ge[i]), color='lightsteelblue', zorder=-2, linewidth=1)
    for i in range(0, len(block_leak_a20_ecml)):
        ax1.plot((i, i), (block_leak_a50_ecml[i], block_leak_a80_ecml[i]), color='lightsteelblue', zorder=-2, linewidth=1)
        ax1.plot((i, i), (block_leak_a20_ecml[i], block_leak_a50_ecml[i]), color='lightsteelblue', zorder=-2, linewidth=1)
    for i in range(0, len(block_leak_a20_wcml)):
        ax2.plot((i, i), (block_leak_a50_wcml[i], block_leak_a80_wcml[i]), color='lightsteelblue', zorder=-2, linewidth=1)
        ax2.plot((i, i), (block_leak_a20_wcml[i], block_leak_a50_wcml[i]), color='lightsteelblue', zorder=-2, linewidth=1)

    ax0.set_ylim(0, 14)
    ax1.set_ylim(0, 14)
    ax2.set_ylim(0, 14)

    # ax1.axvline(265, color='black', alpha=0.5)
    # ax1.axvline(313, color='black', alpha=0.5)
    # ax1.axvline(459, color='black', alpha=0.5)
    # ax1.axvline(541, color='black', alpha=0.5)

    ax2.set_xlabel('Track circuit number')
    ax1.set_ylabel('Total parallel admittance (S)')
    ax0.legend(loc='upper center')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')

    plt.savefig(f'plots/4. block_leak_total_merged.pdf')
    #plt.show()


def block_leakage_changes_merged():
    block_leak_ge = np.load(f'data/resistivity_data/glasgow_edinburgh_falkirk_block_leakage.npz')
    block_leak_a50_ge = block_leak_ge['a50']
    block_leak_wcml = np.load(f'data/resistivity_data/west_coast_main_line_block_leakage.npz')
    block_leak_a50_wcml = block_leak_wcml['a50']
    block_leak_ecml = np.load(f'data/resistivity_data/east_coast_main_line_block_leakage.npz')
    block_leak_a50_ecml = block_leak_ecml['a50']

    markersize = 1
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    block_leak_a50_ge_change = block_leak_a50_ge[1:] - block_leak_a50_ge[0:-1]
    block_leak_a50_ecml_change = block_leak_a50_ecml[1:] - block_leak_a50_ecml[0:-1]
    block_leak_a50_wcml_change = block_leak_a50_wcml[1:] - block_leak_a50_wcml[0:-1]

    ax0.plot(np.arange(0, len(block_leak_a50_ge_change), 1), block_leak_a50_ge_change, color='royalblue', zorder=1, label='Glasgow to Edinburgh via Falkirk')
    ax1.plot(np.arange(0, len(block_leak_a50_ecml_change), 1), block_leak_a50_ecml_change, color='royalblue', zorder=1, label='East Coast Main Line')
    ax2.plot(np.arange(0, len(block_leak_a50_wcml_change), 1), block_leak_a50_wcml_change, color='royalblue', zorder=1, label='West Coast Main Line')

    ax0.set_ylim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax0.set_xlim(-5, len(block_leak_a50_ge_change)+4)
    ax1.set_xlim(-5, len(block_leak_a50_ecml_change)+4)
    ax2.set_xlim(-5, len(block_leak_a50_wcml_change)+4)

    ax0.axhline(0, zorder=-1, linestyle='-', color='black', linewidth=1)
    ax1.axhline(0, zorder=-1, linestyle='-', color='black', linewidth=1)
    ax2.axhline(0, zorder=-1, linestyle='-', color='black', linewidth=1)

    ax2.set_xlabel('Track circuit transition')
    ax1.set_ylabel('Parallel admittance change (S $\mathregular{km^{-1}}$)')
    ax0.legend(loc='upper center')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')

    plt.savefig(f'plots/block_leak_changes_merged.pdf')
    #plt.show()


def block_leakage_hist_merged():
    block_leak_ge = np.load(f'data/resistivity_data/glasgow_edinburgh_falkirk_block_leakage.npz')
    block_leak_a20_ge = block_leak_ge['a20']
    block_leak_a50_ge = block_leak_ge['a50']
    block_leak_a80_ge = block_leak_ge['a80']
    block_leak_wcml = np.load(f'data/resistivity_data/west_coast_main_line_block_leakage.npz')
    block_leak_a20_wcml = block_leak_wcml['a20']
    block_leak_a50_wcml = block_leak_wcml['a50']
    block_leak_a80_wcml = block_leak_wcml['a80']
    block_leak_ecml = np.load(f'data/resistivity_data/east_coast_main_line_block_leakage.npz')
    block_leak_a20_ecml = block_leak_ecml['a20']
    block_leak_a50_ecml = block_leak_ecml['a50']
    block_leak_a80_ecml = block_leak_ecml['a80']

    markersize = 1
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    bins = 200
    ax0.hist(block_leak_a50_ge, bins=bins, label='Glasgow to Edinburgh via Falkirk')
    ax1.hist(block_leak_a50_ecml, bins=bins, label='East Coast Main Line')
    ax2.hist(block_leak_a50_wcml, bins=bins, label='West Coast Main Line')

    ax0.set_xlim(0, 10)
    ax1.set_xlim(0, 10)
    ax2.set_xlim(0, 10)

    ax2.set_xlabel('Parallel admittance (S $\mathregular{km^{-1}}$)')
    ax1.set_ylabel('Count')
    ax0.legend(loc='upper center')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')

    plt.savefig(f'plots/block_leak_hist_merged.pdf')
    #plt.show()


def block_leakage_cumulative_hist_merged():
    block_leak_ge = np.load(f'data/resistivity_data/glasgow_edinburgh_falkirk_block_leakage.npz')
    block_leak_a20_ge = block_leak_ge['a20']
    block_leak_a50_ge = block_leak_ge['a50']
    block_leak_a80_ge = block_leak_ge['a80']
    block_leak_wcml = np.load(f'data/resistivity_data/west_coast_main_line_block_leakage.npz')
    block_leak_a20_wcml = block_leak_wcml['a20']
    block_leak_a50_wcml = block_leak_wcml['a50']
    block_leak_a80_wcml = block_leak_wcml['a80']
    block_leak_ecml = np.load(f'data/resistivity_data/east_coast_main_line_block_leakage.npz')
    block_leak_a20_ecml = block_leak_ecml['a20']
    block_leak_a50_ecml = block_leak_ecml['a50']
    block_leak_a80_ecml = block_leak_ecml['a80']

    markersize = 1
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, hspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    bins = 200
    ax0.hist(block_leak_a50_ge, bins=bins, cumulative=True, label='Glasgow to Edinburgh via Falkirk')
    ax1.hist(block_leak_a50_ecml, bins=bins, cumulative=True, label='East Coast Main Line')
    ax2.hist(block_leak_a50_wcml, bins=bins, cumulative=True, label='West Coast Main Line')

    ax0.set_xlim(0, 10)
    ax1.set_xlim(0, 10)
    ax2.set_xlim(0, 10)

    ax0.axvline(1.6, linestyle='--', color='black')
    ax1.axvline(1.6, linestyle='--', color='black')
    ax2.axvline(1.6, linestyle='--', color='black')

    ax2.set_xlabel('Parallel admittance (S $\mathregular{km^{-1}}$)')
    ax1.set_ylabel('Cumulative count')
    ax0.legend(loc='upper center')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')

    plt.savefig(f'plots/block_leak_cumulative_hist_merged.pdf')
    #plt.show()


def block_leakage_cumulative_hist_single_merged():
    block_leak_ge = np.load(f'data/resistivity_data/glasgow_edinburgh_falkirk_block_leakage.npz')
    block_leak_a20_ge = block_leak_ge['a20']
    block_leak_a50_ge = block_leak_ge['a50']
    block_leak_a80_ge = block_leak_ge['a80']
    block_leak_wcml = np.load(f'data/resistivity_data/west_coast_main_line_block_leakage.npz')
    block_leak_a20_wcml = block_leak_wcml['a20']
    block_leak_a50_wcml = block_leak_wcml['a50']
    block_leak_a80_wcml = block_leak_wcml['a80']
    block_leak_ecml = np.load(f'data/resistivity_data/east_coast_main_line_block_leakage.npz')
    block_leak_a20_ecml = block_leak_ecml['a20']
    block_leak_a50_ecml = block_leak_ecml['a50']
    block_leak_a80_ecml = block_leak_ecml['a80']

    markersize = 1
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 1, hspace=0.15)
    ax0 = fig.add_subplot(gs[0])

    bins = 200
    ax0.hist(block_leak_a50_wcml, bins=bins, alpha=0.75, cumulative=True, histtype="stepfilled", color='tomato', label='West Coast Main Line')
    ax0.hist(block_leak_a50_ecml, bins=bins, alpha=0.75, cumulative=True, histtype="stepfilled", color='khaki', label='East Coast Main Line')
    ax0.hist(block_leak_a50_ge, bins=bins, alpha=0.75, cumulative=True, histtype="stepfilled", color='cornflowerblue', label='Glasgow to Edinburgh via Falkirk')

    ax0.set_xlim(0, 9)
    ax0.set_ylim(0, 1200)

    ax0.axvline(1.6, linestyle='--', color='black')

    ax0.set_xlabel('Parallel admittance (S $\mathregular{km^{-1}}$)')
    ax0.set_ylabel('Cumulative count')
    ax0.legend(loc='upper center', ncols=2)

    plt.savefig(f'plots/5. block_leak_cumulative_hist_single_merged.pdf')
    #plt.show()


def currents(sec):
    markersize = 2
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, 1, hspace=0.1)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])

    e_values = np.linspace(0, 20, 201)
    e_all = np.load(f'e_all_bearings_20.npy.npz')
    ex = e_all['ex'].flatten()
    ey = e_all['ey'].flatten()

    output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey)
    output_uni = model(section_name=sec, ex_uniform=ex, ey_uniform=ey, y_trac=1.6)
    currents = output['i_relays_a']
    currents_all_e = np.zeros((72, len(block_bearings), 201))
    for i_bearing in range(0, len(block_bearings)):
        currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
    currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)
    currents_uni = output_uni['i_relays_a']
    currents_all_e_uni = np.zeros((72, len(block_bearings), 201))
    for i_bearing in range(0, len(block_bearings)):
        currents_all_e_uni[:, i_bearing, :] = currents_uni[i_bearing, :].reshape(72, 201)
    currents_all_e_uni = currents_all_e_uni.reshape(72, len(block_bearings), 201)

    all_min = np.zeros(len(block_bearings))
    all_max = np.zeros(len(block_bearings))
    all_min_uni = np.zeros(len(block_bearings))
    all_max_uni = np.zeros(len(block_bearings))
    for i in range(0, len(block_bearings)):
        all_min[i] = np.min(currents_all_e[:, i, 50].T)
        all_max[i] = np.max(currents_all_e[:, i, 50].T)
        all_min_uni[i] = np.min(currents_all_e_uni[:, i, 50].T)
        all_max_uni[i] = np.max(currents_all_e_uni[:, i, 50].T)

    linewidth = 1
    ax0.plot(np.arange(0, len(block_bearings), 2), all_min[0::2], '-', color='royalblue', alpha=1, linewidth=linewidth)
    ax0.plot(np.arange(0, len(block_bearings), 2), all_max[0::2], '-', color='royalblue', alpha=1, linewidth=linewidth)
    ax0.plot(np.arange(0, len(block_bearings), 2), all_min_uni[0::2], '-', color='tomato', alpha=1, linewidth=linewidth)
    ax0.plot(np.arange(0, len(block_bearings), 2), all_max_uni[0::2], '-', color='tomato', alpha=1, linewidth=linewidth)
    ax0.plot(np.arange(1, len(block_bearings), 2), all_min[1::2], '-', color='royalblue', alpha=1, linewidth=linewidth)
    ax0.plot(np.arange(1, len(block_bearings), 2), all_max[1::2], '-', color='royalblue', alpha=1, linewidth=linewidth)
    ax0.plot(np.arange(1, len(block_bearings), 2), all_min_uni[1::2], '-', color='tomato', alpha=1, linewidth=linewidth)
    ax0.plot(np.arange(1, len(block_bearings), 2), all_max_uni[1::2], '-', color='tomato', alpha=1, linewidth=linewidth)

    ax0.fill_between(np.arange(0, len(block_bearings), 2), all_min[0::2], all_max[0::2], alpha=0.1, color='royalblue')
    ax0.fill_between(np.arange(0, len(block_bearings), 2), all_min_uni[0::2], all_max_uni[0::2], alpha=0.1, color='tomato')
    ax0.fill_between(np.arange(1, len(block_bearings), 2), all_min[1::2], all_max[1::2], alpha=0.1, color='royalblue')
    ax0.fill_between(np.arange(1, len(block_bearings), 2), all_min_uni[1::2], all_max_uni[1::2], alpha=0.1, color='tomato')

    # if line == 'east_coast_main_line':
    #     ax0.axvline(265, color='black', alpha=0.5)
    #     ax0.axvline(313, color='black', alpha=0.5)
    #     ax0.axvline(459, color='black', alpha=0.5)
    #     ax0.axvline(541, color='black', alpha=0.5)

    plt.savefig(f'plots/currents_filled_{sec}.pdf')
    #plt.show()


def thresholds_rs(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 1, hspace=0.1)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])

    e_values = np.linspace(0, 20, 201)
    e_all = np.load(f'e_all_bearings_20.npy.npz')
    ex = e_all['ex'].flatten()
    ey = e_all['ey'].flatten()
    threshold = 0.055

    output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey)
    currents = output['i_relays_a']
    currents_all_e = np.zeros((72, len(block_bearings), 201))
    for i_bearing in range(0, len(block_bearings)):
        currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
    currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)

    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if (first_misoperation_bearing[i] != np.inf) & (first_misoperation_bearing[i] != 0):
            e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds_realistic)), e_thresholds_realistic, s=markersize, marker='X', edgecolor='black', facecolor='orangered', linewidths=0.5, label='Realistic Leakage', zorder=1)

    e_values = np.linspace(0, 20, 201)
    e_all = np.load(f'e_all_bearings_20.npy.npz')
    ex = e_all['ex'].flatten()
    ey = e_all['ey'].flatten()
    threshold = 0.055

    output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey, y_trac=1.6)
    currents = output['i_relays_a']
    currents_all_e = np.zeros((72, len(block_bearings), 201))
    for i_bearing in range(0, len(block_bearings)):
        currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
    currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)

    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if (first_misoperation_bearing[i] != np.inf) & (first_misoperation_bearing[i] != 0):
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='X', edgecolor='black', facecolor='mistyrose', linewidths=0.5, label='Uniform Leakage (1.6 S $\mathregular{km^{-1}}$)', zorder=-1)
    ax0.set_xlim(-5, len(e_thresholds)+5)
    ax0.legend(loc='upper center')
    ax0.set_xlabel('Track circuit number')
    ax0.set_ylabel('Misoperation E (V $\mathregular{km^{-1}}$)')

    plt.savefig(f'plots/thresholds_rs_{sec}.pdf')
    #plt.show()


def thresholds_ws(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    threshold = 0.081
    axles = np.load(f'data/axle_positions/{sec}_train_end_axles_midpoint_a.npy')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, axle_pos_a=ax)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if (first_misoperation_bearing[i] != np.inf) & (first_misoperation_bearing[i] != 0):
            e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds_realistic)), e_thresholds_realistic, s=markersize, marker='X', edgecolor='black', facecolor='honeydew', linewidths=0.5, label='Realistic Leakage')

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
        if (first_misoperation_bearing[i] != np.inf) & (first_misoperation_bearing[i] != 0):
            e_thresholds[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='X', edgecolor='black', facecolor='limegreen', linewidths=0.5, label='Uniform Leakage (1.6 S/km)')
    ax0.set_xlim(-5, len(e_thresholds) + 5)
    ax0.legend(loc='upper center')
    ax0.set_xlabel('Track circuit number')
    ax0.set_ylabel('Misoperation E (V/km)')

    plt.savefig(f'thresholds_ws_{sec}.pdf')
    #plt.show()


def thresholds_rs_dif(sec):
    markersize = 50
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    e_all = np.load(f'data/bearing_e_fields/e_all_bearings_20.npz')
    ex = e_all['ex'].flatten()
    ey = e_all['ey'].flatten()
    threshold = 0.055

    output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey)
    currents = output['i_relays_a']
    currents_all_e = np.zeros((72, len(block_bearings), 201))
    for i_bearing in range(0, len(block_bearings)):
        currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
    currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)

    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds_realistic)), e_thresholds_realistic, s=markersize, marker='X', edgecolor='black', facecolor='orangered', linewidths=0.5, label='Realistic', zorder=5)

    output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey, y_trac=1.6)
    currents = output['i_relays_a']
    currents_all_e = np.zeros((72, len(block_bearings), 201))
    for i_bearing in range(0, len(block_bearings)):
        currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
    currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)
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

    e_dif = e_thresholds_realistic - e_thresholds

    over = np.shape(np.where(e_dif > 0))
    under = np.shape(np.where(e_dif < 0))

    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='X', edgecolor='black', facecolor='mistyrose', linewidths=0.5, label='Uniform (1.6 S $\mathregular{km^{-1}}$)', zorder=4)

    ax1.scatter(range(0, len(e_thresholds)), e_dif, s=markersize, marker='o', edgecolor='black', facecolor='orangered', linewidths=0.5, zorder=5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, zorder=3)

    ax0.set_xlim(-10, len(e_thresholds)+10)
    ax1.set_xlim(-10, len(e_thresholds)+10)
    ax2.set_xlim(-10, len(e_thresholds)+10)

    #ax2.plot(block_bearings, '.', color='tomato')
    ax2.plot(np.arange(0, len(block_bearings[:-1])) + 0.5, block_bearings[1:] - block_bearings[:-1], '.', color='tomato')


    ax0.legend(loc='upper center', ncol=2)
    ax2.set_xlabel('Track circuit number')
    ax2.set_ylabel('Block orientation\n($^\circ$)', multialignment='center')
    ax0.set_ylabel('Misoperation E\n(V $\mathregular{km^{-1}}$)', multialignment='center')
    ax1.set_ylabel('E difference\n(V $\mathregular{km^{-1}}$)', multialignment='center')
    ax0.set_xticks([])
    ax1.grid(axis='x', color='black', alpha=0.5, zorder=1)
    ax2.grid(axis='x', color='black', alpha=0.5, zorder=1)

    plt.savefig(f'plots/6. thresholds_rs_dif_{sec}.pdf')
    plt.show()


def thresholds_ws_dif(sec):
    markersize = 50
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    threshold = 0.081
    axles = np.load(f'data/axle_positions/{sec}_train_end_axles_midpoint_a.npy')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, axle_pos_a=ax)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    e_thresholds_realistic[np.where(e_thresholds_realistic == 0)] = np.nan
    # Plot results
    ax0.scatter(range(0, len(e_thresholds_realistic)), e_thresholds_realistic, s=markersize, marker='X', edgecolor='black', facecolor='honeydew', linewidths=0.5, label='Realistic', zorder=5)

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
    e_thresholds[np.where(e_thresholds == 0)] = np.nan
    # Plot results
    ax0.scatter(range(0, len(e_thresholds)), e_thresholds, s=markersize, marker='X', edgecolor='black', facecolor='limegreen', linewidths=0.5, label='Uniform (1.6 S $\mathregular{km^{-1}}$)', zorder=4)

    ax1.scatter(range(0, len(e_thresholds)), e_thresholds_realistic-e_thresholds, s=markersize, marker='X', edgecolor='black', facecolor='limegreen', linewidths=0.5, zorder=5)
    ax1.axhline(0, color='black', linestyle='--', zorder=3)

    ax0.set_xlim(0, len(e_thresholds))
    ax1.set_xlim(0, len(e_thresholds))

    ax0.legend(loc='upper center')
    ax1.set_xlabel('Track circuit number')
    ax0.set_ylabel('Misoperation E\n(V $\mathregular{km^{-1}}$)', multialignment='center')
    ax1.set_ylabel('E difference\n(V $\mathregular{km^{-1}}$)', multialignment='center')
    ax0.set_xticks([])

    ax0.grid(zorder=1, color='black', alpha=0.5, axis='x')
    ax1.grid(zorder=1, color='black', alpha=0.5, axis='x')

    plt.savefig(f'plots/10. thresholds_ws_dif_{sec}.pdf')
    #plt.show()


def thresholds_rs_solo(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 1, bottom=0.16)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])

    e_values = np.linspace(0, 20, 201)
    e_all = np.load(f'e_all_bearings_20.npy.npz')
    ex = e_all['ex'].flatten()
    ey = e_all['ey'].flatten()
    threshold = 0.055

    output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey)
    currents = output['i_relays_a']
    currents_all_e = np.zeros((72, len(block_bearings), 201))
    for i_bearing in range(0, len(block_bearings)):
        currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
    currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)

    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds_realistic)), e_thresholds_realistic, s=markersize, marker='X', edgecolor='black', facecolor='orangered', linewidths=0.5, label='Realistic Leakage', zorder=1)
    ax0.set_xlim(-5, len(e_thresholds_realistic)+5)

    ax0.set_ylabel('Misoperation E (V/km)')
    ax0.set_xlabel('Track circuit number')

    plt.savefig(f'plots/thresholds_rs_solo_{sec}.pdf')
    #plt.show()


def thresholds_ws_solo(sec):
    markersize = 50
    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 1, bottom=0.16)
    ax0 = fig.add_subplot(gs[0])

    data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    threshold = 0.081
    axles = np.load(f'data/axle_positions/{sec}_train_end_axles_midpoint_a.npy')

    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for a in range(0, 10):
        ax = np.concatenate(axles[a::10])

        for i in range(0, len(bearings)):
            ex_uni = e_values * np.cos(bearings[i])
            ey_uni = e_values * np.sin(bearings[i])
            output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni, axle_pos_a=ax)
            currents_all_e[i, a::10, :] = output['i_relays_a'][a::10, :]
    misoperations_mask = (currents_all_e > threshold) | (currents_all_e < -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
    e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
    for i in range(0, len(first_misoperation_bearing)):
        if first_misoperation_bearing[i] != np.inf:
            e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
        else:
            pass
    # Plot results
    ax0.scatter(range(0, len(e_thresholds_realistic)), e_thresholds_realistic, s=markersize, marker='X', edgecolor='black', facecolor='limegreen', linewidths=0.5, label='Realistic Leakage')
    ax0.set_xlim(-5, len(e_thresholds_realistic) + 5)

    ax0.set_xlabel('Track circuit number')
    ax0.set_ylabel('Misoperation E (V/km)')

    plt.savefig(f'plots/thresholds_ws_solo_{sec}.pdf')
    #plt.show()


def compare_rs():
    lines = ['glasgow_edinburgh_falkirk', 'east_coast_main_line', 'west_coast_main_line']

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, hspace=0.2)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    axs = [ax0, ax1, ax2]

    for i in range(0, len(lines)):
        sec = lines[i]
        ax = axs[i]

        block_lengths = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')['distances']
        block_leak = np.load(f'data/resistivity_data/{sec}_block_leakage.npz')
        block_leak_a50_total = np.multiply(block_leak['a50'], block_lengths)
        data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
        block_bearings = np.rad2deg(data['bearings'])
        e_values = np.linspace(0, 20, 201)
        e_all = np.load(f'e_all_bearings_20.npy.npz')
        ex = e_all['ex'].flatten()
        ey = e_all['ey'].flatten()
        threshold = 0.055

        output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey)
        currents = output['i_relays_a']
        currents_all_e = np.zeros((72, len(block_bearings), 201))
        for i_bearing in range(0, len(block_bearings)):
            currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
        currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)
        misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
        first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
        e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
        for i in range(0, len(first_misoperation_bearing)):
            if first_misoperation_bearing[i] != np.inf:
                e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
            else:
                pass

        output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey, y_trac=1.6)
        currents = output['i_relays_a']
        currents_all_e = np.zeros((72, len(block_bearings), 201))
        for i_bearing in range(0, len(block_bearings)):
            currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
        currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)
        misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
        first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
        e_thresholds_uni = np.full(len(first_misoperation_bearing), np.nan)
        for i in range(0, len(first_misoperation_bearing)):
            if first_misoperation_bearing[i] != np.inf:
                e_thresholds_uni[i] = e_values[int(first_misoperation_bearing[i])]
            else:
                pass

        e_dif = e_thresholds_realistic - e_thresholds_uni
        leak_dif = block_leak_a50_total - (block_lengths * 1.6)
        rho, p_value = stats.spearmanr(e_dif, leak_dif, nan_policy='omit')
        print(f"Spearman's rho: {rho:.4f}")
        print(f"P-value:        {p_value:.4f}")

        ax.plot(range(0, len(e_thresholds_realistic)), min_max_scale(e_dif), color='tomato', zorder=1, label='Electric field threshold difference')
        ax.plot(np.arange(0, len(block_lengths)), min_max_scale(leak_dif), color='royalblue', zorder=1, label='Total block leakage difference')
        ax.set_xlim(0, len(e_thresholds_realistic))
        ax.set_ylim(-0.1, 1.1)
        textstr = '\n'.join((
            r'$R=%.4f$' % (rho,),
            r'$p=%.4f$' % (p_value,)))
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        ax.text(0.1, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.set_xlabel('Track circuit number')
    ax1.set_ylabel('Normalised value')
    ax0.legend(loc='upper center')

    plt.savefig(f'plots/thresholds_leakage_norm_rs.pdf')
    #plt.show()


def compare_rs_vs():
    lines = ['glasgow_edinburgh_falkirk', 'east_coast_main_line', 'west_coast_main_line']

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, hspace=0.2)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    axs = [ax0, ax1, ax2]
    sections = ['Glasgow to Edinburgh via Falkirk', 'East Coast Main Line', 'West Coast Main Line']

    for j in range(0, len(lines)):
        sec = lines[j]
        ax = axs[j]

        block_lengths = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')['distances']
        block_leak = np.load(f'data/resistivity_data/{sec}_block_leakage.npz')
        block_leak_a50_total = np.multiply(block_leak['a50'], block_lengths)

        data = np.load(f'data/rail_data/{sec}/{sec}_distances_bearings.npz')
        block_bearings = np.rad2deg(data['bearings'])
        e_values = np.linspace(0, 20, 201)
        e_all = np.load(f'data/bearing_e_fields/e_all_bearings_20.npz')
        ex = e_all['ex'].flatten()
        ey = e_all['ey'].flatten()
        threshold = 0.055

        output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey)
        currents = output['i_relays_a']
        currents_all_e = np.zeros((72, len(block_bearings), 201))
        for i_bearing in range(0, len(block_bearings)):
            currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
        currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)
        misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
        first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
        e_thresholds_realistic = np.full(len(first_misoperation_bearing), np.nan)
        for i in range(0, len(first_misoperation_bearing)):
            if first_misoperation_bearing[i] != np.inf:
                e_thresholds_realistic[i] = e_values[int(first_misoperation_bearing[i])]
            else:
                pass

        output = model(section_name=sec, ex_uniform=ex, ey_uniform=ey, y_trac=1.6)
        currents = output['i_relays_a']
        currents_all_e = np.zeros((72, len(block_bearings), 201))
        for i_bearing in range(0, len(block_bearings)):
            currents_all_e[:, i_bearing, :] = currents[i_bearing, :].reshape(72, 201)
        currents_all_e = currents_all_e.reshape(72, len(block_bearings), 201)
        misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
        first_misoperation_bearing = np.min(e_first_misoperation_idx, axis=0)
        e_thresholds_uni = np.full(len(first_misoperation_bearing), np.nan)
        for i in range(0, len(first_misoperation_bearing)):
            if first_misoperation_bearing[i] != np.inf:
                e_thresholds_uni[i] = e_values[int(first_misoperation_bearing[i])]
            else:
                pass

        ax.plot(block_leak_a50_total, e_thresholds_realistic, '.', color='tomato', zorder=1, label=sections[j])

    ax2.set_xlabel('Total parallel admittance (S)')
    ax1.set_ylabel('Misoperation E (V $\mathregular{km^{-1}}$)')
    ax0.legend(loc='upper center')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper center')

    plt.savefig(f'plots/9. thresholds_leakage_norm_vs_rs.pdf')
    #plt.show()


def min_max_scale(series):
    return (series - np.nanmin(series)) / (np.nanmax(series) - np.nanmin(series))


def map():
    img = mpimg.imread('data/resistivity_data/resist.jpg')
    extent_osgb = [-31113.793562412757, 690563.7059384637, 871.6070772095118, 1069747.329620562]

    fig = plt.figure(figsize=(8, 10))
    ax = plt.axes(projection=cartopy.crs.OSGB())
    ax.imshow(img, extent=extent_osgb, transform=cartopy.crs.OSGB(), origin="upper")

    data = np.load(f'data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_block_lons_lats.npz')
    ge_lon = data['lons']
    ge_lat = data['lats']

    data = np.load(f'data/rail_data/east_coast_main_line/east_coast_main_line_block_lons_lats.npz')
    ecml_lon = data['lons']
    ecml_lat = data['lats']

    data = np.load(f'data/rail_data/west_coast_main_line/west_coast_main_line_block_lons_lats.npz')
    wcml_lon = data['lons']
    wcml_lat = data['lats']

    ax.plot(ge_lon, ge_lat, transform=cartopy.crs.PlateCarree(), linewidth=3, color="black", zorder=5)
    ax.plot(ecml_lon, ecml_lat, transform=cartopy.crs.PlateCarree(), linewidth=3, color="black", zorder=5)
    ax.plot(wcml_lon, wcml_lat, transform=cartopy.crs.PlateCarree(), linewidth=3, color="black", zorder=5)
    _ge, = ax.plot(ge_lon, ge_lat, transform=cartopy.crs.PlateCarree(), linestyle=':', color="hotpink", zorder=5)
    _ecml, = ax.plot(ecml_lon, ecml_lat, transform=cartopy.crs.PlateCarree(), linestyle=':', color="gold", zorder=5)
    _wcml, = ax.plot(wcml_lon, wcml_lat, transform=cartopy.crs.PlateCarree(), linestyle=':', color="lime", zorder=5)

    ax.scatter(-4.25, 55.86, transform=cartopy.crs.PlateCarree(), marker='h', s=30, zorder=6, edgecolors='black', facecolors='white')
    ax.scatter(-3.19, 55.95, transform=cartopy.crs.PlateCarree(), marker='h', s=30, zorder=6, edgecolors='black', facecolors='white')
    ax.scatter(-0.13, 51.53, transform=cartopy.crs.PlateCarree(), marker='h', s=30, zorder=6, edgecolors='black', facecolors='white')

    ax.scatter(0, 0, marker='s', s=50, zorder=1, edgecolors='black', facecolors='mediumblue', label='0 - 16')
    ax.scatter(0, 0, marker='s', s=50, zorder=1, edgecolors='black', facecolors='cornflowerblue', label='16 - 32')
    ax.scatter(0, 0, marker='s', s=50, zorder=1, edgecolors='black', facecolors='cyan', label='32 - 64')
    ax.scatter(0, 0, marker='s', s=50, zorder=1, edgecolors='black', facecolors='palegreen', label='64 - 125')
    ax.scatter(0, 0, marker='s', s=50, zorder=1, edgecolors='black', facecolors='yellow', label='125 - 250')
    ax.scatter(0, 0, marker='s', s=50, zorder=1, edgecolors='black', facecolors='orange', label='250 - 500')
    ax.scatter(0, 0, marker='s', s=50, zorder=1, edgecolors='black', facecolors='red', label='> 500')

    ax.plot([492000, 524800], [53100, 171200], color='black')
    ax.text(463100, 39000, 'London', color='black')
    ax.plot([173500, 256600], [461000, 655800], color='black')
    ax.text(128000, 445000, 'Glasgow', color='black')
    ax.plot([400800, 330600], [703900, 680500], color='black')
    ax.text(380000, 710000, 'Edinburgh', color='black')

    legend1 = ax.legend(loc='upper right', title=r'Resitivity Range ($\Omega$m)')
    legend2 = ax.legend([_ge, _ecml, _wcml], ['GEvFL', 'ECML', 'WCML'], loc='center right')
    ax.add_artist(legend1)

    ax.set_xlim(100000, 660000)
    ax.set_ylim(0, 1000000)

    #ax.coastlines(resolution="10m")
    plt.savefig(f'plots/map.pdf')
    plt.show()


# mast_resistivity_merged()
# block_leakage_merged()
# block_leakage_total_merged()
# block_leakage_cumulative_hist_single_merged()
# compare_rs()
# compare_rs_vs()
# map()

# for line in ['glasgow_edinburgh_falkirk', 'east_coast_main_line', 'west_coast_main_line']:
#     currents(line)
#     thresholds_rs(line)
#     thresholds_ws(line)
#     thresholds_rs_dif(line)
#     thresholds_ws_dif(line)
#     thresholds_rs_solo(line)
#     thresholds_ws_solo(line)
