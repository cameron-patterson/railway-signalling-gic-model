import numpy as np
import matplotlib.pyplot as plt
from models import model
from matplotlib.gridspec import GridSpec


def plot_storm(section, storm):
    data = np.load(f'data/storm_e_fields/{storm}/{section}_{storm}_e_blocks.npz')
    data_v2 = np.load(f'data/storm_e_fields/{storm}/{section}_{storm}_e_blocks_v2.npz')
    ex_blocks = data['ex_blocks']/1000
    ey_blocks = data['ey_blocks']/1000
    ex_blocks_v2 = data_v2['ex_blocks']/1000
    ey_blocks_v2 = data_v2['ey_blocks']/1000

    axles = np.load(f'data/axle_positions/at_end/axle_positions_two_track_back_axle_at_end_{section}.npz', allow_pickle=True)
    axles_a = axles['axle_pos_a_all']
    axle_start = []
    for i in range(0, len(axles_a)-1):
        axle_start.append(axles['axle_pos_a_all'][i][0])

    currents = model(section_name=section, axle_pos_a=axle_start, ex_blocks=ex_blocks, ey_blocks=ey_blocks)
    currents_v2 = model(section_name=section, axle_pos_a=axle_start, ex_blocks=ex_blocks_v2, ey_blocks=ey_blocks_v2)

    ia = abs(currents['i_relays_a'])
    ia_v2 = abs(currents_v2['i_relays_a'])

    print(np.nanmax(ia_v2 - ia))
    print(np.where((ia_v2 - ia) == np.nanmax(ia_v2 - ia)))
    print(np.nanmin(ia_v2 - ia))
    print(np.where((ia_v2 - ia) == np.nanmin(ia_v2 - ia)))

    t = 1551

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(19, 8))
    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.plot(ia[t, :], 'x', label='Legacy E-field model', color='tomato')
    ax0.plot(ia_v2[t, :], '.', label='E-field model v2', color='cornflowerblue')
    #ax0.axhline(0.055, color='red')
    ax0.axhline(0.081, color='green')
    #ax0.axvline(534, color='black', alpha=0.2, linestyle='--')
    #ax1.axvline(534, color='black', alpha=0.2, linestyle='--')
    ax1.plot(ia_v2[t, :] - ia[t, :], '.', color='steelblue')
    ax0.set_xlim(0, 936)
    ax0.set_ylim(-0.01, 0.09)
    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Current through relay (A)')
    ax0.legend()
    ax1.set_xlim(0, 936)
    ax1.set_xlabel('Block Number')
    ax1.set_ylabel('Current difference $I_{v2} - I_{legacy}$ (A)')
    fig.suptitle('West Coast Main Line; 11 May 2024 01:51')

    plt.show()
    #plt.savefig('gic_wcml_May2024_ws2.pdf')


plot_storm('west_coast_main_line', 'Mar1989')
