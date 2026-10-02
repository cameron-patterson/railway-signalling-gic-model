import numpy as np
from matplotlib.gridspec import GridSpec
from models import model_test_track
import matplotlib.pyplot as plt


def block_length():
    block_lengths = np.arange(0.1, 2.05, 0.05)
    e_fields = np.arange(-10, 10.1, 2.5)
    i1 = np.empty(len(block_lengths))
    i2 = np.empty(len(block_lengths))

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    for j in range(0, len(e_fields)):
        for i in range(0, len(block_lengths)):
            axles = np.array([49*0.8 + block_lengths[i] - 0.01, 49*0.8 + block_lengths[i] - 0.01 - 0.0675, 49*0.8 + block_lengths[i]*2 - 0.01, 49*0.8 + block_lengths[i]*2 - 0.01 - 0.0675])
            outputs = model_test_track(block_length=block_lengths[i], axle_pos_a=axles, axle_pos_b=[], ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[j]]), y_trac=1.6, y_sig=0.1)
            ia = outputs['i_relays_a']
            i1[i] = ia[49]
            i2[i] = ia[50]

        ax0.plot(block_lengths, i1, label=f'E = {e_fields[j]} V/km')
        ax1.plot(block_lengths, i2, label=f'E = {e_fields[j]} V/km')

        ax0.legend()
        ax1.legend()

        ax0.axhline(0.081, color='limegreen', linestyle='--')
        ax1.axhline(0.081, color='limegreen', linestyle='--')
        ax0.axhline(-0.081, color='limegreen', linestyle='--')
        ax1.axhline(-0.081, color='limegreen', linestyle='--')

    plt.show()


def e_by_l():
    block_lengths = np.arange(2, 2.05, 0.05)
    e = 10
    eff_length = block_lengths - 0.0675
    eff_v = e * eff_length
    i1 = np.empty(len(block_lengths))
    i2 = np.empty(len(block_lengths))

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    for i in range(0, len(block_lengths)):
        #axles = np.array([49 * 0.8 + block_lengths[i] - 0.01, 49 * 0.8 + block_lengths[i] - 0.01 - 0.0675, 49 * 0.8 + block_lengths[i] * 2 - 0.01, 49 * 0.8 + block_lengths[i] * 2 - 0.01 - 0.0675])
        #axles = np.array([49 * 0.8 + block_lengths[i] - block_lengths[i]/2, 49 * 0.8 + block_lengths[i] - block_lengths[i]/2 - 0.0675])
        axles = np.array([49 * 0.8 + 0.0225, 49 * 0.8 + 0.0225 + 0.0675])
        axles = np.sort(axles)
        outputs = model_test_track(block_length=block_lengths[i], axle_pos_a=axles, axle_pos_b=[], ex_uniform=np.array([0]), ey_uniform=np.array([e]), y_trac=1.6, y_sig=0.1)
        ia = outputs['i_relays_a']
        i1[i] = ia[49]
        i2[i] = ia[50]

    res1 = eff_v / i1
    res2 = eff_v / i2

    #ax0.plot(eff_length, eff_v)
    #ax1.plot(eff_length, eff_v)

    ax0.plot(eff_length, res1)
    ax1.plot(eff_length, res2)

    #ax0.plot(eff_length, i1)
    #ax1.plot(eff_length, i2)

    plt.show()


def signal_rail_leakage():
    leakages = np.arange(0.01, 1, 0.01)
    e_fields = np.arange(-10, 10.1, 2.5)
    i1 = np.empty(len(leakages))
    i2 = np.empty(len(leakages))

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    axles = np.array([(50 * 0.8) - 0.01, (50 * 0.8) - 0.01 - 0.0675,
                      (51 * 0.8) - 0.01, (51 * 0.8) - 0.01 - 0.0675])

    for j in range(0, len(e_fields)):
        for i in range(0, len(leakages)):
            outputs = model_test_track(axle_pos_a=axles, axle_pos_b=[], ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[j]]), y_trac=1.6, y_sig=leakages[i])
            ia = outputs['i_relays_a']
            i1[i] = ia[49]
            i2[i] = ia[50]

        ax0.plot(leakages, i1, label=f'E = {e_fields[j]} V/km')
        ax1.plot(leakages, i2, label=f'E = {e_fields[j]} V/km')

        ax0.legend()
        ax1.legend()

        ax0.axhline(0.055, color='tomato', linestyle='--')
        ax1.axhline(0.055, color='tomato', linestyle='--')
        ax0.axhline(-0.055, color='tomato', linestyle='--')
        ax1.axhline(-0.055, color='tomato', linestyle='--')

    plt.show()


def traction_rail_leakage_all():
    leakages = np.arange(1, 9, 0.1)
    e_fields = np.arange(-10, 10.1, 2.5)
    i1 = np.empty(len(leakages))
    i2 = np.empty(len(leakages))

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    axles = np.array([(50 * 0.8) - 0.01, (50 * 0.8) - 0.01 - 0.0675,
                      (51 * 0.8) - 0.01, (51 * 0.8) - 0.01 - 0.0675])

    for j in range(0, len(e_fields)):
        for i in range(0, len(leakages)):
            outputs = model_test_track(axle_pos_a=axles, axle_pos_b=[], ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[j]]), y_trac=leakages[i], y_sig=0.1)
            ia = outputs['i_relays_a']
            i1[i] = ia[49]
            i2[i] = ia[50]

        ax0.plot(leakages, i1, label=f'E = {e_fields[j]} V/km')
        ax1.plot(leakages, i2, label=f'E = {e_fields[j]} V/km')

        ax0.legend()
        ax1.legend()

        ax0.axhline(0.055, color='tomato', linestyle='--')
        ax1.axhline(0.055, color='tomato', linestyle='--')
        ax0.axhline(-0.055, color='tomato', linestyle='--')
        ax1.axhline(-0.055, color='tomato', linestyle='--')

    plt.show()


def traction_rail_leakage_two():
    leakages = np.arange(1, 9, 0.1)
    e_fields = np.arange(-10, 10.1, 2.5)
    i1 = np.empty(len(leakages))
    i2 = np.empty(len(leakages))

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    axles = np.array([(50 * 0.8) - 0.01, (50 * 0.8) - 0.01 - 0.0675,
                      (51 * 0.8) - 0.01, (51 * 0.8) - 0.01 - 0.0675])

    for j in range(0, len(e_fields)):
        for i in range(0, len(leakages)):
            outputs = model_test_track(axle_pos_a=axles, axle_pos_b=[], ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[j]]), y_trac_individual=leakages[i], y_trac=1.6, y_sig=0.1)
            ia = outputs['i_relays_a']
            i1[i] = ia[49]
            i2[i] = ia[50]

        ax0.plot(leakages, i1, label=f'E = {e_fields[j]} V/km')
        ax1.plot(leakages, i2, label=f'E = {e_fields[j]} V/km')

        ax0.legend()
        ax1.legend()

        ax0.axhline(0.055, color='tomato', linestyle='--')
        ax1.axhline(0.055, color='tomato', linestyle='--')
        ax0.axhline(-0.055, color='tomato', linestyle='--')
        ax1.axhline(-0.055, color='tomato', linestyle='--')

    plt.show()


def rail_impedance():
    impedances = np.array([0.0289, 0.035, 0.25])
    e_fields = np.arange(-10, 10.1, 2.5)
    i1 = np.empty(len(impedances))
    i2 = np.empty(len(impedances))

    plt.rcParams['font.size'] = '12'
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    axles = np.array([(50 * 0.8) - 0.01, (50 * 0.8) - 0.01 - 0.0675,
                      (51 * 0.8) - 0.01, (51 * 0.8) - 0.01 - 0.0675])

    for j in range(0, len(e_fields)):
        for i in range(0, len(impedances)):
            outputs = model_test_track(axle_pos_a=axles, axle_pos_b=[], ex_uniform=np.array([0]), ey_uniform=np.array([e_fields[j]]), y_trac=1.6, y_sig=0.1, z_trac=impedances[i], z_sig=impedances[i])
            ia = outputs['i_relays_a']
            i1[i] = ia[49]
            i2[i] = ia[50]

        ax0.plot(impedances, i1, '.', label=f'E = {e_fields[j]} V/km')
        ax1.plot(impedances, i2, '.', label=f'E = {e_fields[j]} V/km')

        ax0.legend()
        ax1.legend()

        ax0.axhline(0.055, color='tomato', linestyle='--')
        ax1.axhline(0.055, color='tomato', linestyle='--')
        ax0.axhline(-0.055, color='tomato', linestyle='--')
        ax1.axhline(-0.055, color='tomato', linestyle='--')

    plt.show()


#block_length()
#e_by_l()
#signal_rail_leakage()
#traction_rail_leakage_all()
#traction_rail_leakage_two()
#rail_impedance()
