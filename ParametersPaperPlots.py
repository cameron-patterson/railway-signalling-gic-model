import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from models import model, test_model


def generate_currents_rs(section, y_trac):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])

        output = model(section_name=section, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=y_trac)
        currents_all_e[i] = output['i_relays_a']
        print(i)
    np.save(f'{section}_currents_all_e_rs_{y_trac}.npy', currents_all_e)


def block_length_normalised(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_lengths = data['distances']
    e_values = np.linspace(0, 20, 201)
    threshold = 0.055

    currents_all_e = np.load(f'{section}_currents_all_e_rs_1.6.npy')
    misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e.shape[1])]
    threshold_e_field = np.full(currents_all_e.shape[1], np.nan)
    valid_sections = np.isfinite(min_strength_idx)
    threshold_e_field[valid_sections] = e_values[min_strength_idx[valid_sections].astype(int)]

    # Plot results
    x_range = np.arange(0, len(block_lengths), 1)
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(x_range, (threshold_e_field - np.nanmin(threshold_e_field)) /
             (np.nanmax(threshold_e_field) - np.nanmin(threshold_e_field)),
             label='Minimum Misoperation Electric Field Strength')
    ax0.plot(x_range, (block_lengths - np.nanmin(block_lengths)) /
             (np.nanmax(block_lengths) - np.nanmin(block_lengths)),
             label='Block Lengths')
    ax0.set_xlim(0, len(block_lengths))
    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Normalised Values')
    ax0.legend(loc='upper right')
    plt.show()


def block_length_sorted(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_lengths = data['distances']
    e_values = np.linspace(0, 20, 201)
    threshold = 0.055

    currents_all_e = np.load(f'{section}_currents_all_e_rs_1.6.npy')
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
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(x_range, min_strength_e_field_sorted, 'x')
    ax0.plot(xp, p(xp), color='green')
    ax0.set_xlim(0, len(block_lengths_sorted))
    ax0.set_xlabel('Blocks Sorted by Increasing Length')
    ax0.set_ylabel('Minimum Misoperation Electric Field Strength (V/km)')
    plt.show()


def block_length_sorted_length_on_x(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_lengths = data['distances']
    e_values = np.linspace(0, 20, 201)
    threshold = 0.055

    currents_all_e = np.load(f'{section}_currents_all_e_rs_1.6.npy')
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
    # Fitting line
    # Clean NaN values
    mask = ~np.isnan(block_lengths_sorted) & ~np.isnan(min_strength_e_field_sorted)
    x_range_clean = block_lengths_sorted[mask]
    min_strength_e_field_clean = min_strength_e_field_sorted[mask]
    coefficients = np.polyfit(x_range_clean, min_strength_e_field_clean, 1)
    p = np.poly1d(coefficients)
    xp = np.linspace(min(block_lengths_sorted), max(block_lengths_sorted), 100)
    # Plotting
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(block_lengths_sorted, min_strength_e_field_sorted, 'x')
    ax0.plot(xp, p(xp), color='green')
    ax0.set_xlim(0, 1)
    ax0.set_xlabel('Block Length (km)')
    ax0.set_ylabel('Minimum Misoperation Electric Field Strength (V/km)')
    plt.show()


def block_bearing(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    bearings = np.deg2rad(np.arange(0, 360, 5))
    threshold = 0.055

    currents_all_e = np.load(f'{section}_currents_all_e_rs_1.6.npy')
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
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(np.arange(0, len(block_bearings)), block_bearings, '.', label='Block Bearing')
    ax0.plot(np.arange(0, len(block_bearings)), misoperation_bearings, 'x', label='Minimum Misoperation Electric Field Bearing')
    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Bearing (degrees)')
    ax0.legend(loc='upper right')
    plt.show()


def traction_rail_leakage(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])
    e_values = np.linspace(0, 20, 201)
    # y_tracs = np.array([0.265, 0.530, 1.06, 1.6, 2.46, 4.14, 8.29])
    y_tracs = np.array([0.265, 8.29])
    threshold = 0.055

    min_strength_e_fields = np.full((len(block_bearings), len(y_tracs)), np.nan)
    for i in range(0, len(y_tracs)):
        currents_all_e = np.load(f'{section}_currents_all_e_rs_{y_tracs[i]}.npy')
        misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
        first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
        min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e.shape[1])]
        min_strength_e_field = np.full(currents_all_e.shape[1], np.nan)
        valid_sections = np.isfinite(min_strength_idx)
        min_strength_e_field[valid_sections] = e_values[min_strength_idx[valid_sections].astype(int)]
        min_strength_e_fields[:, i] = min_strength_e_field

    # Plot results
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    for i in range(0, len(y_tracs)):
        ax0.plot(min_strength_e_fields[:, i], label=f'{y_tracs[i]} S/km')
    ax0.legend()
    plt.show()


def traction_rail_leakage_thresholds_shared_c(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_lengths = data['distances']
    block_bearings = np.rad2deg(data['bearings'])
    e_values = np.linspace(0, 20, 201)
    # y_tracs = np.array([0.265, 0.530, 1.06, 1.6, 2.46, 4.14, 8.29])
    y_tracs = np.array([0.265, 8.29])
    threshold = 0.055

    min_strength_e_fields = np.full((len(block_bearings), len(y_tracs)), np.nan)
    y_trac_blocks = np.full((len(block_bearings), len(y_tracs)), np.nan)

    for i in range(len(y_tracs)):
        currents_all_e = np.load(f'{section}_currents_all_e_rs_{y_tracs[i]}.npy')

        misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)

        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)

        first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

        min_strength_idx = e_first_misoperation_idx[
            first_misoperation_bearing,
            np.arange(currents_all_e.shape[1])
        ]

        min_strength_e_field = np.full(currents_all_e.shape[1], np.nan)
        valid_sections = np.isfinite(min_strength_idx)

        min_strength_e_field[valid_sections] = e_values[
            min_strength_idx[valid_sections].astype(int)
        ]

        min_strength_e_fields[:, i] = min_strength_e_field
        y_trac_blocks[:, i] = block_lengths * y_tracs[i]

    # --- COLOR DATA ---
    z = y_trac_blocks

    # Handle zeros / negatives for log scale
    eps = 1e-6
    z_safe = np.where(z <= 0, eps, z)

    # Shared log normalization
    vmin = np.nanmin(z_safe)
    vmax = np.nanmax(z_safe)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # --- PLOTTING ---
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(7, 2, width_ratios=[4, 1], hspace=0.2)

    axes = [fig.add_subplot(gs[i, 0]) for i in range(7)]
    ax_cb = fig.add_subplot(gs[:, 1])

    for i, ax in enumerate(axes):
        sc = ax.scatter(
            range(len(block_lengths)),
            min_strength_e_fields[:, i],
            c=z_safe[:, i],
            cmap='viridis',  # better than rainbow
            norm=norm,
            s=50
        )

        ax.set_ylim(0, 20)

        if i < 6:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Block Number')

    # Shared ylabel in middle
    axes[3].set_ylabel('Minimum Misoperation Electric Field Strength (V/km)')

    # --- COLORBAR ---
    sm = ScalarMappable(norm=norm, cmap='viridis')
    cbar = plt.colorbar(sm, cax=ax_cb)
    cbar.set_label('Total Block Leakage (S)')
    cbar.ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


def traction_rail_leakage_single_threshold(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_lengths = data['distances']
    block_bearings = np.rad2deg(data['bearings'])
    e_values = np.linspace(0, 20, 201)
    # y_tracs = np.array([0.265, 0.530, 1.06, 1.6, 2.46, 4.14, 8.29])
    y_tracs = np.array([0.265, 8.29])
    threshold = 0.055

    min_strength_e_fields = np.full((len(block_bearings), len(y_tracs)), np.nan)
    for i in range(0, len(y_tracs)):
        currents_all_e = np.load(f'{section}_currents_all_e_rs_{y_tracs[i]}.npy')
        misoperations_mask = (currents_all_e < threshold) & (currents_all_e > -threshold)
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
        first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

        min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e.shape[1])]
        min_strength_e_field = np.full(currents_all_e.shape[1], np.nan)
        valid_sections = np.isfinite(min_strength_idx)
        min_strength_e_field[valid_sections] = e_values[min_strength_idx[valid_sections].astype(int)]
        min_strength_e_fields[:, i] = min_strength_e_field

    n_rows = min_strength_e_fields.shape[0]
    # Initialize result arrays
    min_values = np.full(n_rows, np.nan)
    min_indices = np.full(n_rows, -1)  # sentinel for all-NaN rows
    # Mask for rows that are not all NaN
    valid_rows = ~np.isnan(min_strength_e_fields).all(axis=1)
    # Compute min and argmin only for valid rows
    min_values[valid_rows] = np.nanmin(min_strength_e_fields[valid_rows], axis=1)
    min_indices[valid_rows] = np.nanargmin(min_strength_e_fields[valid_rows], axis=1)
    min_y_tracs = y_tracs[min_indices]
    min_y_tracs[min_indices < 0] = np.nan
    min_y_tracs_block = min_y_tracs * block_lengths

    # Plot results
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[4, 1])
    ax0 = fig.add_subplot(gs[0, 0])  # main plot
    ax1 = fig.add_subplot(gs[0, 1])  # colorbar
    sc = ax0.scatter(
        range(0, len(block_lengths)),
        min_values,
        c=min_y_tracs_block,
        cmap='viridis',
        norm=colors.LogNorm(vmin=np.nanmin(min_y_tracs_block[min_y_tracs_block > 0]),
                            vmax=np.nanmax(min_y_tracs_block))
    )
    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Minimum Misoperation Electric Field Strength (V/km)')
    # Colorbar
    cbar = plt.colorbar(sc, cax=ax1)
    cbar.set_label('Total Block Leakage (S)')
    cbar.ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


def traction_rail_leakage_testing_voltage(section):
    # ey_values = np.linspace(0, 20, 201)
    # ex_values = np.zeros(np.shape(ey_values))

    # ey_values = np.linspace(0, 20, 201)
    # ex_values = np.linspace(0, 20, 201)

    ex_values = np.linspace(0, 20, 201)
    ey_values = np.zeros(np.shape(ex_values))

    # y_tracs = np.array([0.265, 0.530, 1.06, 1.6, 2.46, 4.14, 8.29])
    y_tracs = np.array([0.265, 1.6, 8.29])
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_bearings = np.rad2deg(data['bearings'])

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, top=0.8, left=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['orangered', 'limegreen', 'cornflowerblue']
    for i in range(0, len(y_tracs)):
        # Model
        output = model(section, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["V"]
        node_locs_relay_trac_a = output['node_locs_relay_trac_a']
        ax0.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax0.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])
        # Test
        output = test_model(section, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["V"]
        node_locs_relay_trac_a = output['node_locs_relay_trac_a']
        ax1.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax1.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])
        # Model
        output = model(section, ex_uniform=-ex_values, ey_uniform=-ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["V"]
        node_locs_relay_trac_a = output['node_locs_relay_trac_a']
        ax2.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax2.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])
        # Test
        output = test_model(section, ex_uniform=-ex_values, ey_uniform=-ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["V"]
        node_locs_relay_trac_a = output['node_locs_relay_trac_a']
        ax3.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax3.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])

    fig.supxlabel('Block Number')
    fig.supylabel('Traction Rail Potential (V)')

    legend_elements = [Line2D([0], [0], linestyle='-', label='Traction rail leakage = 0.265 S/km', color='orangered'),
                       Line2D([0], [0], linestyle='-', label='Traction rail leakage = 1.6 S/km', color='limegreen'),
                       Line2D([0], [0], linestyle='-', label='Traction rail leakage = 8.29 S/km', color='cornflowerblue'),
                       Line2D([0], [0], linestyle='-', label='Electric field strength = 20 V/km', color='black'),
                       Line2D([0], [0], linestyle='--', label='Electric field strength = 10 V/km', color='black')]

    fig.legend(handles=legend_elements, fancybox=True, loc='center', bbox_to_anchor=(0.5, 0.9), ncol=2)
    plt.show()


def traction_rail_leakage_testing_current(section):
    # ey_values = np.linspace(0, 20, 201)
    # ex_values = np.zeros(np.shape(ey_values))

    # ey_values = np.linspace(0, 20, 201)
    # ex_values = np.linspace(0, 20, 201)

    ex_values = np.linspace(0, 20, 201)
    ey_values = np.zeros(np.shape(ex_values))

    # y_tracs = np.array([0.265, 0.530, 1.06, 1.6, 2.46, 4.14, 8.29])
    y_tracs = np.array([0.265, 1.6, 8.29])

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, top=0.8, left=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['orangered', 'limegreen', 'cornflowerblue']
    for i in range(0, len(y_tracs)):
        # Model
        output = model(section, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        i_relays = output["i_relays_a"]
        ax0.plot(i_relays[:, 200], color=colors[i])
        ax0.plot(i_relays[:, 100], linestyle='--', color=colors[i])
        # Test
        output = test_model(section, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        i_relays = output["i_relays_a"]
        ax1.plot(i_relays[:, 200], color=colors[i])
        ax1.plot(i_relays[:, 100], linestyle='--', color=colors[i])
        # Model
        output = model(section, ex_uniform=-ex_values, ey_uniform=-ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        i_relays = output["i_relays_a"]
        ax2.plot(i_relays[:, 200], color=colors[i])
        ax2.plot(i_relays[:, 100], linestyle='--', color=colors[i])
        # Test
        output = test_model(section, ex_uniform=-ex_values, ey_uniform=-ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        i_relays = output["i_relays_a"]
        ax3.plot(i_relays[:, 200], color=colors[i])
        ax3.plot(i_relays[:, 100], linestyle='--', color=colors[i])
    # ax0.set_xlim(-1, 914)
    # ax1.set_xlim(-1, 914)
    # ax2.set_xlim(-1, 914)
    # ax3.set_xlim(-1, 914)

    fig.supxlabel('Block Number')
    fig.supylabel('Relay Current (A)')

    legend_elements = [Line2D([0], [0], linestyle='-', label='Traction rail leakage = 0.265 S/km', color='orangered'),
                       Line2D([0], [0], linestyle='-', label='Traction rail leakage = 1.6 S/km', color='limegreen'),
                       Line2D([0], [0], linestyle='-', label='Traction rail leakage = 8.29 S/km', color='cornflowerblue'),
                       Line2D([0], [0], linestyle='-', label='Electric field strength = 20 V/km', color='black'),
                       Line2D([0], [0], linestyle='--', label='Electric field strength = 10 V/km', color='black')]

    fig.legend(handles=legend_elements, fancybox=True, loc='center', bbox_to_anchor=(0.5, 0.9), ncol=2)
    plt.show()


def traction_rail_leakage_testing_current_sources(section):
    # ey_values = np.linspace(0, 20, 201)
    # ex_values = np.zeros(np.shape(ey_values))

    # ey_values = np.linspace(0, 20, 201)
    # ex_values = np.linspace(0, 20, 201)

    ex_values = np.linspace(0, 20, 201)
    ey_values = np.zeros(np.shape(ex_values))

    # y_tracs = np.array([0.265, 0.530, 1.06, 1.6, 2.46, 4.14, 8.29])
    y_tracs = np.array([1.6])
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_bearings = data['bearings']

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, top=0.8, left=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['darkviolet']
    for i in range(0, len(y_tracs)):
        # Model
        output = model(section, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["J"]
        node_locs_relay_trac_a = range(0, len(v[:, 100]))

        ax0.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax0.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])
        # Test
        output = test_model(section, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["J"]
        ax1.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax1.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])
        # Model
        output = model(section, ex_uniform=-ex_values, ey_uniform=-ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["J"]
        ax2.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax2.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])
        # Test
        output = test_model(section, ex_uniform=-ex_values, ey_uniform=-ey_values, y_trac=y_tracs[i], electrical_staggering=False)
        v = output["J"]
        ax3.plot(v[node_locs_relay_trac_a, 200], color=colors[i])
        ax3.plot(v[node_locs_relay_trac_a, 100], linestyle='--', color=colors[i])

    fig.supxlabel('Block Number')
    fig.supylabel('Sum of Traction Rail Current Sources (A)')

    legend_elements = [Line2D([0], [0], linestyle='-', label='Electric field strength = 20 V/km', color='black'),
                       Line2D([0], [0], linestyle='--', label='Electric field strength = 10 V/km', color='black')]

    fig.legend(handles=legend_elements, fancybox=True, loc='center', bbox_to_anchor=(0.5, 0.9), ncol=2)
    plt.show()


def line_bearing_difference(section):
    data = np.load(f'data/rail_data/{section}/{section}_block_lons_lats.npz')
    block_lons = data['lons'][:-1]
    block_lats = data['lats'][:-1]

    bearings = np.deg2rad(np.arange(0, 360, 45))
    e_values = np.linspace(0, 20, 201)

    # Plot
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(4, 8, wspace=0)
    ax0 = fig.add_subplot(gs[1:, 0])
    ax1 = fig.add_subplot(gs[1:, 1])
    ax2 = fig.add_subplot(gs[1:, 2])
    ax3 = fig.add_subplot(gs[1:, 3])
    ax4 = fig.add_subplot(gs[1:, 4])
    ax5 = fig.add_subplot(gs[1:, 5])
    ax6 = fig.add_subplot(gs[1:, 6])
    ax7 = fig.add_subplot(gs[1:, 7])

    axs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])

        output = model(section, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6, electrical_staggering=False)
        ia = output['i_relays_a']

        # Create line segments
        points = np.array([block_lons, block_lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create LineCollection
        norm = LogNorm(vmin=1e-3, vmax=1)
        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        lc.set_array(np.abs(ia[:, 101]))
        lc.set_linewidth(2)

        axs[i].add_collection(lc)
        axs[i].set_xlim(block_lons.min()-0.1, block_lons.max()+0.1)
        axs[i].set_ylim(block_lats.min()-0.1, block_lats.max()+0.1)

        axs[i].set_xticks([])
        axs[i].set_yticks([])

    fig.colorbar(lc, ax=axs[-1])
    plt.show()


def new(section):
    def plot_format(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    data = np.load(f'data/rail_data/{section}/{section}_block_lons_lats.npz')
    block_lons = data['lons'][:-1]
    block_lats = data['lats'][:-1]

    bearings = np.deg2rad(np.arange(0, 360, 45))
    e_values = np.linspace(0, 20, 201)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(9, 4, wspace=0, hspace=-0.5)
    ax1 = fig.add_subplot(gs[0:1, 0])
    ax2 = fig.add_subplot(gs[0:1, 1])
    ax3 = fig.add_subplot(gs[0:1, 2])
    ax4 = fig.add_subplot(gs[0:1, 3])

    ax5 = fig.add_subplot(gs[5:6, 0])
    ax6 = fig.add_subplot(gs[5:6, 1])
    ax7 = fig.add_subplot(gs[5:6, 2])
    ax8 = fig.add_subplot(gs[5:6, 3])

    ax9 = fig.add_subplot(gs[2:3, 0])
    ax10 = fig.add_subplot(gs[2:3, 1])
    ax11 = fig.add_subplot(gs[2:3, 2])
    ax12 = fig.add_subplot(gs[2:3, 3])

    ax13 = fig.add_subplot(gs[7:8, 0])
    ax14 = fig.add_subplot(gs[7:8, 1])
    ax15 = fig.add_subplot(gs[7:8, 2])
    ax16 = fig.add_subplot(gs[7:8, 3])

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16]
    for ax in axs:
        plot_format(ax)

    e_index = 101
    norm = LogNorm(vmin=0.01, vmax=0.3, clip=True)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])
        output = model(section, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=1.6)
        ia = output['i_relays_a']
        ia = np.abs(ia)
        # Create line segments
        points = np.array([block_lons, block_lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create LineCollection
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(np.abs(ia[:, e_index]))
        lc.set_linewidth(2)
        axs[i].add_collection(lc)
        axs[i].set_xlim(block_lons.min()-0.1, block_lons.max()+0.1)
        axs[i].set_ylim(block_lats.min()-0.1, block_lats.max()+0.1)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        # Currents
        axs[i + len(bearings)].scatter(np.arange(0, len(ia[::2, e_index]), 1), ia[::2, e_index], marker='^', s=10)
        axs[i + len(bearings)].scatter(np.arange(0, len(ia[1::2, e_index]), 1), ia[1::2, e_index], marker='v', s=10)
        axs[i + len(bearings)].axhline(0.055, color='orangered', linestyle='--')

    sm = ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, location='right', fraction=0.02, pad=0.02)

    plt.show()


def bearing_combi(section):
    data = np.load(f'data/rail_data/{section}/{section}_block_lons_lats.npz')
    block_lons = data['lons'][:-1]
    block_lats = data['lats'][:-1]
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
    block_lengths = data['distances']
    block_bearings = data['bearings']

    points = [(block_lats[0], block_lons[0])]
    for distance in block_lengths:
        destination = geodesic(kilometers=distance).destination(current, 90)
        current = (destination.latitude, destination.longitude)
        points.append(current)

    # Split for plotting
    lats, lons = zip(*points)

    e_bearings = np.deg2rad(np.arange(0, 360, 45))
    e_values = np.linspace(0, 20, 201)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(4, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])

    ey_values = np.linspace(0, 20, 201)
    ex_values = np.zeros(np.shape(ey_values))

    output = model(section, ex_uniform=ex_values, ey_uniform=ey_values, y_trac=1.6, electrical_staggering=False)
    ia = output['i_relays_a']
    v_matrix = output['V']
    j_matrix = output['J']
    node_locs_relay_trac_a = output['node_locs_relay_trac_a']
    node_locs_relay_sig_a = output['node_locs_relay_sig_a']

    e_index = 101

    ax1.plot(block_lons, block_lats)
    ax2.plot()
    ax1.scatter(block_lons, block_lats, s=3)
    ax3.plot()
    plt.show()


sec = 'glasgow_edinburgh_falkirk'
# y_tracs = np.array([0.265, 0.530, 1.06, 1.6, 2.46, 4.14, 8.29])
# for y in y_tracs:
#     generate_currents_rs(sec, y)
# block_length_normalised(sec)
# block_length_sorted(sec)
# block_length_sorted_length_on_x(sec)
# block_bearing(sec)
# traction_rail_leakage(sec)
# traction_rail_leakage_thresholds_shared_c(sec)
# traction_rail_leakage_single_threshold(sec)
# traction_rail_leakage_testing_voltage(sec)
# traction_rail_leakage_testing_current(sec)
# traction_rail_leakage_testing_current_sources(sec)
# line_bearing_difference(sec)
# new(sec)
bearing_combi(sec)
