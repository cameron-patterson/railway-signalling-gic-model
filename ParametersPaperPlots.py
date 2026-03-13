import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from models import model

section = 'east_coast_main_line'
data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')
block_lengths = data['distances']
block_bearings = np.rad2deg(data['bearings'])
currents_all_e_standard = np.load(f'{section}_currents_all_e_rs_1.6.npy')
bearings = np.deg2rad(np.arange(0, 360, 5))
e_values = np.linspace(0, 20, 201)
# y_tracs = np.array([4.14, 8.29])
y_tracs = np.array([0.265, 0.530, 1.06, 2.46, 4.14, 8.29])
threshold = 0.055


def generate_currents_rs(y_trac):
    currents_all_e = np.full((len(bearings), len(block_bearings), len(e_values)), np.nan)
    for i in range(0, len(bearings)):
        ex_uni = e_values * np.cos(bearings[i])
        ey_uni = e_values * np.sin(bearings[i])

        output = model(section_name=section, ex_uniform=ex_uni, ey_uniform=ey_uni, y_trac=y_trac, electrical_staggering=False)
        currents_all_e[i] = output['i_relays_a']
        print(i)
    np.save(f'{section}_currents_all_e_rs_{y_trac}.npy', currents_all_e)


def block_length_normalised():
    misoperations_mask = currents_all_e_standard < threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_standard.shape[1])]
    threshold_e_field = np.full(currents_all_e_standard.shape[1], np.nan)
    valid_sections = np.isfinite(min_strength_idx)
    threshold_e_field[valid_sections] = e_values[
        min_strength_idx[valid_sections].astype(int)
    ]

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


def block_length_sorted():
    misoperations_mask = currents_all_e_standard < threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_standard.shape[1])]
    min_strength_e_field = np.full(currents_all_e_standard.shape[1], np.nan)
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
    coefficients = np.polyfit(x_range_clean, min_strength_e_field_clean, 3)
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


def block_length_sorted_length_on_x():
    misoperations_mask = currents_all_e_standard < threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_standard.shape[1])]
    min_strength_e_field = np.full(currents_all_e_standard.shape[1], np.nan)
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
    coefficients = np.polyfit(x_range_clean, min_strength_e_field_clean, 3)
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


def block_bearing():
    misoperations_mask = currents_all_e_standard < threshold
    e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
    has_misoperation_value = misoperations_mask.any(axis=2)
    e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
    first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)
    min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e_standard.shape[1])]
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


def traction_rail_leakage():
    min_strength_e_fields = np.full((len(block_bearings), len(y_tracs)), np.nan)
    for i in range(0, len(y_tracs)):
        currents_all_e = np.load(f'{section}_currents_all_e_rs_{y_tracs[i]}.npy')
        misoperations_mask = currents_all_e < threshold
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


def traction_rail_leakage_thresholds_shared_c():
    min_strength_e_fields = np.full((len(block_bearings), len(y_tracs)), np.nan)
    y_trac_blocks = np.full((len(block_bearings), len(y_tracs)), np.nan)
    for i in range(0, len(y_tracs)):
        currents_all_e = np.load(f'{section}_currents_all_e_rs_{y_tracs[i]}.npy')
        misoperations_mask = currents_all_e < threshold
        e_first_misoperation_idx = misoperations_mask.argmax(axis=2)
        has_misoperation_value = misoperations_mask.any(axis=2)
        e_first_misoperation_idx = np.where(has_misoperation_value, e_first_misoperation_idx, np.inf)
        first_misoperation_bearing = np.argmin(e_first_misoperation_idx, axis=0)

        min_strength_idx = e_first_misoperation_idx[first_misoperation_bearing, np.arange(currents_all_e.shape[1])]
        min_strength_e_field = np.full(currents_all_e.shape[1], np.nan)
        valid_sections = np.isfinite(min_strength_idx)
        min_strength_e_field[valid_sections] = e_values[min_strength_idx[valid_sections].astype(int)]
        min_strength_e_fields[:, i] = min_strength_e_field
        y_trac_blocks[:, i] = block_lengths * y_tracs[i]

    z = y_trac_blocks  # color value for each series

    # Shared color limits
    vmin, vmax = z.min(), z.max()

    # Plot results
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, width_ratios=[4, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[:, 1])

    # Plot each series individually with consistent vmin/vmax
    ax0.scatter(range(0, len(block_lengths)), min_strength_e_fields[:, 0], c=z[:, 0], cmap='rainbow_r', vmin=vmin, vmax=vmax, s=50)
    ax1.scatter(range(0, len(block_lengths)), min_strength_e_fields[:, 1], c=z[:, 1], cmap='rainbow_r', vmin=vmin, vmax=vmax, s=50)

    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Minimum Misoperation Electric Field Strength (V/km)')
    ax0.set_title('Multiple Series with Shared Colorbar')

    # Create shared colorbar using ScalarMappable
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='rainbow_r')
    cbar = plt.colorbar(sm, cax=ax_cb)
    cbar.set_label('Total Block Leakage (S)')
    cbar.ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


def traction_rail_leakage_single_threshold():
    min_strength_e_fields = np.full((len(block_bearings), len(y_tracs)), np.nan)
    for i in range(0, len(y_tracs)):
        currents_all_e = np.load(f'{section}_currents_all_e_rs_{y_tracs[i]}.npy')
        misoperations_mask = currents_all_e < threshold
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
    sc = ax0.scatter(range(0, len(block_lengths)), min_values, c=min_y_tracs_block, cmap='rainbow_r')
    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Minimum Misoperation Electric Field Strength (V/km)')
    # Colorbar
    cbar = plt.colorbar(sc, cax=ax1)
    cbar.set_label('Total Block Leakage (S)')
    cbar.ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


#for y in [0.265, 0.530, 1.06, 2.46, 4.14, 8.29]:
    # generate_currents_rs(y)
# block_length_normalised()
# block_length_sorted()
# block_length_sorted_length_on_x()
# block_bearing()
traction_rail_leakage()
# traction_rail_leakage_thresholds_shared_c()
# traction_rail_leakage_single_threshold()
