import numpy as np
import matplotlib.pyplot as plt


def generate_simplified_axle_positions_midpoint(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')

    block_lengths = data['distances']
    block_lengths_sum = np.cumsum(data['distances'])
    block_lengths_sum_midpoints = block_lengths_sum - (block_lengths / 2)
    train_rear_axles_at_midpoint_a = block_lengths_sum_midpoints
    train_front_axles_from_midpoint_a = train_rear_axles_at_midpoint_a + 0.2595
    train_rear_axles_at_midpoint_b = block_lengths_sum_midpoints
    train_front_axles_from_midpoint_b = train_rear_axles_at_midpoint_b - 0.2595

    # Find if any train fronts overlap the end of the line and remove those trains
    over_pos = np.where(train_front_axles_from_midpoint_a > block_lengths_sum[-1])[0]
    if len(over_pos) > 0:
        train_rear_axles_at_midpoint_a = train_rear_axles_at_midpoint_a[:over_pos[0]]
        train_front_axles_from_midpoint_a = train_front_axles_from_midpoint_a[:over_pos[0]]
    under_pos = np.where(train_front_axles_from_midpoint_b < 0)[0]
    if len(under_pos) > 0:
        train_rear_axles_at_midpoint_b = train_rear_axles_at_midpoint_b[int(under_pos[-1] + 1):]
        train_front_axles_from_midpoint_b = train_front_axles_from_midpoint_b[int(under_pos[-1] + 1):]

    # plt.plot(block_lengths_sum, np.full(len(block_lengths_sum), 1), '.')
    # plt.plot(train_rear_axles_at_midpoint_a, np.full(len(train_rear_axles_at_midpoint_a), 1), '>')
    # plt.plot(train_front_axles_from_midpoint_a, np.full(len(train_front_axles_from_midpoint_a), 1), '>')
    # plt.plot(block_lengths_sum, np.full(len(block_lengths_sum), -1), '.')
    # plt.plot(train_rear_axles_at_midpoint_b, np.full(len(train_rear_axles_at_midpoint_b), -1), '<')
    # plt.plot(train_front_axles_from_midpoint_b, np.full(len(train_front_axles_from_midpoint_b), -1), '<')
    # plt.show()

    train_end_axles_midpoint_a = np.full((len(train_rear_axles_at_midpoint_a), 2), np.nan)
    train_end_axles_midpoint_a[:, 0] = train_rear_axles_at_midpoint_a
    train_end_axles_midpoint_a[:, 1] = train_front_axles_from_midpoint_a

    train_end_axles_midpoint_b = np.full((len(train_rear_axles_at_midpoint_b), 2), np.nan)
    train_end_axles_midpoint_b[:, 0] = train_rear_axles_at_midpoint_b
    train_end_axles_midpoint_b[:, 1] = train_front_axles_from_midpoint_b

    np.save(f'{section}_train_end_axles_midpoint_a.npy', train_end_axles_midpoint_a)
    np.save(f'{section}_train_end_axles_midpoint_b.npy', train_end_axles_midpoint_b)


def generate_simplified_axle_positions_exiting(section):
    data = np.load(f'data/rail_data/{section}/{section}_distances_bearings.npz')

    block_lengths = data['distances']
    block_lengths_sum = np.cumsum(data['distances'])
    train_rear_axles_at_exiting_a = block_lengths_sum - 0.01
    train_front_axles_from_exiting_a = train_rear_axles_at_exiting_a + 0.2595
    train_rear_axles_at_exiting_b = block_lengths_sum[:-1] + 0.01
    train_front_axles_from_exiting_b = train_rear_axles_at_exiting_b - 0.2595

    # Find if any train fronts overlap the end of the line and remove those trains
    over_pos = np.where(train_front_axles_from_exiting_a > block_lengths_sum[-1])[0]
    if len(over_pos) > 0:
        train_rear_axles_at_exiting_a = train_rear_axles_at_exiting_a[:over_pos[0]]
        train_front_axles_from_exiting_a = train_front_axles_from_exiting_a[:over_pos[0]]
    under_pos = np.where(train_front_axles_from_exiting_b < 0)[0]
    if len(under_pos) > 0:
        train_rear_axles_at_exiting_b = train_rear_axles_at_exiting_b[int(under_pos[-1] + 1):]
        train_front_axles_from_exiting_b = train_front_axles_from_exiting_b[int(under_pos[-1] + 1):]

    # plt.plot(block_lengths_sum, np.full(len(block_lengths_sum), 1), '.')
    # plt.plot(train_rear_axles_at_exiting_a, np.full(len(train_rear_axles_at_exiting_a), 1), '>')
    # plt.plot(train_front_axles_from_exiting_a, np.full(len(train_front_axles_from_exiting_a), 1), '>')
    # plt.plot(block_lengths_sum, np.full(len(block_lengths_sum), -1), '.')
    # plt.plot(train_rear_axles_at_exiting_b, np.full(len(train_rear_axles_at_exiting_b), -1), '<')
    # plt.plot(train_front_axles_from_exiting_b, np.full(len(train_front_axles_from_exiting_b), -1), '<')
    # plt.show()

    train_end_axles_exiting_a = np.full((len(train_rear_axles_at_exiting_a), 2), np.nan)
    train_end_axles_exiting_a[:, 0] = train_rear_axles_at_exiting_a
    train_end_axles_exiting_a[:, 1] = train_front_axles_from_exiting_a

    train_end_axles_exiting_b = np.full((len(train_rear_axles_at_exiting_b), 2), np.nan)
    train_end_axles_exiting_b[:, 0] = train_rear_axles_at_exiting_b
    train_end_axles_exiting_b[:, 1] = train_front_axles_from_exiting_b

    np.save(f'{section}_train_end_axles_exiting_a.npy', train_end_axles_exiting_a)
    np.save(f'{section}_train_end_axles_exiting_b.npy', train_end_axles_exiting_b)
    pass


for s in ['glasgow_edinburgh_falkirk', 'east_coast_main_line', 'west_coast_main_line']:
    generate_simplified_axle_positions_midpoint(s)
    generate_simplified_axle_positions_exiting(s)
