import numpy as np


def gen_e_fields_for_all_bearings():
    bearings = np.deg2rad(np.arange(0, 360, 5))
    e_values = np.linspace(0, 20, 201)
    ex_uni_all = np.zeros((len(bearings), len(e_values)))
    ey_uni_all = np.zeros((len(bearings), len(e_values)))
    for i in range(0, len(bearings)):
        ex_uni_all[i, :] = e_values * np.cos(bearings[i])
        ey_uni_all[i, :] = e_values * np.sin(bearings[i])
    np.savez(f'e_all_bearings_20.npy', ex=ex_uni_all, ey=ey_uni_all)
    pass


gen_e_fields_for_all_bearings()