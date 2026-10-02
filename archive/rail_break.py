import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from models import model, model_rail_break


def rail_break(sec):
    ex_uni = np.array([0, 5])
    ey_uni = np.array([0, 5])
    output = model_rail_break(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni)
    i_break = output['i_relays_a']
    output = model(section_name=sec, ex_uniform=ex_uni, ey_uniform=ey_uni)
    i_normal = output['i_relays_a']

    plt.plot(i_normal[:, 0], '.', color='blue')
    plt.plot(i_break[:, 0], 'x', color='blue')
    plt.plot(i_normal[:, 1], '.', color='orange')
    plt.plot(i_break[:, 1], 'x', color='orange')
    plt.axhline(0.055, color="red")
    plt.axhline(-0.055, color="red")
    plt.axhline(0.081, color="green")
    plt.axhline(-0.081, color="green")

    plt.show()


rail_break("glasgow_edinburgh_falkirk")
