
import colorsys
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank


def colorize(z):
    magni = np.abs(z)
    phase = np.angle(z)

    h = (phase + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + magni ** 0.3)
    s = 0.8

    c = np.vectorize(colorsys.hls_to_rgb)(h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1, 2, 0)
    return c


def plot_scaling_functions(filterbank: dict):
    assert filterbank is not None

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    scales = filterbank['phi']['j']
    fig, axs = plt.subplots(ncols=scales)
    fig.suptitle(f"Scaling functions for scales $j$")
    for scale, ax in zip(range(scales), axs):
        ft = filterbank['phi'][scale]
        fw = np.fft.fftshift(np.fft.fft2(ft))
        fw = np.abs(fw)

        ax.imshow(fw, cmap="gray_r")
        ax.set_title(f"$j$ = {scale}")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    fig.show()


def plot_wavelet_functions(filterbank: dict):
    assert filterbank is not None

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # count J and K over all filters metadata
    scales = len(set([f['j'] for f in filterbank['psi']]))
    angles = len(set([f['theta'] for f in filterbank['psi']]))
    assert len(filterbank['psi']) == (scales * angles)

    fig, axs = plt.subplots(scales, angles, sharex=True, sharey=True)
    fig.suptitle(R"Wavelets (magnitude) for each scale $j$ and angle $\theta$")
    for filter in filterbank['psi']:
        scale = filter['j']
        angle = filter['theta']
        ft = filter[0]
        fw = np.fft.fftshift(np.fft.fft2(ft))

        ax = axs[scale][angle]
        ax.imshow(colorize(fw))
        # ax.imshow(np.abs(fw))
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_title(f"$j = {scale}$ \n $\\theta={angle}$")

    fig.show()

    fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
    fig.suptitle(R"Wavelets (phase) for each scale $j$ and angle $\theta$")
    for filter in filterbank['psi']:
        scale = filter['j']
        angle = filter['theta']
        ft = filter[0]
        fw = np.fft.fftshift(np.fft.fft2(ft))

        ax = axs[scale][angle]
        ax.imshow(np.angle(fw))
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_title(f"$j = {scale}$ \n $\\theta={angle}$")

    fig.show()


if __name__ == "__main__":
    M = 64 #32
    J = 4 #3
    L = 8
    fb: Final = filter_bank(M, M, J, L=L)

    # print("keys:", fb['psi'][0].keys())
    # print("psis:", len(fb["psi"]))
    # print("psi.j:", fb['psi'][0]["j"])
    # print("psi.theta:", fb['psi'][0]["theta"])

    plot_scaling_functions(fb)
    plot_wavelet_functions(fb)

    plt.show()
