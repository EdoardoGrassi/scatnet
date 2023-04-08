
import colorsys
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from kymatio.scattering2d.filter_bank import filter_bank

from models.scatnet import ScatNet2D


def colorize2(z: npt.NDArray[np.complex128]):
    magni = np.abs(z)
    phase = np.angle(z)

    h = (phase + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + magni ** 0.3)
    s = 0.8

    c = np.vectorize(colorsys.hls_to_rgb)(h,l,s) # --> tuple
    print("shape:", c)
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    print("shape:", c.shape)
    c = c.squeeze().transpose(1, 2, 0)
    return c

def colorize(z: npt.NDArray[np.complex128]):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(colorsys.hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.squeeze().transpose(1, 2, 0) 
    return c


def plot_scaling_functions(filterbank: dict):
    assert filterbank is not None

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # scales = filterbank['phi']['j']
    # fig, axs = plt.subplots(ncols=scales)
    # fig.suptitle(f"Scaling functions for scales $j$")
    # for scale, ax in zip(range(scales), axs):
    #     ft = filterbank['phi'][scale]
    #     fw = np.fft.fftshift(np.fft.fft2(ft))
    #     fw = np.abs(fw)

    #     ax.imshow(fw, cmap="gray_r")
    #     ax.set_title(f"$j$ = {scale}")
    #     ax.xaxis.set_ticks([])
    #     ax.yaxis.set_ticks([])

    scales = filterbank['j']
    fig, axs = plt.subplots(nrows=1, ncols=scales)
    fig.suptitle(f"Scaling functions for scales $j$")
    for scale, ax in zip(range(scales), axs):
        ft = filterbank['levels'][scale]
        fw = np.fft.fftshift(np.fft.fft2(ft))
        fw = np.abs(fw)

        ax.imshow(fw, cmap="gray_r")
        ax.set_title(f"$j$ = {scale}")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    # fig.show()
    fig.savefig('docs/images/scaling.png')


def plot_wavelet_functions(filterbank: list[dict]):
    assert filterbank is not None

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # count J and K over all filters metadata
    scales = len(set([f['j'] for f in filterbank]))
    angles = len(set([f['theta'] for f in filterbank]))
    assert len(filterbank) == (scales * angles)

    fig, axs = plt.subplots(nrows=scales, ncols=angles, sharex=True, sharey=True)
    fig.suptitle(R"Wavelets (magnitude) for each scale $j$ and angle $\theta$")
    for filter in filterbank:
        print("filter[0]", filter['levels'][0].shape)
        scale: int = filter['j']
        angle: int = filter['theta']
        ft = filter['levels'][0]
        fw = np.fft.fftshift(np.fft.fft2(ft))

        ax = axs[scale][angle]
        # ax.imshow(colorize(fw))
        ax.imshow(np.abs(fw), cmap='gray')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_title(f"$j = {scale}$ \n $\\theta={angle}$")

    fig.show()

    fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
    fig.suptitle(R"Wavelets (phase) for each scale $j$ and angle $\theta$")
    for filter in filterbank:
        scale: int = filter['j']
        angle: int = filter['theta']
        ft = filter['levels'][0]
        fw = np.fft.fftshift(np.fft.fft2(ft))

        ax = axs[scale][angle]
        ax.imshow(np.angle(fw), cmap='gray')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_title(f"$j = {scale}$ \n $\\theta={angle}$")

    fig.show()


if __name__ == "__main__":
    model = ScatNet2D(shape=(1, 32, 32), classes=101)
    phis, psis = model.features.load_filters()
    # #print("phis:", phis)
    # print("psis:", "levels=", len(psis[0]['levels']))

    M = 43
    J = 4
    L = 8
    fb: Final = filter_bank(M, M, J=J, L=L)
    # phis, psis = fb['phi'], fb['psi']
    # print("keys:", fb['psi'][0].keys())
    # print("psis:", len(fb['psi']))
    # print("psi.j:", fb['psi'][0]['j'])
    # print("psi.theta:", fb['psi'][0]['theta'])

    plot_scaling_functions(phis)
    plot_wavelet_functions(psis)

    plt.show()
