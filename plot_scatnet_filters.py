import colorsys
from typing import Final

import torchvision
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from kymatio.scattering2d.filter_bank import filter_bank

from models.scatnet import ScatNet


def colorize2(z: npt.NDArray[np.complex128]):
    magni = np.abs(z)
    phase = np.angle(z)

    h = (phase + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + magni ** 0.3)
    s = 0.8

    c = np.vectorize(colorsys.hls_to_rgb)(h, l, s)  # --> tuple
    print("shape:", c)
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    print("shape:", c.shape)
    c = c.squeeze().transpose(1, 2, 0)
    return c


def colorize(z: npt.NDArray[np.complex128]):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(colorsys.hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.squeeze().transpose(1, 2, 0)
    return c


def plot_scaling_functions(fig: plt.Figure, filters: dict):
    assert filters is not None

    scales = filters['j']
    # fig.suptitle(f"Scaling functions for scales $j$")
    axes = fig.subplots(nrows=scales, ncols=1)
    for scale in range(scales):
        ft = filters['levels'][scale]
        fw = np.fft.fftshift(np.fft.fft2(ft))

        ax0 = axes[scale]
        ax0.imshow(fw.real, cmap="gray")
        # ax0.set_title(f"$j$ = {scale}")
        ax0.axis('off')

        # ax1 = axes[scale][1]
        # ax1.imshow(fw.imag, cmap="gray")
        # ax1.set_title(f"$j$ = {scale}")
        # ax1.axis('off')


def plot_wavelet_functions(fig0: plt.Figure, fig1: plt.Figure, filters: list[dict]):
    assert filters is not None

    # count J and K over all filters metadata
    scales = len(set([f['j'] for f in filters]))
    angles = len(set([f['theta'] for f in filters]))
    assert len(filters) == (scales * angles)

    axs0 = fig0.subplots(nrows=scales, ncols=angles)
    # fig0.suptitle(R"Wavelets ($R$) for each scale $j$ and angle $\theta$")

    axs1 = fig1.subplots(nrows=scales, ncols=angles)
    # fig1.suptitle(R"Wavelets ($I$) for each scale $j$ and angle $\theta$")

    for filter in filters:
        # print("filter[0]", filter['levels'][0].shape)
        scale: int = filter['j']
        angle: int = filter['theta']
        ft = filter['levels'][0]
        fw = np.fft.fftshift(np.fft.fft2(ft))

        ax0 = axs0[scale][angle]
        ax0.axis('off')
        ax0.imshow(fw.real, cmap='gray') #, aspect='auto')
        # ax0.set_title(f"$j = {scale}$ \n $\\theta={angle}$")

        ax1 = axs1[scale][angle]
        ax1.axis('off')
        ax1.imshow(fw.imag, cmap='gray') #, aspect='auto')
        # ax1.set_title(f"$j = {scale}$ \n $\\theta={angle}$")


def plot_w_real(fig: plt.Figure, filters: list[dict]):
    assert len(filters) > 0

    real = []
    for filter in filters:
        ft = filter['levels'][0]
        fw = np.fft.fftshift(np.fft.fft2(ft))
        real += fw.real

    grid = torchvision.utils.make_grid(real, normalize=True)
    fig.axis('off')
    fig.imshow(grid.numpy().transpose((1, 2, 0)), cmap='gray')


def plot_w_imag(fig: plt.Figure, filters: list[dict]):
    assert len(filters) > 0

    real = []
    for filter in filters:
        ft = filter['levels'][0]
        fw = np.fft.fftshift(np.fft.fft2(ft))
        real += fw.imag

    grid = torchvision.utils.make_grid(real, normalize=True)
    fig.axis('off')
    fig.imshow(grid.numpy().transpose((1, 2, 0)), cmap='gray')



def main():
    model = ScatNet(shape=(1, 32, 32), classes=10)
    phis, psis = model.features.load_filters()

    M = 32  # 43
    J = 4
    L = 8
    fb: Final = filter_bank(M, M, J=J, L=L)
    phis, psis = fb['phi'], fb['psi']

    mainfig = plt.figure(layout='tight')
    mainfig.tight_layout(pad=0.1)
    subfigs = mainfig.subfigures(ncols=3, width_ratios=[1, L, L])


    plot_scaling_functions(subfigs[0], phis)
    # subfigs[0].subplots_adjust(wspace=0.1, hspace=0.1)
    # fig0.show()
    # fig0.savefig('report/images/scatnet-scaling-filters.png')

    # f = plt.figure()
    plot_wavelet_functions(subfigs[1], subfigs[2], psis)
    # fig1.show()
    # fig1.savefig('report/images/scatnet-wavelet-filters.png')
    subfigs[1].subplots_adjust(wspace=0, hspace=0)
    #f.set_aspect('equal')

    mainfig.savefig('report/images/scatnet-filters.png')
    plt.show()


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    main()
