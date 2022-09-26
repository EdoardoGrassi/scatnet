import os
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from kymatio.datasets import fetch_fsdd
from kymatio.numpy import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory


def plot_1st_order_filter(phi_f: dict, psi1_f: list):
    fig, ax = plt.subplots()
    ax.set_title(f'First order filters (Q = {Q})')
    ax.set_xlabel(R'$\omega$')
    ax.set_ylabel(R'$\hat\psi_j(\omega)$')

    ax.plot(np.arange(T) / T, phi_f[0], 'r')
    for psi_f in psi1_f:
        ax.plot(np.arange(T) / T, psi_f[0], 'b')

    fig.show()
    fig.savefig('data/images/scaling-filters-1d.png')


def plot_2nd_order_filters(phi_f: dict, psi2_f: list):
    fig, ax = plt.subplots()
    ax.set_title(R'Second-order filters ($Q = 1$)', fontsize=18)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(R'$\omega$', fontsize=18)
    ax.set_ylabel(R'$\hat\psi_j(\omega)$', fontsize=18)

    ax.plot(np.arange(T) / T, phi_f[0], 'r')
    for psi_f in psi2_f:
        ax.plot(np.arange(T) / T, psi_f[0], 'b')

    fig.show()
    fig.savefig('data/images/wavelet-filters-1d.png')


def plot_low_pass(psi1_f: list):
    fig, ax = plt.subplots()
    # ax.update({"text.usetex": True})
    ax.set_title(f'First-order filter - Time domain (Q = {Q})', fontsize=12)
    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel(r'$\psi(t)$', fontsize=18)
    ax.legend(["$\Re{\psi$}_real", "$\Im{\psi$}_imag"])

    psi_t = np.fft.ifft(psi1_f[-1][0])
    psi_t_r = np.real(psi_t)
    psi_t_i = np.imag(psi_t)
    ax.plot(np.concatenate((psi_t_r[-2**8:], psi_t_r[:2**8])), 'b')
    ax.plot(np.concatenate((psi_t_i[-2**8:], psi_t_i[:2**8])), 'r')

    fig.show()
    fig.savefig('data/images/scaling-function-t-1d.png')


if __name__ == '__main__':

    T: Final = 2 ** 13
    J: Final = 5
    Q: Final = 8
    phi_f, psi1_f, psi2_f, _ = scattering_filter_factory(np.log2(T), J, Q)

    with plt.ion():
        plot_1st_order_filter(phi_f, psi1_f)
        plot_2nd_order_filters(phi_f, psi2_f)
        plot_low_pass(psi1_f)

    # example with a sample audio file
    dataset: Final = fetch_fsdd(verbose=True)

    file_path = os.path.join(
        dataset['path_dataset'], sorted(dataset['files'])[0])
    _, x = scipy.io.wavfile.read(file_path)
    x = x / np.max(np.abs(x))


    T: Final = x.shape[-1]
    J: Final = 6
    Q: Final = 16
    scattering = Scattering1D(J=J, shape=T, Q=Q)

    def analyze():
        Sx = scattering(x)

        meta = scattering.meta()
        o0 = np.where(meta['order'] == 0)
        o1 = np.where(meta['order'] == 1)
        o2 = np.where(meta['order'] == 2)

        plt.figure(figsize=(8, 2))
        plt.plot(x)
        plt.title('Original signal')

        plt.figure(figsize=(8, 8))
        plt.subplot(3, 1, 1)
        plt.plot(Sx[o0][0])
        plt.title(R'$0^{th}$-order scattering coefficients')

        plt.subplot(3, 1, 2)
        plt.imshow(Sx[o1], aspect='auto')
        plt.title(R'$1^{st}$-order scattering coefficients')

        plt.subplot(3, 1, 3)
        plt.imshow(Sx[o2], aspect='auto')
        plt.title(R'$2^{nd}$-order scattering coefficients')

        plt.show()

    analyze()