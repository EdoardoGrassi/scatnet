import itertools
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from kymatio import Scattering2D
from matplotlib import gridspec
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torchvision

from curet.curet import Curet


def plot_1st_order(scattering, fig: plt.Figure, ax: plt.Axes):
    assert all(x is not None for x in (scattering, fig, ax))

    scaler: Final = mpl.colors.Normalize(
        scattering.min(), scattering.max(), clip=True)
    mapper: Final = cm.ScalarMappable(norm=scaler, cmap="gray")
    # Mapper of coefficient amplitude to a grayscale color for visualisation.

    l_offset = int(L - L / 2 - 1)  # follow same ordering as Kymatio for angles
    for row, col in itertools.product(range(window_rows), range(window_columns)):
        ax = fig.add_subplot(gs_order_1[row, col], projection='polar')
        ax.axis('off')
        coefficients = scattering[:, row, col]

        for j, l in itertools.product(range(J), range(L)):
            coeff = coefficients[l + j * L]
            color = mapper.to_rgba(coeff)
            angle = (l_offset - l) * np.pi / L
            radius = 2 ** (-j - 1)
            ax.bar(x=angle,
                   height=radius,
                   width=np.pi / L,
                   bottom=radius,
                   color=color)
            ax.bar(x=angle + np.pi,
                   height=radius,
                   width=np.pi / L,
                   bottom=radius,
                   color=color)


def plot_2nd_order(scattering, fig: plt.Figure, ax: plt.Axes):
    assert all(x is not None for x in (scattering, fig, ax))

    scaler: Final = mpl.colors.Normalize(
        scattering.min(), scattering.max(), clip=True)
    mapper: Final = cm.ScalarMappable(norm=scaler, cmap="gray")
    # Mapper of coefficient amplitude to a grayscale color for visualisation.

    l_offset = int(L - L / 2 - 1)  # follow same ordering as Kymatio for angles
    for row, col in itertools.product(range(window_rows), range(window_columns)):
        ax = fig.add_subplot(gs_order_2[row, col], projection='polar')
        ax.axis('off')
        coefficients = scattering[:, row, col]

        for j1 in range(J - 1):
            for j2 in range(j1 + 1, J):
                for l1, l2 in itertools.product(range(L), range(L)):
                    coeff_index = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                        (j1 * (J - 1) - j1 * (j1 - 1) // 2)
                    # indexing a bit complex which follows the order used by Kymatio to compute
                    # scattering coefficients
                    coeff = coefficients[coeff_index]
                    color = mapper.to_rgba(coeff)
                    # split along angles first-order quadrants in L quadrants, using same ordering
                    # as Kymatio (clockwise) and center (with the 0.5 offset)
                    angle = (l_offset - l1) * np.pi / L + \
                        (L // 2 - l2 - 0.5) * np.pi / (L ** 2)
                    radius = 2 ** (-j1 - 1)
                    # equal split along radius is performed through height variable
                    ax.bar(x=angle,
                           height=radius / 2 ** (J - 2 - j1),
                           width=np.pi / L ** 2,
                           bottom=radius +
                           (radius / 2 ** (J - 2 - j1)) * (J - j2 - 1),
                           color=color)
                    ax.bar(x=angle + np.pi,
                           height=radius / 2 ** (J - 2 - j1),
                           width=np.pi / L ** 2,
                           bottom=radius +
                           (radius / 2 ** (J - 2 - j1)) * (J - j2 - 1),
                           color=color)


if __name__ == "__main__":

    BATCH_SIZE: Final = 2

    dataset: Final = Curet(Path(R"C:\data\curet\data"))
    print("Curet classes:", dataset.classes)

    # img_name = Path.cwd() / "images/digit.png"

    # src_img = Image.open(img_name).convert('L').resize((32, 32))
    # src_img = np.array(src_img)

    loader: Final = DataLoader(dataset, batch_size=BATCH_SIZE)
    images, labels = next(iter(loader))
    print(images.shape)
    print("labels:", labels)

    fig, axs = plt.subplots(ncols=len(images))
    for ax, label, image in zip(axs, labels, images):
        index = dataset.classes[label.item()]
        ax.set_title(label)
        ax.imshow(image.squeeze())

    fig.show()
    plt.pause(0.001)

    grid = torchvision.utils.make_grid(images)

    SHOWCASE_CLASSES = [0, 1, 2, 3]
    indices = [dataset.targets == x for x in SHOWCASE_CLASSES]
    data = Subset(dataset, indices)

    L: Final = 6 # angles
    J: Final = 3 # scales

    images, labels = next(iter(loader))
    for image, label in zip(images, labels):
        print("img shape: ", image.shape)
        image = image.squeeze()
        scattering: Final = Scattering2D(J=J, shape=image.shape,
                                L=L, max_order=2, frontend='numpy')

        image = image.numpy().astype(np.float32) / 255.

        scat_coeffs = scattering(image)
        print("coeffs shape: ", scat_coeffs.shape)
        # Invert colors
        scat_coeffs = -scat_coeffs

        len_order_1 = J*L
        order_1 = scat_coeffs[1:1+len_order_1, :, :]

        len_order_2 = (J*(J-1)//2)*(L**2)
        order_2 = scat_coeffs[1+len_order_1:, :, :]

        # Retrieve spatial size
        window_rows, window_columns = scat_coeffs.shape[1:]
        print("nb of (order 1, order 2) coefficients: ", (len_order_1, len_order_2))

        # Define figure size and grid on which to plot input digit image, first-order and second-order scattering coefficients
        fig = plt.figure(figsize=(47, 15))
        spec = fig.add_gridspec(ncols=3, nrows=1)

        gs = gridspec.GridSpec(1, 3, wspace=0.1)
        gs_order_1 = gridspec.GridSpecFromSubplotSpec(
            window_rows, window_columns, subplot_spec=gs[1])
        gs_order_2 = gridspec.GridSpecFromSubplotSpec(
            window_rows, window_columns, subplot_spec=gs[2])

        # Start by plotting input digit image and invert colors
        ax = plt.subplot(gs[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(255 - image, cmap='gray', interpolation='nearest', aspect='auto')

        ax = plt.subplot(gs[1])
        ax.set_xticks([])
        ax.set_yticks([])
        plot_1st_order(order_1, fig, ax)

        ax = plt.subplot(gs[2])
        ax.set_xticks([])
        ax.set_yticks([])
        plot_2nd_order(order_2, fig, ax)

        plt.show()
