from typing import Final

import torch.utils.data

from curet.data import CLASSES, SAMPLES_PER_CLASS, Curet, curet_meta_table


class CuretSubset(torch.utils.data.Subset):
    def __init__(self, dataset: Curet, max_view_angles: tuple[int, int]) -> None:
        assert all(angle > 0 for angle in max_view_angles)

        view_and_lumi: Final = curet_meta_table()

        # TODO: check correctness of angle range check
        visible: Final = lambda x: abs(x) <= max_view_angles[1]
        samples: Final = [i for i in range(1, SAMPLES_PER_CLASS + 1) \
                            if visible(view_and_lumi[i][1])]
        indices: Final = [s + i * len(CLASSES) for s in samples for i in range(len(CLASSES))]

        super().__init__(dataset, indices)
