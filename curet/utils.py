from typing import Final, Sequence

import torch.utils.data

from curet.data import SAMPLES_PER_CLASS, Curet, curet_meta_table


class CuretViewSubset(torch.utils.data.Subset):
    def __init__(self, dataset: Curet, max_view_angles: tuple[int, int]) -> None:
        assert all(angle > 0 for angle in max_view_angles)

        view_and_lumi: Final = curet_meta_table()

        # TODO: check correctness of angle range check
        visible: Final = lambda x: abs(x) <= max_view_angles[1]
        samples: Final = [i for i in range(1, SAMPLES_PER_CLASS + 1)
                          if visible(view_and_lumi[i][1])]
        nclasses = len(dataset.classes)
        indices: Final = [
            s + i * nclasses for s in samples for i in range(nclasses)]

        super().__init__(dataset, indices)


class CuretClassSubset(torch.utils.data.Subset):
    def __init__(self, dataset: Curet, classes: Sequence[int]) -> None:
        numclasses = len(dataset.classes)
        indices = [s + i * numclasses for s in range(SAMPLES_PER_CLASS)
                   for i in classes]
        super().__init__(dataset, indices)
