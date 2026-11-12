"""State-vector labels and variable slices."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateLayout:
    variable_slices: dict[str, slice]
    labels: tuple[str, ...]

    @classmethod
    def from_mode_counts(cls, mode_counts: dict[str, int]) -> StateLayout:
        if not mode_counts:
            raise ValueError("state layout requires at least one variable")
        slices = {}
        labels = []
        start = 0
        for variable, count in mode_counts.items():
            if not variable or count < 1:
                raise ValueError("state variables need names and positive mode counts")
            slices[variable] = slice(start, start + count)
            labels.extend(f"{variable}:{mode:03d}" for mode in range(count))
            start += count
        return cls(slices, tuple(labels))

    @property
    def size(self) -> int:
        return len(self.labels)

    def slice_for(self, variable: str) -> slice:
        try:
            return self.variable_slices[variable]
        except KeyError as exc:
            raise KeyError(f"variable is absent from state layout: {variable}") from exc
