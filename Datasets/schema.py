from typing import List, Tuple

class DataDimension:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size

class DataSchema:
    def __init__(self, dimensions: List[DataDimension]):
        self.dimensions = dimensions

    def get_shape(self) -> Tuple[int, ...]:
        return tuple(dim.size for dim in self.dimensions)
