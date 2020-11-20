from flax import struct


@struct.dataclass
class MeanFunction:
    pass


@struct.dataclass
class Zero(MeanFunction):
    pass
