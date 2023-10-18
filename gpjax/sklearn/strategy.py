from dataclasses import dataclass


@dataclass
class AbstractStrategy:
    pass


@dataclass
class ExactInference(AbstractStrategy):
    pass


@dataclass
class VariationalInference(AbstractStrategy):
    pass


@dataclass
class MCMCInference(AbstractStrategy):
    pass
