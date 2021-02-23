from chex import dataclass
from typing import Optional


@dataclass
class Likelihood:
    name: Optional[str] = 'Likelihood'


@dataclass
class Gaussian:
    name: Optional[str] = 'Gaussian'
