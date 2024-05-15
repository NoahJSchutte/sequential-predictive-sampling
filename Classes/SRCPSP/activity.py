from dataclasses import dataclass
from typing import Dict, Set


@dataclass
class Activity:
    id: str
    resource_requirement: Dict[str, int]
    duration: int
    successor_ids: Set[str]


