from dataclasses import dataclass
from typing import Dict
import numpy as np

from Classes.Super.scenario import Scenario


@dataclass
class ScenarioSVRPTW(Scenario):
    id: str
    demand_per_customer: Dict[str, float]  # customer.id, demand
    travel_time_matrix: np.array  # size customer x customer
    customer_exists: Dict[str, bool]  # customer.id, exists
