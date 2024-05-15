from dataclasses import dataclass
from typing import Dict
from Classes.Super.scenario import Scenario


@dataclass
class ScenarioSRCPSP(Scenario):
    id: str
    duration_per_activity: Dict[str, float]
