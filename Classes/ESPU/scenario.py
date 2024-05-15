from typing import List, Dict
from dataclasses import dataclass
from Classes.Super.scenario import Scenario


@dataclass
class ScenarioESPU(Scenario):
    id: int
    elective_surgery_durations: Dict[str, List[float]]  # per type
    emergency_surgery_arrival_times: Dict[str, List[float]]  # per day
    emergency_surgery_durations: Dict[str, List[float]]  # per type

    #def get_elective_surgery_durations(self, block_type_name):
    #    return self.elective_surgery_durations[block_type_name]

    #def reset_emergencies(self):
    #    for day in self.emergency_surgery_durations:
    #        self.emergency_surgery_arrival_times[day] = list()
    #        self.emergency_surgery_durations[day] = list()

