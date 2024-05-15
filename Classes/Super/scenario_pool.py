from Classes.Super.instance import Instance
from Classes.Super.scenario import Scenario

from typing import List, Set


class ScenarioPool:
    def __init__(
            self,
            instance: Instance
    ):
        self.instance = instance
        self.scenarios: List[Scenario] = list()

    def __len__(self):
        return len(self.scenarios)

    def get_instance(self):
        return self.instance

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)

    def get_scenario(self, index: int):
        return self.scenarios[index]

    def get_scenarios(self):
        return self.scenarios

    def get_scenario_ids(self) -> List[int]:
        return [int(scenario.id) for scenario in self.scenarios]

    def set_scenarios(self, scenario_list: List[Scenario]):
        self.scenarios = scenario_list

    def get_scenario_ids_set(self) -> Set[int]:
        return set(self.get_scenario_ids())

    def get_scenario_set(self):
        return set(self.scenarios)

    def union(self, other_pool: 'ScenarioPool'):
        this_scenario_set = self.get_scenario_set()
        other_scenario_set = other_pool.get_scenario_set()
        union_set = this_scenario_set.union(other_scenario_set)
        union_scenario_pool = ScenarioPool(self.instance)
        for scenario in union_set:
            union_scenario_pool.add_scenario(scenario)

        return union_scenario_pool

    def difference(self, other_pool: 'ScenarioPool'):
        this_scenario_set = self.get_scenario_set()
        other_scenario_set = other_pool.get_scenario_set()
        difference_set = this_scenario_set.difference(other_scenario_set)
        difference_scenario_pool = ScenarioPool(self.instance)
        for scenario in difference_set:
            difference_scenario_pool.add_scenario(scenario)

        return difference_scenario_pool
