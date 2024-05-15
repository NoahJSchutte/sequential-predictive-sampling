from Classes.Super.instance import Instance
from Classes.Super.solution import Solution
from Classes.Super.scenario_pool import ScenarioPool
from Classes.tracker import Tracker


class Simulator:
    def __init__(
            self,
            instance: Instance
    ):
        self.instance = instance

    def evaluate(self, solution: Solution, scenario_pool: ScenarioPool, tracker: Tracker = None,
                 solution_index: int = None) -> float:
        return 0
