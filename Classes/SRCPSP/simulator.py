from Classes.Super.simulator import Simulator
from Classes.SRCPSP.activity import Activity
from Classes.SRCPSP.instance import InstanceSRCPSP
from Classes.SRCPSP.scenario_pool import ScenarioPoolSRCPSP
from Classes.SRCPSP.scenario import ScenarioSRCPSP
from Classes.SRCPSP.solution import SolutionSRCPSP
from Classes.tracker import Tracker

from typing import Dict, Set


class SimulatorSRCPSP(Simulator):
    def __init__(
            self,
            instance: InstanceSRCPSP
    ):
        Simulator.__init__(self, instance)
        self.instance = instance
        self.started_activities: Set[str] = set()  # Set[activity.id]
        self.finished_activities: Set[str] = set()  # Set[activity.id]
        self.activity_start_times: Dict[str, float] = dict()
        self.current_activity_index: int = 0  # the index of the last started activity from the solution list
        self.current_time: float = 0

    def reset(self):
        self.started_activities = set()
        self.finished_activities = set()
        self.activity_start_times = dict()
        self.current_time = 0
        self.current_activity_index = 0

    def evaluate(self, solution: SolutionSRCPSP, scenario_pool: ScenarioPoolSRCPSP, tracker: Tracker = None,
                 solution_index: int = None) -> float:
        sum_objectives = 0
        for scenario in scenario_pool.get_scenarios():
            project_duration = self.evaluate_scenario(solution, scenario)
            if tracker:
                if solution_index is None:
                    tracker.set_objective(int(scenario.id), solution.id, project_duration)
                else:
                    tracker.set_objective_on_location(int(scenario.id), solution_index, project_duration)

            sum_objectives += project_duration

        return sum_objectives / len(scenario_pool)

    def evaluate_scenario(self, solution: SolutionSRCPSP, scenario: ScenarioSRCPSP) -> float:
        self.reset()
        project_finished = False
        while not project_finished:
            self.finish_activities(scenario)
            successfully_started_activity = True
            while successfully_started_activity:
                successfully_started_activity = self.start_next_activity(solution)
            project_finished = self.advance_time(scenario)

        return self.current_time

    def get_activity_start_times(self):
        return self.activity_start_times

    def advance_time(self, scenario: ScenarioSRCPSP) -> bool:
        active_activity_ids = self.started_activities.difference(self.finished_activities)
        earliest_finish_time = float('inf')
        for activity_id in active_activity_ids:
            finish_time = self.get_finish_time(activity_id, scenario)
            earliest_finish_time = min(earliest_finish_time, finish_time)

        if earliest_finish_time < float('inf'):
            self.current_time = earliest_finish_time
            return False
        else:
            return True

    def get_finish_time(self, activity_id: str, scenario: ScenarioSRCPSP) -> float:
        return self.activity_start_times[activity_id] + scenario.duration_per_activity[activity_id]

    def finish_activities(self, scenario: ScenarioSRCPSP):
        active_activity_ids = self.started_activities.difference(self.finished_activities)
        for activity_id in active_activity_ids:
            if self.get_finish_time(activity_id, scenario) <= self.current_time:
                self.finished_activities.add(activity_id)

    def start_next_activity(self, solution: SolutionSRCPSP) -> bool:
        if self.current_activity_index < len(solution.activities):
            current_activity = solution.activities[self.current_activity_index]
            if self.enough_resources_available(current_activity):
                if self.predecessors_finished(current_activity):
                    self.started_activities.add(current_activity.id)
                    self.activity_start_times[current_activity.id] = self.current_time
                    self.current_activity_index += 1
                    return True
        return False

    def enough_resources_available(self, activity: Activity) -> bool:
        current_availabilities = self.current_resources_available()
        for resource_id in self.instance.get_resource_ids():
            if current_availabilities[resource_id] < activity.resource_requirement[resource_id]:
                return False

        return True

    def current_resources_available(self) -> Dict[str, int]:
        active_activity_ids = self.started_activities.difference(self.finished_activities)
        resource_availabilities = self.instance.get_resource_availabilities()
        current_availabilities = resource_availabilities.copy()
        for activity_id in active_activity_ids:
            activity = self.instance.get_activity(activity_id)
            for resource_id in self.instance.get_resource_ids():
                current_availabilities[resource_id] -= activity.resource_requirement[resource_id]

        for resource_id in self.instance.get_resource_ids():
            if current_availabilities[resource_id] < 0:
                print('Error: Somehow there are more resources being used than available')

        return current_availabilities

    def project_finished(self) -> bool:
        return len(self.finished_activities) == len(self.instance.get_activities())

    def predecessors_finished(self, activity: Activity) -> bool:
        predecessor_ids = self.instance.get_predecessors(activity)
        for predecessor_id in predecessor_ids:
            if predecessor_id not in self.finished_activities:
                return False

        return True
