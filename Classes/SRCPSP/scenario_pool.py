import numpy as np
import random
from typing import List

from Classes.SRCPSP.instance import InstanceSRCPSP
from Classes.SRCPSP.scenario import ScenarioSRCPSP
from Classes.Super.scenario_pool import ScenarioPool

from Functions.stats_functions import draw_from_distribution, get_inverse_cdf


class ScenarioPoolSRCPSP(ScenarioPool):
    def __init__(
            self,
            instance: InstanceSRCPSP
    ):
        ScenarioPool.__init__(self, instance)
        self.instance = instance
        self.scenarios: List[ScenarioSRCPSP] = list()

    def generate_scenarios(self, size: int, distribution_rule, seed: int):
        np.random.seed(seed)
        distribution = distribution_rule.__name__[:-1]

        for i in range(size):
            duration_per_activity = dict()
            for activity in self.instance.get_activities():
                mean = activity.duration
                if mean == 0:
                    duration_per_activity[activity.id] = 0
                else:
                    duration_per_activity[activity.id] = draw_from_distribution(distribution, distribution_rule(mean))
            scenario = ScenarioSRCPSP(str(i), duration_per_activity)
            self.scenarios.append(scenario)

    def generate_scenarios_average(self, dummy: bool):
        duration_per_activity = dict()
        for activity in self.instance.get_activities():
            duration_per_activity[activity.id] = activity.duration
        scenario = ScenarioSRCPSP('0', duration_per_activity)
        self.scenarios.append(scenario)

    def generate_scenarios_descriptive(self, size: int, distribution_rule, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        distribution = distribution_rule.__name__[:-1]
        possible_durations_per_activity = list()
        for activity in self.instance.get_activities():
            mean = activity.duration
            distribution_parameters = distribution_rule(mean)
            possible_durations = list()
            for i in range(size):
                if mean == 0:
                    possible_durations.append(0)
                else:
                    percentile = (i + 0.5) / size
                    inverse_cdf_value = get_inverse_cdf(distribution, percentile, parameters=distribution_parameters)
                    possible_durations.append(inverse_cdf_value)

            possible_durations_per_activity.append(possible_durations)

        for i in range(size):
            duration_per_activity = dict()
            for j, activity in enumerate(self.instance.get_activities()):
                activity_realization = random.choice(possible_durations_per_activity[j])
                possible_durations_per_activity[j].remove(activity_realization)
                duration_per_activity[activity.id] = activity_realization
            scenario = ScenarioSRCPSP(str(i), duration_per_activity)
            self.scenarios.append(scenario)





