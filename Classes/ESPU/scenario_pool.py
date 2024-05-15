from Classes.ESPU.scenario import ScenarioESPU
from Classes.ESPU.instance import InstanceESPU
from Classes.Super.scenario_pool import ScenarioPool
from Functions.stats_functions import get_mean, get_inverse_cdf

from typing import List, Set
import random
import numpy as np
import pandas as pd
import copy


class ScenarioPoolESPU(ScenarioPool):
    def __init__(
            self,
            instance: InstanceESPU  # scenario pool is dependent on the instance
    ):
        ScenarioPool.__init__(self, instance)
        self.instance = instance
        self.scenarios: List[ScenarioESPU] = list()

    def get_size(self):
        return len(self.scenarios)

    def generate_scenarios(self, number_of_scenarios: int = 10000, seed: int = 5):
        np.random.seed(seed)
        for i in range(number_of_scenarios):
            elective_surgery_dict = dict()
            emergency_times, emergency_durations = self.draw_emergency_surgeries(self.instance.day_start_time,
                                                                                 self.instance.day_finish_time)
            for block_type_name, block_type in self.instance.block_types.items():
                number_of_elective_surgeries = round(self.instance.instance_size*block_type.percentage_of_surgeries)
                elective_surgery_draws = getattr(np.random, block_type.distribution)(
                    *block_type.parameters, number_of_elective_surgeries)
                elective_surgery_dict[block_type_name] = list(elective_surgery_draws)
            self.scenarios.append(ScenarioESPU(i, copy.deepcopy(elective_surgery_dict), emergency_times, emergency_durations))

    def generate_scenarios_average(self, emergencies: bool = True):
        arrival_time_emergency = get_mean(self.instance.emergency_arrival_distribution, self.instance.emergency_arrival_parameters)
        duration_emergency = self.instance.emergency_duration_mean
        arrivals = {}
        durations = {}
        current_time = self.instance.day_start_time + arrival_time_emergency / 2
        arrival_times = list()
        emergency_durations = list()
        if emergencies:
            while current_time < self.instance.day_finish_time:
                arrival_times.append(current_time)
                emergency_durations.append(duration_emergency)
                current_time += arrival_time_emergency

        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            arrivals[day] = arrival_times.copy()
            durations[day] = emergency_durations.copy()

        elective_surgery_dict = dict()
        for block_type_name, block_type in self.instance.block_types.items():
            number_of_elective_surgeries = round(self.instance.instance_size*block_type.percentage_of_surgeries)
            elective_surgery_dict[block_type_name] = list()
            elective_realization = get_mean(block_type.distribution, block_type.parameters)
            for j in range(number_of_elective_surgeries):
                elective_surgery_dict[block_type_name].append(elective_realization)

        scenario = ScenarioESPU(0, copy.deepcopy(elective_surgery_dict), arrivals, durations)
        self.scenarios.append(scenario)

    def generate_scenarios_descriptive(self, size: int, seed: int):
        np.random.seed(seed)
        random.seed(seed)

        # emergency arrivals, first get upper bound
        day_time = self.instance.day_finish_time - self.instance.day_start_time
        mean = get_mean(self.instance.emergency_arrival_distribution, self.instance.emergency_arrival_parameters)
        expected_number_of_arrivals = 5*day_time/mean
        upper_bound_number_of_arrivals = int(10*expected_number_of_arrivals)

        # emergency arrivals, possible realizations
        possible_arrival_times_per_emergency_per_day = dict()
        possible_durations_per_emergency_per_day = dict()
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            possible_arrival_times_per_emergency: List[List[float]] = list()
            possible_durations_per_emergency = list()
            for emergency in range(upper_bound_number_of_arrivals):
                possible_arrivals_emergency: List[float] = list()
                possible_durations_emergency = list()
                for i in range(size):
                    percentile = (i + 0.5) / size
                    inverse_cdf_arrivals = get_inverse_cdf(self.instance.emergency_arrival_distribution,
                                                           percentile,
                                                           self.instance.emergency_arrival_parameters)
                    inverse_cdf_durations = get_inverse_cdf(self.instance.emergency_duration_distribution,
                                                            percentile,
                                                            self.instance.emergency_duration_parameters)
                    possible_arrivals_emergency.append(inverse_cdf_arrivals)
                    possible_durations_emergency.append(inverse_cdf_durations)
                possible_arrival_times_per_emergency.append(possible_arrivals_emergency)
                possible_durations_per_emergency.append(possible_durations_emergency)
            possible_arrival_times_per_emergency_per_day[day] = possible_arrival_times_per_emergency
            possible_durations_per_emergency_per_day[day] = possible_durations_per_emergency

        # elective surgery durations
        possible_durations_per_elective = dict()
        for block_type_name, block_type in self.instance.block_types.items():
            number_of_elective_surgeries = round(self.instance.instance_size*block_type.percentage_of_surgeries)
            possible_durations_per_elective[block_type_name] = list()
            for j in range(number_of_elective_surgeries):
                possible_durations = list()
                for i in range(size):
                    percentile = (i + 0.5) / size
                    inverse_cdf_value = get_inverse_cdf(block_type.distribution, percentile, block_type.parameters)
                        #eval(f'{block_type.distribution}.ppt')(percentile, *block_type.parameters)
                    possible_durations.append(inverse_cdf_value)
                possible_durations_per_elective[block_type_name].append(possible_durations)

        # create actual scenarios
        for i in range(size):
            # pick elective durations
            elective_surgery_dict = dict()
            for block_type_name, block_type in self.instance.block_types.items():
                number_of_elective_surgeries = round(self.instance.instance_size*block_type.percentage_of_surgeries)
                elective_surgery_dict[block_type_name] = list()
                for j in range(number_of_elective_surgeries):
                    elective_realization = random.choice(possible_durations_per_elective[block_type_name][j])
                    possible_durations_per_elective[block_type_name][j].remove(elective_realization)
                    elective_surgery_dict[block_type_name].append(elective_realization)

            # pick emergency durations and arrivals
            arrivals = {}
            durations = {}
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                time = self.instance.day_start_time
                arrival_times = list()
                emergency_durations = list()
                j = 0
                while time < self.instance.day_finish_time:
                    new_emergency_arrival_time = random.choice(possible_arrival_times_per_emergency_per_day[day][j])
                    new_emergency_duration = random.choice(possible_durations_per_emergency_per_day[day][j])
                    time = time + new_emergency_arrival_time
                    arrival_times.append(time)
                    emergency_durations.append(new_emergency_duration)
                    j += 1

                arrival_times.pop()  # remove the last time as it is outside the day.end_time
                emergency_durations.pop()
                arrivals[day] = arrival_times
                durations[day] = emergency_durations

            scenario = ScenarioESPU(i, copy.deepcopy(elective_surgery_dict), arrivals, durations)
            self.scenarios.append(scenario)

    def save_scenario_pool(self):
        #Save the scenario pool
        result = []
        for scenario in self.scenarios:
            result.append({
                "Scenario": scenario.id,
                "Duration": scenario.emergency_surgery_durations,
                "ArrivalTime": scenario.emergency_surgery_arrival_times,
                "ElectiveDuration": scenario.elective_surgery_durations,
            })
        result = pd.DataFrame(result)
        result.to_csv("testFiles/Pools/ScenarioPool" + str(len(self.scenarios)) + ".csv", 
                        index=False, 
                        sep=";")

    def draw_emergency_surgeries(self, day_start_time, day_finish_time):
        arrivals = {}
        durations = {}      
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            time = day_start_time
            arrival_times = list()
            emergency_durations = list()
            while time < day_finish_time:
                new_emergency_arrival_time = getattr(np.random, self.instance.emergency_arrival_distribution)(
                    *self.instance.emergency_arrival_parameters)
                new_emergency_duration = getattr(np.random, self.instance.emergency_duration_distribution)(
                    *self.instance.emergency_duration_parameters)
                time = time + new_emergency_arrival_time
                arrival_times.append(time)
                emergency_durations.append(new_emergency_duration)

            arrival_times.pop()  # remove the last time as it is outside the day.end_time
            emergency_durations.pop()
            arrivals[day] = arrival_times
            durations[day] = emergency_durations

        return arrivals, durations


