from Classes.SRCPSP.scenario_pool import ScenarioPool
from Functions.general_functions import save_object

from typing import Dict

import numpy as np


class Tracker:
    def __init__(
            self,
            scenario_pool: ScenarioPool,
            base_solution_len=100000
    ):
        self.main_scenario_pool = scenario_pool  # this pool should never be adjusted, as it hold all scenarios
        self.main_pool_size = len(scenario_pool)

        self.scenario_per_solution_matrix = np.zeros((base_solution_len, len(scenario_pool)))
        self.final_decomposition: (np.array, np.array) = None
        # the indexes of these matrix are the iterations (each iteration has a solution) and the scenarios
        self.solution_ids = list()  # track the solution ids, for the scenarios the ids are equal to the indices
        self.base_solution_len = base_solution_len
        self.iteration_space = base_solution_len

        self.candidates_x = list()      # x-axis, iteration number
        self.candidates_y = list()
        self.currents_x = list()
        self.currents_y = list()
        self.bests_x = list()
        self.bests_y = list()

        self.factorization_time = list()
        self.factorization_time_iteration = list()
        self.temperature = list()
        self.temperature_iteration = list()
        self.probability_thresholds = list()
        self.probability_thresholds_iteration = list()
        self.scenario_distributions = list()
        self.iteration_number = 0
        self.elapsed_time = 0

        self.evaluations_per_iteration = list()
        self.probabilities_positive = list()
        self.probabilities_negative = list()
        self.variances = dict()
        variances = ['lb', 'ub', 'uv', 'mg', 'nk', 'cv', 'nv', 'cf', 'nvi', 'cvi']
        for variance_id in variances:
            self.variances[variance_id] = list()

    def set_objective_matrix(self, scenario_id, iteration, objective):
        if not iteration < self.iteration_space:
            extra_rows = np.zeros((self.base_solution_len, len(self.main_scenario_pool)))
            self.scenario_per_solution_matrix = np.concatenate((self.scenario_per_solution_matrix, extra_rows))
        self.scenario_per_solution_matrix[iteration, scenario_id] = objective

    def get_row(self, index: int):
        return self.scenario_per_solution_matrix[index, :]

    def erase_row(self, index: int):
        self.scenario_per_solution_matrix[index, :] = 0

    def get_train_rows(self, number_of_rows: int, iteration: int):
        return self.scenario_per_solution_matrix[(iteration - number_of_rows):iteration, :]

    def set_objective(self, scenario_id, solution_id, objective):
        self.solution_ids.append(solution_id)
        self.scenario_per_solution_matrix[int(solution_id), scenario_id] = objective

    def set_objective_on_location(self, scenario_id, index, objective):
        # this one actually sets it on the index you give to it
        self.scenario_per_solution_matrix[index, scenario_id] = objective

    def get_number_of_evaluations_per_iteration(self, iteration):
        return (self.scenario_per_solution_matrix > 0).sum(axis=1)[:iteration]

    def get_number_of_evaluations(self):
        return (self.scenario_per_solution_matrix > 0).sum()

    def get_current_scenario_per_solution_matrix(self, last_evaluated_scenario, iteration: int,
                                                 number_of_solutions=100000):
        solution_start_index = max(0, iteration + 1 - number_of_solutions)
        return self.scenario_per_solution_matrix[solution_start_index:iteration+1, 0:last_evaluated_scenario]

    def add_current_objective(self, objective: float, iteration: int):
        self.currents_y.append(objective)
        self.currents_x.append(iteration)

    def add_candidate_objective(self, objective: float, iteration: int):
        self.candidates_y.append(objective)
        self.candidates_x.append(iteration)

    def add_best_objective(self, objective: float, iteration: int):
        self.bests_y.append(objective)
        self.bests_x.append(iteration)

    def add_variance(self, value: float, id: str):
        self.variances[id].append(value)

    def add_factorization_time(self, time: float, iteration: int):
        self.factorization_time.append(time)
        self.factorization_time_iteration.append(iteration)

    def get_variance_count(self, id: str):
        return len(self.variances[id])

    def add_evaluations(self, number_of_evaluations: int):
        self.evaluations_per_iteration.append(number_of_evaluations)

    def add_probabilities(self, postive_probability: float, negative_probability: float):
        self.probabilities_positive.append(postive_probability)
        self.probabilities_negative.append(negative_probability)

    def add_final_decomposition(self, solution_values: np.array, scenario_distribution: np.array):
        self.final_decomposition = (solution_values, scenario_distribution)

    def add_temperature(self, temperature: float, iteration: int):
        self.temperature.append(temperature)
        self.temperature_iteration.append(iteration)

    def add_elapsed_time(self, elapsed_time: float):
        self.elapsed_time = elapsed_time

    def add_iteration_number(self, iteration_number: int):
        self.iteration_number = iteration_number

    def add_probability_thresholds(self, positive: float, negative: float, iteration: int):
        self.probability_thresholds.append((positive, negative))
        self.probability_thresholds_iteration.append(iteration)

    def add_scenario_distribution(self, scenario_distribution: np.array):
        self.scenario_distributions.append(scenario_distribution.copy())
        
    def return_results(self, basic: bool = False) -> Dict:
        results = dict()
        results['objective'] = self.currents_y[-1]
        results['number accepted'] = len(self.currents_y)
        results['number of evaluations'] = self.get_number_of_evaluations()
        results['number of iterations'] = self.iteration_number
        results['elapsed time'] = self.elapsed_time
        if not basic:
            results['candidates_x'] = self.candidates_x
            results['candidates_y'] = self.candidates_y
            results['currents_x'] = self.currents_x
            results['currents_y'] = self.currents_y
            results['bests_x'] = self.bests_x
            results['bests_y'] = self.bests_y
            results['variances'] = self.variances
            results['evaluations per iteration'] = self.evaluations_per_iteration
            results['probabilities positive'] = self.probabilities_positive
            results['probabilities negative'] = self.probabilities_negative
            column_sum = self.scenario_per_solution_matrix.sum(axis=1)
            last_non_empty = np.argmax(column_sum <= 0)
            results['matrix'] = self.scenario_per_solution_matrix[:last_non_empty, :]
            results['final decomposition'] = self.final_decomposition
            results['factorization time'] = self.factorization_time
            results['factorization time iteration'] = self.factorization_time_iteration
            results['temperature'] = self.temperature
            results['temperature iteration'] = self.temperature_iteration
            results['probability thresholds'] = self.probability_thresholds
            results['probability thresholds iteration'] = self.probability_thresholds_iteration
            results['scenario distributions'] = self.scenario_distributions

        return results

    def save_results(self, directory, file_name, basic: bool = False) -> None:
        results = self.return_results(basic)
        save_object(results, directory, file_name)



        



