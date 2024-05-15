from Classes.SRCPSP.simulator import Simulator
from Classes.SRCPSP.solution import Solution
from Classes.SRCPSP.solution_manager import SolutionManager
from Classes.SRCPSP.scenario_pool import ScenarioPool
from Classes.tracker import Tracker
from Classes.SimulatedAnnealing.simulated_annealing import SimulatedAnnealing

from Functions.stats_functions import norm_cdf, get_t_bound, get_t_bounds
from typing import List, Tuple

import numpy as np
import random
import time


class SeqPreSA(SimulatedAnnealing):
    def __init__(
            self,
            budget: int,
            initial_solution: Solution,
            solution_manager: SolutionManager,
            simulator: Simulator,
            scenario_pool: ScenarioPool,
            tracker: Tracker,
            budget_type: str = 'iterations',  # 'iterations', 'evaluations', 'time'
            seed: int = 0,
            initial_temperature: float = 100,
            final_temperature: float = 0.1,
            fixed_solution_size: int = 10000,
            scenarios_per_sample_iteration: int = 2,
            use_best: bool = True,
            update_temperature_expected_iterations: bool = False,
            speed_up_factorization: bool = True,
            decision_criterion: str = 'optimal',  # 'optimal', 'probabilities', 'intervals'
            decision_probabilities: Tuple[float, float, float, float] = (0.5, 1, 0.5, 0),
            informative_method: str = 'updates',
            print_updates: bool = True
    ):
        SimulatedAnnealing.__init__(self, budget, initial_solution, solution_manager, simulator,
                                    scenario_pool, budget_type, seed, initial_temperature, final_temperature,
                                    use_best, print_updates)
        self.final_temperature = final_temperature
        self.simulator = simulator
        self.scenario_pool = scenario_pool
        self.solution_manager = solution_manager
        self.tracker: Tracker = tracker
        self.fixed_solution_size = fixed_solution_size
        self.scenarios_per_sample_iteration = scenarios_per_sample_iteration
        self.update_temperature_expected_iterations = update_temperature_expected_iterations
        self.speed_up_factorization = speed_up_factorization
        self.print_updates = print_updates
        self.informative_method = informative_method

        self.end = '\n' if print_updates else '\r'

        # Initialization
        random.seed(seed)
        self.current_solution: Solution = initial_solution
        self.best_solution: Solution = initial_solution
        self.candidate_solution: Solution = initial_solution

        self.current_objective = self.first_evaluation(self.current_solution)
        self.best_objective = self.current_objective
        self.candidate_objective: float = float('inf')
        
        self.current_variance = 0
        self.best_variance = 0
        self.candidate_variance = 0

        self.initial_temperature = initial_temperature  # self.best_objective * (1-1.1)/np.log(0.5)
        self.temperature = self.initial_temperature
        self.current_iteration = 0
        self.solution_index_shift = 0

        self.best_row_from_tracker = 0
        self.solution_distribution: np.array = None
        self.scenario_distribution: np.array = self.tracker.get_current_scenario_per_solution_matrix(
            len(self.scenario_pool), self.current_iteration).reshape(-1)
        self.current_matrix: np.array = None
        self.current_matrix_evaluated: np.array = None  # in the second dimension only scenarios that have an evaluation
        self.current_seen_scenario_ids: np.array = [i for i in range(len(scenario_pool))]
        self.estimated_objectives: np.array = None

        self.current_non_evaluated_scenario_ids = list()
        self.candidate_non_evaluated_scenario_ids = list()

        # Decision criterion
        self.decision_criterion = decision_criterion  # 'optimal', 'probability'
        self.sample_more = False

        self.start_positive_probability = decision_probabilities[0]
        self.end_positive_probability = decision_probabilities[1]
        self.positive_probability = self.start_positive_probability
        self.start_negative_probability = decision_probabilities[2]
        self.end_negative_probability = decision_probabilities[3]
        self.negative_probability = self.start_negative_probability

    def add_tracker(self, tracker: Tracker):
        self.tracker = tracker

    def search(self) -> Solution:
        self.start_time = time.time()
        while self.budget_left():
            factorization_time = 0
            self.current_iteration += 1
            self.solution_index_shift = max(0, self.current_iteration - self.fixed_solution_size)
            self.candidate_solution = self.solution_manager.get_candidate(self.current_solution)
            self.candidate_solution.id = self.current_iteration
            number_of_evaluations = self.evaluate_and_accept()
            self.total_number_of_evaluations += number_of_evaluations
            self.elapsed_time = time.time() - self.start_time
            self.update_temperature()
            if self.informative_method == 'updates':
                self.update_scenario_distribution()
            self.tracker.add_candidate_objective(self.candidate_objective, self.current_iteration)
            self.tracker.add_evaluations(number_of_evaluations)
            if self.print_updates:
                self.print_iteration_results(number_of_evaluations, factorization_time)
            self.tracker.add_scenario_distribution(self.scenario_distribution)

        self.tracker.add_final_decomposition(self.solution_distribution, self.scenario_distribution)
        self.tracker.add_elapsed_time(self.elapsed_time)
        self.tracker.add_iteration_number(self.current_iteration)

        return self.best_solution

    def update_scenario_distribution(self, update_percentage: float = 0.9):
        candidate_evaluations = self.tracker.get_row(self.candidate_solution.id)
        candidate_evaluated_scenarios = candidate_evaluations > 0
        average_candidate_evaluations = candidate_evaluations[candidate_evaluated_scenarios].mean()
        average_scenario_distribution = self.scenario_distribution[candidate_evaluated_scenarios].mean()
        ratio = average_scenario_distribution / average_candidate_evaluations
        self.scenario_distribution[candidate_evaluated_scenarios] = \
            update_percentage * candidate_evaluations[candidate_evaluated_scenarios] * ratio + \
            (1 - update_percentage) * self.scenario_distribution[candidate_evaluated_scenarios]

    def set_current_matrix(self):
        self.current_matrix = self.tracker.get_current_scenario_per_solution_matrix(len(self.scenario_pool),
                                                                                    self.current_iteration-1,
                                                                                    self.fixed_solution_size)
        if self.use_best and (self.best_solution.id < (self.current_iteration-self.fixed_solution_size)):
            best_extra_row = self.tracker.get_current_scenario_per_solution_matrix(len(self.scenario_pool),
                                                                                   self.best_solution.id,
                                                                                   1)
            self.current_matrix = np.vstack([self.current_matrix, best_extra_row])
        if self.current_solution.id < (self.current_iteration-self.fixed_solution_size):
            current_extra_row = self.tracker.get_current_scenario_per_solution_matrix(len(self.scenario_pool),
                                                                                      self.current_solution.id,
                                                                                      1)
            self.current_matrix = np.vstack([self.current_matrix, current_extra_row])

        scenario_evaluated = (self.current_matrix.sum(axis=0) > 0)
        self.current_matrix_evaluated = self.current_matrix[:, scenario_evaluated]
        self.current_seen_scenario_ids = [i for i, x in enumerate(scenario_evaluated) if x]
        if len(self.current_seen_scenario_ids) != self.current_matrix_evaluated.shape[1]:
            print('here')

    def update_objectives(self):
        estimated_matrix = np.matmul(self.solution_distribution.reshape(-1, 1),
                                     self.scenario_distribution[self.scenario_distribution > 0].reshape(1, -1))
        estimated_matrix[self.current_matrix_evaluated > 0] = \
            self.current_matrix_evaluated[self.current_matrix_evaluated > 0]
        objectives = estimated_matrix.mean(axis=1)
        if self.current_matrix.shape[0] > self.fixed_solution_size + 1:  # never happens if self.use_best == False
            self.best_objective = objectives[-2]
            self.current_objective = objectives[-1]
        elif self.current_matrix.shape[0] > self.fixed_solution_size:
            if self.use_best:
                self.best_objective = objectives[-1]
                self.current_objective = objectives[self.current_solution.id-self.solution_index_shift]
            else:
                self.current_objective = objectives[-1]
        else:  # self.current_matrix.shape[0] == self.fixed_solution_size
            if self.use_best:
                self.best_objective = objectives[self.best_solution.id-self.solution_index_shift]
            self.current_objective = objectives[self.current_solution.id-self.solution_index_shift]

    def update_temperature(self):
        if (self.decision_criterion == 'probability') or (self.decision_criterion == 'intervals'):
            self.update_decision_probabilities(update_function='log', base=100)
        else:
            if self.temperature > self.final_temperature:
                if self.update_temperature_expected_iterations:
                    average_evaluations = self.total_number_of_evaluations / self.current_iteration
                    evaluations_left = self.budget - self.total_number_of_evaluations
                    expected_iterations = self.current_iteration + evaluations_left / average_evaluations
                    self.temperature = self.initial_temperature * \
                        np.divide(self.final_temperature, self.initial_temperature) \
                        ** (self.current_iteration/expected_iterations)
                else:
                    self.temperature = self.initial_temperature * \
                        np.divide(self.final_temperature, self.initial_temperature) \
                        ** self.get_budget_ratio()
            self.tracker.add_temperature(self.temperature, self.current_iteration)

    def update_decision_probabilities(self, update_function: str = 'log',  # 'linear', 'log', 'geometric'
                                      base: int = 10):  
        # Note that when not linearly, negative is assumed to be the exact opposite of positive
        if self.positive_probability < self.end_positive_probability:
            if update_function == 'linear':
                self.positive_probability = self.start_positive_probability + self.get_budget_ratio() * \
                                            (self.end_positive_probability - self.start_positive_probability)
            elif update_function == 'log':
                self.positive_probability = self.start_positive_probability + \
                                            (np.log(1+(base-1)*self.get_budget_ratio()) / np.log(base)) * \
                                            (self.end_positive_probability - self.start_positive_probability)
            else:  # update_function == 'geometric'
                self.positive_probability = self.start_positive_probability * \
                                            np.divide(self.end_positive_probability, self.start_positive_probability) \
                                            ** self.get_budget_ratio()
        if self.negative_probability > self.end_negative_probability:
            if update_function == 'linear':
                self.negative_probability = self.start_negative_probability - self.get_budget_ratio() * \
                                            (self.end_negative_probability - self.start_negative_probability)
            elif update_function == 'log':
                inverse_negative_probability = (1 - self.start_negative_probability) + \
                                            (np.log(1+(base-1)*self.get_budget_ratio()) / np.log(base)) * \
                                            (self.start_negative_probability - self.end_negative_probability)
                self.negative_probability = 1 - inverse_negative_probability
            else:  # update_function == 'geometric'
                self.negative_probability = self.start_negative_probability * \
                                            np.divide(self.end_negative_probability, self.start_negative_probability) \
                                            ** self.get_budget_ratio()

        self.tracker.add_probability_thresholds(self.positive_probability, self.negative_probability,
                                                self.current_iteration)

    def accepted(self):
        self.current_solution = self.candidate_solution
        self.current_objective = self.candidate_objective
        self.current_variance = self.candidate_variance
        self.current_non_evaluated_scenario_ids = self.candidate_non_evaluated_scenario_ids
        self.tracker.add_current_objective(self.current_objective, self.current_iteration)

        if self.use_best and (self.current_objective < self.best_objective):
            self.best_solution = self.current_solution
            self.best_objective = self.current_objective
            self.tracker.add_best_objective(self.best_objective, self.current_iteration)
    
    def sample_candidate_or_current(self, loop_times: int) -> bool:  # returns True for candidate, False for current
        if loop_times == 1:
            return True
        elif self.informative_method == 'updates':
            candidate_ratio = len(self.candidate_non_evaluated_scenario_ids) / \
                              (len(self.candidate_non_evaluated_scenario_ids) +
                               len(self.current_non_evaluated_scenario_ids))
        else:
            candidate_ratio = self.candidate_variance / \
                            (self.candidate_variance + self.current_variance)
        random_uniform = np.random.uniform()
        return candidate_ratio > random_uniform
    
    def first_evaluation(self, solution: Solution):
        return self.simulator.evaluate(solution, self.scenario_pool, tracker=self.tracker)

    def sample_random(self, non_evaluated_ids: List[int], scenarios_per_sample_iteration: int):
        scenario_ids = [i for i in range(len(self.scenario_pool))
                        if (i in non_evaluated_ids) and (i in self.current_seen_scenario_ids)]
        new_pool = ScenarioPool(self.scenario_pool.get_instance())
        for i in range(scenarios_per_sample_iteration):
            if len(scenario_ids) > 0:
                random_index = np.random.randint(0, len(scenario_ids))
                scenario_id = scenario_ids.pop(random_index)
                non_evaluated_ids.remove(scenario_id)
                new_pool.add_scenario(self.scenario_pool.get_scenario(scenario_id))
            else:
                break

        return new_pool

    def sample_random_and_unseen(self, non_evaluated_ids: List[int]):
        id_unseen = (self.current_matrix.sum(axis=0) <= 0)
        unseen_ids = [i for i, x in enumerate(id_unseen) if x]
        if self.scenarios_per_sample_iteration == 1:
            new_pool = self.sample_random(non_evaluated_ids, 2)
        else:
            new_pool = self.sample_random(non_evaluated_ids, self.scenarios_per_sample_iteration)
        random_index = np.random.randint(0, len(unseen_ids))
        unseen_scenario_id = unseen_ids[random_index]
        non_evaluated_ids.remove(unseen_scenario_id)
        new_pool.add_scenario(self.scenario_pool.get_scenario(unseen_scenario_id))

        return new_pool

    def sample_informative(self, non_evaluated_ids: List[int], sample_candidate: bool) -> (float, float, List[int]):
        not_evaluated_and_seen = [i for i in range(len(self.scenario_pool))
                              if (i in non_evaluated_ids) and (i in self.current_seen_scenario_ids)]
        evaluated_all_seen = (len(not_evaluated_and_seen) == 0)
        if not evaluated_all_seen:
            if sample_candidate:
                if (self.fixed_solution_size < float('inf')) and \
                        (len(non_evaluated_ids) == len(self.scenario_pool)) and \
                        (self.current_matrix_evaluated.shape[1] < len(self.scenario_pool)):
                    new_pool = self.sample_random_and_unseen(non_evaluated_ids)
                elif (len(non_evaluated_ids) == len(self.scenario_pool)) and (self.scenarios_per_sample_iteration == 1):
                    new_pool = self.sample_random(non_evaluated_ids, 2)
                else:
                    new_pool = self.sample_random(non_evaluated_ids, self.scenarios_per_sample_iteration)
                self.simulator.evaluate(self.candidate_solution, new_pool, tracker=self.tracker)
            else:  # sample_current
                new_pool = self.sample_random(non_evaluated_ids, self.scenarios_per_sample_iteration)
                self.simulator.evaluate(self.current_solution, new_pool, tracker=self.tracker,
                                        solution_index=self.current_solution.id)
            number_of_evaluations = len(new_pool)
        else:
            number_of_evaluations = 0

        index_for_tracker = self.candidate_solution.id if sample_candidate else self.current_solution.id
        objectives_per_scenario = self.tracker.get_row(index_for_tracker)
        if not evaluated_all_seen:
            objectives_exist = objectives_per_scenario > 0
            distribution_exists = self.scenario_distribution > 0
            both_exist = objectives_exist & distribution_exists
            coefficient_estimate = self.get_coefficient_estimate(objectives_per_scenario)
            estimated_objectives = self.scenario_distribution * coefficient_estimate
            estimated_objectives[both_exist] = objectives_per_scenario[both_exist]
            updated_objective = estimated_objectives[distribution_exists].mean()
            estimated_variance = self.get_variance_coefficient(objectives_per_scenario, coefficient_estimate)
        else:
            updated_objective = objectives_per_scenario.mean()
            estimated_variance = 0

        return updated_objective, estimated_variance, non_evaluated_ids, number_of_evaluations

    def sample_informative_updates(self, non_evaluated_ids: List[int], sample_candidate: bool, loop_times: int) -> \
            (float, float, List[int]):
        if len(non_evaluated_ids) > 0:
            if sample_candidate and (len(non_evaluated_ids) == len(self.scenario_pool)) and \
                    (self.scenarios_per_sample_iteration == 1):
                new_pool = self.sample_random(non_evaluated_ids, 2)
                self.simulator.evaluate(self.candidate_solution, new_pool, tracker=self.tracker)
            elif sample_candidate:
                new_pool = self.sample_random(non_evaluated_ids, self.scenarios_per_sample_iteration)
                self.simulator.evaluate(self.candidate_solution, new_pool, tracker=self.tracker)
            else:  # sample_current
                new_pool = self.sample_random(non_evaluated_ids, self.scenarios_per_sample_iteration)
                self.simulator.evaluate(self.current_solution, new_pool, tracker=self.tracker,
                                        solution_index=self.current_solution.id)
            number_of_evaluations = len(new_pool)
        else:
            number_of_evaluations = 0

        current_objectives = self.tracker.get_row(self.current_solution.id)
        candidate_objectives = self.tracker.get_row(self.candidate_solution.id)
        current_exists = current_objectives > 0
        candidate_exists = candidate_objectives > 0
        if sum(current_exists) + sum(candidate_exists) == 2*len(self.scenario_pool):
            self.estimated_objectives = candidate_objectives
            self.current_objective = self.scenario_distribution.mean()
            self.candidate_objective = self.estimated_objectives.mean()
            self.current_variance = 0
            self.candidate_variance = 0
        else:
            if sample_candidate:
                candidate_coefficient_estimate = self.get_coefficient_estimate(candidate_objectives)
                self.estimated_objectives = self.scenario_distribution * candidate_coefficient_estimate
                self.estimated_objectives[candidate_exists] = candidate_objectives[candidate_exists]
                self.candidate_objective = self.estimated_objectives.mean()
                self.candidate_variance = self.get_variance_coefficient(candidate_objectives,
                                                                   candidate_coefficient_estimate)  # here for variance
            if (loop_times == 1) or (not sample_candidate):
                current_coefficient_estimate = self.get_coefficient_estimate(current_objectives)
                self.estimated_objectives = self.scenario_distribution * current_coefficient_estimate
                self.estimated_objectives[current_exists] = current_objectives[current_exists]
                self.current_objective = self.estimated_objectives.mean()
                self.current_variance = self.get_variance_coefficient(current_objectives,
                                                                 current_coefficient_estimate)  # here for variance

        return non_evaluated_ids, number_of_evaluations

    def evaluate_and_accept(self) -> int:
        # Updates: self.candidate_objective, self.candidate_solution variance
        self.candidate_non_evaluated_scenario_ids = self.scenario_pool.get_scenario_ids()
        loop_times, total_evaluations, previous_difference, previous_variance = 0, 0, 0, 0

        while True:
            loop_times += 1
            sample_candidate = self.sample_candidate_or_current(loop_times)
            if sample_candidate:
                self.candidate_non_evaluated_scenario_ids, number_of_evaluations = \
                    self.sample_informative_updates(self.candidate_non_evaluated_scenario_ids, sample_candidate,
                                                    loop_times)
            else:
                self.current_non_evaluated_scenario_ids, number_of_evaluations = \
                    self.sample_informative_updates(self.current_non_evaluated_scenario_ids, sample_candidate,
                                                    loop_times)

            total_evaluations += number_of_evaluations
            if loop_times == 1:
                previous_variance = self.candidate_variance + self.current_variance

            if self.decision_criterion == 'optimal':
                sample_more = self.decide_optimal(previous_difference, previous_variance, loop_times)
                previous_difference = self.candidate_objective - self.current_objective
                previous_variance = self.candidate_variance + self.current_variance
            elif self.decision_criterion == 'probability':
                sample_more = self.decide_probability()
            else:  # self.decision_criterion == 'intervals'
                sample_more = self.decide_intervals_updates()

            if not sample_more:
                return total_evaluations

    def decide_intervals_updates(self) -> bool:
        if self.candidate_variance <= 0:
            if self.candidate_objective < self.current_objective:
                self.is_accepted = True
                self.accepted()
            else:
                self.is_accepted = False
            return False
        else:
            degrees = len(self.scenario_pool) - len(self.candidate_non_evaluated_scenario_ids) - 1
            lower_bound, upper_bound = get_t_bounds(self.candidate_objective, self.candidate_variance, degrees,
                                                    self.positive_probability)
            if self.current_objective > upper_bound:
                self.is_accepted = True
                self.accepted()
                return False
            elif self.current_objective < lower_bound:
                self.is_accepted = False
                return False
            else:
                return True
            
    def decide_intervals(self) -> bool:
        if self.candidate_variance + self.current_variance > 0:
            candidate_has_potential = self.candidate_objective < self.current_objective
            degrees_current = len(self.scenario_pool) - len(self.current_non_evaluated_scenario_ids) - 1
            degrees_candidate = len(self.scenario_pool) - len(self.candidate_non_evaluated_scenario_ids) - 1
            if candidate_has_potential:
                confidence = self.positive_probability
                lb_current = get_t_bound(self.current_objective, self.current_variance, degrees_current, confidence,
                                         upper=False)
                ub_candidate = get_t_bound(self.candidate_objective, self.candidate_variance, degrees_candidate,
                                           confidence, upper=True)
                if self.print_updates:
                    print(f'{candidate_has_potential} current: {lb_current}, '
                          f'candidate: {ub_candidate}', end=self.end)
                if lb_current > ub_candidate:
                    self.is_accepted = True
                    self.accept()
                    return False
            else:
                confidence = (1 - self.negative_probability)
                ub_current = get_t_bound(self.current_objective, self.current_variance,
                                         degrees_current, confidence, upper=True)
                lb_candidate = get_t_bound(self.candidate_objective, self.candidate_variance,
                                           degrees_candidate, confidence, upper=False)
                if self.print_updates:
                    print(f'{candidate_has_potential} current: {ub_current}, '
                          f'candidate: {lb_candidate}', end=self.end)
                if lb_candidate > ub_current:
                    self.is_accepted = False
                    return False
            return True
        else:
            if self.candidate_objective < self.current_objective:
                self.is_accepted = True
                self.accepted()
            else: 
                self.is_accepted = False
            return False

    def decide_optimal(self, previous_difference: float, previous_variance: float, loop_times: int) -> bool:
        adjust_variance = True
        current_difference = self.candidate_objective - self.current_objective
        if adjust_variance:
            degrees_current = len(self.scenario_pool) - len(self.current_non_evaluated_scenario_ids) - 1
            degrees_candidate = len(self.scenario_pool) - len(self.candidate_non_evaluated_scenario_ids) - 1
            if degrees_current > 2:
                current_variance = self.current_variance * degrees_current / (degrees_current - 2)
            else:
                current_variance = self.current_variance * 6
            if degrees_candidate > 2:
                candidate_variance = self.candidate_variance * degrees_candidate / (degrees_candidate - 2)
            else:
                candidate_variance = self.candidate_variance * 6
            current_variance = candidate_variance + current_variance
        else:
            current_variance = self.candidate_variance + self.current_variance
        if current_variance <= 0:
            acceptance_probability = current_difference < 0
            rejection_probability = int(not acceptance_probability)
        else:
            acceptance_probability = min(1,
                                         np.exp(-2 * (current_difference + current_variance / (2 * self.temperature)) *
                                                (previous_difference + current_variance / (2 * self.temperature)) /
                                                current_variance))
            rejection_probability = int(current_difference >= 0)
            if self.print_updates:
                self.print_sampling_results(loop_times, current_difference, acceptance_probability,
                                            rejection_probability, self.end)

        random_uniform_accept = np.random.uniform()
        random_uniform_reject = np.random.uniform()
        if acceptance_probability >= random_uniform_accept:
            self.is_accepted = True
            self.accepted()
            return False
        elif rejection_probability >= random_uniform_reject:
            # Note: The simple rejection rule doesn't work that well when evaluating small sample steps
            self.is_accepted = False
            return False
        else:  # continue
            return True

    def decide_probability(self) -> bool:
        current_difference = self.candidate_objective - self.current_objective
        current_variance = self.candidate_variance + self.current_variance
        if current_variance > 0:
            prob_better = norm_cdf(current_difference, current_variance)
        else:
            prob_better = float(current_difference < 0)
            # Var(X-Y) = Var(X) + Var(Y) - Cov(X, Y), so without Cov it is an upper bound

        if self.print_updates:
            print(f'Probability that better: {prob_better}')
        random_uniform_accept = np.random.uniform()
        if (prob_better >= self.positive_probability) and (prob_better >= random_uniform_accept):
            self.is_accepted = True
            self.accepted()
            return False
        elif prob_better <= self.negative_probability:  # and (prob_better <= random_uniform_reject):
            self.is_accepted = False
            return False
        else:
            return True

    def get_coefficient_estimate(self, objectives_per_scenario):
        both_exist = (objectives_per_scenario > 0) & (self.scenario_distribution > 0)
        coefficient_estimate = np.matmul(self.scenario_distribution[both_exist].T,
                                         objectives_per_scenario[both_exist]) / \
            np.matmul(self.scenario_distribution[both_exist].T, self.scenario_distribution[both_exist])

        return coefficient_estimate

    def get_variance_coefficient(self, objectives_per_scenario: np.array,
                                 coefficient: float):
        certain = objectives_per_scenario > 0
        uncertain = np.invert(certain)
        errors = (coefficient * self.scenario_distribution[certain] - objectives_per_scenario[certain]) ** 2
        error_estimate = errors.sum() / (certain.sum() - 1)
        variance_denominator = (np.matmul(self.scenario_distribution[certain].T, self.scenario_distribution[certain]))
        sum_uncertain = self.scenario_distribution[uncertain].sum() ** 2

        return error_estimate * (uncertain.sum() + sum_uncertain / variance_denominator) / len(
            self.scenario_distribution) ** 2

    def print_iteration_results(self, number_of_evaluations: int, factorization_time: float):
        acceptance = 'accepted' if self.is_accepted else 'rejected'
        best_addition = f'best {round(self.best_objective, 2)}, ' if self.use_best else ''
        print(f'Iteration {self.current_iteration}: '
              f'temp {round(self.temperature, 4)}, '
              f'candidate ({round(self.candidate_objective, 2)}, {round(self.candidate_variance**0.5, 1)}), '
              f'current ({round(self.current_objective, 2)}, {round(self.current_variance**0.5, 1)}), '
              f'{best_addition}'
              f'evaluations {number_of_evaluations}, '
              f'{acceptance}, '
              f'factorization time {round(factorization_time, 2)}', end=self.end)

    @staticmethod
    def print_sampling_results(loop_times: int, difference: float, acceptance_probability: float,
                               rejection_probability: float, end: str):
        print(f'In loop {loop_times}, '
              f'the difference is {round(difference, 0)}, '
              f'acceptance probability: {round(acceptance_probability, 2)}, '
              f'rejection probability: {round(rejection_probability, 2)}', end=end)

