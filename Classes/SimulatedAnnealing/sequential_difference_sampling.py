from Classes.SRCPSP.simulator import Simulator
from Classes.SRCPSP.solution import Solution
from Classes.SRCPSP.solution_manager import SolutionManager
from Classes.SRCPSP.scenario_pool import ScenarioPool
from Classes.tracker import Tracker
from Classes.SimulatedAnnealing.simulated_annealing import SimulatedAnnealing

from Functions.general_functions import save_object
from typing import List

import numpy as np
import time


class SeqDifSA(SimulatedAnnealing):
    def __init__(
            self,
            budget: int,
            initial_solution: Solution,
            solution_manager: SolutionManager,
            simulator: Simulator,
            scenario_pool: ScenarioPool,
            tracker: Tracker = None,
            budget_type: str = 'iterations',
            seed: int = 0,
            initial_temperature: float = 100,
            final_temperature: float = 0.01,
            scenarios_per_sample_iteration: int = 1,
            use_best: bool = True,
            print_updates: bool = True,
            assumed_variance: float = 0,
            use_updating_variance: bool = True
    ):
        SimulatedAnnealing.__init__(self, budget, initial_solution, solution_manager, simulator,
                                    scenario_pool, budget_type, seed, initial_temperature, final_temperature,
                                    use_best, print_updates)
        self.tracker = tracker
        self.scenarios_per_sample_iteration = scenarios_per_sample_iteration
        self.assumed_variance = assumed_variance
        self.use_updating_variance = use_updating_variance
        self.variance = self.assumed_variance
        self.update_variance_counter = 0

    def search(self) -> Solution:
        self.start_time = time.time()
        while self.budget_left():
            self.current_iteration += 1
            self.candidate_solution = self.solution_manager.get_candidate(self.current_solution)
            self.candidate_solution.id = self.current_iteration
            number_of_evaluations = self.evaluate_and_accept()
            self.total_number_of_evaluations += number_of_evaluations
            self.elapsed_time = time.time() - self.start_time
            self.update_temperature()
            self.tracker.add_candidate_objective(self.candidate_objective, self.current_iteration)
            if self.print_updates:
                self.print_iteration_results(number_of_evaluations)

        if self.tracker is not None:
            self.tracker.add_elapsed_time(self.elapsed_time)
            self.tracker.add_iteration_number(self.current_iteration)

        return self.best_solution

    def sample(self, non_evaluated_scenario_ids: List[int]) -> ScenarioPool:
        new_pool = ScenarioPool(self.scenario_pool.get_instance())
        for i in range(self.scenarios_per_sample_iteration):
            if len(non_evaluated_scenario_ids) > 0:
                random_index = np.random.randint(0, len(non_evaluated_scenario_ids))
                scenario_id = non_evaluated_scenario_ids.pop(random_index)
                new_pool.add_scenario(self.scenario_pool.get_scenario(scenario_id))
            else:
                break

        return new_pool

    def update_variance(self, observed_difference: float):
        if self.update_variance_counter == 0:
            self.variance = observed_difference**2
        elif self.update_variance_counter == 1:
            self.variance = (self.variance**0.5 + observed_difference)**2
        else:
            mean_absolute_difference = (self.variance * (self.update_variance_counter - 1))**0.5
            mean_absolute_difference += observed_difference
            self.variance = mean_absolute_difference**2 / self.update_variance_counter
        self.update_variance_counter += 1

    def evaluate_and_accept(self) -> int:
        non_evaluated_scenario_ids = self.scenario_pool.get_scenario_ids()
        loop_times, total_evaluations, previous_difference = 0, 0, 0
        self.tracker.erase_row(self.current_solution.id)

        while True:
            loop_times += 1
            if len(non_evaluated_scenario_ids) == 0:
                current_difference = self.candidate_objective - self.current_objective
                acceptance_probability = int(current_difference < 0)
                rejection_probability = int(not acceptance_probability)
            else:
                new_pool = self.sample(non_evaluated_scenario_ids)
                total_evaluations += 2*len(new_pool)
                self.simulator.evaluate(self.candidate_solution, new_pool, tracker=self.tracker)
                self.simulator.evaluate(self.current_solution, new_pool, tracker=self.tracker)

                objectives_current = self.tracker.get_row(self.current_solution.id)
                objectives_candidate = self.tracker.get_row(self.candidate_solution.id)
                self.current_objective = objectives_current[objectives_current > 0].mean()
                self.candidate_objective = objectives_candidate[objectives_candidate > 0].mean()
                current_difference = self.candidate_objective - self.current_objective
                if self.use_updating_variance and (loop_times == 1):
                    self.update_variance(current_difference)

                acceptance_probability = min(1,
                                             np.exp(-2*(current_difference + self.variance/(2*self.temperature)) *
                                                    (previous_difference + self.variance/(2*self.temperature)) /
                                                    self.variance))
                rejection_probability = int(current_difference >= 0)
            if self.print_updates:
                self.print_sampling_results(loop_times, current_difference, acceptance_probability,
                                            rejection_probability)
            random_uniform_accept = np.random.uniform()
            random_uniform_reject = np.random.uniform()
            if acceptance_probability >= random_uniform_accept:
                self.accepted()
                self.is_accepted = True
                return total_evaluations
            elif rejection_probability >= random_uniform_reject:
                # Note: The simple rejection rule doesn't work that well when evaluating small sample steps
                self.is_accepted = False
                return total_evaluations
            else:  # continue
                previous_difference = current_difference

    def print_iteration_results(self, number_of_evaluations: int):
        acceptance = 'accepted' if self.is_accepted else 'rejected'
        best_addition = f'best {round(self.best_objective, 2)}, ' if self.use_best else ''
        print(f'Iteration {self.current_iteration}: '
              f'temp {round(self.temperature, 2)}, '
              f'candidate {round(self.candidate_objective, 2)}, '
              f'current {round(self.current_objective, 2)}, '
              f'{best_addition}'
              f'evaluations {number_of_evaluations}, '
              f'{acceptance}', end='\r')

    def save_solution(self, directory, file_name) -> None:
        # Note we save the current solution as this is how this method works
        save_object(self.current_solution, directory, file_name)

    @staticmethod
    def print_sampling_results(loop_times: int, difference: float, acceptance_probability: float,
                               rejection_probability: float):
        print(f'In loop {loop_times}, '
              f'the difference is {round(difference, 0)}, '
              f'acceptance probability: {round(acceptance_probability, 2)}, '
              f'rejection probability: {round(rejection_probability, 2)}')
