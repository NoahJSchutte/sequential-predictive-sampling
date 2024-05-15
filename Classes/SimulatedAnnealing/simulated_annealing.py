from Classes.SRCPSP.simulator import Simulator
from Classes.SRCPSP.solution import Solution
from Classes.SRCPSP.solution_manager import SolutionManager
from Classes.SRCPSP.scenario_pool import ScenarioPool
from Classes.tracker import Tracker

from Functions.general_functions import save_object

import random
import numpy as np
import time


class SimulatedAnnealing:
    def __init__(
            self,
            budget: int,
            initial_solution: Solution,
            solution_manager: SolutionManager,
            simulator: Simulator,
            scenario_pool: ScenarioPool,
            budget_type: str = 'iterations',
            seed: int = 0,
            initial_temperature: float = 100,
            final_temperature: float = 0.01,
            use_best: bool = True,
            print_updates: bool = True,
            compute_initial_temperature: bool = False
    ):

        self.initial_temperature = initial_temperature  # self.best_objective * (1-1.1)/np.log(0.5)
        self.final_temperature = final_temperature
        self.budget = budget
        self.budget_type = budget_type
        self.simulator = simulator
        self.scenario_pool = scenario_pool
        self.solution_manager = solution_manager
        self.tracker: Tracker = None
        self.use_best = use_best
        self.print_updates = print_updates

        # Initialization
        random.seed(seed)
        self.current_solution: Solution = initial_solution
        self.best_solution: Solution = initial_solution
        self.candidate_solution: Solution = None

        self.current_objective = self.evaluate(self.current_solution)
        self.best_objective = self.current_objective
        self.candidate_objective: float
        self.temperature = self.initial_temperature  # currently temperature is always relative to the budget
        self.current_iteration = 0
        self.total_number_of_evaluations = 0
        self.elapsed_time = 0
        self.start_time = 0
        self.is_accepted = False

    def add_tracker(self, tracker: Tracker):
        self.tracker = tracker

    def budget_left(self):
        if self.budget_type == 'iterations':
            return self.current_iteration < self.budget
        elif self.budget_type == 'evaluations':
            return self.total_number_of_evaluations < self.budget
        elif self.budget_type == 'time':
            return self.elapsed_time < self.budget

    def get_budget_ratio(self):
        if self.budget_type == 'iterations':
            return self.current_iteration / self.budget
        elif self.budget_type == 'evaluations':
            return self.total_number_of_evaluations / self.budget
        elif self.budget_type == 'time':
            return self.elapsed_time / self.budget

    def search(self) -> Solution:
        self.start_time = time.time()
        while self.budget_left():
            self.current_iteration += 1
            self.candidate_solution = self.solution_manager.get_candidate(self.current_solution)
            self.candidate_solution.id = self.current_iteration
            self.candidate_objective = self.evaluate(self.candidate_solution)
            self.total_number_of_evaluations += len(self.scenario_pool)
            self.elapsed_time = time.time() - self.start_time
            self.is_accepted = self.accept()
            self.update_temperature()
            if self.tracker is not None:
                self.tracker.add_candidate_objective(self.candidate_objective, self.current_iteration)
            if self.print_updates:
                self.print(self.candidate_objective)

        if self.tracker is not None:
            self.tracker.add_elapsed_time(self.elapsed_time)
            self.tracker.add_iteration_number(self.current_iteration)

        return self.best_solution

    def evaluate(self, solution: Solution) -> float:
        objective = self.simulator.evaluate(solution, self.scenario_pool)

        return objective

    def update_temperature(self):
        if self.temperature > self.final_temperature:
            self.temperature = self.initial_temperature * np.divide(self.final_temperature, self.initial_temperature) \
                               ** self.get_budget_ratio()
        if self.tracker is not None:
            self.tracker.add_temperature(self.temperature, self.current_iteration)

    def accept(self):
        if random.uniform(0, 1) <= np.exp(np.divide((self.current_objective-self.candidate_objective),
                                                    self.temperature)):
            self.accepted()
            return True
        else:
            return False

    def accepted(self):
        self.current_solution = self.candidate_solution
        self.current_objective = self.candidate_objective
        if self.tracker is not None:
            self.tracker.add_current_objective(self.current_objective, self.current_iteration)

        if self.use_best and (self.current_objective < self.best_objective):
            self.best_solution = self.current_solution
            self.best_objective = self.current_objective
            if self.tracker is not None:
                self.tracker.add_best_objective(self.best_objective, self.current_iteration)

    def print(self, candidate_objective: float):
        acceptance = 'accepted' if self.is_accepted else 'rejected'
        best_addition = f'best {round(self.best_objective, 2)}, ' if self.use_best else ''
        print(f'Iteration {self.current_iteration}: temp {round(self.temperature, 2)}, '
              f'candidate {round(candidate_objective, 2)}, '
              f'current {round(self.current_objective, 2)}, '
              f'{best_addition}{acceptance}', end='\r')

    def save_solution(self, directory, file_name) -> None:
        if self.use_best:
            save_object(self.best_solution, directory, file_name)
        else:
            save_object(self.current_solution, directory, file_name)