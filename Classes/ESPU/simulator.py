from Classes.ESPU.solution import SolutionESPU
from Classes.ESPU.day import Day
from Classes.ESPU.sim_block import SimBlock
from Classes.ESPU.sim_surgery import SimSurgery
from Classes.ESPU.instance import InstanceESPU
from Classes.ESPU.scenario import ScenarioESPU
from Classes.ESPU.scenario_pool import ScenarioPool
from Classes.tracker import Tracker
from Classes.Super.simulator import Simulator
from typing import List, Dict

import numpy as np
import random


class SimulatorESPU(Simulator):
    def __init__(
            self,
            instance: InstanceESPU
    ):
        Simulator.__init__(self, instance)
        self.instance = instance
        self.current_time: float = 0
        self.sim_blocks: Dict[str, List[SimBlock]] = dict()
        self.emergency_surgeries: List[SimSurgery] = list()
        self.policy = 'greedy'
        self.print_status = False

    def reset(self):
        self.current_time = 0
        self.sim_blocks = dict()
        self.emergency_surgeries = {}

    def set_policy(self, policy: str): # currently working: greedy and deterministic
        self.policy = policy

    def set_emergency_surgery_distributions(self, emergency_surgery_duration_distribution,
                                            emergency_surgery_duration_parameters,
                                            emergency_surgery_arrival_distribution,
                                            emergency_surgery_arrival_rate):
        self.instance.emergency_duration_distribution = emergency_surgery_duration_distribution
        self.instance.emergency_duration_parameters = emergency_surgery_duration_parameters
        self.instance.emergency_arrival_distribution = emergency_surgery_arrival_distribution
        self.instance.emergency_arrival_rate = emergency_surgery_arrival_rate

    def evaluate(self, solution: SolutionESPU, scenario_pool: ScenarioPool, tracker: Tracker = None, 
                 solution_index: int = None, print_status=False) -> float:
        sum_objectives = 0
        solution.reset_day_objectives()
        for scenario in scenario_pool.scenarios:
            objective = self.simulate_full_solution(solution, scenario)
            if tracker:
                if solution_index is None:
                    tracker.set_objective(scenario.id, solution.id, objective)
                else:
                    tracker.set_objective_on_location(scenario.id, solution_index, objective)
            sum_objectives += objective

            if print_status:
                print(f'Scenario {scenario.id} has an objective of {objective}')

        average_objective = sum_objectives / len(scenario_pool)
        solution.set_simulated_objective(average_objective)  # when the evaluator is used this gets overwritten

        return average_objective

    def evaluate_adjusted_day(self, solution: SolutionESPU, scenario_pool: ScenarioPool):
        self.reset()
        not_scheduling_costs = (self.instance.instance_size - solution.get_number_of_scheduled_surgeries()) * \
                               self.instance.not_scheduling_costs
        day = solution.get_adjusted_day()
        sum_objectives = 0
        for scenario in scenario_pool.scenarios:
            costs, info, criteria = self.simulate_day(day, scenario)
            sum_objectives += costs

        day_objective = sum_objectives / len(scenario_pool)
        day.set_simulated_objective(day_objective)

        return solution.update_simulated_objective_based_on_days(not_scheduling_costs)

    def simulate_full_solution(self, solution: SolutionESPU, scenario: ScenarioESPU):
        self.reset()
        not_scheduling_costs = (self.instance.instance_size - solution.get_number_of_scheduled_surgeries()) * \
            self.instance.not_scheduling_costs
        objective = not_scheduling_costs

        for day_name, day in solution.days.items():
            day_costs, day_info, day_criteria = self.simulate_day(day, scenario)
            day.adjust_simulated_objective(day_costs)
            objective += day_costs

        return objective

    def simulate_day(self, day: Day, scenario: ScenarioESPU):
        # create objects that are needed
        self.current_time = 0
        self.sim_blocks[day.name] = []    # empty sim_blocks
        for name, block in day.blocks.items():
            sim_block = SimBlock(block, self.instance)
            self.sim_blocks[day.name].append(sim_block)

        for sim_block in self.sim_blocks[day.name]:
            sim_block.set_surgery_durations(scenario)
            sim_block.set_policy(self.policy)

        self.set_emergency_surgeries(scenario)

        while self.emergency_surgeries[day.name]:  # or while not end day
            emergency_surgery = self.emergency_surgeries[day.name].pop(0)
            next_decision_point = max(emergency_surgery.get_scheduled_start_time(), self.current_time)
            self.current_time = next_decision_point
            self.apply_policy(day, emergency_surgery, next_decision_point)

        # When there are no surgeries left we still need to finish all surgeries to be able to het costs
        certain_finish_time = float('inf')
        for block in self.sim_blocks[day.name]:
            block.move_to_next_decision_point(certain_finish_time)

        costs, info, criteria = self.calculate_realized_costs(day)

        return costs, info, criteria

    def apply_policy(self, day: Day, emergency_surgery, next_decision_point):
        at_least_one_block_is_free = False
        for sim_block in self.sim_blocks[day.name]:
            sim_block.move_to_next_decision_point(next_decision_point)
            if not sim_block.has_ongoing_surgery:
                at_least_one_block_is_free = True

        if self.policy == 'greedy':
            free_blocks = list()
            for block in self.sim_blocks[day.name]:
                if not block.has_ongoing_surgery:
                    free_blocks.append(block)

            if free_blocks:
                if len(free_blocks) == 1:
                    best_block = free_blocks[0]
                else:
                    best_block = self.pick_best_free_block_greedy(free_blocks)
            else:
                best_block = self.pick_best_occupied_block_greedy(day)
            best_block.add_emergency_surgery(emergency_surgery)
            cancellations = 0
            if self.print_status:
                print(f'Best block is {best_block.block.block_type.name} with {cancellations} cancellations')
        elif self.policy == 'random':
            best_block = random.choice(self.sim_blocks[day.name])
            cancellations = random.randint(0, best_block.get_number_of_surgeries_left(including_ongoing=False,
                                                                                        includes_emergencies=False))
            best_block.cancel_surgeries(cancellations)
            best_block.add_emergency_surgery(emergency_surgery)
        else:  # All other policies currently use the same procedure
            best_block, cancellations = self.pick_best_block(day)
            best_block.cancel_surgeries(cancellations)
            best_block.add_emergency_surgery(emergency_surgery)
            #print(f'{cancellations} cancellations were done')


    def ask_policy(self, day: Day, next_decision_point):
        for sim_block in self.sim_blocks[day.name]:
            sim_block.move_to_next_decision_point(next_decision_point)

        if self.policy == 'greedy' or self.policy == 'random':
            print('Asking for a policy is not possible for the greedy or random policy')
            return None
        else:
            best_block, cancellations = self.pick_best_block(day)
            return best_block, cancellations

    def apply_decision(self, day: Day, emergency_surgery, block_id: int, cancellations: int):
        # This function should be run after ask_policy, as in ask_policy the blocks current state is updated
        chosen_block = self.get_sim_block(day, block_id)
        chosen_block.cancel_surgeries(cancellations)
        chosen_block.add_emergency_surgery(emergency_surgery)

    def get_sim_block(self, day: Day, block_id: int):
        for sim_block in self.sim_blocks[day.name]:
            if int(sim_block.get_block_key()) == block_id:
                return sim_block.get_block()
        print('There was no matching block to the provided id')
        return None

    def get_next_free_block_time(self, day:Day):
        next_free_block_time = float('inf')
        for sim_block in self.sim_blocks[day.name]:
            free_time = sim_block.get_next_free_time()
            if free_time < next_free_block_time:
                next_free_block_time = free_time

        return next_free_block_time

    def pick_best_free_block_greedy(self, free_blocks):
        minimum_total_expected_duration = float('inf')
        best_block = free_blocks[0]
        for block in free_blocks:
            block_expected_duration = block.get_remaining_total_expected_duration()
            if block_expected_duration < minimum_total_expected_duration:
                minimum_total_expected_duration = block_expected_duration
                best_block = block

        return best_block

    def pick_best_occupied_block_greedy(self, day:Day):
        # we only get here if all blocks are occupied
        best_expected_remaining_duration = float('inf')
        best_block = self.sim_blocks[day.name][0]
        for block in self.sim_blocks[day.name]:
            expected_remaining_duration = block.get_next_expected_finish_time()
            if expected_remaining_duration < best_expected_remaining_duration:
                best_expected_remaining_duration = expected_remaining_duration
                best_block = block

        return best_block

    def pick_best_block(self, day:Day):
        best_marginal_costs = float('inf')
        best_cancellations = 0
        best_block = self.sim_blocks[day.name][0]
        for sim_block in self.sim_blocks[day.name]:
            if not sim_block.has_emergency_scheduled():
                marginal_costs, cancellations = sim_block.get_marginal_costs_and_cancellations()
                if marginal_costs < best_marginal_costs:
                    best_marginal_costs = marginal_costs
                    best_cancellations = cancellations
                    best_block = sim_block
        if self.print_status and False:
            print(f'Best block is {best_block.block.block_type.name} with {best_cancellations} cancellations and '
                  f'marginal costs of {best_marginal_costs}')
        return best_block, best_cancellations

    def calculate_realized_costs(self, day:Day):  # sim blocks are from a certain day
        total_costs = 0
        info = {"waiting_costs": 0, "emergency_waiting_costs": 0, "over_costs": 0, "idle_costs": 0,
                "cancellation_costs": 0}
        criteria = {"waiting": 0, "emergency_waiting": 0, "over": 0, "idle": 0,
                "cancellation": 0}
        for sim_block in self.sim_blocks[day.name]:
            additional_cost, block_info, block_criteria = sim_block.calculate_realized_costs()
            total_costs += additional_cost
            for cost in ["waiting_costs", "emergency_waiting_costs", "over_costs", "idle_costs", "cancellation_costs"]:
                info[cost] += block_info[cost]
            for crit in ["waiting", "emergency_waiting", "over", "idle",
                         "cancellation"]:
                criteria[crit] += block_criteria[crit]
        return total_costs, info, criteria

    def draw_emergency_surgeries(self, day: Day):
        time = day.start_time
        while time < day.end_time:
            new_emergency_arrival_time = getattr(np.random, self.instance.emergency_arrival_distribution)(
                *self.instance.emergency_arrival_parameters)
            new_emergency_duration = getattr(np.random, self.instance.emergency_duration_distribution)(
                *self.instance.emergency_duration_parameters)
            time = time + new_emergency_arrival_time
            new_emergency_surgery = SimSurgery(time, day, is_emergency=True)
            new_emergency_surgery.set_realized_duration(new_emergency_duration)
            self.emergency_surgeries.append(new_emergency_surgery)
        self.emergency_surgeries.pop()  # remove the last time as it is outside the day.end_time

    def set_emergency_surgeries(self, scenario: ScenarioESPU):
        emergency_arrivals = scenario.emergency_surgery_arrival_times
        emergency_durations = scenario.emergency_surgery_durations
        for day in emergency_arrivals:
            self.emergency_surgeries[day] = []
            for i, arrival_time in enumerate(emergency_arrivals[day]):
                duration = emergency_durations[day][i]
                new_emergency_surgery = SimSurgery(arrival_time, day, is_emergency=True)
                new_emergency_surgery.set_realized_duration(duration)
                self.emergency_surgeries[day].append(new_emergency_surgery)

    @staticmethod
    def pick_best_free_block_strategy(free_blocks: List[SimBlock]):
        # note might be nicer to move the sim_block part in the loop to SimBlock
        current_best_marginal_costs = 100000
        for sim_block in free_blocks:
            marginal_costs, cancellations = sim_block.get_marginal_costs_and_cancellations()
            if marginal_costs < current_best_marginal_costs:
                best_block = sim_block
                best_block_cancellations = cancellations

        return best_block, best_block_cancellations

