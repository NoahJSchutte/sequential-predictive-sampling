import random
from typing import Set
from Classes.Super.solution_manager import SolutionManager
from Classes.Super.solution import Solution
from Classes.SRCPSP.solution import SolutionSRCPSP
from Classes.SRCPSP.instance import InstanceSRCPSP


class SolutionManagerSRCPSP(SolutionManager):
    def __init__(
            self,
            instance: InstanceSRCPSP,
            required_swaps: int = 1,
            maximum_swaps: int = 1,
            allow_only_new_solutions: bool = False
    ):
        SolutionManager.__init__(self, instance)
        self.instance = instance
        self.found_solution_representations: Set[str] = set()
        self.required_swaps = required_swaps
        self.maximum_swaps = maximum_swaps
        self.allow_only_new_solutions = allow_only_new_solutions

    def add_solution_representation(self, solution: SolutionSRCPSP) -> None:
        solution_representation = self.get_solution_representation(solution)
        self.found_solution_representations.add(solution_representation)

    def get_candidate(self, solution: SolutionSRCPSP) -> Solution:
        number_of_activities = len(solution.activities)
        potential_swaps = list(range(1, number_of_activities-2))  # we force exclusion of start and end dummy activities
        # and we swap with the next activity
        swap_success = False
        enough_swaps = False
        number_of_successful_swaps = 0
        while not enough_swaps or not swap_success:
            potential_swap_index = random.randint(0, len(potential_swaps)-1)
            swap_index = potential_swaps[potential_swap_index]
            if not self.instance.have_precedence_relationship(solution.activities[swap_index],
                                                              solution.activities[swap_index+1]):
                solution_activities_copy = solution.activities.copy()
                solution_activities_copy.insert(swap_index, solution_activities_copy.pop(swap_index+1))
                solution = SolutionSRCPSP(solution_activities_copy)
                if self.allow_only_new_solutions:
                    number_of_successful_swaps += 1
                    if self.equals_earlier_found(solution):
                        swap_success = False
                    else:
                        swap_success = True
                else:
                    number_of_successful_swaps += 1
                    swap_success = True
            if len(potential_swaps) > 0:
                potential_swaps.pop(potential_swap_index)
            else:
                print('Error: No swap possibilities left, so current solution is returned')
                break
            if number_of_successful_swaps >= self.required_swaps:
                coin_flip = random.randint(0, 1)
                if coin_flip or (number_of_successful_swaps >= self.maximum_swaps):
                    enough_swaps = True

        return solution

    def equals_earlier_found(self, solution: SolutionSRCPSP) -> bool:
        solution_representation = self.get_solution_representation(solution)
        earlier_found = solution_representation in self.found_solution_representations
        self.found_solution_representations.add(solution_representation)

        return earlier_found

    def generate_random_solution(self, seed: int = 0):
        random.seed(seed)
        activities = self.instance.get_activities()
        activity_list = list()
        while len(activity_list) < len(activities):
            potential_activities = list()
            for activity in activities:
                if activity not in activity_list:
                    predecessors = self.instance.get_predecessors(activity)
                    for chosen_activity in activity_list:
                        predecessors = predecessors.difference({chosen_activity.id})
                    if len(predecessors) == 0:
                        potential_activities.append(activity)
            random_activity = random.choice(potential_activities)
            activity_list.append(random_activity)

        return SolutionSRCPSP(activity_list)

    @staticmethod
    def get_solution_representation(solution: SolutionSRCPSP) -> str:
        solution_representation = ''
        for activity in solution.activities:
            solution_representation += f'{activity.id}_'
        solution_representation = solution_representation[:-1]

        return solution_representation
