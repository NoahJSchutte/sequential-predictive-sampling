import random
import pandas as pd
import numpy as np

from Classes.Super.solution_manager import SolutionManager
from Classes.Super.solution import Solution
from Classes.ESPU.solution import SolutionESPU
from Classes.ESPU.instance import InstanceESPU
from Classes.ESPU.block import Block
from Classes.ESPU.day import Day

from Functions.ESPU.mutation_functions import only_spread, remove_one_random, add_one_random


class SolutionManagerESPU(SolutionManager):
    def __init__(
            self,
            instance: InstanceESPU
    ):
        SolutionManager.__init__(self, instance)
        self.instance = instance
        self.nr_heuristics = {"add_remove": 3, "spread": 5}
        self.heuristics_names = {"add_remove": ["same_nr", "add", "remove"],
                                 "spread": ["spread_increasing", "slack_begin", "slack_end", "shift_left", "shift_right"]}
        self.heuristic_values = {"add_remove": [i for i in range(self.nr_heuristics["add_remove"])],
                                 "spread": [i for i in range(self.nr_heuristics["spread"])]}
        
    def get_candidate(self, solution: SolutionESPU) -> Solution:
        # Select
        mutator_id = np.random.choice(self.heuristic_values["add_remove"])
        spread_id = np.random.choice(self.heuristic_values["spread"])

        # Correct if we can't add currently
        solution.count_surgeries()
        if self.heuristics_names["add_remove"][mutator_id] == "add":
            if sum(solution.nr_surgeries.values()) == self.instance.instance_size:
                mutator_id = 0
                #if self.print_detailed:
                #    print("WARNING: solution is full, can not add")
        elif self.heuristics_names["add_remove"][mutator_id] == "remove" or \
                self.heuristics_names["add_remove"][mutator_id] == "same_nr":
            if sum(solution.nr_surgeries.values()) == 0:
                mutator_id = 1
                #if self.print_detailed:
                #    print("WARNING: solution is empty, can not remove or have same nr")

        # Translate mutator_idator
        mutator = self.heuristics_names["add_remove"][mutator_id]
        spread = self.heuristics_names["spread"][spread_id]

        if mutator == "same_nr":
            candidate = only_spread(solution, spread) # only apply spread heuristic when no surgery is added
        elif mutator == "remove":
            candidate = remove_one_random(solution, spread="spread_equally", slack_begin=False,
                                          slack_end=False)
        else:  # mutator == "add":
            candidate = add_one_random(solution, instance_info=self.instance, spread="spread_equally", 
                                       slack_begin=False, slack_end=False)

        candidate.count_surgeries()
        
        return candidate

    def generate_random_solution(self, seed: int) -> SolutionESPU:
        random.seed(seed)

        # Read input blocks
        blocks = pd.read_csv("Data/Input/ESPU/Blocks.csv", delimiter=";")

        # Initialize empty solution
        nr_surgeries = {"CARD": 0, "GASTRO": 0, "GYN": 0, "MED": 0, "ORTH": 0, "URO": 0}
        solution = SolutionESPU(nr_surgeries)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_dict = {}
        instance_nr_patients = {}
        for d in days:
            day = Day(name=d, start_time=0, end_time=8 * 60)
            day_dict[d] = day

        # Iterate over different block types [CARD, GASTRO, GYN, MED, ORTH, URO]
        for block in self.instance.block_types:
            type_class = self.instance.block_types[block]
            nr_patients = self.instance.block_types[block].nr_of_surgeries  # nr of patients
            instance_nr_patients[block] = nr_patients
            blocks_of_type = blocks.loc[blocks['TYPE'] == block]["BLOCK"].to_list()  # blocks with that type

            # Initialize assignment dictionary
            assignment_dict = {}
            for b in blocks_of_type:
                assignment_dict[b] = 0
            assignment_dict["not assigned"] = 0

            # Divide total number of patients of certain types between different
            # blocks of that type + not assigned possibility
            for i in range(nr_patients):
                assignment = random.choice(blocks_of_type + ["not assigned"])
                assignment_dict[assignment] += 1

            # Add surgery times to block when surgeries are assigned
            # Add block classes to days
            for b in blocks_of_type:
                day = blocks.loc[blocks['BLOCK'] == b]["DAY"].values[0]
                block_class = Block(block_type=type_class, key=b, end_time=8 * 60)
                nr_surgeries = assignment_dict[b]
                surgery_start_times = sorted(random.sample(range(0, 8 * 60), nr_surgeries))
                block_class.start_times_surgeries = [float(start_time) for start_time in surgery_start_times]
                day_dict[day].add_block(block_class)

        # Add day dictionary to solution
        solution.days = day_dict

        # Recount nr of surgeries
        solution.count_surgeries()

        return solution
