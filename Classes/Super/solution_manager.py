from Classes.Super.instance import Instance
from Classes.Super.solution import Solution
import random

class SolutionManager:
    def __init__(
            self,
            instance: Instance
    ):
        self.instance = instance

    def get_candidate(self, solution: Solution) -> Solution:
        return solution

    def generate_random_solution(self, seed: int) -> Solution:
        return Solution()

    def set_seed(self, seed: int):
        random.seed(seed)


