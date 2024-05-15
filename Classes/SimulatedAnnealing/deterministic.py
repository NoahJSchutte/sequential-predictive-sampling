from Classes.SRCPSP.simulator import Simulator
from Classes.SRCPSP.solution import Solution
from Classes.SRCPSP.solution_manager import SolutionManager
from Classes.SRCPSP.scenario_pool import ScenarioPool
from Classes.SimulatedAnnealing.simulated_annealing import SimulatedAnnealing


class DeterministicSA(SimulatedAnnealing):
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
            print_updates: bool = True
    ):
        SimulatedAnnealing.__init__(self, budget, initial_solution, solution_manager, simulator,
                                    scenario_pool, budget_type, seed, initial_temperature, final_temperature,
                                    use_best, print_updates)


