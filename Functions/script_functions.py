from typing import List
import numpy as np
import random

from Classes.tracker import Tracker
from Classes.Super.solution import Solution
from Classes.SRCPSP.instance import InstanceSRCPSP
from Classes.SRCPSP.simulator import SimulatorSRCPSP
from Classes.SRCPSP.solution import SolutionSRCPSP
from Classes.SRCPSP.solution_manager import SolutionManagerSRCPSP
from Classes.SRCPSP.scenario_pool import ScenarioPoolSRCPSP
from Classes.ESPU.instance import InstanceESPU
from Classes.ESPU.simulator import SimulatorESPU
from Classes.ESPU.solution import SolutionESPU
from Classes.ESPU.solution_manager import SolutionManagerESPU
from Classes.ESPU.scenario_pool import ScenarioPoolESPU
from Classes.SVRPTW.instance import InstanceSVRPTW
from Classes.SVRPTW.simulator import SimulatorSVRPTW
from Classes.SVRPTW.solution import SolutionSVRPTW
from Classes.SVRPTW.solution_manager import SolutionManagerSVRPTW
from Classes.SVRPTW.scenario_pool import ScenarioPoolSVRPTW

from Functions.import_functions import read_instance_SRCPSP, read_instance_ESPU, read_instance_SVRPTW
from Functions.distribution_rules import exponential1, beta1, beta2, uniform1, uniform2
from Functions.general_functions import load_object


def get_problem_specific_objects(problem: str, seed: int,
                                 instance_parameters: List, solution_manager_parameters: List,
                                 scenario_generation_parameters: List, pool_sampling: str = 'random',
                                 evaluation_budget: int = 100000, no_emergencies: bool = False):
    instance = globals()[f'read_instance_{problem}'](*instance_parameters)
    simulator = globals()[f'Simulator{problem}'](instance)
    solution_manager = globals()[f'SolutionManager{problem}'](instance, *solution_manager_parameters)
    solution_manager.set_seed(seed)
    initial_solution = solution_manager.generate_random_solution(seed=seed)
    initial_solution.id = 0
    scenario_pool = globals()[f'ScenarioPool{problem}'](instance)
    if pool_sampling == 'descriptive':
        scenario_pool.generate_scenarios_descriptive(*scenario_generation_parameters)
    elif pool_sampling == 'average':
        scenario_pool.generate_scenarios_average(not no_emergencies)
    else:
        scenario_pool.generate_scenarios(*scenario_generation_parameters)
    tracker = Tracker(scenario_pool, base_solution_len=evaluation_budget+1)

    return instance, simulator, solution_manager, initial_solution, scenario_pool, tracker


def validate_solution(solution: Solution, problem: str, seed: int, pool_size: int, pool_sampling: str,
                      instance_directory: str, par_1, par_2, par_3, par_4, par_5, no_emergencies: bool = False)\
        -> float:
    random.seed(seed)
    np.random.seed(seed)
    if problem == 'SRCPSP':  # note: the usage of the parameters is a bit ugly here
        distribution_rule = globals()[par_4]
        instance_parameters = [instance_directory, par_1, par_2, par_3]
        solution_manager_parameters = [par_5, par_5, False]
        scenario_generation_parameters = [pool_size, distribution_rule, seed]
    elif problem == 'ESPU':
        instance_parameters = [par_1, par_2, par_3]
        solution_manager_parameters = []
        scenario_generation_parameters = [pool_size, seed]
    else: # problem == 'SVRPTW:
        demand_distribution_rule = globals()[par_3[0]]
        travel_time_distribution_rule = globals()[par_3[1]]
        instance_parameters = [par_1, par_2[0], par_2[1], par_2[2], par_4]
        solution_manager_parameters = []
        scenario_generation_parameters = [pool_size, demand_distribution_rule,
                                          travel_time_distribution_rule, seed]

    instance, simulator, solution_manager, initial_solution, test_pool, tracker = \
        get_problem_specific_objects(problem, seed, instance_parameters,
                                     solution_manager_parameters,
                                     scenario_generation_parameters,
                                     pool_sampling=pool_sampling,
                                     no_emergencies=no_emergencies)
    evaluation = simulator.evaluate(solution, test_pool)

    return evaluation
