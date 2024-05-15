# This script is used to run experiments.
# After the experiments are run, use the same settings within ### ### and run validate_results.py to get results.
import random
import numpy as np
import sys
import os

directory = sys.path[0]
new_path = directory.split('/Scripts')[0]
sys.path.append(new_path)

from Classes.SimulatedAnnealing.deterministic import DeterministicSA
from Classes.SimulatedAnnealing.sequential_predictive_sampling import SeqPreSA
from Classes.SimulatedAnnealing.sequential_difference_sampling import SeqDifSA

from Functions.general_functions import create_parser
from Functions.script_functions import get_problem_specific_objects
from Functions.distribution_rules import exponential1, beta1, beta2, uniform1, uniform2

run_locally = True

if not run_locally:
    parser = create_parser()
    args = parser.parse_args()

########################################################################################################################
# problem
problem = 'ESPU'
instance_directory = f'Data/Input/{problem}'
save_results = True

# problem configurations
if problem == 'SRCPSP':
    instance_sizes = [30]  # 30, 120 for SRCPSP, 70, 100, 150, 200 for ESPU
    instance_configurations = [1]
    instance_numbers = [1, 2, 3, 4, 5]
    distribution_configurations = ['exponential1']
    swaps = [1]
    problem_parameters_values = [instance_sizes, instance_configurations, instance_numbers,
                                 distribution_configurations, swaps]
    problem_parameters_names = ['instance size', 'instance configuration', 'instance number',
                                'distribution configuration', 'swaps']
    problem_parameters_ids = ['is', 'ic', 'in', 'dc', 'sw']
elif problem == 'ESPU':
    instance_sizes = [70, 100, 150, 200]
    emergency_arrival_rates = [4]
    costs_configurations = [4]
    problem_parameters_values = [instance_sizes, emergency_arrival_rates, costs_configurations]
    problem_parameters_names = ['instance size', 'emergency arrival rate', 'costs configuration']
    problem_parameters_ids = ['is', 'er', 'cc']
else:  # problem == 'SVRPTW'
    instance_ids = ['R101', 'R102', 'R103', 'R105']
    costs_configurations = [(50, 50, 1)]
    distribution_rules = [('exponential1', 'exponential1')]
    instance_configurations = [(5, 50, 1, False, True)]
    problem_parameters_values = [instance_ids, costs_configurations, distribution_rules, instance_configurations]
    problem_parameters_names = ['instance id', 'costs configuration', 'uncertainty configuration',
                                'instance configuration']
    problem_parameters_ids = ['ii', 'cc', 'uc', 'ic']
    instance_to_configuration = {'R101': (9, 50, 1, False, False),
                                 'R102': (11, 50, 1, False, False),
                                 'R103': (7, 50, 1, False, False),
                                 'R105': (5, 50, 1, False, False),
                                 'R104': (5, 50, 1, False, False)}


while len(problem_parameters_names) < 5:  # needed for the for-loop structure
    problem_parameters_names.append('')
    problem_parameters_ids.append('')
    problem_parameters_values.append([''])

# model configurations
# method: informative, optimal, deterministic
# pool sampling: average, descriptive, random
# train pool size: 1, 10, 100
configurations = [{'method': 'seqpre',              'pool sampling': 'descriptive',          'train pool size': 100},
                  {'method': 'deterministic',       'pool sampling': 'average',              'train pool size': 1},
                  {'method': 'seqdif',              'pool sampling': 'descriptive',          'train pool size': 100}
                  #{'method': 'deterministic',       'pool sampling': 'descriptive',          'train pool size': 10},
                  #{'method': 'deterministic',       'pool sampling': 'descriptive',          'train pool size': 100},
                  #{'method': 'deterministic',       'pool sampling': 'random',               'train pool size': 1},
                  #{'method': 'deterministic',       'pool sampling': 'random',               'train pool size': 10},
                  #{'method': 'deterministic',       'pool sampling': 'random',               'train pool size': 100},
                  ]

# algorithm settings
initial_temperature = 144
final_temperature = 0.01
budgets = [5000]
budget_type = 'evaluations'  # 'iterations', 'evaluations'
use_best = False

# informative (seqpre) sampling specific
scenarios_per_sample_iteration_informative = 2
update_temperature_expected_iterations = False
fixed_solution_size = 50
speed_up_factorization = True
decision_criteria = ['optimal']  # optimal, probability, intervals
decision_probabilities = (0.9, 0.9999, 0.5, 0.5)
informative_methods = ['updates']  # updates is the only method

# optimal (seqdif) sampling specific
scenarios_per_sample_iteration_optimal = 1
assumed_variance = 10
use_updating_variance = True
########################################################################################################################

# from here different from validate_results.py
if run_locally:  # set random seeds and affinity parameter (-1 default)
    random_seeds = range(0, 1)
    affinity_parameter = -1
else:
    random_seeds = range(args.seed_start, args.seed_end+1)
    affinity_parameter = args.affinity
    if affinity_parameter > -1:
        os.sched_setaffinity(0, {affinity_parameter})

solution_directory = 'Data/Output/Solutions'
results_directory = 'Data/Output/Results'
print_updates = True
basic_results = True

for configuration in configurations:
    method = configuration['method']
    pool_sampling = configuration['pool sampling']
    train_pool_size = configuration['train pool size']
    for budget in budgets:
        if method != 'seqpre':
            used_decision_criteria = ['']
            used_informative_methods = ['']
        else:
            used_decision_criteria = decision_criteria
            used_informative_methods = informative_methods
        for decision_criterion in used_decision_criteria:
            for informative_method in used_informative_methods:
                if method == 'seqpre':
                    name_addition_method = f'_fs={fixed_solution_size}_sf={speed_up_factorization}' \
                                            f'_si={scenarios_per_sample_iteration_informative}_im={informative_method}' \
                                            f'_dc={decision_criterion}' \
                                            f'_pr={int(100 * decision_probabilities[0])}{int(100 * decision_probabilities[1])}' \
                                            f'{int(100 * decision_probabilities[2])}{int(100 * decision_probabilities[3])}'
                elif method == 'seqdif':
                    name_addition_method = f'_av={assumed_variance}_uv={use_updating_variance}' \
                                            f'_si={scenarios_per_sample_iteration_optimal}'
                else:
                    name_addition_method = ''
                for par_1 in problem_parameters_values[0]:
                    for par_2 in problem_parameters_values[1]:
                        for par_3 in problem_parameters_values[2]:
                            for par_4 in problem_parameters_values[3]:
                                for par_5 in problem_parameters_values[4]:
                                    if problem == 'SVRPTW':
                                        par_4 = instance_to_configuration[par_1]
                                    name_addition_problem = ''
                                    selected_values = [par_1, par_2, par_3, par_4, par_5]
                                    for i in range(5):
                                        if len(problem_parameters_ids[i]) > 0:
                                            name_addition_problem = f'{name_addition_problem}_' \
                                                                    f'{problem_parameters_ids[i]}={selected_values[i]}'
                                    for seed in random_seeds:
                                        random.seed(seed)
                                        np.random.seed(seed)
                                        if problem == 'SRCPSP':  # note: the usage of the parameters is a bit ugly here
                                            distribution_rule = locals()[par_4]
                                            instance_parameters = [instance_directory, par_1, par_2, par_3]
                                            solution_manager_parameters = [par_5, par_5, False]
                                            scenario_generation_parameters = [train_pool_size, distribution_rule, seed]
                                        elif problem == 'ESPU':
                                            instance_parameters = [par_1, par_2, par_3]
                                            solution_manager_parameters = []
                                            scenario_generation_parameters = [train_pool_size, seed]
                                        else:
                                            demand_distribution_rule = locals()[par_3[0]]
                                            travel_time_distribution_rule = locals()[par_3[1]]
                                            instance_parameters = [par_1, par_2[0], par_2[1], par_2[2], par_4]
                                            solution_manager_parameters = []
                                            scenario_generation_parameters = [train_pool_size, demand_distribution_rule,
                                                                              travel_time_distribution_rule, seed]

                                        instance, simulator, solution_manager, initial_solution, scenario_pool, tracker = \
                                            get_problem_specific_objects(problem, seed, instance_parameters,
                                                                         solution_manager_parameters,
                                                                         scenario_generation_parameters,
                                                                         pool_sampling=pool_sampling,
                                                                         evaluation_budget=budget)
                                        file_name = f'p={problem}_m={method}' \
                                                    f'_s={pool_sampling}_n={train_pool_size}' \
                                                    f'_r={seed}_ti={initial_temperature}_tf={final_temperature}' \
                                                    f'_bt={budget_type}_b={budget}{name_addition_method}' \
                                                    f'{name_addition_problem}'
                                        if method == 'deterministic':
                                            algorithm = DeterministicSA(budget=budget,
                                                                        initial_solution=initial_solution,
                                                                        solution_manager=solution_manager,
                                                                        scenario_pool=scenario_pool,
                                                                        budget_type=budget_type,
                                                                        simulator=simulator,
                                                                        seed=seed,
                                                                        initial_temperature=initial_temperature,
                                                                        final_temperature=final_temperature,
                                                                        use_best=use_best,
                                                                        print_updates=print_updates)
                                            algorithm.add_tracker(tracker)
                                        elif method == 'seqpre':
                                            algorithm = SeqPreSA(
                                                budget=budget,
                                                initial_solution=initial_solution,
                                                solution_manager=solution_manager,
                                                scenario_pool=scenario_pool,
                                                budget_type=budget_type,
                                                simulator=simulator,
                                                tracker=tracker,
                                                seed=seed,
                                                scenarios_per_sample_iteration=scenarios_per_sample_iteration_informative,
                                                fixed_solution_size=fixed_solution_size,
                                                initial_temperature=initial_temperature,
                                                final_temperature=final_temperature,
                                                use_best=use_best,
                                                update_temperature_expected_iterations=update_temperature_expected_iterations,
                                                speed_up_factorization=speed_up_factorization,
                                                decision_criterion=decision_criterion,
                                                decision_probabilities=decision_probabilities,
                                                informative_method=informative_method,
                                                print_updates=print_updates)
                                        else: #if method == 'seqdif':
                                            algorithm = SeqDifSA(
                                                budget=budget,
                                                initial_solution=initial_solution,
                                                solution_manager=solution_manager,
                                                scenario_pool=scenario_pool,
                                                budget_type=budget_type,
                                                simulator=simulator,
                                                tracker=tracker,
                                                seed=seed,
                                                scenarios_per_sample_iteration=scenarios_per_sample_iteration_optimal,
                                                initial_temperature=initial_temperature,
                                                final_temperature=final_temperature,
                                                assumed_variance=assumed_variance,
                                                use_updating_variance=use_updating_variance,
                                                use_best=use_best,
                                                print_updates=print_updates)
                                        algorithm.search()
                                        algorithm.save_solution(solution_directory, file_name)
                                        if save_results:
                                            tracker.save_results(results_directory, file_name, basic=basic_results)
                                        print(f'Completed {file_name}')





