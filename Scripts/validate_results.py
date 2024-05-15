# This script is used to validate the results obtained from run_experiment.py.
# Use the same settings as in run_experiment.py within ### ### here to validate them.
import pandas as pd
import sys

directory = sys.path[0]
new_path = directory.split('/Scripts')[0]
sys.path.append(new_path)

from Functions.general_functions import load_object
from Functions.script_functions import validate_solution

# validation settings
problem = 'ESPU'
validate = True
validate_train = False
load_results = True
basic_results = True
test_pool_size = 1000
test_seed = 1995
validation_name = f'{problem}.csv'

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
    instance_ids = ['R101', 'R102', 'R103', 'R104', 'R105']
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
# method: seqpre, seqdif, deterministic
# pool sampling: average, descriptive, random
# train pool size: 1, 10, 10
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
budgets = [5000] # 25000, 100000]
budget_type = 'evaluations'  # 'iterations', 'evaluations'
use_best = False

# informative (seqpre) sampling specific
scenarios_per_sample_iteration_informative = 2
update_temperature_expected_iterations = False
fixed_solution_size = 50
speed_up_factorization = True
decision_criteria = ['optimal']  # optimal, probability, intervals
decision_probabilities = (0.9, 0.9999, 0.5, 0.5)
informative_methods = ['updates']

# optimal (seqdif) sampling specific
scenarios_per_sample_iteration_optimal = 1
assumed_variance = 10
use_updating_variance = True
########################################################################################################################

# from here different from run_experiment.py
random_seeds = range(0, 1)
solution_directory = 'Data/Output/Solutions'
results_directory = 'Data/Output/Results'
validations_directory = 'Data/Output/Validations'

validation_list = list()
non_existent = 0
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
                                    for train_seed in random_seeds:
                                        file_name = f'p={problem}_m={method}' \
                                                    f'_s={pool_sampling}_n={train_pool_size}' \
                                                    f'_r={train_seed}_ti={initial_temperature}_tf={final_temperature}' \
                                                    f'_bt={budget_type}_b={budget}{name_addition_method}' \
                                                    f'{name_addition_problem}'
                                        solution = load_object(solution_directory, file_name)
                                        if solution is None:
                                            non_existent += 1
                                            print(f'Solution {file_name} does not exist')
                                        else:
                                            if validate:
                                                test_evaluation = validate_solution(solution,
                                                                                    problem,
                                                                                    test_seed,
                                                                                    test_pool_size,
                                                                                    'random',
                                                                                    instance_directory,
                                                                                    par_1, par_2, par_3, par_4, par_5)
                                            else:
                                                test_evaluation = None
                                            if validate_train:
                                                train_evaluation = validate_solution(solution,
                                                                                     problem,
                                                                                     train_seed,
                                                                                     train_pool_size,
                                                                                     pool_sampling,
                                                                                     instance_directory,
                                                                                     par_1, par_2, par_3, par_4, par_5)
                                            else:
                                                train_evaluation = None
                                            validation_stats = {
                                                'problem': problem,
                                                'method': method,
                                                'budget': budget,
                                                'pool sampling': pool_sampling,
                                                'train pool size': train_pool_size,
                                                'train seed': train_seed,
                                                'test pool size': test_pool_size,
                                                'test seed': test_seed,
                                                'test objective': test_evaluation,
                                                'train objective': train_evaluation,
                                                'method settings': name_addition_method,
                                                'instance size': par_1
                                            }
                                            if load_results:
                                                results = load_object(results_directory, file_name)
                                                if results is None:
                                                    print(f'Results {file_name} do not exist')
                                                else:
                                                    if basic_results:
                                                        results_stats = results
                                                    else:
                                                        train_objective = results['bests_y'][-1] if use_best else results['currents_y'][-1]
                                                        train_objective_iteration = results['bests_x'][-1] if use_best \
                                                            else results['currents_x'][-1]
                                                        results_stats = {
                                                            'number of iterations': len(results['candidates_x']),
                                                            'number of accepted solutions': len(results['currents_x']),
                                                            'train objective': train_objective,
                                                            'best objective iteration': train_objective_iteration,
                                                            'elapsed time': results['elapsed time']
                                                        }
                                                    validation_stats = {**validation_stats, **results_stats}

                                            for i in range(5):
                                                if len(problem_parameters_names[i]) > 0:
                                                    validation_stats[problem_parameters_names[i]] = selected_values[i]
                                            validation_list.append(validation_stats)
                                            print(f'Validation number {len(validation_list)} finished')
print(f'There we {non_existent} not existing solutions')
df = pd.DataFrame(validation_list)
df.to_csv(f'{validations_directory}/{validation_name}')



