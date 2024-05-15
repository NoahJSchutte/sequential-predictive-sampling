# Notes
# Currently it is hardcoded that everything is in integers

# Imports
from typing import Tuple
from Classes.SRCPSP.activity import Activity
from Classes.SRCPSP.instance import InstanceSRCPSP

from Classes.ESPU.instance import InstanceESPU
from Classes.ESPU.type import Type

from Classes.SVRPTW.customer import Customer
from Classes.SVRPTW.instance import InstanceSVRPTW

import pandas as pd


def process_file(directory, filename):
    with open(f'{directory}/{filename}') as file:
        # Initialization
        # contrary to the file, activities start at 0 (as that is the dummy activity)
        activity_ids = set()
        resources = set()  # resources start at 1
        resource_availabilities = dict()
        resource_requirements = dict()
        durations = dict()
        activity_successors = dict()

        finished = False

        while not finished:
            line = file.readline()
            if line == 'RESOURCES\n':
                line = file.readline()  # first line has (renewable) resources
                splitted_line = line.split()
                for i in range(1, int(splitted_line[3]) + 1):
                    resources.add(str(i))
            elif line == 'PRECEDENCE RELATIONS:\n':
                file.readline()  # skip line with titles
                line = file.readline()
                while line[0] != '*':
                    splitted_line = line.split()
                    activity_id = str(int(splitted_line[0]) - 1)
                    activity_ids.add(activity_id)
                    successors = set([str(int(s) - 1) for s in splitted_line[3:]])

                    activity_successors[activity_id] = successors
                    line = file.readline()
            elif line[0] == '-':
                line = file.readline()
                while line[0] != '*':
                    splitted_line = line.split()
                    activity_id = str(int(splitted_line[0]) - 1)

                    duration = int(splitted_line[2])
                    durations[activity_id] = duration

                    resource_requirement_list = splitted_line[3:]
                    resource_requirements[activity_id] = dict()
                    for resource, resource_requirement in enumerate(resource_requirement_list, 1):
                        resource_requirements[activity_id][str(resource)] = int(resource_requirement)
                    line = file.readline()
            elif line == 'RESOURCEAVAILABILITIES:\n':
                file.readline()
                line = file.readline()
                resource_availabilities_list = line.split()
                for resource, availability in enumerate(resource_availabilities_list, 1):
                    resource_availabilities[str(resource)] = int(availability)
                finished = True

        activities = list()
        activity_ids = sorted(list(activity_ids))
        for activity_id in activity_ids:
            activity = Activity(activity_id,
                                resource_requirements[activity_id],
                                durations[activity_id],
                                activity_successors[activity_id])
            activities.append(activity)
        instance = InstanceSRCPSP(filename[:-3], activities, resources, resource_availabilities)

        return instance


def read_instance_SRCPSP(directory, size, configuration, number):
    size_folder = f'j{size}'
    name = f'j{size}{configuration}_{number}.sm'
    instance = process_file(f'{directory}/{size_folder}', name)

    return instance


def read_instance_ESPU(instance_size: int, emergency_arrival_rate: int, costs_configuration: int,
                       dist_file: str = "Distributions.csv", percentages_file: str = "Percentages_waitlist.csv",
                       directory: str = "Data/Input/ESPU/"):
    costs_csv = f'{directory}Costs_{costs_configuration}.csv'
    dist_csv = f'{directory}{dist_file}'
    percentages_csv = f'{directory}{percentages_file}'
    df_dist = pd.read_csv(dist_csv)
    df_percentages = pd.read_csv(percentages_csv, sep=';')
    df_costs = pd.read_csv(costs_csv, sep=";", header=None, index_col=0).to_dict()[1]

    emergency_duration_distribution = df_dist.loc[df_dist['Type'] == 'EMERGENCY', 'Optimal distribution'].values[0]
    emergency_duration_parameters = [*eval(df_dist.loc[df_dist['Type'] == 'EMERGENCY', 'Parameters'].values[0])]

    instance = InstanceESPU(instance_size=instance_size,
                            not_scheduling_costs=df_costs["NOTSCHEDULING"],
                            cancellation_costs=df_costs["CANCELLING"],
                            waiting_costs=df_costs["ELECTIVEWAITINGTIME"],
                            waiting_emergency_costs=df_costs["EMERGENCYWAITINGTIME"],
                            over_time_costs=df_costs["OVERTIME"],
                            idle_time_costs=df_costs["IDLETIME"],
                            emergency_duration_distribution=emergency_duration_distribution,
                            emergency_duration_parameters=emergency_duration_parameters,
                            emergency_arrival_distribution='exponential',
                            emergency_arrival_parameters=[int(480/emergency_arrival_rate)])
    for idx in df_dist.index:
        block_type = df_dist.loc[idx, 'Type']
        if block_type != 'EMERGENCY':
            percentage = df_percentages.loc[df_percentages['Type'] == block_type, "Percentage"]
            percentage_as_float = float(percentage.values[0].replace(',', '.'))
            instance.add_type(Type(block_type, df_dist.loc[idx, 'Optimal distribution'],
                              [*eval(df_dist.loc[idx, 'Parameters'])],
                              percentage_as_float,
                              round(percentage_as_float*instance.instance_size),
                              exp_rate=df_dist.loc[idx, 'Exponential rate']))

    return instance


def read_instance_SVRPTW(instance_id: str,
                         not_scheduled_customer_costs: float, missed_customer_costs: float,
                         distance_costs: float, override_tuple: Tuple[int, int, float, bool],
                         directory: str = 'Data/Input/SVRPTW/Solomon'):
    filename = f'{instance_id}.txt'
    with open(f'{directory}/{filename}') as file:
        customer_list = list()
        finished = False
        while not finished:
            line = file.readline()
            if line == 'NUMBER     CAPACITY\n':
                line = file.readline()
                vehicle_number_str, capacity_str = line.split()
            elif line == '':
                finished = True
            else:
                splitted_line = line.split()
                if len(splitted_line) > 0:
                    if splitted_line[0].isnumeric():
                        new_customer = Customer(splitted_line[0],
                                                int(splitted_line[1]),
                                                int(splitted_line[2]),
                                                int(splitted_line[3]),
                                                int(splitted_line[4]),
                                                int(splitted_line[5]),
                                                int(splitted_line[6]))
                        customer_list.append(new_customer)
        if override_tuple[0] > 0:
            vehicle_number = override_tuple[0]
        else:
            vehicle_number = int(vehicle_number_str)
        if override_tuple[1] > 0:
            capacity = override_tuple[1]
        else:
            capacity = int(capacity_str)
        instance = InstanceSVRPTW(customer_list, vehicle_number, capacity,
                                  not_scheduled_customer_costs, missed_customer_costs, distance_costs,
                                  exists_prob=override_tuple[2], absolute_missed_customers=override_tuple[3],
                                  schedule_all_customers=override_tuple[4])
        return instance












