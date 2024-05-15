import numpy as np
import random
from typing import Dict, List

from Classes.Super.scenario_pool import ScenarioPool
from Classes.SVRPTW.instance import InstanceSVRPTW
from Classes.SVRPTW.scenario import ScenarioSVRPTW
from Classes.SVRPTW.customer import Customer

from Functions.stats_functions import draw_from_distribution, get_inverse_cdf


class ScenarioPoolSVRPTW(ScenarioPool):
    def __init__(
            self,
            instance: InstanceSVRPTW
    ):
        ScenarioPool.__init__(self, instance)
        self.instance = instance
        self.scenarios: List[ScenarioSVRPTW]

    def generate_scenarios(self, size: int, demand_distribution_rule, travel_time_distribution_rule, seed: int = 5):
        random.seed(seed)
        np.random.seed(seed)
        demand_distribution = demand_distribution_rule.__name__[:-1]
        travel_time_distribution = travel_time_distribution_rule.__name__[:-1]
        
        for i in range(size):
            demand_per_customer = dict()
            travel_time_matrix = np.zeros((self.instance.get_number_of_customers(including_depot=True),
                                           self.instance.get_number_of_customers(including_depot=True)))
            customer_exists = dict()
            for customer in self.instance.get_customers(including_depot=True):
                customer_exists[customer.id] = bool(np.random.binomial(1, self.instance.exists_prob))
                demand = max(self.instance.capacity,
                             draw_from_distribution(demand_distribution, demand_distribution_rule(customer.demand)))
                demand_per_customer[customer.id] = demand
                for other_customer in self.instance.get_customers(including_depot=True):
                    travel_time_mean = self.get_travel_time(customer, other_customer)
                    travel_time = draw_from_distribution(travel_time_distribution,
                                                         travel_time_distribution_rule(travel_time_mean))
                    travel_time_matrix[int(customer.id), int(other_customer.id)] = travel_time
            customer_exists['0'] = True  # depot always exists
            scenario = ScenarioSVRPTW(str(i), demand_per_customer, travel_time_matrix, customer_exists)
            self.scenarios.append(scenario)

    def generate_scenarios_average(self, dummy: bool):
        demand_per_customer = dict()
        travel_time_matrix = np.zeros((self.instance.get_number_of_customers(including_depot=True),
                                       self.instance.get_number_of_customers(including_depot=True)))
        existence_per_customer = dict()
        for customer in self.instance.get_customers(including_depot=True):
            demand_per_customer[customer.id] = customer.demand
            existence_per_customer[customer.id] = bool(round(self.instance.exists_prob))
            for other_customer in self.instance.get_customers(including_depot=True):
                travel_time = self.get_travel_time(customer, other_customer)
                travel_time_matrix[int(customer.id), int(customer.id)] = travel_time
        scenario = ScenarioSVRPTW('0', demand_per_customer, travel_time_matrix, existence_per_customer)
        self.scenarios.append(scenario)

    def generate_scenarios_descriptive(self, size: int, demand_distribution_rule, travel_time_distribution_rule, 
                                       seed: int = 5):
        random.seed(seed)
        np.random.seed(seed)
        demand_distribution = demand_distribution_rule.__name__[:-1]
        travel_time_distribution = travel_time_distribution_rule.__name__[:-1]
        
        possible_demands_per_customer: List[List[float]] = list()
        possible_existences_per_customer: List[List[bool]] = list()
        for customer in self.instance.get_customers(including_depot=True):
            mean = customer.demand
            demand_distribution_parameters = demand_distribution_rule(mean)
            possible_demands = list()
            possible_existences = list()
            for i in range(size):
                if mean == 0:
                    possible_demands.append(0)
                    possible_existences.append(True)
                else:
                    percentile = (i + 0.5) / size
                    inverse_cdf_value = get_inverse_cdf(demand_distribution, percentile,
                                                        parameters=demand_distribution_parameters)
                    possible_demand = max(self.instance.capacity, inverse_cdf_value)
                    possible_demands.append(possible_demand)
                    possible_existence = percentile < self.instance.exists_prob
                    possible_existences.append(possible_existence)
            possible_demands_per_customer.append(possible_demands)
            possible_existences_per_customer.append(possible_existences)

        possible_travel_times_per_arc: Dict[(str, str): List[float]] = dict()
        for customer1 in self.instance.get_customers(including_depot=True):
            for customer2 in self.instance.get_customers(including_depot=True):
                travel_time_mean = self.get_travel_time(customer1, customer2)
                travel_time_distribution_parameters = travel_time_distribution_rule(travel_time_mean)
                possible_travel_times = list()
                for i in range(size):
                    if travel_time_mean == 0:
                        possible_travel_times.append(0)
                    else:
                        percentile = (i + 0.5) / size
                        inverse_cdf_value = get_inverse_cdf(travel_time_distribution, percentile,
                                                            parameters=travel_time_distribution_parameters)
                        possible_travel_times.append(inverse_cdf_value)
                possible_travel_times_per_arc[(customer1.id, customer2.id)] = possible_travel_times

        for i in range(size):
            demand_per_customer = dict()
            existence_per_customer = dict()
            travel_time_matrix = np.zeros((self.instance.get_number_of_customers(including_depot=True),
                                           self.instance.get_number_of_customers(including_depot=True)))
            for j, customer in enumerate(self.instance.get_customers(including_depot=True)):
                demand_realization = random.choice(possible_demands_per_customer[j])
                possible_demands_per_customer[j].remove(demand_realization)
                demand_per_customer[customer.id] = demand_realization
                existence_realization = random.choice(possible_existences_per_customer[j])
                possible_existences_per_customer[j].remove(existence_realization)
                existence_per_customer[customer.id] = existence_realization
                for other_customer in self.instance.get_customers(including_depot=True):
                    travel_time_realization = random.choice(possible_travel_times_per_arc[(customer.id,
                                                                                           other_customer.id)])
                    possible_travel_times_per_arc[(customer.id, other_customer.id)].remove(travel_time_realization)
                    travel_time_matrix[int(customer.id), int(other_customer.id)] = travel_time_realization
            scenario = ScenarioSVRPTW(str(i), demand_per_customer, travel_time_matrix, existence_per_customer)
            self.scenarios.append(scenario)

    @staticmethod
    def get_travel_time(customer1: Customer, customer2: Customer, speed: float = 1):  # speed in distance/time
        distance_customers = ((customer1.x_coord - customer2.x_coord)**2 +
                              (customer1.y_coord - customer2.y_coord)**2)**0.5
        return distance_customers / speed



