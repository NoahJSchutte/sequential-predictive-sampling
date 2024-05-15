from Classes.Super.simulator import Simulator
from Classes.SVRPTW.instance import InstanceSVRPTW
from Classes.SVRPTW.solution import SolutionSVRPTW
from Classes.SVRPTW.scenario_pool import ScenarioPoolSVRPTW
from Classes.SVRPTW.scenario import ScenarioSVRPTW
from Classes.SVRPTW.customer import Customer
from Classes.SVRPTW.route import Route
from Classes.tracker import Tracker


class SimulatorSVRPTW(Simulator):
    def __init__(
            self,
            instance: InstanceSVRPTW
    ):
        Simulator.__init__(self, instance)
        self.instance = instance
        self.current_time: float = 0

    def evaluate(self, solution: SolutionSVRPTW, scenario_pool: ScenarioPoolSVRPTW, tracker: Tracker = None,
                 solution_index: int = None) -> float:
        sum_objectives = 0
        for scenario in scenario_pool.get_scenarios():
            non_scheduling_costs, missed_customer_costs, traversed_distance_costs = \
                self.evaluate_scenario(solution, scenario)
            objective_costs = non_scheduling_costs + missed_customer_costs + traversed_distance_costs
            #print(f'Not scheduling: {non_scheduling_costs}, missed customer: {missed_customer_costs},'
            #      f'traversed distance" {traversed_distance_costs}')
            if tracker:
                if solution_index is None:
                    tracker.set_objective(int(scenario.id), solution.id, objective_costs)
                else:
                    tracker.set_objective_on_location(int(scenario.id), solution_index, objective_costs)

            sum_objectives += objective_costs

        return sum_objectives / len(scenario_pool)

    def evaluate_scenario(self, solution: SolutionSVRPTW, scenario: ScenarioSVRPTW):
        if self.instance.absolute_missed_customers:
            non_scheduling_costs = self.instance.not_scheduled_customer_costs * \
                                   solution.get_number_of_unscheduled_customers()
        else:
            unscheduled_customers = solution.get_unscheduled_customers()
            depot = self.instance.depot
            unscheduled_distance = 0
            for customer in unscheduled_customers:
                distance = 2*self.get_distance(depot, customer)
                unscheduled_distance += distance
            non_scheduling_costs = unscheduled_distance * self.instance.distance_costs

        missed_customer_costs = 0
        traversed_distance_costs = 0
        for route in solution.get_routes():
            traversed_distance, missed_customers, missed_customer_distance = self.simulate_route(route, scenario)
            if self.instance.absolute_missed_customers:
                missed_customer_costs += missed_customers * self.instance.missed_customer_costs
            else:
                missed_customer_costs += missed_customer_distance * self.instance.distance_costs
            traversed_distance_costs += traversed_distance * self.instance.distance_costs

        return non_scheduling_costs, missed_customer_costs, traversed_distance_costs

    def simulate_route(self, route: Route, scenario: ScenarioSVRPTW):
        missed_customers = 0
        missed_customer_distance = 0
        current_time = 0
        traversed_distance = 0
        current_capacity = self.instance.capacity
        depot = self.instance.depot
        current_customer = self.instance.depot
        next_customer = depot  # in case the route is empty
        for i, next_customer in enumerate(route.get_customers()):
            if scenario.customer_exists[next_customer.id]:
                if current_time > depot.due_date:
                    travel_back_distance = scenario.travel_time_matrix[int(current_customer.id), int(depot.id)]
                    traversed_distance += travel_back_distance
                    missed_customers += len(route) - i
                    for missed_customer in route.get_customers()[i:]:
                        if scenario.customer_exists[missed_customer.id]:
                            missed_customer_distance += 2*self.get_distance(depot, missed_customer)
                    return traversed_distance, missed_customers, missed_customer_distance
                distance_to_customer = self.get_distance(current_customer, next_customer)
                traversed_distance += distance_to_customer
                current_time += scenario.travel_time_matrix[int(current_customer.id), int(next_customer.id)]
                if current_time < next_customer.ready_time:
                    current_time = next_customer.ready_time + next_customer.service_time
                elif current_time <= next_customer.due_date:
                    customer_demand = scenario.demand_per_customer[next_customer.id]
                    if current_capacity > customer_demand:
                        current_time += next_customer.service_time
                        current_capacity -= customer_demand
                    else:  # refill capacity
                        travel_back_and_forth_distance = 2*self.get_distance(next_customer, depot)
                        traversed_distance += travel_back_and_forth_distance
                        travel_time_back = scenario.travel_time_matrix[int(next_customer.id), int(depot.id)]
                        travel_time_forth = scenario.travel_time_matrix[int(depot.id), int(next_customer.id)]
                        current_time += travel_time_back + travel_time_forth
                        current_capacity = self.instance.capacity
                        if current_time <= next_customer.due_date:
                            current_time += next_customer.service_time
                            current_capacity -= customer_demand
                        else:
                            missed_customers += 1
                            missed_customer_distance += 2*self.get_distance(depot, next_customer)
                else:
                    missed_customers += 1
                    missed_customer_distance += 2*self.get_distance(depot, next_customer)
        travel_back_distance = scenario.travel_time_matrix[int(next_customer.id), int(depot.id)]
        traversed_distance += travel_back_distance

        return traversed_distance, missed_customers, missed_customer_distance

    @staticmethod
    def get_distance(customer1: Customer, customer2: Customer):
        distance_customers = ((customer1.x_coord - customer2.x_coord)**2 +
                              (customer1.y_coord - customer2.y_coord)**2)**0.5
        return distance_customers

