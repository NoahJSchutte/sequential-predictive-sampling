import copy
import random

from Classes.Super.solution_manager import SolutionManager
from Classes.SVRPTW.solution import SolutionSVRPTW
from Classes.SVRPTW.instance import InstanceSVRPTW
from Classes.SVRPTW.route import Route
from Classes.SVRPTW.customer import Customer


class SolutionManagerSVRPTW(SolutionManager):
    def __init__(
            self,
            instance: InstanceSVRPTW,
            seed: int = 1995
    ):
        SolutionManager.__init__(self, instance)
        self.instance = instance
        random.seed(seed)

    def generate_random_solution(self, seed: int):
        routes = list()
        customers = copy.deepcopy(self.instance.get_customers(including_depot=False))

        first_time = True
        while len(customers) > 0:
            for i in range(self.instance.get_vehicle_number()):
                if first_time:
                    new_route = Route()
                    routes.append(new_route)
                if len(customers) > 0:
                    random_index = random.randint(0, len(customers)-1)
                    random_customer = customers.pop(random_index)
                    routes[i].add_customer(random_customer)
            if first_time:
                first_time = False

        solution = SolutionSVRPTW(routes, list(), id=0)
        return solution

    def get_candidate(self, solution: SolutionSVRPTW):
        if self.instance.schedule_all_customers:
            return self.get_candidate_all_customers(solution)
        else:
            solution_routes = copy.deepcopy(solution.get_routes())
            unscheduled_customers = copy.deepcopy(solution.get_unscheduled_customers())
            for route in solution_routes:
                random_draw = random.uniform(0, 1)
                if len(unscheduled_customers) == 0:
                    if (random_draw < 1/2) and (len(route) > 0):  # remove customer
                        customer_index = random.randint(0, len(route)-1)
                        removed_customer = route.remove_customer(customer_index)
                        unscheduled_customers.append(removed_customer)
                    # else do nothing
                else:
                    if random_draw < 1/3:  # add customer
                        customer_index = random.randint(0, len(unscheduled_customers)-1)
                        customer_to_add = unscheduled_customers.pop(customer_index)
                        self.add_customer(route, customer_to_add)
                    elif (random_draw < 2/3) and (len(route) > 0):  # remove customer
                        customer_index = random.randint(0, len(route)-1)
                        removed_customer = route.remove_customer(customer_index)
                        unscheduled_customers.append(removed_customer)
                # swap
                if len(route) > 1:
                    swap_index = random.randint(0, len(route)-2)
                    removed_customer = route.remove_customer(swap_index)
                    route.add_customer(removed_customer, swap_index+1)

            return SolutionSVRPTW(solution_routes, unscheduled_customers)

    def get_candidate_all_customers(self, solution: SolutionSVRPTW, do_nothing_probability: float = 0.33):
        solution_routes = copy.deepcopy(solution.get_routes())
        unscheduled_customers = list()  # there are initially never unscheduled customers
        for route in solution_routes:
            if len(route) > 0:
                random_draw = random.uniform(0, 1)
                if random_draw > do_nothing_probability:
                    customer_index = random.randint(0, len(route)-1)
                    removed_customer = route.remove_customer(customer_index)
                    unscheduled_customers.append(removed_customer)
                # else do nothing
        for customer in unscheduled_customers:
            random_route = random.choice(solution_routes)
            self.add_customer(random_route, customer)
        for route in solution_routes:
            if len(route) > 1:
                random_draw = random.uniform(0, 1)
                if random_draw > do_nothing_probability:
                    swap_index = random.randint(0, len(route)-2)
                    removed_customer = route.remove_customer(swap_index)
                    route.add_customer(removed_customer, swap_index+1)
        return SolutionSVRPTW(solution_routes, list())

    @staticmethod
    def add_customer(route: Route, new_customer: Customer):
        customer_window_mean = (new_customer.due_date + new_customer.ready_time) / 2
        later_window = 0
        for customer in route.get_customers():
            window_mean = (customer.due_date + customer.ready_time) / 2
            if customer_window_mean > window_mean:
                later_window += 1
        route.add_customer(new_customer, later_window)



