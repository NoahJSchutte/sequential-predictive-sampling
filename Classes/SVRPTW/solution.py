from typing import List

from Classes.Super.solution import Solution
from Classes.SVRPTW.route import Route
from Classes.SVRPTW.customer import Customer


class SolutionSVRPTW(Solution):
    def __init__(
            self,
            routes: List[Route],
            unscheduled_customers: List[Customer],
            id: int = -1
    ):
        Solution.__init__(self, id)
        self.routes = routes
        self.unscheduled_customers = unscheduled_customers

    def add_id(self, id: int):
        self.id = id

    def get_number_of_routes(self):
        return len(self.routes)

    def get_route(self, index: int):
        return self.routes[index]

    def get_routes(self):
        return self.routes

    def get_unscheduled_customers(self):
        return self.unscheduled_customers

    def get_number_of_unscheduled_customers(self):
        return len(self.unscheduled_customers)



