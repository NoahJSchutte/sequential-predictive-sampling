from typing import List

from Classes.Super.instance import Instance
from Classes.SVRPTW.customer import Customer


class InstanceSVRPTW(Instance):
    def __init__(
            self,
            customers: List[Customer],
            vehicle_number: int,
            capacity: int,
            not_scheduled_customer_costs: float,
            missed_customer_costs: float,
            distance_costs: float = 1,
            exists_prob: float = 1,
            absolute_missed_customers: bool = False,
            schedule_all_customers: bool = False
    ):
        Instance.__init__(self)
        self.customers: List[Customer] = customers
        self.depot: Customer = self.customers[0]
        self.vehicle_number: int = vehicle_number
        self.capacity: int = capacity
        self.not_scheduled_customer_costs: float = not_scheduled_customer_costs
        self.missed_customer_costs: float = missed_customer_costs
        self.distance_costs: float = distance_costs
        self.exists_prob: float = exists_prob
        self.absolute_missed_customers: bool = absolute_missed_customers
        self.schedule_all_customers: bool = schedule_all_customers

    def get_customers(self, including_depot: bool = False):
        if including_depot:
            return self.customers
        else:
            return self.customers[1:]

    def get_number_of_customers(self, including_depot: bool = False):
        if including_depot:
            return len(self.customers)
        else:
            return len(self.customers) - 1

    def get_vehicle_number(self):
        return self.vehicle_number

