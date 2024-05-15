from typing import List
from Classes.SVRPTW.customer import Customer


class Route:
    def __init__(
            self
    ):
        self.customers: List[Customer] = list()

    def add_customer(self, customer: Customer, index: int = -1):
        if index == -1:
            self.customers.append(customer)
        else:
            self.customers.insert(index, customer)

    def get_customers(self):
        return self.customers

    def remove_customer(self, index: int) -> Customer:

        return self.customers.pop(index)

    def __len__(self):
        return len(self.customers)