from dataclasses import dataclass


@dataclass
class Customer:
    id: str
    x_coord: int
    y_coord: int
    demand: int
    ready_time: int
    due_date: int
    service_time: int
