from typing import List

from Classes.SRCPSP.activity import Activity
from Classes.Super.solution import Solution


class SolutionSRCPSP(Solution):
    def __init__(
            self,
            activities: List[Activity],
            id: int = -1
    ):
        Solution.__init__(self, id)
        self.activities = activities




