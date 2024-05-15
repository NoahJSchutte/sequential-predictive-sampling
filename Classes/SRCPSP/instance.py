from Classes.SRCPSP.activity import Activity
from Classes.Super.instance import Instance
from typing import Dict, List, Set


class InstanceSRCPSP(Instance):
    def __init__(
            self,
            id: str,
            activities: List[Activity],  # Activity.id, Activity
            resource_ids: Set[str],
            resource_availabilities: Dict[str, int]
    ):
        Instance.__init__(self)
        self.id = id
        self.activities = activities
        self.resource_ids = resource_ids
        self.resource_availabilities = resource_availabilities
        self.activities_dict = dict()
        for activity in self.activities:
            self.activities_dict[activity.id] = activity

        self.activity_successors: Dict[str, Set[str]] = dict()  # str: activity_id
        self.activity_predecessors: Dict[str, Set[str]] = dict()  # str: activity_id
        self.get_activity_successors_dict()
        self.get_activity_predecessors_dict()

    def get_activity_successors_dict(self):
        for activity in self.activities:
            self.activity_successors[activity.id] = set()
            for successor_id in activity.successor_ids:
                self.activity_successors[activity.id].add(successor_id)

        # we get all the indirect successors as well
        for activity in self.activities:
            successor_ids = self.get_all_successors(activity.id)
            self.activity_successors[activity.id].union(successor_ids)
            
    def get_all_successors(self, activity_id: str) -> Set[str]:
        successor_ids = self.activity_successors[activity_id]
        for successor_id in successor_ids:
            successor_ids = successor_ids.union(self.get_all_successors(successor_id))
            
        return successor_ids

    def get_successors(self, activity: Activity):
        return self.activity_successors[activity.id]

    def get_activity_predecessors_dict(self):
        for activity in self.activities:
            self.activity_predecessors[activity.id] = set()
        for activity in self.activities:
            successor_ids = self.activity_successors[activity.id]
            for successor_id in successor_ids:
                self.activity_predecessors[successor_id].add(activity.id)

        # we get all the indirect predecessors as well
        for activity in self.activities:
            predecessor_ids = self.get_all_predecessors(activity.id)
            self.activity_predecessors[activity.id].union(predecessor_ids)

    def get_all_predecessors(self, activity_id: str) -> Set[str]:
        predecessor_ids = self.activity_predecessors[activity_id]
        for predecessor_id in predecessor_ids:
            predecessor_ids = predecessor_ids.union(self.get_all_predecessors(predecessor_id))

        return predecessor_ids

    def get_predecessors(self, activity: Activity):
        return self.activity_predecessors[activity.id]

    def get_activity(self, activity_id: str):
        return self.activities_dict[activity_id]

    def get_activities(self):
        return self.activities

    def get_resource_ids(self):
        return self.resource_ids

    def get_resource_availabilities(self):
        return self.resource_availabilities

    def have_precedence_relationship(self, predecessor: Activity, successor: Activity) -> bool:
        return successor.id in self.activity_successors[predecessor.id]

    def get_size(self):
        return len(self.activities) - 2


