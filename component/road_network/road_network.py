class RoadNetwork:
    def __init__(self):
        self.lanes = {}

    def add_lane(self, lane):
        self.lanes.append(lane)

    def get_lane(self, lane_id):
        return self.lanes.get(lane_id)