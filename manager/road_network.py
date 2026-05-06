from factory.lane_factory import LaneFactory

class RoadNetwork:
    def __init__(self, config):
        self.config = config
        self.lanes = {}
        self.setup_lanes(self.config)

    def add_lane(self, lane):
        self.lanes[lane.lane_id] = lane

    def get_lane(self, lane_id):
        return self.lanes.get(lane_id)

    def get_left_lane(self, lane_id):
        lane = self.get_lane(lane_id)
        if lane and lane.left_lane_id:
            return self.get_lane(lane.left_lane_id)
        return None

    def get_right_lane(self, lane_id):
        lane = self.get_lane(lane_id)
        if lane and lane.right_lane_id:
            return self.get_lane(lane.right_lane_id)
        return None

    def is_within_bounds(self, s, d):
        """s, d座標が道路ネットワークの範囲内にあるかを判定するメソッド"""
        for lane in self.lanes.values():
            if 0 <= s <= lane.s_coords[-1] and -lane.width / 2 <= d <= lane.width / 2:
                return True
        return False

    def _setup_lanes(self):
        lane_factory = LaneFactory()
        for lane_config in self.config.road_network.lanes:
            lane = lane_factory.create_lane(lane_config)
            self.add_lane(lane)