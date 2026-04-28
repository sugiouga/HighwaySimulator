class RoadNetwork:
    def __init__(self):
        self.lanes = {}

    def add_lane(self, lane):
        self.lanes.append(lane)

    def get_lane(self, lane_id):
        return self.lanes.get(lane_id)

    def is_within_bounds(self, s, d):
        """s, d座標が道路ネットワークの範囲内にあるかを判定するメソッド"""
        for lane in self.lanes.values():
            if 0 <= s <= lane.s_coords[-1] and -lane.width / 2 <= d <= lane.width / 2:
                return True
        return False