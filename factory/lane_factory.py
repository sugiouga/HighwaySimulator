from component.lane import Lane

class LaneFactory:
    def __init__(self, config):
        self.config = config

    def create_lane(self, lane_config):
        lane_id = lane_config['lane_id']
        waypoints = lane_config['waypoints']
        width = self.config.road_network.lane_width
        return Lane(lane_id, waypoints, width)