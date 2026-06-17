from component.lane.lane import Lane

class LaneFactory:
    def __init__(self, config):
        self.config = config

    def create_lane(self, lane_config):
        lane_id = lane_config['lane_id']
        waypoints = lane_config['waypoints']
        width = self.config.road_network.lane_width
        lane = Lane(lane_id, waypoints, width)
        # support adjacency keys from config (adj_left / adj_right)
        left_adj = lane_config.get('adj_left') if isinstance(lane_config, dict) else getattr(lane_config, 'adj_left', None)
        right_adj = lane_config.get('adj_right') if isinstance(lane_config, dict) else getattr(lane_config, 'adj_right', None)
        if left_adj:
            lane.left_lane_id = left_adj
        if right_adj:
            lane.right_lane_id = right_adj
        return lane