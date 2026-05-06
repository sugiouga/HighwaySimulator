from config_loader import MasterConfig
from manager.traffic_manager import TrafficManager
from manager.road_network import RoadNetwork
from observer.visualizer import Visualizer


def run_simulation(config: MasterConfig):
    road_network = RoadNetwork(config)
    traffic_manager = TrafficManager(road_network, config, dt=config.simulation.time_step)

    if config.visualization.enable:
        traffic_manager.add_observer(Visualizer(config))

    while traffic_manager.current_time < config.simulation.total_time:
        traffic_manager.step()

    return traffic_manager


def main():
    config = MasterConfig.from_yaml("config.yaml")
    run_simulation(config)


if __name__ == "__main__":
    main()
