import os
import sys


def configure_carla_paths():
    """Add CARLA PythonAPI paths used by both 0.9.10 and newer releases."""
    carla_root = os.getenv("CARLA_ROOT")
    if carla_root is None:
        raise ValueError("CARLA_ROOT must be defined.")

    carla_root = os.path.abspath(os.path.expanduser(carla_root))
    paths = [
        os.path.join(carla_root, "PythonAPI"),
        os.path.join(carla_root, "PythonAPI", "carla"),
        os.path.join(carla_root, "PythonAPI", "carla", "agents"),
    ]
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)


def make_global_route_planner(carla_map, sampling_resolution=0.5):
    """Create a GlobalRoutePlanner across CARLA 0.9.10 and 0.9.13 APIs."""
    configure_carla_paths()

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ImportError:
        from navigation.global_route_planner import GlobalRoutePlanner

    try:
        from navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    except ImportError:
        GlobalRoutePlannerDAO = None

    if GlobalRoutePlannerDAO is None:
        return GlobalRoutePlanner(carla_map, sampling_resolution)

    planner = GlobalRoutePlanner(
        GlobalRoutePlannerDAO(carla_map, sampling_resolution=sampling_resolution)
    )
    if hasattr(planner, "setup"):
        planner.setup()
    return planner
