"""
Box Pushing Robot - CS152 Final Project

Main entry point for the box-pushing mission.

Architecture:
- simulation.py : PyBullet setup, environment creation
- kb.py         : Knowledge base (beliefs about the world)
- mapping.py    : Sensor interpretation (LiDAR → occupancy)
- planning.py   : Path planning (A*, exploration, box delivery)
- robot.py      : Low-level robot control (wheels, sensors)
- mission.py    : Mission execution (plan → actions → movements)
- main.py       : Entry point (this file)
"""

import simulation
import kb
from robot import RobotController, EAST
from mission import MissionController


def main():
    """Run the box-pushing mission."""
    print("=" * 60)
    print("Box Pushing Robot - CS152 Final Project")
    print("=" * 60)

    # Connect to PyBullet
    print("\n[1] Connecting to PyBullet...")
    try:
        client_id = simulation.connect(gui=True)
        if client_id < 0:
            raise RuntimeError("PyBullet GUI connection failed")
        print("    Connected to GUI!")
        realtime = False
    except Exception:
        print("    GUI unavailable, using DIRECT mode")
        client_id = simulation.connect(gui=False)
        if client_id < 0:
            raise RuntimeError("PyBullet DIRECT connection failed")
        realtime = False

    # Create environment
    print("\n[2] Creating environment...")
    env = simulation.create_environment(room_size=5.0)
    print(
        f"    Grid: {env['grid_size']}x{env['grid_size']} cells ({simulation.CELL_SIZE}m each)"
    )
    print(f"    Box at (world grid): {env['box_grid']} (unknown to robot)")
    print(f"    Goal at (world grid): {env['goal_grid']} (known to robot)")

    # Robot's internal frame: starts at (0,0)
    # Convert world coordinates to internal frame
    pybullet_start = (1, 1)  # Robot spawn position in PyBullet world grid

    # Goal position in internal frame (relative to robot start)
    goal_internal = (
        env["goal_grid"][0] - pybullet_start[0],
        env["goal_grid"][1] - pybullet_start[1],
    )
    # Box position in internal frame (debug-only; robot does not know it)
    box_internal = (
        env["box_grid"][0] - pybullet_start[0],
        env["box_grid"][1] - pybullet_start[1],
    )

    # Initialize Knowledge Base (cells start as UNKNOWN for exploration)
    print("\n[3] Initializing Knowledge Base...")
    knowledge = kb.KnowledgeBase()
    # Constrain the KB to the real room bounds in the INTERNAL frame.
    # PyBullet grid is [0..grid_size-1]; internal grid = pybullet - pybullet_start.
    min_x = -pybullet_start[0]
    max_x = env["grid_size"] - 1 - pybullet_start[0]
    min_y = -pybullet_start[1]
    max_y = env["grid_size"] - 1 - pybullet_start[1]
    knowledge.set_bounds(min_x, max_x, min_y, max_y)
    knowledge.set_goal(*goal_internal)
    print("    All cells start as UNKNOWN (exploration mode)")
    print(f"    Goal marked at: {goal_internal} (internal frame)")
    print(f"    Box (debug only): {box_internal} (internal frame)")

    # Load robot at PyBullet position (internal frame is (0,0))
    start_heading = EAST
    print(f"\n[4] Loading robot at PyBullet grid {pybullet_start}...")
    robot_id = simulation.load_robot(pybullet_start, start_heading)

    # KB uses internal frame: robot starts at (0,0)
    knowledge.set_robot(0, 0, start_heading)

    # Create controller (internal frame starts at 0,0)
    controller = RobotController(
        robot_id,
        knowledge,
        wall_ids=env["wall_ids"],
    )

    # Let simulation settle
    simulation.settle()

    # Initial scan to populate KB around start position
    print("\n[5] Initial LiDAR scan...")
    controller.scan_lidar(visualize=realtime)
    knowledge.print_grid()

    # Create mission controller
    mission = MissionController(
        controller,
        knowledge,
        box_id=env["box_id"],
    )

    # Run the full mission
    print("\n[6] Starting mission...")
    success = mission.run_full_mission(realtime=realtime)

    # Final state
    print("\n" + "=" * 60)
    if success:
        print("MISSION SUCCESSFUL!")
    else:
        print("MISSION FAILED")
    print("=" * 60)

    knowledge.print_grid()

    # Keep window open for inspection (GUI mode only)
    if realtime:
        import pybullet

        print("\nPress Ctrl+C to exit...")
        try:
            while pybullet.isConnected():
                simulation.step()
        except KeyboardInterrupt:
            pass
        pybullet.disconnect()


if __name__ == "__main__":
    main()
