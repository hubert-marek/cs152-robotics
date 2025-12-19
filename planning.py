"""
Path planning algorithms for box-pushing robot.

Contains:
- A* for point-to-point navigation
- Frontier-based exploration (find the box)
- Box pushing strategy (deliver box to goal)
- Path to actions conversion
"""

from __future__ import annotations

import time
from heapq import heappush, heappop
from typing import Optional

from metrics import get_metrics

from kb import (
    KnowledgeBase,
    NORTH,
    EAST,
    SOUTH,
    WEST,
    DIRECTION_VECTORS,
    UNKNOWN,
    FREE,
    OCC,
)

# Defensive fallback: ensure UNKNOWN exists even if an older/stale version of this
# file (or a partial import edit) ends up running.
try:
    UNKNOWN
except NameError:
    UNKNOWN = -1


# Maps (dx, dy) to heading constant
DELTA_TO_HEADING = {
    (0, 1): NORTH,
    (1, 0): EAST,
    (0, -1): SOUTH,
    (-1, 0): WEST,
}

# Maps heading to (dx, dy)
HEADING_TO_DELTA = DIRECTION_VECTORS


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def turns_between(from_heading: int, to_heading: int) -> int:
    """
    Count minimum number of 90° turns to go from one heading to another.

    Returns 0, 1, or 2 (since max is 180° = 2 turns).
    """
    diff = abs(to_heading - from_heading) % 4
    if diff == 3:
        diff = 1  # Going the other way is shorter
    return diff


def astar(
    kb: KnowledgeBase,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    start_heading: Optional[int] = None,
    allow_unknown: bool = False,
    turn_cost: float = 0.0,
    search_type: str = "navigation",
) -> Optional[list[tuple[int, int]]]:
    """
    A* search on 4-connected grid with optional turn penalty.

    Input:
        -kb: KnowledgeBase with occupancy information
        -start: (x, y) start cell
        -goal: (x, y) goal cell
        -start_heading: initial heading (NORTH/EAST/SOUTH/WEST)
        -allow_unknown: if True, treat UNKNOWN cells as traversable
        -turn_cost: penalty per 90 degree turn (0.5 = half cell cost)
        -search_type: label for metrics ("navigation", "exploration", "sokoban")
    Output:
        -path: List of (x,y) cells from start to goal, or None if no path
    """
    search_start_time = time.perf_counter()
    nodes_expanded = 0

    if start == goal:
        # Record trivial search
        search_time_ms = (time.perf_counter() - search_start_time) * 1000
        get_metrics().record_astar_search(
            nodes_expanded=0,
            path_length=1,
            found_path=True,
            search_time_ms=search_time_ms,
            start=start,
            goal=goal,
            search_type=search_type,
        )
        return [start]

    # Get starting heading
    if start_heading is None:
        if kb.robot is not None:
            start_heading = kb.robot_heading
        else:
            start_heading = EAST  # Default

    # Check if goal is reachable - but allow navigating TO box cell
    # because pushing requires moving into the box position
    if not kb.is_traversable(goal[0], goal[1], allow_unknown=allow_unknown):
        if kb.box != goal:
            # Record failed search
            search_time_ms = (time.perf_counter() - search_start_time) * 1000
            get_metrics().record_astar_search(
                nodes_expanded=0,
                path_length=0,
                found_path=False,
                search_time_ms=search_time_ms,
                start=start,
                goal=goal,
                search_type=search_type,
            )
            return None

    # State includes heading because turning has a cost for diff-drive robots
    # This is different from standard grid A* which only tracks (x,y)
    initial_state = (start[0], start[1], start_heading)

    # Priority queue: (f_score, g_score, state, path)
    # Using f as primary sort key for A* optimality
    open_set: list[
        tuple[float, float, tuple[int, int, int], list[tuple[int, int]]]
    ] = []
    heappush(open_set, (manhattan_distance(start, goal), 0.0, initial_state, [start]))

    # Track visited states (x,y,heading)
    visited: set[tuple[int, int, int]] = set()

    # Main loop
    while open_set:
        # Pop the state with the lowest f_score
        _, g, state, path = heappop(open_set)
        x, y, heading = state

        # Goal check - we only care about position, not final heading
        # Return the path if the goal is reached
        if (x, y) == goal:
            # Record successful search
            search_time_ms = (time.perf_counter() - search_start_time) * 1000
            get_metrics().record_astar_search(
                nodes_expanded=nodes_expanded,
                path_length=len(path),
                found_path=True,
                search_time_ms=search_time_ms,
                start=start,
                goal=goal,
                search_type=search_type,
            )
            return path

        # Skip if the state has been visited
        if state in visited:
            continue
        visited.add(state)
        nodes_expanded += 1

        # Expand 4-connected neighbors
        for neighbor in kb.neighbors4(x, y):
            nx, ny = neighbor

            # Calculate which direction we need to face to move there
            dx, dy = nx - x, ny - y
            required_heading = DELTA_TO_HEADING.get((dx, dy))
            if required_heading is None:
                continue

            neighbor_state = (nx, ny, required_heading)
            if neighbor_state in visited:
                continue

            # Check if we can actually move there
            if not kb.is_traversable(nx, ny, allow_unknown=allow_unknown):
                if neighbor != goal:  # Can move to goal even if occupied
                    continue

            # Cost calculation: 1 for the move itself, plus turn penalty
            # turns_between returns 0, 1, or 2 (max 180 degrees = 2 turns)
            num_turns = turns_between(heading, required_heading)
            edge_cost = 1.0 + turn_cost * num_turns

            new_g = g + edge_cost
            # Manhattan distance is admissible heuristic for 4-connected grid (used as f_score)
            new_f = new_g + manhattan_distance(neighbor, goal)
            new_path = path + [neighbor]
            # Push the neighbor state with the new f_score
            heappush(open_set, (new_f, new_g, neighbor_state, new_path))

    # Record failed search
    search_time_ms = (time.perf_counter() - search_start_time) * 1000
    get_metrics().record_astar_search(
        nodes_expanded=nodes_expanded,
        path_length=0,
        found_path=False,
        search_time_ms=search_time_ms,
        start=start,
        goal=goal,
        search_type=search_type,
    )
    return None  # No path found


def path_to_actions(
    path: list[tuple[int, int]],
    start_heading: int,
) -> list[str]:
    """
    Convert a cell path to a list of actions.

    Args:
        path: List of (x, y) cells (from A*)
        start_heading: Initial heading (NORTH/EAST/SOUTH/WEST)

    Returns:
        List of action strings: 'turn_left', 'turn_right', 'move_forward'
    """
    if len(path) < 2:
        return []

    actions: list[str] = []
    current_heading = start_heading

    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]

        target_heading = DELTA_TO_HEADING.get((dx, dy))
        if target_heading is None:
            raise ValueError(f"Invalid move delta: ({dx}, {dy})")

        # Turn to face target heading (shortest path)
        while current_heading != target_heading:
            # Compute turn direction: +1 = right, -1 = left (mod 4)
            diff = (target_heading - current_heading) % 4
            if diff == 1:
                actions.append("turn_right")
                current_heading = (current_heading + 1) % 4
            elif diff == 3:
                actions.append("turn_left")
                current_heading = (current_heading - 1) % 4
            elif diff == 2:
                # 180° turn - pick either direction
                actions.append("turn_right")
                current_heading = (current_heading + 1) % 4
            else:
                break  # Already facing correct direction

        actions.append("move_forward")

    return actions


def find_nearest_frontier(
    kb: KnowledgeBase,
) -> Optional[tuple[int, int]]:
    """
    Find the nearest frontier cell (UNKNOWN adjacent to FREE) from robot position.

    Uses BFS for simplicity (could use A* distance for better accuracy).

    Returns:
        (x, y) of nearest frontier, or None if no frontiers exist.
    """
    if kb.robot is None:
        return None

    frontiers = kb.frontiers()
    if not frontiers:
        return None

    robot_pos = (kb.robot_x, kb.robot_y)

    # Find closest by Manhattan distance
    nearest = min(frontiers, key=lambda f: manhattan_distance(robot_pos, f))
    return nearest


def plan_exploration_step(
    kb: KnowledgeBase,
    *,
    turn_cost: float = 0.5,
) -> Optional[list[str]]:
    """
    Plan actions to reach the nearest frontier cell.

    Args:
        kb: KnowledgeBase with robot pose and occupancy
        turn_cost: penalty per 90° turn (default 0.5 = half a cell cost per turn)

    Returns:
        List of actions to execute, or None if no reachable frontier.
    """
    if kb.robot is None:
        return None

    robot_pos = (kb.robot_x, kb.robot_y)

    # Try frontiers in increasing heuristic distance until we find a reachable one.
    frontiers = list(kb.frontiers())
    if not frontiers:
        return None

    frontiers.sort(key=lambda f: manhattan_distance(robot_pos, f))

    for frontier in frontiers:
        # Move TO a FREE cell adjacent to the frontier (not into UNKNOWN)
        approaches = [
            (nx, ny) for (nx, ny) in kb.neighbors4(*frontier) if kb.is_free(nx, ny)
        ]
        if not approaches:
            continue
        approaches.sort(key=lambda p: manhattan_distance(robot_pos, p))

        for approach in approaches:
            path = astar(
                kb, robot_pos, approach, turn_cost=turn_cost, search_type="exploration"
            )
            if path is not None:
                return path_to_actions(path, kb.robot_heading)

    return None


# =============================================================================
# SOKOBAN-STYLE BOX DELIVERY PLANNING
# =============================================================================


def plan_box_delivery(
    kb: KnowledgeBase,
    *,
    turn_cost: float = 0.5,
    max_states: int = 50000,
) -> Optional[list[str]]:
    """
    Plan complete action sequence to deliver box to goal using A*.

    This is Sokoban-style planning: we search over combined robot+box state
    to find optimal sequence of moves. The key insight is that pushing is
    just a special case of moving forward - when you move into the box cell,
    both the robot and the box move in that direction.

    Input:
        -kb: KnowledgeBase with robot/box/goal positions
        -turn_cost: penalty per 90 degree turn (0.5 = half cell cost)
        -max_states: search budget to prevent infinite loops
    Output:
        -actions: List of ['turn_left', 'turn_right', 'move_forward'],
                  or None if no solution
    """
    search_start_time = time.perf_counter()

    if kb.robot is None or kb.box is None or kb.goal is None:
        return None

    goal_pos = kb.goal
    start_box = kb.box
    start_pos = (kb.robot_x, kb.robot_y)

    if start_box == goal_pos:  # Already done
        search_time_ms = (time.perf_counter() - search_start_time) * 1000
        get_metrics().record_astar_search(
            nodes_expanded=0,
            path_length=0,
            found_path=True,
            search_time_ms=search_time_ms,
            start=start_pos,
            goal=goal_pos,
            search_type="sokoban",
        )
        return []

    # 5D state: robot position + heading + box position
    # This captures everything needed to determine valid moves
    start_state = (kb.robot_x, kb.robot_y, kb.robot_heading, start_box[0], start_box[1])

    def heuristic(state: tuple[int, int, int, int, int]) -> float:
        """
        Manhattan distance from box to goal.
        This is admissible because box must move at least this many cells.
        """
        # Manhattan distance from box to goal
        _, _, _, bx, by = state
        return manhattan_distance((bx, by), goal_pos)

    def static_cell_state(x: int, y: int) -> int:
        """
        Occupancy state for *static* obstacles only.

        The KB marks the (initial) box cell as OCC, but the box is a dynamic object
        tracked in the planner state. Treat that specific cell as FREE here so we
        don't permanently block it during planning.
        """
        if kb.box is not None and (x, y) == kb.box:
            return FREE
        return kb.get_cell(x, y)

    def is_cell_free(x: int, y: int, box_pos: tuple[int, int]) -> bool:
        """
        I use optimistic planning - UNKNOWN cells are allowed for robot
        movement. This prevents the planner from getting stuck on partially
        explored maps. If we hit an obstacle, we just replan.
        """
        if (x, y) == box_pos:
            return False  # Can't walk through the box
        state = static_cell_state(x, y)
        return state == FREE or state == UNKNOWN

    def is_cell_empty_for_box(x: int, y: int) -> bool:
        """Check if cell is valid for box to move into (not wall, not goal-blocking)."""
        # Box can go anywhere that's not an obstacle
        state = kb.get_cell(x, y)
        return state != OCC

    open_set: list[tuple[float, float, tuple[int, int, int, int, int], list[str]]] = []
    # Push the start state with the heuristic value
    heappush(open_set, (heuristic(start_state), 0.0, start_state, []))
    # Track visited states
    visited: set[tuple[int, int, int, int, int]] = set()
    # Track number of states explored
    states_explored = 0

    # Main loop
    while open_set and states_explored < max_states:
        # Pop the state with the lowest f_score
        _, g, state, actions = heappop(open_set)
        # Unpack the state
        rx, ry, heading, bx, by = state
        states_explored += 1

        # Goal - box at goal position (robot position doesn't matter)
        if (bx, by) == goal_pos:
            # Record successful search
            search_time_ms = (time.perf_counter() - search_start_time) * 1000
            get_metrics().record_astar_search(
                nodes_expanded=states_explored,
                path_length=len(actions),
                found_path=True,
                search_time_ms=search_time_ms,
                start=start_pos,
                goal=goal_pos,
                search_type="sokoban",
            )
            return actions

        if state in visited:
            continue
        visited.add(state)

        # Action 1: turn_left
        new_heading = (heading - 1) % 4
        new_state = (rx, ry, new_heading, bx, by)
        if new_state not in visited:
            new_g = g + turn_cost
            new_f = new_g + heuristic(new_state)
            heappush(open_set, (new_f, new_g, new_state, actions + ["turn_left"]))

        # Action 2: turn_right
        new_heading = (heading + 1) % 4
        new_state = (rx, ry, new_heading, bx, by)
        if new_state not in visited:
            new_g = g + turn_cost
            new_f = new_g + heuristic(new_state)
            heappush(open_set, (new_f, new_g, new_state, actions + ["turn_right"]))

        # Action 3: move_forward
        dx, dy = HEADING_TO_DELTA[heading]
        ahead_x, ahead_y = rx + dx, ry + dy

        # Case A: Moving into empty cell (not the box)
        if (ahead_x, ahead_y) != (bx, by):
            if is_cell_free(ahead_x, ahead_y, (bx, by)):
                new_state = (ahead_x, ahead_y, heading, bx, by)
                if new_state not in visited:
                    new_g = g + 1.0
                    new_f = new_g + heuristic(new_state)
                    heappush(
                        open_set, (new_f, new_g, new_state, actions + ["move_forward"])
                    )

        # Case B: Pushing the box
        elif (ahead_x, ahead_y) == (bx, by):
            # Box would move to cell beyond it
            box_dest_x, box_dest_y = bx + dx, by + dy

            # Check if box destination is valid (must be FREE, not UNKNOWN)
            # Optimistic: allow pushing into UNKNOWN cells (not just FREE)
            # Trust replanning to handle unexpected obstacles discovered en route
            # This reduces exploration time while single-wall yaw correction keeps us localized
            box_dest_state = static_cell_state(box_dest_x, box_dest_y)
            if box_dest_state != OCC:  # FREE or UNKNOWN both OK
                # Push is valid: robot moves to box position, box moves forward
                new_state = (ahead_x, ahead_y, heading, box_dest_x, box_dest_y)
                if new_state not in visited:
                    # Pushing costs 1 (same as moving)
                    new_g = g + 1.0
                    new_f = new_g + heuristic(new_state)
                    heappush(
                        open_set, (new_f, new_g, new_state, actions + ["move_forward"])
                    )

    # Record failed search
    search_time_ms = (time.perf_counter() - search_start_time) * 1000
    get_metrics().record_astar_search(
        nodes_expanded=states_explored,
        path_length=0,
        found_path=False,
        search_time_ms=search_time_ms,
        start=start_pos,
        goal=goal_pos,
        search_type="sokoban",
    )
    return None  # No solution found within state limit
