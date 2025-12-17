from __future__ import annotations

from math import cos, pi, sin
from typing import Callable, Iterable, Optional

from kb import KnowledgeBase, ORIENTATION_ANGLES


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    # compares angles (e.g., "closest ray to cardinal direction", shortest-turn logic).
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def bresenham_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Grid traversal from (x0,y0) to (x1,y1), inclusive endpoints.
    We will use this to mark the cells that are free along the ray from LiDAR."""
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def integrate_lidar(
    kb: KnowledgeBase,
    scan_data: Iterable[tuple[float, float, int]],
    *,
    cell_size: float,
    max_range: float,
    is_obstacle: Optional[Callable[[int], bool]] = None,
    actual_robot_internal: Optional[tuple[int, int, float]] = None,
) -> None:
    """
    Mapping update rule: integrate LiDAR rays into the KB occupancy grid.

    Args:
        kb: KnowledgeBase instance (must have kb.robot set)
        scan_data: iterable of (angle_rad, distance_m, obj_id) where angle is in robot frame
        cell_size: meters per grid cell
        max_range: LiDAR max range in meters
        is_obstacle: optional predicate on obj_id; if provided, only hits where
            is_obstacle(obj_id) is True are treated as OCC at the ray endpoint.
            Useful for filtering out non-walls (e.g., ignore goal markers).
        actual_robot_internal: optional (grid_x, grid_y, yaw_rad) of actual robot position
            in internal frame. If provided, uses this instead of KB robot position to
            calculate hit locations. This fixes frame offset issues when internal pose
            drifts from actual.
    """
    if kb.robot is None:
        raise ValueError("Robot pose not set")

    # Use actual position if provided, otherwise fall back to KB position
    if actual_robot_internal is not None:
        rx, ry, heading_yaw = actual_robot_internal
    else:
        rx, ry = kb.robot.x, kb.robot.y
        heading_yaw = ORIENTATION_ANGLES[kb.robot.heading]

    for ang, dist_m, obj_id in scan_data:
        # Clamp distance to match range of LiDAR
        dist_m = min(max_range, max(0.0, dist_m))

        # Convert ray direction (robot frame) into world angle using discrete heading.
        world_ang = normalize_angle(heading_yaw + ang)
        dx_m = cos(world_ang) * dist_m
        dy_m = sin(world_ang) * dist_m

        # Convert distance to grid cells
        tx = rx + int(round(dx_m / cell_size))
        ty = ry + int(round(dy_m / cell_size))

        # Mark the cells that are free along the ray from LiDAR
        cells = bresenham_cells(rx, ry, tx, ty)
        if not cells:
            continue

        # Check if the ray hit something - Lidar returns -1 if no hit until max range.
        # IMPORTANT: If we hit a *non-obstacle* object (e.g., the box), we should NOT
        # mark the endpoint as FREE; we only know the space *up to* the hit is free.
        hit = obj_id != -1 and dist_m < max_range * 0.999
        treat_as_obstacle = hit and (is_obstacle(obj_id) if is_obstacle else True)

        # Mark FREE along the ray excluding endpoint (so we can mark endpoint OCC on hits).
        for cx, cy in cells[:-1]:
            kb.mark_free(cx, cy)

        ex, ey = cells[-1]
        if (ex, ey) == (rx, ry):
            continue

        # Endpoint handling:
        # - If we hit an obstacle: endpoint is occupied.
        # - If we did not hit anything (max range): endpoint is free (known clear up to range).
        # - If we hit something but don't treat it as an obstacle: leave endpoint unchanged.
        if treat_as_obstacle:
            kb.mark_occupied(ex, ey)
        elif not hit:
            kb.mark_free(ex, ey)
