"""
Mission Controller - Executes plans using waypoint navigation.

Converts planning actions to waypoints, then navigates using
continuous pose estimation with sensor fusion:
- Actions → Waypoints (positions + headings)
- Waypoint navigation with pose tracking
- LiDAR recalibration at each waypoint
- Manages exploration and box delivery missions
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import pybullet

import planning
import simulation
from simulation import CELL_SIZE, TIME_STEP
from robot import DIRECTION_VECTORS, ORIENTATION_NAMES
from metrics import get_metrics

if TYPE_CHECKING:
    from robot import RobotController
    from kb import KnowledgeBase


class MissionController:
    """
    Executes high-level missions using planning algorithms.

    Handles:
    - Converting plans to batched movements
    - Reactive control during execution
    - Position/yaw correction at checkpoints
    - Mission state management (exploring, pushing box, etc.)
    """

    def __init__(
        self,
        robot: "RobotController",
        kb: "KnowledgeBase",
        box_id: int,
    ):
        """
        Initialize mission controller.

        Args:
            robot: RobotController for low-level movement
            kb: KnowledgeBase wrapper
            box_id: PyBullet body ID of the box (for detection)
        """
        self.robot = robot
        self.kb = kb
        self.box_id = box_id

    def execute_actions(self, actions: list[str], realtime: bool = True) -> bool:
        """
        Execute a list of planning actions by converting to waypoints.

        Simulates the actions to compute waypoints, then navigates
        using continuous pose estimation.

        Args:
            actions: List of action strings from planning
            realtime: If True, add visualization delays

        Returns:
            True if all actions completed successfully
        """
        if not actions:
            return True

        # Planners return actions (FORWARD/LEFT/RIGHT) rather than coordinates
        # because the Sokoban solver needs explicit state transitions. The conversion
        # overhead here is minimal and keeps the planner interface consistent.

        # Convert actions to waypoints (simulate to find where we end up)
        waypoints = self._actions_to_waypoints(actions)

        for i, (wx, wy, heading) in enumerate(waypoints):
            # Navigate to waypoint
            reached = self.robot.move_to_waypoint(
                wx, wy, target_heading=heading, realtime=realtime
            )
            get_metrics().record_waypoint_result(reached)
            if not reached:
                return False

            # Update KB to match
            self._sync_kb_to_position(wx, wy, heading)

            # Recalibrate at waypoint. move_to_waypoint handles mid-motion corrections,
            # but we do a full recalibration here for strategic checkpoints.
            _ = self.robot.scan_lidar(visualize=realtime, recalibrate=True)

        return True

    def _actions_to_waypoints(self, actions: list[str]) -> list[tuple[int, int, int]]:
        """
        Convert action sequence to waypoints with headings.

        Returns list of (grid_x, grid_y, heading) tuples.
        Only includes waypoints where position changes or heading
        matters (e.g., before a move after turns).
        """
        waypoints = []
        x, y = self.kb.robot_x, self.kb.robot_y
        heading = self.kb.robot_heading

        for action in actions:
            if action == "turn_left":
                heading = (heading - 1) % 4
            elif action == "turn_right":
                heading = (heading + 1) % 4
            elif action == "move_forward":
                dx, dy = DIRECTION_VECTORS[heading]
                x, y = x + dx, y + dy
                waypoints.append((x, y, heading))
            elif action == "move_backward":
                dx, dy = DIRECTION_VECTORS[heading]
                x, y = x - dx, y - dy
                waypoints.append((x, y, heading))

        # Consolidate consecutive waypoints with same heading
        # (turns without moves don't need intermediate waypoints)
        if waypoints:
            consolidated = [waypoints[0]]
            for wp in waypoints[1:]:
                # Only add if position changed
                if wp[:2] != consolidated[-1][:2]:
                    consolidated.append(wp)
                else:
                    # Update heading at same position
                    consolidated[-1] = wp
            return consolidated

        # If no moves, but heading changed (turn-only), add current pos with new heading
        if heading != self.kb.robot_heading:
            return [(x, y, heading)]

        return []

    def _sync_kb_to_position(self, x: int, y: int, heading: int):
        """
        Sync KB robot position to target waypoint position.

        We set the KB to the intended waypoint (planner frame). After this, a LiDAR
        scan with recalibration can refine the robot's internal pose estimate, and
        the scan logic will keep KB pose aligned to that estimate.
        """
        # Sync KB to TARGET waypoint
        self.kb.set_robot(x, y, heading)

    def explore_until_box_found(
        self, realtime: bool = True, max_steps: int = 100
    ) -> bool:
        """
        Explore using frontier-based exploration until box is found and delivery path exists.

        Uses optimistic planning: attempts delivery as soon as box is found, allowing
        pushes into UNKNOWN cells. Replanning handles any obstacles discovered mid-push.

        Args:
            realtime: If True, add visualization delays
            max_steps: Maximum exploration steps before giving up

        Returns:
            True if box was found and delivery path exists, False otherwise
        """

        box_found = False

        for step in range(max_steps):
            # Scan with recalibration, then check for box
            scan_data = self.robot.scan_lidar(visualize=realtime, recalibrate=True)

            if self._check_for_box(scan_data):
                if not box_found:
                    box_found = True

                # Try to plan delivery immediately (optimistic planning)
                delivery_plan = planning.plan_box_delivery(self.kb)
                if delivery_plan is not None:
                    return True

            # Plan next exploration step
            actions = planning.plan_exploration_step(self.kb)

            if actions is None:
                # No more frontiers - do final scan
                scan_data = self.robot.scan_lidar(visualize=realtime, recalibrate=True)
                if self._check_for_box(scan_data):
                    if planning.plan_box_delivery(self.kb) is not None:
                        return True
                return box_found

            if not self.execute_actions(actions, realtime):
                return False

        return box_found

    def _check_for_box(self, scan_data) -> bool:
        """
        Check if the box is visible in the current scan.

        Uses the robot's internal pose estimate (which should be recalibrated
        before calling this) to calculate box position.
        """
        for angle, distance, obj_id in scan_data:
            # Uses PyBullet body ID for box identification. A real robot would
            # classify objects by size/shape from LiDAR returns (box ~0.5m vs walls).
            if obj_id == self.box_id:
                # Use INTERNAL pose (recalibrated) - not PyBullet ground truth
                pose = self.robot.get_pose()
                robot_x = pose["x"]
                robot_y = pose["y"]
                robot_yaw = pose["yaw"]

                # Calculate box hit position using trig
                abs_angle = robot_yaw + angle
                hit_x = robot_x + distance * math.cos(abs_angle)
                hit_y = robot_y + distance * math.sin(abs_angle)

                # Convert to internal grid
                box_grid_x = int(round((hit_x - CELL_SIZE / 2) / CELL_SIZE))
                box_grid_y = int(round((hit_y - CELL_SIZE / 2) / CELL_SIZE))

                if self.kb.box_found:
                    self.kb.move_box(box_grid_x, box_grid_y)
                else:
                    self.kb.set_box(box_grid_x, box_grid_y)
                return True

        return False

    def deliver_box_to_goal(self, realtime: bool = True) -> bool:
        """
        Plan and execute box delivery to goal using Sokoban-style planning.

        Returns:
            True if box successfully delivered, False otherwise
        """

        if not self.kb.box_found:
            return False

        # Sync KB robot position to actual continuous pose before planning
        pose = self.robot.get_pose()
        self.kb.set_robot(pose["grid_x"], pose["grid_y"], self.kb.robot_heading)

        if self.kb.is_box_at_goal():
            return True

        actions = planning.plan_box_delivery(self.kb)

        if actions is None:
            return False

        # Execute the plan, updating box position after each push
        return self._execute_delivery_plan(actions, realtime)

    def _execute_delivery_plan(
        self, actions: list[str], realtime: bool, replan_depth: int = 0
    ) -> bool:
        """
        Execute a delivery plan using sensor-guided navigation.

        Uses only:
        1. Robot's internal pose estimate (from sensors)
        2. KB's belief of box position (from LiDAR detection)
        3. Bump sensor for contact detection during push
        4. LiDAR to re-detect box after each push

        Args:
            actions: List of actions from planner
            realtime: Whether to add visualization delays
            replan_depth: Current recursion depth (for loop prevention)
        """
        MAX_REPLAN_DEPTH = 10

        if actions is None or replan_depth >= MAX_REPLAN_DEPTH:
            return False

        def _replan_delivery(*, reason: str, heading_for_kb: int) -> bool:
            """Replan delivery from current estimated pose and LiDAR box observation."""
            get_metrics().record_replanning(reason)
            if replan_depth >= MAX_REPLAN_DEPTH:
                return False

            # Refresh box estimate
            scan_data = self.robot.scan_lidar(visualize=True, recalibrate=True)
            box_found = self._update_box_from_lidar(scan_data)

            if not box_found:
                # Back up and scan again (may be occluded)
                self.robot.move("backward", realtime)
                scan_data = self.robot.scan_lidar(visualize=True, recalibrate=True)
                self._update_box_from_lidar(scan_data)

            # Sync robot pose from sensors
            pose = self.robot.get_pose()
            self.kb.set_robot(pose["grid_x"], pose["grid_y"], heading_for_kb)

            new_plan = planning.plan_box_delivery(self.kb)
            if new_plan is None:
                return False

            return self._execute_delivery_plan(new_plan, realtime, replan_depth + 1)

        # Execute the plan
        for i, action in enumerate(actions):
            robot_x, robot_y = self.kb.robot_x, self.kb.robot_y
            current_heading = self.kb.robot_heading

            # Check if this move will push the box
            will_push = False
            if action == "move_forward":
                dx, dy = DIRECTION_VECTORS[current_heading]
                ahead = (robot_x + dx, robot_y + dy)
                if ahead == (self.kb.box_x, self.kb.box_y):
                    will_push = True
                    new_box_x = self.kb.box_x + dx
                    new_box_y = self.kb.box_y + dy

            # Execute actions
            if action == "turn_left":
                # Safety: back up if in contact with box
                if self.robot.is_in_contact_with(self.box_id):
                    self.robot.move("backward", realtime)
                    backup_pose = self.robot.get_pose()
                    self.kb.set_robot(
                        backup_pose["grid_x"], backup_pose["grid_y"], current_heading
                    )

                new_heading = (current_heading - 1) % 4
                reached = self.robot.move_to_waypoint(
                    robot_x, robot_y, new_heading, realtime
                )
                if not reached:
                    return _replan_delivery(
                        reason="turn_left failed", heading_for_kb=current_heading
                    )
                self.robot.scan_lidar(visualize=True, recalibrate=True)
                self.kb.set_robot(robot_x, robot_y, new_heading)

            elif action == "turn_right":
                # Safety: back up if in contact with box
                if self.robot.is_in_contact_with(self.box_id):
                    self.robot.move("backward", realtime)
                    backup_pose = self.robot.get_pose()
                    self.kb.set_robot(
                        backup_pose["grid_x"], backup_pose["grid_y"], current_heading
                    )

                new_heading = (current_heading + 1) % 4
                reached = self.robot.move_to_waypoint(
                    robot_x, robot_y, new_heading, realtime
                )
                if not reached:
                    return _replan_delivery(
                        reason="turn_right failed", heading_for_kb=current_heading
                    )
                self.robot.scan_lidar(visualize=True, recalibrate=True)
                self.kb.set_robot(robot_x, robot_y, new_heading)

            elif action == "move_forward":
                dx, dy = DIRECTION_VECTORS[current_heading]
                target_x = robot_x + dx
                target_y = robot_y + dy

                if will_push:
                    # Push with bump sensor feedback
                    pushed = self._push_box_with_verification(
                        target_x,
                        target_y,
                        current_heading,
                        new_box_x,
                        new_box_y,
                        realtime,
                    )
                    if pushed:
                        # Verify push via LiDAR
                        scan_data = self.robot.scan_lidar(
                            visualize=True, recalibrate=True
                        )
                        box_found = self._update_box_from_lidar(scan_data)

                        if not box_found:
                            # Back up and scan again (robot may occlude the box)
                            self.robot.move("backward", realtime)
                            scan_data = self.robot.scan_lidar(
                                visualize=True, recalibrate=True
                            )
                            self._update_box_from_lidar(scan_data)

                        if (self.kb.box_x, self.kb.box_y) != (new_box_x, new_box_y):
                            return _replan_delivery(
                                reason=f"push verification failed",
                                heading_for_kb=current_heading,
                            )
                    else:
                        return _replan_delivery(
                            reason="push failed",
                            heading_for_kb=current_heading,
                        )
                else:
                    # Regular movement (non-push)
                    reached = self.robot.move_to_waypoint(
                        target_x, target_y, current_heading, realtime
                    )
                    if not reached:
                        return _replan_delivery(
                            reason=f"move failed",
                            heading_for_kb=current_heading,
                        )

                    # Verify we reached the right cell
                    pose = self.robot.get_pose()
                    if (pose["grid_x"], pose["grid_y"]) != (target_x, target_y):
                        return _replan_delivery(
                            reason=f"position drift",
                            heading_for_kb=current_heading,
                        )

                self.kb.set_robot(target_x, target_y, current_heading)

            else:
                raise ValueError(f"Unknown delivery action: {action}")

            # Check if done
            if self.kb.is_box_at_goal():
                return True

        # Plan exhausted without reaching goal
        if not self.kb.is_box_at_goal():
            return False

        return True

    def _update_box_from_lidar(self, scan_data) -> bool:
        """
        Re-detect box position using LiDAR scan (realistic sensing).

        Uses the robot's INTERNAL pose estimate (which should be recalibrated
        before calling this) to calculate box position.

        Returns:
            True if box was detected in scan
        """
        for angle, distance, obj_id in scan_data:
            if obj_id == self.box_id:
                # Use INTERNAL pose (recalibrated) - not PyBullet ground truth
                pose = self.robot.get_pose()
                robot_x = pose["x"]
                robot_y = pose["y"]
                robot_yaw = pose["yaw"]

                # Calculate box hit position using trig
                abs_angle = robot_yaw + angle
                hit_x = robot_x + distance * math.cos(abs_angle)
                hit_y = robot_y + distance * math.sin(abs_angle)

                # Convert to internal grid
                box_grid_x_raw = int(round((hit_x - CELL_SIZE / 2) / CELL_SIZE))
                box_grid_y_raw = int(round((hit_y - CELL_SIZE / 2) / CELL_SIZE))
                box_grid_x = box_grid_x_raw
                box_grid_y = box_grid_y_raw

                # Heuristic: if the ray direction is diagonal but rounding produced a
                # 4-neigh (cardinal) adjacent cell, "promote" it to the diagonal cell.
                rx, ry = pose["grid_x"], pose["grid_y"]
                ux, uy = math.cos(abs_angle), math.sin(abs_angle)
                diag = abs(ux) > 0.35 and abs(uy) > 0.35
                manhattan = abs(box_grid_x - rx) + abs(box_grid_y - ry)
                promoted = False
                if diag and manhattan == 1:
                    if box_grid_x == rx:
                        box_grid_x += 1 if ux > 0 else -1
                        promoted = True
                    if box_grid_y == ry:
                        box_grid_y += 1 if uy > 0 else -1
                        promoted = True

                # Update KB
                if self.kb.box_found:
                    self.kb.move_box(box_grid_x, box_grid_y)
                else:
                    self.kb.set_box(box_grid_x, box_grid_y)
                return True
        return False

    def _push_box_with_verification(
        self,
        target_x: int,
        target_y: int,
        heading: int,
        expected_box_x: int,
        expected_box_y: int,
        realtime: bool,
    ) -> bool:
        """
        Push the box by driving into it for ~1 cell distance while in contact.

        1. Drive forward until bump sensor detects contact
        2. KEEP DRIVING while in contact, tracking ROBOT's traveled distance
        3. Stop when robot has traveled ~1 cell while pushing
        4. Use LiDAR afterward to detect where box ended up

        Returns:
            True if push was executed (contact maintained)
        """
        push_distance = CELL_SIZE * 1.0  # Push robot forward ~1 cell
        max_approach_steps = 400  # Steps to reach box
        max_push_steps = 600  # Steps while pushing
        push_speed = 10.0  # Constant driving speed

        dx, dy = DIRECTION_VECTORS[heading]
        target_angle = math.atan2(dy, dx)

        # Phase 1: Drive straight toward box
        self.robot._reset_odometry()
        contact_made = False

        for step in range(max_approach_steps):
            self.robot.update_pose(TIME_STEP)

            if self.robot.is_in_contact_with(self.box_id):
                contact_made = True
                break

            yaw_error = target_angle - self.robot.pose_yaw
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi

            correction = max(-3.0, min(3.0, 4.0 * yaw_error))
            self.robot._set_drive_wheels(push_speed, correction, realtime)

            simulation.step(1)
            if realtime:
                time.sleep(TIME_STEP)

        if not contact_made:
            self.robot._stop()
            return False

        # Phase 2: Keep pushing with heading hold
        push_start_x = self.robot.pose_x
        push_start_y = self.robot.pose_y
        contact_lost_count = 0
        robot_pushed_distance = 0.0

        for step in range(max_push_steps):
            self.robot.update_pose(TIME_STEP)

            robot_pushed_distance = math.sqrt(
                (self.robot.pose_x - push_start_x) ** 2
                + (self.robot.pose_y - push_start_y) ** 2
            )

            in_contact = self.robot.is_in_contact_with(self.box_id)
            if not in_contact:
                contact_lost_count += 1
                if contact_lost_count > 50:
                    break
            else:
                contact_lost_count = 0

            if robot_pushed_distance >= push_distance:
                break

            yaw_error = target_angle - self.robot.pose_yaw
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi

            correction = max(-3.0, min(3.0, 4.0 * yaw_error))
            self.robot._set_drive_wheels(push_speed, correction, realtime)

            simulation.step(1)
            if realtime:
                time.sleep(TIME_STEP)

        # Record push metrics
        box_pos, _ = pybullet.getBasePositionAndOrientation(self.box_id)
        box_grid_x = int(round((box_pos[0] - CELL_SIZE / 2) / CELL_SIZE))
        box_grid_y = int(round((box_pos[1] - CELL_SIZE / 2) / CELL_SIZE))

        push_success = robot_pushed_distance >= push_distance * 0.5
        get_metrics().record_push_attempt(
            success=push_success,
            expected_pos=(expected_box_x, expected_box_y),
            actual_pos=(box_grid_x, box_grid_y),
            distance_pushed=robot_pushed_distance,
            contact_maintained=(contact_lost_count <= 50),
        )

        return push_success

    def run_full_mission(self, realtime: bool = True) -> bool:
        """
        Run the complete mission: explore → find box → deliver to goal.

        Returns:
            True if mission completed successfully
        """
        metrics = get_metrics()
        metrics.start_mission()

        metrics.start_phase("exploration")
        exploration_success = self.explore_until_box_found(realtime)
        metrics.end_phase("exploration")

        if not exploration_success:
            metrics.end_mission(success=False)
            return False

        metrics.start_phase("delivery")
        delivery_success = self.deliver_box_to_goal(realtime)
        metrics.end_phase("delivery")

        if not delivery_success:
            metrics.end_mission(success=False)
            return False

        metrics.end_mission(success=True)
        return True
