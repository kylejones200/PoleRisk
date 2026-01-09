"""
Crew and logistics constraints modeling for utility pole maintenance.

Models truck rolls, travel time, depots, skill mix, and outage windows.
This turns maintenance schedules into executable plans that respect real-world
resource constraints, not just budgets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:
    from geopy.distance import geodesic

    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logger.warning("geopy not available. Install with: pip install geopy")


class CrewType(Enum):
    """Types of maintenance crews."""

    LINE_CREW = "line_crew"  # Power line maintenance
    STRUCTURAL_CREW = "structural_crew"  # Pole inspection/repair
    EMERGENCY_RESPONSE = "emergency_response"  # Emergency repairs
    SPECIALIZED = "specialized"  # Specialized equipment (crane, etc.)


class WorkComplexity(Enum):
    """Work complexity levels."""

    SIMPLE = "simple"  # Quick inspection
    MODERATE = "moderate"  # Standard maintenance
    COMPLEX = "complex"  # Replacement or major repair
    EMERGENCY = "emergency"  # Emergency response


@dataclass
class Depot:
    """Depot/warehouse location for crews and materials."""

    depot_id: str
    name: str
    latitude: float
    longitude: float
    address: str

    # Capacity
    max_crews: int
    current_crews: int = 0
    material_capacity: Dict[str, float] = field(
        default_factory=dict
    )  # material_type -> quantity

    # Operating hours
    operating_hours_start: int = 7  # 7 AM
    operating_hours_end: int = 19  # 7 PM
    timezone: str = "America/Los_Angeles"


@dataclass
class Crew:
    """Crew definition with capabilities and constraints."""

    crew_id: str
    crew_type: CrewType
    depot_id: str

    # Capabilities
    skill_mix: List[str]  # ["pole_inspection", "pole_repair", "pole_replacement"]
    certifications: List[str]  # Required certifications
    equipment: List[str]  # Available equipment

    # Capacity
    max_work_hours_per_day: float = 10.0
    max_work_hours_per_week: float = 50.0

    # Location and status
    current_latitude: Optional[float] = None
    current_longitude: Optional[float] = None
    current_status: str = "available"  # available, assigned, working, off_duty

    # Scheduling constraints
    unavailable_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)

    # Cost
    hourly_rate: float = 150.0
    travel_cost_per_mile: float = 0.50


@dataclass
class WorkTask:
    """Work task with location and requirements."""

    task_id: str
    pole_id: str
    work_type: str  # "inspection", "repair", "replacement"
    work_complexity: WorkComplexity

    # Location
    latitude: float
    longitude: float
    address: str

    # Requirements
    required_skills: List[str]
    required_certifications: List[str]
    required_equipment: List[str]
    estimated_duration_hours: float
    materials_needed: Dict[str, float]  # material_type -> quantity

    # Timing constraints
    earliest_start: datetime
    latest_completion: datetime
    preferred_work_window: Optional[Tuple[datetime, datetime]] = None
    outage_required: bool = False
    outage_customers_affected: int = 0

    # Priority
    priority: str = "medium"  # critical, high, medium, low
    risk_score: float = 0.5


@dataclass
class CrewAssignment:
    """Assignment of crew to work task."""

    task_id: str
    crew_id: str
    depot_id: str

    # Scheduling
    assigned_start_time: datetime
    assigned_end_time: datetime
    travel_time_hours: float
    work_time_hours: float
    total_time_hours: float

    # Cost
    travel_cost: float
    work_cost: float
    total_cost: float

    # Feasibility
    can_complete_on_time: bool
    conflicts: List[str] = field(default_factory=list)  # List of constraint violations


class CrewLogisticsOptimizer:
    """Optimize crew assignments respecting logistics constraints."""

    def __init__(self):
        """Initialize crew logistics optimizer."""
        self.depots: Dict[str, Depot] = {}
        self.crews: Dict[str, Crew] = {}
        self.assignments: List[CrewAssignment] = []

        # Default travel speed (mph)
        self.avg_travel_speed_mph = 30.0

        # Setup time at work site (minutes)
        self.setup_time_minutes = 15.0

        # Default distances
        self.max_travel_distance_miles = 100.0  # Max reasonable travel distance

    def add_depot(self, depot: Depot):
        """Add a depot to the system."""
        self.depots[depot.depot_id] = depot

    def add_crew(self, crew: Crew):
        """Add a crew to the system."""
        self.crews[crew.crew_id] = crew

    def optimize_assignments(
        self,
        tasks: List[WorkTask],
        planning_horizon_days: int = 7,
        start_date: datetime = None,
    ) -> List[CrewAssignment]:
        """
        Optimize crew assignments for a set of work tasks.

        Uses greedy heuristic with constraint checking.
        """
        if start_date is None:
            start_date = datetime.now()

        # Sort tasks by priority and deadline
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (
                self._priority_weight(t.priority),
                t.latest_completion,
                t.risk_score,
            ),
            reverse=True,
        )

        assignments = []
        crew_schedules = {crew_id: [] for crew_id in self.crews.keys()}

        for task in sorted_tasks:
            # Find best crew assignment
            best_assignment = None
            best_score = float("inf")

            for crew_id, crew in self.crews.items():
                if crew.current_status != "available":
                    continue

                # Check if crew has required skills
                if not self._crew_can_handle_task(crew, task):
                    continue

                # Try to assign task to crew
                assignment = self._try_assign_task(
                    crew,
                    task,
                    start_date,
                    planning_horizon_days,
                    crew_schedules[crew_id],
                )

                if assignment and assignment.can_complete_on_time:
                    # Score assignment (lower is better)
                    score = (
                        assignment.total_cost
                        + assignment.travel_time_hours * 100.0  # Penalize travel
                        + len(assignment.conflicts)
                        * 1000.0  # Heavy penalty for conflicts
                    )

                    if score < best_score:
                        best_score = score
                        best_assignment = assignment

            if best_assignment:
                assignments.append(best_assignment)
                crew_schedules[best_assignment.crew_id].append(best_assignment)

        self.assignments = assignments
        return assignments

    def _crew_can_handle_task(self, crew: Crew, task: WorkTask) -> bool:
        """Check if crew has required skills and equipment."""
        # Check skills
        if not all(skill in crew.skill_mix for skill in task.required_skills):
            return False

        # Check certifications
        if not all(
            cert in crew.certifications for cert in task.required_certifications
        ):
            return False

        # Check equipment (if required)
        if task.required_equipment:
            if not all(eq in crew.equipment for eq in task.required_equipment):
                return False

        return True

    def _try_assign_task(
        self,
        crew: Crew,
        task: WorkTask,
        start_date: datetime,
        planning_horizon_days: int,
        existing_assignments: List[CrewAssignment],
    ) -> Optional[CrewAssignment]:
        """Try to assign a task to a crew, checking constraints."""

        # Calculate travel time
        depot = self.depots[crew.depot_id]
        travel_time, travel_distance = self._calculate_travel_time(
            (depot.latitude, depot.longitude),
            (task.latitude, task.longitude),
            existing_assignments,
        )

        # Check travel distance constraint
        if travel_distance > self.max_travel_distance_miles:
            return None

        # Find earliest available time slot
        work_start, work_end = self._find_available_time_slot(
            crew,
            task,
            start_date,
            planning_horizon_days,
            travel_time,
            existing_assignments,
        )

        if work_start is None:
            return None

        # Calculate costs
        travel_cost = travel_distance * crew.travel_cost_per_mile
        work_cost = task.estimated_duration_hours * crew.hourly_rate
        total_cost = travel_cost + work_cost

        # Check constraints
        conflicts = self._check_constraints(
            crew, task, work_start, work_end, existing_assignments
        )

        assignment = CrewAssignment(
            task_id=task.task_id,
            crew_id=crew.crew_id,
            depot_id=crew.depot_id,
            assigned_start_time=work_start,
            assigned_end_time=work_end,
            travel_time_hours=travel_time,
            work_time_hours=task.estimated_duration_hours,
            total_time_hours=travel_time * 2
            + task.estimated_duration_hours,  # Round trip
            travel_cost=travel_cost,
            work_cost=work_cost,
            total_cost=total_cost,
            can_complete_on_time=len(conflicts) == 0,
            conflicts=conflicts,
        )

        return assignment

    def _calculate_travel_time(
        self,
        start_location: Tuple[float, float],
        end_location: Tuple[float, float],
        existing_assignments: List[CrewAssignment] = None,
    ) -> Tuple[float, float]:
        """Calculate travel time and distance between locations."""
        if GEOPY_AVAILABLE:
            # Calculate distance using geodesic distance
            distance_km = geodesic(start_location, end_location).kilometers
            distance_miles = distance_km * 0.621371
        else:
            # Fallback: simple haversine approximation
            import math

            lat1, lon1 = start_location
            lat2, lon2 = end_location
            R = 6371  # Earth radius in km
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(lat1))
                * math.cos(math.radians(lat2))
                * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.asin(math.sqrt(a))
            distance_km = R * c
            distance_miles = distance_km * 0.621371

        # Account for real-world travel (not straight-line)
        # Add 20% overhead for road network
        actual_distance = distance_miles * 1.2

        # Calculate travel time
        travel_hours = actual_distance / self.avg_travel_speed_mph

        return travel_hours, actual_distance

    def _find_available_time_slot(
        self,
        crew: Crew,
        task: WorkTask,
        start_date: datetime,
        planning_horizon_days: int,
        travel_time: float,
        existing_assignments: List[CrewAssignment],
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Find an available time slot for the task."""

        # Sort existing assignments by start time
        sorted_assignments = sorted(
            existing_assignments, key=lambda a: a.assigned_start_time
        )

        current_time = max(start_date, task.earliest_start)
        end_time = current_time + timedelta(days=planning_horizon_days)
        end_time = min(end_time, task.latest_completion)

        # Try each day in the planning horizon
        depot = self.depots[crew.depot_id]
        current_day = current_time.replace(
            hour=depot.operating_hours_start, minute=0, second=0
        )

        while current_day < end_time:
            # Check if within operating hours
            day_start = current_day.replace(hour=depot.operating_hours_start, minute=0)
            day_end = current_day.replace(hour=depot.operating_hours_end, minute=0)

            # Account for travel time
            work_start = day_start + timedelta(hours=travel_time)
            work_end = work_start + timedelta(hours=task.estimated_duration_hours)

            # Check if work fits in day
            if work_end <= day_end:
                # Check if overlaps with existing assignments
                overlaps = False
                for existing in sorted_assignments:
                    if self._time_overlaps(
                        (work_start, work_end),
                        (existing.assigned_start_time, existing.assigned_end_time),
                    ):
                        overlaps = True
                        break

                if not overlaps:
                    # Check crew availability constraints
                    if not self._crew_available_at_time(crew, work_start, work_end):
                        current_day += timedelta(days=1)
                        continue

                    return work_start, work_end

            current_day += timedelta(days=1)

        return None, None

    def _time_overlaps(
        self, slot1: Tuple[datetime, datetime], slot2: Tuple[datetime, datetime]
    ) -> bool:
        """Check if two time slots overlap."""
        return slot1[0] < slot2[1] and slot2[0] < slot1[1]

    def _crew_available_at_time(
        self, crew: Crew, start_time: datetime, end_time: datetime
    ) -> bool:
        """Check if crew is available during the time period."""
        # Check unavailable periods
        for unavail_start, unavail_end in crew.unavailable_periods:
            if self._time_overlaps(
                (start_time, end_time), (unavail_start, unavail_end)
            ):
                return False

        # Check weekly hour limits (simplified)
        # In full implementation, would check week boundaries

        return True

    def _check_constraints(
        self,
        crew: Crew,
        task: WorkTask,
        work_start: datetime,
        work_end: datetime,
        existing_assignments: List[CrewAssignment],
    ) -> List[str]:
        """Check all constraints and return list of violations."""
        conflicts = []

        # Check crew capacity
        hours_today = sum(
            a.work_time_hours
            for a in existing_assignments
            if a.assigned_start_time.date() == work_start.date()
        )
        if hours_today + task.estimated_duration_hours > crew.max_work_hours_per_day:
            conflicts.append(
                f"Exceeds daily hours limit: {hours_today + task.estimated_duration_hours:.1f} > {crew.max_work_hours_per_day}"
            )

        # Check deadline
        if work_end > task.latest_completion:
            conflicts.append(
                f"Exceeds latest completion: {work_end} > {task.latest_completion}"
            )

        # Check outage windows (if required)
        if task.outage_required:
            # In full implementation, would check outage coordination
            pass

        return conflicts

    def _priority_weight(self, priority: str) -> int:
        """Convert priority to numeric weight (higher = more important)."""
        weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return weights.get(priority, 2)

    def calculate_total_cost(self, assignments: List[CrewAssignment] = None) -> float:
        """Calculate total cost of assignments."""
        if assignments is None:
            assignments = self.assignments
        return sum(a.total_cost for a in assignments)

    def calculate_total_travel_time(
        self, assignments: List[CrewAssignment] = None
    ) -> float:
        """Calculate total travel time."""
        if assignments is None:
            assignments = self.assignments
        return sum(a.travel_time_hours * 2 for a in assignments)  # Round trip

    def generate_daily_schedule(self, date: datetime) -> pd.DataFrame:
        """Generate daily schedule for a specific date."""
        daily_assignments = [
            a for a in self.assignments if a.assigned_start_time.date() == date.date()
        ]

        schedule_data = []
        for assignment in daily_assignments:
            crew = self.crews[assignment.crew_id]
            schedule_data.append(
                {
                    "crew_id": assignment.crew_id,
                    "crew_type": crew.crew_type.value,
                    "task_id": assignment.task_id,
                    "start_time": assignment.assigned_start_time,
                    "end_time": assignment.assigned_end_time,
                    "travel_time_hours": assignment.travel_time_hours,
                    "work_time_hours": assignment.work_time_hours,
                    "total_cost": assignment.total_cost,
                    "depot": assignment.depot_id,
                }
            )

        return pd.DataFrame(schedule_data)

    def optimize_outage_windows(
        self, tasks: List[WorkTask]
    ) -> Dict[str, List[datetime]]:
        """Optimize outage windows for tasks requiring outages."""
        outage_tasks = [t for t in tasks if t.outage_required]

        # Group by customer impact area (simplified - in reality would use GIS)
        # Try to batch outages in same area
        outage_schedule = {}

        for task in outage_tasks:
            # Find optimal outage window (typically early morning, low demand)
            # Preferred: 2-6 AM on weekdays
            task_date = task.earliest_start.date()
            outage_window_start = datetime.combine(
                task_date, datetime.min.time().replace(hour=2)
            )
            outage_window_end = datetime.combine(
                task_date, datetime.min.time().replace(hour=6)
            )

            outage_schedule[task.task_id] = [outage_window_start, outage_window_end]

        return outage_schedule
