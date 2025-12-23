"""
Inspection realism modeling for utility pole inspections.

Models missed defects, inspector bias, inspection intervals, and probability
of detection by inspection method. This increases credibility with field teams
by accounting for real-world inspection limitations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InspectionMethod(Enum):
    """Available inspection methods."""
    VISUAL = "visual"
    SOUNDING = "sounding"  # Hammer sounding
    BORING = "boring"  # Drill boring
    RESISTOGRAPH = "resistograph"  # Resistograph drilling
    SONIC_TESTING = "sonic_testing"  # Sonic/ultrasonic testing
    LIDAR = "lidar"  # LiDAR scanning
    MOISTURE_METER = "moisture_meter"
    GROUND_PENETRATING_RADAR = "ground_penetrating_radar"
    LOAD_TESTING = "load_testing"
    STRUCTURAL_ENGINEERING_REVIEW = "structural_engineering_review"


class DefectType(Enum):
    """Types of defects that can be detected."""
    SURFACE_DECAY = "surface_decay"
    INTERNAL_DECAY = "internal_decay"
    GROUNDLINE_DECAY = "groundline_decay"
    CRACK = "crack"
    SPLIT = "split"
    CHECK = "check"  # Longitudinal separation in wood
    HARDWARE_FAILURE = "hardware_failure"
    CORROSION = "corrosion"
    FOUNDATION_ISSUE = "foundation_issue"
    LEAN = "lean"
    OVERLOAD = "overload"


@dataclass
class InspectionCapability:
    """Capability of an inspection method to detect defects."""
    method: InspectionMethod
    detection_probability: float  # Base probability (0.0 to 1.0)
    false_positive_rate: float  # Rate of false positives
    depth_penetration_m: float  # How deep it can detect
    cost_per_pole: float  # Estimated cost in dollars
    time_per_pole_minutes: float  # Time required per pole
    skill_level_required: str  # "basic", "intermediate", "advanced", "expert"


@dataclass
class InspectorProfile:
    """Profile of an inspector with skill and bias characteristics."""
    inspector_id: str
    experience_years: float
    certification_level: str  # "certified", "senior", "expert"
    specialization: Optional[str] = None  # "wood", "concrete", "steel", "composite"
    
    # Bias factors (1.0 = no bias, >1.0 = over-detection, <1.0 = under-detection)
    detection_bias: float = 1.0  # Overall detection rate modifier
    severity_bias: float = 1.0  # Tendency to over/under-estimate severity
    
    # Skill factors (0.0 to 1.0)
    visual_skill: float = 0.8
    equipment_skill: float = 0.7
    documentation_skill: float = 0.75
    
    # Consistency (lower = more variable)
    consistency: float = 0.85  # 0.0 to 1.0


@dataclass
class Defect:
    """A defect that may or may not be detected."""
    defect_type: DefectType
    severity: float  # 0.0 (minor) to 1.0 (critical)
    location: str  # "groundline", "mid_pole", "top", etc.
    depth_cm: float  # How deep the defect is
    actual_present: bool = True  # Whether defect actually exists
    detected: bool = False  # Whether it was detected
    detection_method: Optional[InspectionMethod] = None
    detection_confidence: Optional[float] = None


@dataclass
class InspectionResult:
    """Results of an inspection with realism modeling."""
    pole_id: str
    inspection_date: datetime
    inspector: InspectorProfile
    methods_used: List[InspectionMethod]
    
    # Defects found
    defects_detected: List[Defect]
    
    # Quality metrics
    detection_rate: float  # Fraction of actual defects detected
    false_positive_rate: float  # Fraction of non-defects flagged
    inspection_confidence: float  # Overall confidence in results
    
    # Missed defects (for analysis)
    defects_missed: List[Defect] = field(default_factory=list)
    
    # Inspection metadata
    inspection_conditions: Optional[Dict] = None  # weather, lighting, etc.
    time_spent_minutes: Optional[float] = None


class InspectionRealismModel:
    """Model for realistic inspection outcomes with biases and limitations."""
    
    def __init__(self):
        """Initialize inspection realism model."""
        
        # Define detection capabilities by method and defect type
        # Format: (defect_type, method) -> detection_probability
        self.detection_matrix = self._initialize_detection_matrix()
        
        # Define inspection method capabilities
        self.method_capabilities = self._initialize_method_capabilities()
        
        # Environmental factors affecting inspection
        self.environmental_factors = {
            'poor_lighting': 0.15,  # Reduction in visual detection
            'rain': 0.25,  # Reduction in all outdoor methods
            'wind': 0.10,  # Reduction in precision methods
            'vegetation_obstruction': 0.30,  # Reduction in visual/groundline
            'access_difficulty': 0.20,  # Reduction if hard to access
        }
    
    def simulate_inspection(self, pole_id: str, 
                           actual_defects: List[Defect],
                           inspector: InspectorProfile,
                           methods: List[InspectionMethod],
                           inspection_date: datetime = None,
                           environmental_conditions: Optional[Dict] = None) -> InspectionResult:
        """
        Simulate a realistic inspection with detection probabilities.
        
        Args:
            pole_id: ID of the pole being inspected
            actual_defects: List of defects that actually exist
            inspector: Inspector profile with skill/bias
            methods: Inspection methods to use
            inspection_date: Date of inspection
            environmental_conditions: Environmental factors affecting inspection
            
        Returns:
            InspectionResult with detected and missed defects
        """
        inspection_date = inspection_date or datetime.now()
        environmental_conditions = environmental_conditions or {}
        
        detected_defects = []
        missed_defects = []
        
        # Simulate detection for each actual defect
        for defect in actual_defects:
            detected = False
            detection_method = None
            detection_confidence = 0.0
            
            # Try each inspection method
            for method in methods:
                # Calculate detection probability for this method/defect combination
                base_prob = self._get_detection_probability(defect, method)
                
                # Apply inspector skill
                skill_modifier = self._get_skill_modifier(inspector, method)
                
                # Apply environmental factors
                env_modifier = self._calculate_environmental_modifier(
                    method, environmental_conditions
                )
                
                # Apply inspector bias
                bias_modifier = inspector.detection_bias
                
                # Apply consistency (random variation)
                consistency_factor = np.random.normal(
                    inspector.consistency, 
                    0.1 * (1 - inspector.consistency)
                )
                consistency_factor = np.clip(consistency_factor, 0.0, 1.0)
                
                # Calculate final detection probability
                detection_prob = (
                    base_prob * 
                    skill_modifier * 
                    env_modifier * 
                    bias_modifier * 
                    consistency_factor
                )
                detection_prob = np.clip(detection_prob, 0.0, 1.0)
                
                # Determine if defect is detected
                if np.random.random() < detection_prob:
                    detected = True
                    detection_method = method
                    detection_confidence = detection_prob
                    break  # Defect detected, no need to try other methods
            
            # Create defect copy with detection status
            defect_copy = Defect(
                defect_type=defect.defect_type,
                severity=defect.severity * inspector.severity_bias,  # Apply severity bias
                location=defect.location,
                depth_cm=defect.depth_cm,
                actual_present=defect.actual_present,
                detected=detected,
                detection_method=detection_method,
                detection_confidence=detection_confidence,
            )
            
            if detected:
                detected_defects.append(defect_copy)
            else:
                missed_defects.append(defect_copy)
        
        # Simulate false positives (defects reported that don't exist)
        false_positives = self._simulate_false_positives(
            methods, inspector, environmental_conditions, len(actual_defects)
        )
        detected_defects.extend(false_positives)
        
        # Calculate quality metrics
        detection_rate = len([d for d in detected_defects if d.actual_present]) / max(len(actual_defects), 1)
        false_positive_rate = len(false_positives) / max(len(detected_defects), 1)
        
        # Overall confidence based on methods used and inspector skill
        inspection_confidence = self._calculate_inspection_confidence(
            methods, inspector, environmental_conditions
        )
        
        # Estimate time spent
        time_spent = sum([
            self.method_capabilities[method].time_per_pole_minutes 
            for method in methods
        ])
        
        return InspectionResult(
            pole_id=pole_id,
            inspection_date=inspection_date,
            inspector=inspector,
            methods_used=methods,
            defects_detected=detected_defects,
            defects_missed=missed_defects,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            inspection_confidence=inspection_confidence,
            inspection_conditions=environmental_conditions,
            time_spent_minutes=time_spent,
        )
    
    def calculate_optimal_inspection_interval(self, 
                                            pole_risk_score: float,
                                            defect_growth_rate: float = 0.1) -> int:
        """
        Calculate optimal inspection interval based on risk and defect growth.
        
        Args:
            pole_risk_score: Overall risk score (0.0 to 1.0)
            defect_growth_rate: How fast defects worsen per year (0.0 to 1.0)
            
        Returns:
            Optimal interval in days
        """
        # Base interval: 1 year for low risk, 6 months for high risk
        base_interval_days = 365 - (pole_risk_score * 183)
        
        # Adjust for defect growth rate
        # Faster growth = more frequent inspections needed
        growth_adjustment = 1.0 - (defect_growth_rate * 0.5)
        adjusted_interval = base_interval_days * growth_adjustment
        
        # Round to common intervals (30, 60, 90, 180, 365 days)
        intervals = [30, 60, 90, 180, 365]
        optimal = min(intervals, key=lambda x: abs(x - adjusted_interval))
        
        return optimal
    
    def recommend_inspection_methods(self, 
                                    defect_types: List[DefectType],
                                    budget_per_pole: float = 500.0) -> List[InspectionMethod]:
        """
        Recommend inspection methods based on defect types and budget.
        
        Args:
            defect_types: Types of defects to look for
            budget_per_pole: Available budget per pole in dollars
            
        Returns:
            List of recommended inspection methods
        """
        # Always start with visual (cheapest)
        recommended = [InspectionMethod.VISUAL]
        remaining_budget = budget_per_pole - self.method_capabilities[InspectionMethod.VISUAL].cost_per_pole
        
        # Sort methods by effectiveness for these defect types
        method_scores = {}
        for method in InspectionMethod:
            if method == InspectionMethod.VISUAL:
                continue
            
            # Calculate effectiveness score for this method
            score = 0.0
            cost = self.method_capabilities[method].cost_per_pole
            
            for defect_type in defect_types:
                base_prob = self.detection_matrix.get((defect_type, method), 0.0)
                score += base_prob
            
            # Score per dollar
            if cost > 0:
                method_scores[method] = score / cost
        
        # Add methods by score until budget exhausted
        for method, score_per_dollar in sorted(method_scores.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True):
            cost = self.method_capabilities[method].cost_per_pole
            if cost <= remaining_budget:
                recommended.append(method)
                remaining_budget -= cost
        
        return recommended
    
    def _get_detection_probability(self, defect: Defect, method: InspectionMethod) -> float:
        """Get base detection probability for defect/method combination."""
        return self.detection_matrix.get((defect.defect_type, method), 0.0)
    
    def _get_skill_modifier(self, inspector: InspectorProfile, method: InspectionMethod) -> float:
        """Get skill modifier based on inspector capabilities."""
        if method == InspectionMethod.VISUAL:
            return inspector.visual_skill
        elif method in [InspectionMethod.RESISTOGRAPH, 
                       InspectionMethod.SONIC_TESTING,
                       InspectionMethod.BORING]:
            return inspector.equipment_skill
        else:
            return (inspector.visual_skill + inspector.equipment_skill) / 2.0
    
    def _calculate_environmental_modifier(self, 
                                        method: InspectionMethod,
                                        conditions: Dict) -> float:
        """Calculate environmental impact on detection."""
        modifier = 1.0
        
        # Visual methods most affected by lighting
        if method == InspectionMethod.VISUAL:
            if conditions.get('poor_lighting'):
                modifier *= (1.0 - self.environmental_factors['poor_lighting'])
            if conditions.get('rain'):
                modifier *= (1.0 - self.environmental_factors['rain'])
        
        # Groundline methods affected by vegetation
        if method in [InspectionMethod.VISUAL, InspectionMethod.SOUNDING]:
            if conditions.get('vegetation_obstruction'):
                modifier *= (1.0 - self.environmental_factors['vegetation_obstruction'])
        
        # All methods affected by access
        if conditions.get('access_difficulty'):
            modifier *= (1.0 - self.environmental_factors['access_difficulty'])
        
        return modifier
    
    def _simulate_false_positives(self, 
                                  methods: List[InspectionMethod],
                                  inspector: InspectorProfile,
                                  conditions: Dict,
                                  num_actual_defects: int) -> List[Defect]:
        """Simulate false positive detections."""
        false_positives = []
        
        for method in methods:
            capability = self.method_capabilities[method]
            base_fp_rate = capability.false_positive_rate
            
            # Adjust for inspector skill (better inspectors = fewer false positives)
            adjusted_fp_rate = base_fp_rate * (1.0 - inspector.equipment_skill * 0.5)
            
            # Some methods have higher false positive rates
            if np.random.random() < adjusted_fp_rate:
                # Create a false positive defect
                false_positives.append(Defect(
                    defect_type=DefectType.SURFACE_DECAY,  # Common false positive
                    severity=np.random.uniform(0.2, 0.5),  # Usually low severity
                    location="groundline",
                    depth_cm=0.0,
                    actual_present=False,
                    detected=True,
                    detection_method=method,
                    detection_confidence=adjusted_fp_rate,
                ))
        
        return false_positives
    
    def _calculate_inspection_confidence(self,
                                       methods: List[InspectionMethod],
                                       inspector: InspectorProfile,
                                       conditions: Dict) -> float:
        """Calculate overall confidence in inspection results."""
        # Base confidence from methods used
        method_confidence = np.mean([
            self.method_capabilities[m].detection_probability 
            for m in methods
        ])
        
        # Inspector skill contribution
        inspector_confidence = (
            inspector.visual_skill * 0.4 + 
            inspector.equipment_skill * 0.4 + 
            inspector.consistency * 0.2
        )
        
        # Environmental impact
        env_penalty = sum([
            self.environmental_factors.get(k, 0) * v 
            for k, v in conditions.items() 
            if isinstance(v, bool) and v
        ]) / max(len(conditions), 1)
        
        overall_confidence = (
            method_confidence * 0.5 + 
            inspector_confidence * 0.5
        ) * (1.0 - env_penalty)
        
        return np.clip(overall_confidence, 0.0, 1.0)
    
    def _initialize_detection_matrix(self) -> Dict[Tuple[DefectType, InspectionMethod], float]:
        """Initialize detection probability matrix."""
        matrix = {}
        
        # Visual inspection
        matrix[(DefectType.SURFACE_DECAY, InspectionMethod.VISUAL)] = 0.85
        matrix[(DefectType.CRACK, InspectionMethod.VISUAL)] = 0.90
        matrix[(DefectType.SPLIT, InspectionMethod.VISUAL)] = 0.95
        matrix[(DefectType.LEAN, InspectionMethod.VISUAL)] = 1.0
        matrix[(DefectType.HARDWARE_FAILURE, InspectionMethod.VISUAL)] = 0.80
        matrix[(DefectType.CORROSION, InspectionMethod.VISUAL)] = 0.70
        
        # Sounding (hammer test)
        matrix[(DefectType.GROUNDLINE_DECAY, InspectionMethod.SOUNDING)] = 0.70
        matrix[(DefectType.SURFACE_DECAY, InspectionMethod.SOUNDING)] = 0.60
        matrix[(DefectType.INTERNAL_DECAY, InspectionMethod.SOUNDING)] = 0.50
        
        # Boring
        matrix[(DefectType.GROUNDLINE_DECAY, InspectionMethod.BORING)] = 0.95
        matrix[(DefectType.INTERNAL_DECAY, InspectionMethod.BORING)] = 0.90
        
        # Resistograph
        matrix[(DefectType.GROUNDLINE_DECAY, InspectionMethod.RESISTOGRAPH)] = 0.98
        matrix[(DefectType.INTERNAL_DECAY, InspectionMethod.RESISTOGRAPH)] = 0.95
        matrix[(DefectType.SURFACE_DECAY, InspectionMethod.RESISTOGRAPH)] = 0.85
        
        # Sonic testing
        matrix[(DefectType.INTERNAL_DECAY, InspectionMethod.SONIC_TESTING)] = 0.90
        matrix[(DefectType.CRACK, InspectionMethod.SONIC_TESTING)] = 0.85
        
        # Moisture meter
        matrix[(DefectType.GROUNDLINE_DECAY, InspectionMethod.MOISTURE_METER)] = 0.75
        matrix[(DefectType.SURFACE_DECAY, InspectionMethod.MOISTURE_METER)] = 0.70
        
        # LiDAR
        matrix[(DefectType.LEAN, InspectionMethod.LIDAR)] = 0.98
        matrix[(DefectType.CRACK, InspectionMethod.LIDAR)] = 0.80
        
        # Load testing
        matrix[(DefectType.OVERLOAD, InspectionMethod.LOAD_TESTING)] = 1.0
        
        # Structural engineering review
        matrix[(DefectType.OVERLOAD, InspectionMethod.STRUCTURAL_ENGINEERING_REVIEW)] = 0.95
        matrix[(DefectType.CRACK, InspectionMethod.STRUCTURAL_ENGINEERING_REVIEW)] = 0.90
        matrix[(DefectType.FOUNDATION_ISSUE, InspectionMethod.STRUCTURAL_ENGINEERING_REVIEW)] = 0.85
        
        # Ground penetrating radar
        matrix[(DefectType.FOUNDATION_ISSUE, InspectionMethod.GROUND_PENETRATING_RADAR)] = 0.85
        
        return matrix
    
    def _initialize_method_capabilities(self) -> Dict[InspectionMethod, InspectionCapability]:
        """Initialize inspection method capabilities."""
        capabilities = {}
        
        capabilities[InspectionMethod.VISUAL] = InspectionCapability(
            method=InspectionMethod.VISUAL,
            detection_probability=0.70,
            false_positive_rate=0.15,
            depth_penetration_m=0.0,  # Surface only
            cost_per_pole=50.0,
            time_per_pole_minutes=15.0,
            skill_level_required="basic",
        )
        
        capabilities[InspectionMethod.SOUNDING] = InspectionCapability(
            method=InspectionMethod.SOUNDING,
            detection_probability=0.65,
            false_positive_rate=0.20,
            depth_penetration_m=0.1,
            cost_per_pole=100.0,
            time_per_pole_minutes=20.0,
            skill_level_required="intermediate",
        )
        
        capabilities[InspectionMethod.BORING] = InspectionCapability(
            method=InspectionMethod.BORING,
            detection_probability=0.90,
            false_positive_rate=0.05,
            depth_penetration_m=0.5,
            cost_per_pole=300.0,
            time_per_pole_minutes=45.0,
            skill_level_required="advanced",
        )
        
        capabilities[InspectionMethod.RESISTOGRAPH] = InspectionCapability(
            method=InspectionMethod.RESISTOGRAPH,
            detection_probability=0.95,
            false_positive_rate=0.03,
            depth_penetration_m=1.0,
            cost_per_pole=400.0,
            time_per_pole_minutes=60.0,
            skill_level_required="advanced",
        )
        
        capabilities[InspectionMethod.SONIC_TESTING] = InspectionCapability(
            method=InspectionMethod.SONIC_TESTING,
            detection_probability=0.85,
            false_positive_rate=0.10,
            depth_penetration_m=2.0,
            cost_per_pole=500.0,
            time_per_pole_minutes=90.0,
            skill_level_required="expert",
        )
        
        capabilities[InspectionMethod.LIDAR] = InspectionCapability(
            method=InspectionMethod.LIDAR,
            detection_probability=0.80,
            false_positive_rate=0.08,
            depth_penetration_m=0.0,  # Surface geometry
            cost_per_pole=200.0,
            time_per_pole_minutes=10.0,  # Fast but requires processing
            skill_level_required="intermediate",
        )
        
        capabilities[InspectionMethod.MOISTURE_METER] = InspectionCapability(
            method=InspectionMethod.MOISTURE_METER,
            detection_probability=0.70,
            false_positive_rate=0.15,
            depth_penetration_m=0.2,
            cost_per_pole=75.0,
            time_per_pole_minutes=10.0,
            skill_level_required="basic",
        )
        
        capabilities[InspectionMethod.GROUND_PENETRATING_RADAR] = InspectionCapability(
            method=InspectionMethod.GROUND_PENETRATING_RADAR,
            detection_probability=0.80,
            false_positive_rate=0.12,
            depth_penetration_m=3.0,
            cost_per_pole=600.0,
            time_per_pole_minutes=120.0,
            skill_level_required="expert",
        )
        
        capabilities[InspectionMethod.LOAD_TESTING] = InspectionCapability(
            method=InspectionMethod.LOAD_TESTING,
            detection_probability=0.95,
            false_positive_rate=0.02,
            depth_penetration_m=0.0,  # Structural, not depth-based
            cost_per_pole=1000.0,
            time_per_pole_minutes=180.0,
            skill_level_required="expert",
        )
        
        capabilities[InspectionMethod.STRUCTURAL_ENGINEERING_REVIEW] = InspectionCapability(
            method=InspectionMethod.STRUCTURAL_ENGINEERING_REVIEW,
            detection_probability=0.90,
            false_positive_rate=0.05,
            depth_penetration_m=0.0,  # Analysis-based
            cost_per_pole=800.0,
            time_per_pole_minutes=240.0,  # Requires detailed analysis
            skill_level_required="expert",
        )
        
        return capabilities

