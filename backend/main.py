from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
import json
from typing import List, Dict, Any, Optional
import math
from collections import deque
from enum import Enum
from datetime import datetime

app = FastAPI()

# This lets your frontend talk to the backend without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class ShotType(Enum):
    OVERHAND = "overhand"
    SIDEARM = "sidearm"
    RISER = "riser"
    UNKNOWN = "unknown"

# ============== PLAYER PROFILE & BREAKDOWN CLASSES ==============

class PlayerBreakdown:
    """Detailed breakdown of a player's performance"""
    def __init__(self, player_id: str = "player_1"):
        self.player_id = player_id
        self.shots_taken = 0
        self.dominant_shot_type = "unknown"
        self.average_form_score = 0
        self.total_form_score = 0
        
        # Detailed metrics storage
        self.elbow_angles = []
        self.layback_angles = []
        self.h2ss_values = []
        self.hand_velocities = []
        self.trunk_rotations = []
        self.arm_extensions = []
        
        # Shot-by-shot breakdown
        self.shot_breakdowns = []
        
        # Strengths and weaknesses
        self.strengths = []
        self.areas_for_improvement = []
        
        # Video metadata
        self.video_duration = 0
        self.frames_processed = 0
        self.analysis_timestamp = datetime.now().isoformat()
    
    def add_shot(self, shot_data: Dict):
        """Add a shot to the player's breakdown"""
        self.shots_taken += 1
        self.shot_breakdowns.append({
            "shot_number": self.shots_taken,
            "timestamp": shot_data.get("timestamp", 0),
            "shot_type": shot_data.get("shot_type", "unknown"),
            "metrics": {
                "load_angle": shot_data.get("load_angle", 0),
                "release_angle": shot_data.get("release_angle", 0),
                "angle_change": shot_data.get("angle_change", 0),
                "shot_speed": shot_data.get("shot_speed", 0),
                "max_layback": shot_data.get("max_layback", 0),
                "max_h2ss": shot_data.get("max_h2ss", 0),
                "max_hand_velocity": shot_data.get("max_hand_velocity", 0),
                "trunk_rotation": shot_data.get("trunk_rotation", 0),
                "arm_extension_style": shot_data.get("arm_extension_style", "unknown"),
                "arm_path_type": shot_data.get("arm_path_type", "unknown")
            },
            "form_score": shot_data.get("form_score", 0),
            "feedback": shot_data.get("feedback", [])
        })
        
        # Update aggregates
        self.total_form_score += shot_data.get("form_score", 0)
        self.average_form_score = self.total_form_score / self.shots_taken
        
        # Track dominant shot type
        type_counts = {}
        for shot in self.shot_breakdowns:
            st = shot["shot_type"]
            type_counts[st] = type_counts.get(st, 0) + 1
        self.dominant_shot_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"
    
    def add_frame_metrics(self, metrics: Dict):
        """Add frame-by-frame metrics for temporal analysis"""
        if "elbow_angle" in metrics:
            self.elbow_angles.append(metrics["elbow_angle"])
        if "layback" in metrics:
            self.layback_angles.append(metrics["layback"])
        if "h2ss" in metrics:
            self.h2ss_values.append(metrics["h2ss"])
        if "hand_velocity" in metrics:
            self.hand_velocities.append(metrics["hand_velocity"])
        if "trunk_rotation" in metrics:
            self.trunk_rotations.append(metrics["trunk_rotation"])
        if "arm_extension" in metrics:
            self.arm_extensions.append(metrics["arm_extension"])
    
    def analyze_strengths_weaknesses(self):
        """Analyze player's strengths and areas for improvement"""
        self.strengths = []
        self.areas_for_improvement = []
        
        if not self.shot_breakdowns:
            return
        
        # Average metrics across all shots
        avg_layback = np.mean([s["metrics"]["max_layback"] for s in self.shot_breakdowns])
        avg_h2ss = np.mean([s["metrics"]["max_h2ss"] for s in self.shot_breakdowns])
        avg_hand_velocity = np.mean([s["metrics"]["max_hand_velocity"] for s in self.shot_breakdowns])
        avg_trunk = np.mean([s["metrics"]["trunk_rotation"] for s in self.shot_breakdowns])
        avg_form_score = np.mean([s["form_score"] for s in self.shot_breakdowns])
        
        # Analyze consistency
        if len(self.shot_breakdowns) >= 3:
            score_variance = np.std([s["form_score"] for s in self.shot_breakdowns])
            if score_variance < 5:
                self.strengths.append("Exceptional consistency across shots")
            elif score_variance < 10:
                self.strengths.append("Good shot-to-shot consistency")
            else:
                self.areas_for_improvement.append("Work on consistency between shots")
        
        # Layback analysis
        if avg_layback >= 100:
            self.strengths.append(f"Elite forearm layback ({avg_layback:.0f}°) - generates massive whip")
        elif avg_layback >= 80:
            self.strengths.append(f"Good forearm layback ({avg_layback:.0f}°)")
        else:
            self.areas_for_improvement.append(f"Improve forearm layback (current: {avg_layback:.0f}°, target: 100°+)")
        
        # H2SS analysis
        if avg_h2ss >= 30:
            self.strengths.append(f"Elite core torque with {avg_h2ss:.0f}° hip-shoulder separation")
        elif avg_h2ss >= 20:
            self.strengths.append(f"Good core engagement ({avg_h2ss:.0f}° H2SS)")
        else:
            self.areas_for_improvement.append(f"Increase hip-shoulder separation (current: {avg_h2ss:.0f}°, target: 30°+)")
        
        # Hand velocity
        if avg_hand_velocity > 2.0:
            self.strengths.append("Explosive hand speed at release")
        elif avg_hand_velocity > 1.5:
            self.strengths.append("Good hand speed")
        else:
            self.areas_for_improvement.append("Generate more hand speed through late acceleration")
        
        # Trunk rotation
        if avg_trunk > 45:
            self.strengths.append("Excellent trunk rotation - using full core")
        elif avg_trunk > 30:
            self.strengths.append("Good core rotation")
        else:
            self.areas_for_improvement.append("Increase trunk rotation for more power")
        
        # Arm extension style
        extension_styles = [s["metrics"]["arm_extension_style"] for s in self.shot_breakdowns]
        if "charlotte_north" in extension_styles:
            self.strengths.append("Charlotte North style arm extension - maximum whip")
        elif "t_rex_arm" in extension_styles:
            self.areas_for_improvement.append("Stop T-Rex arming - extend arms away from body")
        
        # Shot type proficiency
        shot_types = [s["shot_type"] for s in self.shot_breakdowns]
        primary_type = max(set(shot_types), key=shot_types.count)
        type_shots = [s for s in self.shot_breakdowns if s["shot_type"] == primary_type]
        avg_type_score = np.mean([s["form_score"] for s in type_shots])
        
        if avg_type_score >= 80:
            self.strengths.append(f"Elite {primary_type} mechanics")
        elif avg_type_score >= 70:
            self.strengths.append(f"Solid {primary_type} fundamentals")
        else:
            self.areas_for_improvement.append(f"Refine {primary_type} technique")
    
    def get_breakdown(self) -> Dict:
        """Get complete player breakdown"""
        self.analyze_strengths_weaknesses()
        
        # Calculate trends
        trends = {}
        if len(self.shot_breakdowns) >= 3:
            recent_scores = [s["form_score"] for s in self.shot_breakdowns[-3:]]
            earlier_scores = [s["form_score"] for s in self.shot_breakdowns[:3]]
            
            if np.mean(recent_scores) > np.mean(earlier_scores):
                trends["form_trend"] = "improving"
            elif np.mean(recent_scores) < np.mean(earlier_scores):
                trends["form_trend"] = "declining"
            else:
                trends["form_trend"] = "stable"
        
        return {
            "player_id": self.player_id,
            "shots_analyzed": self.shots_taken,
            "dominant_shot_type": self.dominant_shot_type,
            "average_form_score": round(self.average_form_score, 1),
            "strengths": self.strengths,
            "areas_for_improvement": self.areas_for_improvement,
            "shot_by_shot": self.shot_breakdowns,
            "trends": trends,
            "video_metrics": {
                "total_frames": self.frames_processed,
                "video_duration_seconds": self.video_duration,
                "analysis_timestamp": self.analysis_timestamp
            },
            "average_metrics": {
                "avg_layback": round(np.mean(self.layback_angles), 1) if self.layback_angles else 0,
                "avg_h2ss": round(np.mean(self.h2ss_values), 1) if self.h2ss_values else 0,
                "avg_hand_velocity": round(np.mean(self.hand_velocities), 2) if self.hand_velocities else 0,
                "avg_trunk_rotation": round(np.mean(self.trunk_rotations), 1) if self.trunk_rotations else 0,
                "avg_elbow_angle": round(np.mean(self.elbow_angles), 1) if self.elbow_angles else 0
            }
        }

# ============== EXISTING ANALYSIS FUNCTIONS ==============

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# ============== FOREARM LAYBACK ANALYSIS ==============
def calculate_forearm_layback(shoulder, elbow, wrist):
    """
    Calculate forearm layback angle - the angle between forearm and upper arm
    Elite players like Jarrod Neumann achieve 120°+
    """
    # Vector from elbow to shoulder (upper arm)
    upper_arm = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
    
    # Vector from elbow to wrist (forearm)
    forearm = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
    
    # Normalize
    upper_arm_norm = upper_arm / (np.linalg.norm(upper_arm) + 1e-6)
    forearm_norm = forearm / (np.linalg.norm(forearm) + 1e-6)
    
    # Calculate angle
    dot = np.dot(upper_arm_norm, forearm_norm)
    angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    
    # Determine if wrist is behind elbow (positive layback)
    # Project forearm onto upper arm direction
    projection = np.dot(forearm, upper_arm_norm)
    
    if projection < 0:  # Wrist behind elbow
        return 180 - angle  # Layback angle
    else:
        return angle  # Regular elbow angle

def analyze_layback(layback_history):
    """Analyze forearm layback throughout the shot"""
    if len(layback_history) < 10:
        return {
            "max_layback": 0,
            "avg_layback": 0,
            "quality": "unknown",
            "feedback": []
        }
    
    laybacks = list(layback_history)[-30:]  # Last 30 frames
    max_layback = max(laybacks)
    avg_layback = np.mean(laybacks)
    
    # Thresholds from the article
    if max_layback >= 120:
        quality = "elite"
        feedback = [{
            "type": "excellent",
            "message": f"🔥 ELITE LAYBACK! You achieved {max_layback:.0f}° - Jarrod Neumann level!",
            "detail": "This creates massive whip through the zone."
        }]
    elif max_layback >= 100:
        quality = "good"
        feedback = [{
            "type": "good",
            "message": f"Good layback at {max_layback:.0f}°. You're generating solid whip.",
            "detail": "Work on holding this position longer to increase power."
        }]
    elif max_layback >= 80:
        quality = "average"
        feedback = [{
            "type": "moderate",
            "message": f"Average layback at {max_layback:.0f}°. Room for improvement.",
            "detail": "Focus on letting your wrist 'lay back' more during windup."
        }]
    else:
        quality = "needs_work"
        feedback = [{
            "type": "needs_work",
            "message": f"Limited layback ({max_layback:.0f}°). You're missing potential whip.",
            "detail": "Drill: Practice 'loading' the stick with a loose wrist - let it fall back."
        }]
    
    return {
        "max_layback": float(max_layback),
        "avg_layback": float(avg_layback),
        "quality": quality,
        "feedback": feedback
    }

# ============== HIP-TO-SHOULDER SEPARATION (H2SS) ==============
def calculate_h2ss(landmarks):
    """
    Calculate Hip-to-Shoulder Separation - the rotational difference between hips and shoulders
    Elite players: 30-40° for males, 20-30° for females (Charlotte North level)
    """
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # Vectors
    hip_vector = np.array([right_hip.x - left_hip.x, right_hip.y - left_hip.y])
    shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y])
    
    # Normalize
    hip_norm = hip_vector / (np.linalg.norm(hip_vector) + 1e-6)
    shoulder_norm = shoulder_vector / (np.linalg.norm(shoulder_vector) + 1e-6)
    
    # Calculate angle
    dot = np.dot(hip_norm, shoulder_norm)
    angle = np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))
    
    return angle

def analyze_h2ss(h2ss_history, is_female=True):
    """Analyze hip-to-shoulder separation during loading phase"""
    if len(h2ss_history) < 10:
        return {
            "max_h2ss": 0,
            "avg_h2ss": 0,
            "quality": "unknown",
            "feedback": []
        }
    
    h2ss_values = list(h2ss_history)[-30:]
    max_h2ss = max(h2ss_values)
    avg_h2ss = np.mean(h2ss_values)
    
    # Thresholds based on gender
    if is_female:
        elite = 30  # Charlotte North level
        good = 20
    else:
        elite = 40  # Pro male level
        good = 30
    
    if max_h2ss >= elite:
        quality = "elite"
        feedback = [{
            "type": "excellent",
            "message": f"🔥 ELITE CORE TORQUE! {max_h2ss:.0f}° hip-shoulder separation.",
            "detail": "You're creating massive rotational energy in your core."
        }]
    elif max_h2ss >= good:
        quality = "good"
        feedback = [{
            "type": "good",
            "message": f"Good H2SS at {max_h2ss:.0f}°. You're generating solid torque.",
            "detail": "Try to maintain this separation longer through the load phase."
        }]
    elif max_h2ss >= 15:
        quality = "average"
        feedback = [{
            "type": "moderate",
            "message": f"Average H2SS at {max_h2ss:.0f}°. More torque = more power.",
            "detail": "Focus on keeping shoulders back while hips rotate forward."
        }]
    else:
        quality = "needs_work"
        feedback = [{
            "type": "needs_work",
            "message": f"Limited H2SS ({max_h2ss:.0f}°). You're not fully loading your core.",
            "detail": "Drill: Practice 'separating' - rotate hips forward while keeping shoulders back."
        }]
    
    return {
        "max_h2ss": float(max_h2ss),
        "avg_h2ss": float(avg_h2ss),
        "quality": quality,
        "feedback": feedback
    }

# ============== HAND ACCELERATION ==============
def calculate_hand_velocity(wrist_positions, fps):
    """Calculate hand velocity over time"""
    if len(wrist_positions) < 3:
        return []
    
    velocities = []
    for i in range(1, len(wrist_positions)):
        p1 = wrist_positions[i-1]
        p2 = wrist_positions[i]
        dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        velocities.append(dist * fps)  # Convert to per-second velocity
    
    return velocities

def analyze_hand_acceleration(wrist_positions_pixels, fps, shot_start, shot_end):
    """Analyze hand acceleration during the shot"""
    if len(wrist_positions_pixels) < 10:
        return {
            "max_velocity": 0,
            "max_acceleration": 0,
            "timing": "unknown",
            "feedback": []
        }
    
    # Focus on shot frames
    shot_positions = list(wrist_positions_pixels)[shot_start:shot_end+1]
    
    if len(shot_positions) < 5:
        return {
            "max_velocity": 0,
            "max_acceleration": 0,
            "timing": "unknown",
            "feedback": []
        }
    
    # Calculate velocities
    velocities = calculate_hand_velocity(shot_positions, fps)
    
    if not velocities:
        return {
            "max_velocity": 0,
            "max_acceleration": 0,
            "timing": "unknown",
            "feedback": []
        }
    
    # Calculate accelerations
    accelerations = []
    for i in range(1, len(velocities)):
        accelerations.append((velocities[i] - velocities[i-1]) * fps)
    
    max_velocity = max(velocities)
    max_acceleration = max(accelerations) if accelerations else 0
    
    # When does peak velocity occur? (should be late in the shot)
    peak_idx = np.argmax(velocities)
    peak_position = peak_idx / len(velocities) if velocities else 0
    
    if peak_position > 0.7:
        timing = "excellent"
        timing_msg = "Perfect timing! Hand accelerates at the very end - maximum whip."
    elif peak_position > 0.5:
        timing = "good"
        timing_msg = "Good timing, but try to delay acceleration even more."
    else:
        timing = "early"
        timing_msg = "Hand accelerating too early - you're losing potential whip."
    
    feedback = [{
        "type": "metric",
        "message": f"Peak hand velocity: {max_velocity:.1f} units/sec"
    }, {
        "type": "metric",
        "message": f"Peak acceleration: {max_acceleration:.1f} units/sec²"
    }, {
        "type": "timing",
        "message": timing_msg
    }]
    
    return {
        "max_velocity": float(max_velocity),
        "max_acceleration": float(max_acceleration),
        "timing": timing,
        "feedback": feedback
    }

# ============== TRUNK ROTATION ==============
def calculate_trunk_rotation(landmarks):
    """Calculate trunk rotation angle"""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    
    # Shoulder line
    shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y])
    
    # Hip line
    hip_vector = np.array([right_hip.x - left_hip.x, right_hip.y - left_hip.y])
    
    # Target direction (assume toward top of frame)
    target = np.array([0, -1])
    
    # Calculate shoulder angle to target
    shoulder_angle = np.degrees(np.arccos(
        np.dot(shoulder_vector, target) / (np.linalg.norm(shoulder_vector) + 1e-6)
    ))
    
    # Calculate hip angle to target
    hip_angle = np.degrees(np.arccos(
        np.dot(hip_vector, target) / (np.linalg.norm(hip_vector) + 1e-6)
    ))
    
    return {
        "shoulder_rotation": float(shoulder_angle),
        "hip_rotation": float(hip_angle),
        "trunk_rotation": float(abs(shoulder_angle - hip_angle))
    }

def analyze_trunk_rotation(rotation_history):
    """Analyze trunk rotation throughout the shot"""
    if len(rotation_history) < 10:
        return {
            "max_rotation": 0,
            "sequencing": "unknown",
            "feedback": []
        }
    
    rotations = list(rotation_history)[-30:]
    max_rotation = max([r["trunk_rotation"] for r in rotations])
    
    # Check sequencing (shoulders should follow hips)
    shoulder_peaks = [r["shoulder_rotation"] for r in rotations]
    hip_peaks = [r["hip_rotation"] for r in rotations]
    
    shoulder_peak_idx = np.argmax(shoulder_peaks)
    hip_peak_idx = np.argmax(hip_peaks)
    
    if hip_peak_idx < shoulder_peak_idx:
        sequencing = "ideal"
        seq_msg = "Perfect kinetic sequence: hips lead, shoulders follow"
    else:
        sequencing = "needs_work"
        seq_msg = "Sequencing off - shoulders rotating before hips"
    
    feedback = [{
        "type": "metric",
        "message": f"Trunk rotation range: {max_rotation:.1f}°"
    }, {
        "type": "sequence",
        "message": seq_msg
    }]
    
    if max_rotation > 45:
        feedback.append({
            "type": "excellent",
            "message": "🔥 Excellent trunk mobility! You're using your whole core."
        })
    elif max_rotation > 30:
        feedback.append({
            "type": "good",
            "message": "Good trunk rotation. Engage your core even more."
        })
    else:
        feedback.append({
            "type": "needs_work",
            "message": "Limited trunk rotation. Focus on rotating through your core."
        })
    
    return {
        "max_rotation": float(max_rotation),
        "sequencing": sequencing,
        "feedback": feedback
    }

# ============== EXISTING ANALYSIS FUNCTIONS ==============

def determine_shot_type_from_motion(shoulder_history, wrist_history, angle_history) -> str:
    """
    Determine shot type by analyzing the entire motion, not just the release point
    A true riser starts LOW (wrist near hips) and finishes HIGH
    Sidearm stays relatively level throughout
    Overhand stays high throughout
    """
    if len(wrist_history) < 10 or len(shoulder_history) < 10 or len(angle_history) < 10:
        return "unknown"
    
    # Get wrist positions throughout the motion (last 30 frames or so)
    wrists = list(wrist_history)[-30:]
    shoulders = list(shoulder_history)[-30:]
    angles = list(angle_history)[-30:]
    
    # Find the lowest point the wrist reached during the motion
    lowest_wrist_y = max([w.get('y', 0) for w in wrists])  # Higher y = lower in frame
    highest_wrist_y = min([w.get('y', 0) for w in wrists])  # Lower y = higher in frame
    
    # Get average shoulder height
    avg_shoulder_y = np.mean([s.get('y', 0) for s in shoulders])
    
    # Calculate how much the wrist moved vertically
    vertical_movement = lowest_wrist_y - highest_wrist_y
    
    # Check if wrist ever dropped significantly below shoulder (riser characteristic)
    dropped_below_shoulder = lowest_wrist_y > avg_shoulder_y + 0.15
    
    # Check minimum angle during motion (for loading)
    min_angle = min(angles) if angles else 180
    max_angle = max(angles) if angles else 0
    
    # Calculate average angle during the motion
    avg_angle = np.mean(angles) if angles else 0
    
    # RISER: starts low (wrist below shoulder), has large vertical movement, angle loads then extends
    if dropped_below_shoulder and vertical_movement > 0.2 and min_angle < 80:
        return "riser"
    
    # SIDEARM: stays relatively level, angle around 90-140 throughout
    elif vertical_movement < 0.2 and max_angle > 100 and avg_angle > 80 and avg_angle < 140:
        return "sidearm"
    
    # OVERHAND: stays high throughout, angle loads then extends to high release
    elif highest_wrist_y < avg_shoulder_y - 0.1 and max_angle > 140:
        return "overhand"
    
    # Default based on release angle if unclear
    else:
        # Look at the last few frames (release)
        recent_angles = angles[-5:] if len(angles) >= 5 else angles
        release_angle = np.mean(recent_angles) if recent_angles else 0
        
        if release_angle > 140:
            # High release angle - could be overhand or riser
            if lowest_wrist_y > avg_shoulder_y:
                return "riser"
            else:
                return "overhand"
        else:
            return "sidearm"

def analyze_arm_extension(shoulder_history, elbow_history, wrist_history, angle_history):
    """
    Analyze how extended the arm is throughout the shot
    Measures the horizontal distance between shoulder and wrist to detect:
    - Full extension (Charlotte North style): arm away from body, creating massive arc
    - T-Rex arm: arm tucked close to body, limited power generation
    """
    if len(wrist_history) < 10 or len(shoulder_history) < 10:
        return {
            "max_extension": 0,
            "avg_extension": 0,
            "extension_style": "unknown",
            "feedback": []
        }
    
    # Convert to lists
    wrists = list(wrist_history)[-30:]
    shoulders = list(shoulder_history)[-30:]
    elbows = list(elbow_history)[-30:] if elbow_history else []
    angles = list(angle_history)[-30:] if angle_history else []
    
    # Calculate horizontal distances (shoulder to wrist)
    horizontal_distances = []
    for i in range(len(wrists)):
        if i < len(shoulders) and i < len(wrists):
            # Get x-coordinates (normalized 0-1)
            shoulder_x = shoulders[i].get('x', 0) if isinstance(shoulders[i], dict) else 0
            wrist_x = wrists[i].get('x', 0) if isinstance(wrists[i], dict) else 0
            
            # Horizontal distance (how far arm is from body)
            horiz_dist = abs(wrist_x - shoulder_x)
            horizontal_distances.append(horiz_dist)
    
    if not horizontal_distances:
        return {
            "max_extension": 0,
            "avg_extension": 0,
            "extension_style": "unknown",
            "feedback": []
        }
    
    # Calculate extension metrics
    max_extension = max(horizontal_distances)
    avg_extension = np.mean(horizontal_distances)
    
    # Check if arm is fully extended at any point
    fully_extended = max_extension > 0.35  # Arm is far from body
    
    # Check if arm is consistently extended
    consistently_extended = avg_extension > 0.25
    
    # Analyze the shape of the motion
    # For Charlotte North style, we want to see the arm extend AWAY from body
    # creating that big arc, not just up and down
    
    # Track how extension changes over time
    extension_trend = "stable"
    if len(horizontal_distances) > 5:
        first_half = np.mean(horizontal_distances[:len(horizontal_distances)//2])
        second_half = np.mean(horizontal_distances[len(horizontal_distances)//2:])
        
        if second_half > first_half + 0.1:
            extension_trend = "increasing"  # Arm moving away from body
        elif first_half > second_half + 0.1:
            extension_trend = "decreasing"  # Arm coming back to body
    
    # Determine extension style
    if fully_extended and consistently_extended:
        if extension_trend == "increasing":
            extension_style = "charlotte_north"  # Full extension creating massive arc
        else:
            extension_style = "full_extension"
    elif avg_extension > 0.2:
        extension_style = "moderate_extension"
    else:
        extension_style = "t_rex_arm"  # Arm tucked close to body
    
    # Generate feedback
    feedback = []
    
    if extension_style == "charlotte_north":
        feedback.append({
            "type": "excellent",
            "message": "🔥 CHARLOTTE NORTH STYLE! Your arm fully extends away from your body, creating that massive arc for maximum torque.",
            "detail": f"Max extension: {max_extension:.2f} (normalized units)"
        })
        feedback.append({
            "type": "tip",
            "message": "Notice how your arm sweeps outward, not just upward. This creates the whip effect that generates elite power."
        })
    elif extension_style == "full_extension":
        feedback.append({
            "type": "good",
            "message": "Good arm extension! You're getting your arm away from your body.",
            "detail": f"Max extension: {max_extension:.2f}"
        })
        feedback.append({
            "type": "tip",
            "message": "To reach Charlotte North level, try to extend even MORE horizontally - think about reaching away from your body, not just up."
        })
    elif extension_style == "moderate_extension":
        feedback.append({
            "type": "moderate",
            "message": "Moderate arm extension. You're getting some separation, but could extend more.",
            "detail": f"Avg extension: {avg_extension:.2f}"
        })
        feedback.append({
            "type": "drill",
            "message": "Drill: Practice 'reaching out' shots where you focus on keeping your arm away from your body throughout the motion."
        })
    else:  # t_rex_arm
        feedback.append({
            "type": "needs_work",
            "message": "⚠️ T-Rex arm detected! Your arm is staying too close to your body.",
            "detail": f"Avg extension: {avg_extension:.2f} (should be > 0.25)"
        })
        feedback.append({
            "type": "drill",
            "message": "Drill: Wall drills focusing on keeping your stick away from your body. Imagine you're trying to paint a wide arc on the wall."
        })
        feedback.append({
            "type": "tip",
            "message": "Charlotte North generates power by creating distance between her hands and body. The further out, the more whip!"
        })
    
    # Add horizontal vs vertical analysis
    if len(angles) > 0 and len(horizontal_distances) > 0:
        # Check if extension is more horizontal or vertical
        max_angle_idx = np.argmax(angles) if angles else 0
        if max_angle_idx < len(horizontal_distances):
            extension_at_release = horizontal_distances[max_angle_idx]
            
            if extension_at_release > 0.3:
                feedback.append({
                    "type": "insight",
                    "message": "Excellent horizontal extension at release - this creates the 'whip' effect."
                })
            elif extension_at_release < 0.15:
                feedback.append({
                    "type": "insight",
                    "message": "Your arm is close to your body at release. More horizontal extension would add power."
                })
    
    return {
        "max_extension": float(max_extension),
        "avg_extension": float(avg_extension),
        "extension_style": extension_style,
        "extension_trend": extension_trend,
        "fully_extended": fully_extended,
        "consistently_extended": consistently_extended,
        "feedback": feedback
    }

def analyze_arm_path(shoulder_history, wrist_history, angle_history):
    """
    Analyze the path the arm takes during the shot
    Visualizes whether the arm moves in an arc (Charlotte North) or straight line (T-Rex)
    """
    if len(wrist_history) < 15 or len(shoulder_history) < 15:
        return {
            "path_type": "unknown",
            "arc_size": 0,
            "feedback": []
        }
    
    wrists = list(wrist_history)[-30:]
    shoulders = list(shoulder_history)[-30:]
    
    # Get x and y coordinates
    wrist_x = [w.get('x', 0) for w in wrists if isinstance(w, dict)]
    wrist_y = [w.get('y', 0) for w in wrists if isinstance(w, dict)]
    shoulder_x = [s.get('x', 0) for s in shoulders if isinstance(s, dict)]
    
    if len(wrist_x) < 10:
        return {"path_type": "unknown", "arc_size": 0, "feedback": []}
    
    # Calculate the arc of the motion
    # For Charlotte North, we want to see the wrist move in a wide arc
    # For T-Rex, it's more of a straight line
    
    # Calculate the area covered by the wrist movement
    x_range = max(wrist_x) - min(wrist_x)
    y_range = max(wrist_y) - min(wrist_y)
    
    # Calculate path curvature
    # Fit a quadratic to the path and check the coefficient
    if len(wrist_x) > 5:
        # Normalize time
        t = np.linspace(0, 1, len(wrist_x))
        
        # Fit quadratic to x vs t and y vs t
        x_coeffs = np.polyfit(t, wrist_x, 2)
        y_coeffs = np.polyfit(t, wrist_y, 2)
        
        # The quadratic coefficients tell us about curvature
        x_curve = abs(x_coeffs[0])
        y_curve = abs(y_coeffs[0])
        
        # Total curvature
        total_curve = x_curve + y_curve
    else:
        total_curve = 0
    
    # Arc size (area of bounding box)
    arc_size = x_range * y_range
    
    # Determine path type
    if arc_size > 0.15 and total_curve > 0.1:
        path_type = "wide_arc"  # Charlotte North style - big sweeping motion
        feedback = [{
            "type": "excellent",
            "message": "🔥 Wide arc detected! Your arm sweeps in a big curve, maximizing the whip effect."
        }]
    elif arc_size > 0.1:
        path_type = "moderate_arc"
        feedback = [{
            "type": "good",
            "message": "Good arc in your motion. Try to make it even wider for more power."
        }]
    else:
        path_type = "straight_line"
        feedback = [{
            "type": "needs_work",
            "message": "⚠️ Your arm moves in a straight line. Charlotte North creates power with a wide, sweeping arc.",
            "drill": "Practice 'painting a rainbow' with your stick - focus on the curve, not the straight line."
        }]
    
    return {
        "path_type": path_type,
        "arc_size": float(arc_size),
        "x_range": float(x_range),
        "y_range": float(y_range),
        "curvature": float(total_curve),
        "feedback": feedback
    }

def detect_shot_phases(angle_history, wrist_history, shoulder_history):
    """
    Analyze the entire shot motion to identify key phases
    Returns dict with phase timings and characteristics
    """
    if len(angle_history) < 20 or len(wrist_history) < 20 or len(shoulder_history) < 20:
        return None
    
    angles = list(angle_history)
    wrists = list(wrist_history)
    shoulders = list(shoulder_history)
    
    # Find loading phase (minimum angle)
    min_angle_idx = np.argmin(angles)
    min_angle = angles[min_angle_idx]
    
    # Find release phase (maximum angle)
    max_angle_idx = np.argmax(angles)
    max_angle = angles[max_angle_idx]
    
    # Find lowest wrist position (start of riser)
    wrist_y_values = [w.get('y', 0) for w in wrists]
    lowest_wrist_idx = np.argmax(wrist_y_values) if wrist_y_values else 0
    
    # Find highest wrist position (follow-through)
    highest_wrist_idx = np.argmin(wrist_y_values) if wrist_y_values else 0
    
    # Calculate shot speed
    frames_between = max_angle_idx - min_angle_idx
    shot_speed = (max_angle - min_angle) / frames_between if frames_between > 0 else 0
    
    return {
        "load_angle": float(min_angle),
        "load_frame": int(min_angle_idx),
        "release_angle": float(max_angle),
        "release_frame": int(max_angle_idx),
        "lowest_wrist_frame": int(lowest_wrist_idx),
        "highest_wrist_frame": int(highest_wrist_idx),
        "shot_speed": float(shot_speed),
        "frames_between": int(frames_between),
        "wrist_dropped": bool(wrist_y_values[lowest_wrist_idx] > shoulders[lowest_wrist_idx].get('y', 0) + 0.15) if shoulders else False,
        "vertical_movement": float(wrist_y_values[lowest_wrist_idx] - wrist_y_values[highest_wrist_idx]) if wrists else 0
    }

def determine_shot_type_at_release(shoulder, elbow, wrist, angle) -> str:
    """
    Temporary shot type for real-time display during the shot
    """
    # Compare y-coordinates (smaller y = higher in frame)
    wrist_y = wrist.y
    shoulder_y = shoulder.y
    elbow_y = elbow.y
    
    # Calculate the difference (negative means wrist is above shoulder)
    height_diff = wrist_y - shoulder_y
    
    # Check elbow position relative to shoulder (for sidearm detection)
    elbow_out = abs(elbow.x - shoulder.x)  # How far out the elbow is
    
    # Overhand: wrist clearly above shoulder AND elbow not flared out
    if height_diff < -0.15 and elbow_out < 0.2:
        return "overhand"
    
    # Sidearm: wrist level with shoulder OR elbow flared out (even if wrist drops a bit)
    elif (abs(height_diff) <= 0.2 or elbow_out > 0.2) and angle > 90:
        return "sidearm"
    
    # Riser: wrist clearly below shoulder AND elbow not flared out
    elif height_diff > 0.15 and elbow_out < 0.2:
        return "riser"
    
    # Default based on angle if unclear
    elif angle > 120:
        if wrist_y < shoulder_y:
            return "overhand"
        else:
            return "riser"
    else:
        return "sidearm"

def get_body_center(landmarks) -> Optional[tuple]:
    """Calculate the center of the body (between shoulders and hips)"""
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        
        return (center_x, center_y)
    except:
        return None

def get_primary_person_from_frame(frame, results):
    """
    Identify the primary person in a single frame
    """
    if not results.pose_landmarks:
        return None
    
    h, w, _ = frame.shape
    landmarks = results.pose_landmarks.landmark
    
    # Calculate body size (shoulder width * height range)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    
    shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
    body_height = abs(left_shoulder.y - left_hip.y) * h
    body_size = shoulder_width * body_height
    
    # Calculate distance from center
    center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
    center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
    distance_from_center = abs(center_x - 0.5) + abs(center_y - 0.5)
    
    # Score: bigger is better, closer to center is better
    score = (body_size / 10000) + (1 - distance_from_center) * 2
    
    return {
        "score": score,
        "landmarks": landmarks,
        "center": (center_x, center_y),
        "body_size": body_size
    }

def detect_lacrosse_shot(angle_history: deque, shoulder_history: deque, elbow_history: deque, wrist_history: deque, fps: int = 30) -> Dict:
    """
    Detect a lacrosse shot based on the characteristic motion pattern
    """
    
    if len(angle_history) < 25:
        return {"shot_detected": False}
    
    # Convert deques to lists
    angles = list(angle_history)
    shoulders = list(shoulder_history)
    elbows = list(elbow_history)
    wrists = list(wrist_history)
    
    # Calculate rolling averages to smooth noise
    window_size = 7
    smoothed_angles = []
    for i in range(len(angles) - window_size + 1):
        smoothed_angles.append(np.mean(angles[i:i+window_size]))
    
    if len(smoothed_angles) < 15:
        return {"shot_detected": False}
    
    # Look for shot pattern in recent frames
    recent_angles = smoothed_angles[-25:] if len(smoothed_angles) > 25 else smoothed_angles
    recent_shoulders = shoulders[-25:] if len(shoulders) > 25 else shoulders
    recent_wrists = wrists[-25:] if len(wrists) > 25 else wrists
    
    # Find the minimum (loading phase) and maximum (release)
    min_idx = np.argmin(recent_angles)
    max_idx = np.argmax(recent_angles)
    
    # A real shot MUST have min before max
    if max_idx <= min_idx:
        return {"shot_detected": False}
    
    min_val = recent_angles[min_idx]
    max_val = recent_angles[max_idx]
    angle_change = max_val - min_val
    frames_between = max_idx - min_idx
    
    # Calculate shot speed
    shot_speed = angle_change / frames_between if frames_between > 0 else 0
    
    # STRICT criteria for a real shot
    min_angle_change = 70
    min_release_angle = 120
    min_shot_speed = 8
    
    is_valid_shot = (
        angle_change >= min_angle_change and
        max_val >= min_release_angle and
        frames_between >= 4 and
        frames_between <= 12 and
        shot_speed >= min_shot_speed and
        min_val <= 80
    )
    
    # Validate the angle pattern
    if is_valid_shot and min_idx > 0 and max_idx < len(recent_angles) - 1:
        before_min = recent_angles[min_idx - 1] if min_idx > 0 else min_val + 10
        after_min = recent_angles[min_idx + 1] if min_idx < len(recent_angles) - 1 else min_val - 10
        before_max = recent_angles[max_idx - 1] if max_idx > 0 else max_val - 10
        
        decreasing_to_min = before_min > min_val
        increasing_from_min = after_min > min_val
        increasing_to_max = before_max < max_val
        
        if not (decreasing_to_min and increasing_from_min and increasing_to_max):
            is_valid_shot = False
    
    if is_valid_shot:
        # Determine shot type from full motion
        shot_type = determine_shot_type_from_motion(
            shoulder_history, 
            wrist_history, 
            angle_history
        )
        
        # Get detailed phase info
        phases = detect_shot_phases(angle_history, wrist_history, shoulder_history)
        
        # Analyze arm extension (Charlotte North vs T-Rex)
        arm_extension_analysis = analyze_arm_extension(
            shoulder_history, 
            elbow_history,
            wrist_history, 
            angle_history
        )
        
        # Analyze arm path
        arm_path_analysis = analyze_arm_path(
            shoulder_history,
            wrist_history,
            angle_history
        )
        
        return {
            "shot_detected": True,
            "shot_type": shot_type,
            "load_angle": float(min_val),
            "release_angle": float(max_val),
            "angle_change": float(angle_change),
            "frames": int(frames_between),
            "shot_speed": float(shot_speed),
            "shot_start_frame": int(max(0, min_idx - 2)),
            "shot_end_frame": int(min(len(recent_angles) - 1, max_idx + 3)),
            "phases": phases,
            "arm_extension": arm_extension_analysis,
            "arm_path": arm_path_analysis
        }
    
    return {"shot_detected": False}

def analyze_lacrosse_form(angles_over_time: List[float], shots: List[Dict], 
                          detected_shot_type: str, fps: int,
                          layback_analysis=None, h2ss_analysis=None,
                          hand_accel_analysis=None, trunk_analysis=None) -> Dict[str, Any]:
    """
    Analyze lacrosse shooting form based on elite standards
    """
    
    if not angles_over_time or len(angles_over_time) < 30:
        return {
            "rating": "Insufficient Data",
            "score": 0,
            "comparison": "Need more video footage",
            "main_feedback": "Please upload a longer video (at least 2-3 seconds)",
            "shot_type": detected_shot_type,
            "drills": [],
            "metrics": {
                "avg_angle": 0,
                "max_angle": 0,
                "min_angle": 0,
                "angle_range": 0,
                "shots_detected": 0,
                "form_score": 0,
                "avg_load_angle": 0,
                "avg_release_angle": 0,
                "avg_explosion": 0,
                "consistency_rating": 0
            },
            "shots_analyzed": 0
        }
    
    # Calculate basic metrics
    avg_angle = np.mean(angles_over_time)
    max_angle = np.max(angles_over_time)
    min_angle = np.min(angles_over_time)
    angle_range = max_angle - min_angle
    
    # Charlotte North standards for different shot types
    standards = {
        "overhand": {
            "load_angle": {"elite": 45, "good": 55, "average": 65},
            "release_angle": {"elite": 160, "good": 150, "average": 140},
            "shot_explosion": {"elite": 115, "good": 100, "average": 85},
            "name": "Overhand"
        },
        "sidearm": {
            "load_angle": {"elite": 60, "good": 70, "average": 80},
            "release_angle": {"elite": 140, "good": 130, "average": 120},
            "shot_explosion": {"elite": 80, "good": 70, "average": 60},
            "name": "Sidearm"
        },
        "riser": {
            "load_angle": {"elite": 50, "good": 60, "average": 70},
            "release_angle": {"elite": 165, "good": 155, "average": 145},
            "shot_explosion": {"elite": 115, "good": 100, "average": 85},
            "name": "Riser"
        }
    }
    
    shot_style = detected_shot_type if detected_shot_type in standards else "overhand"
    shot_standards = standards[shot_style]
    
    # Calculate form score
    form_score = 50
    deductions = []
    
    # Filter shots
    valid_shots = []
    arm_extensions = []
    arm_paths = []
    
    if shots:
        for shot in shots:
            if (shot["release_angle"] >= 100 and 
                shot["angle_change"] >= 50 and
                shot.get("shot_speed", 0) >= 5):
                valid_shots.append(shot)
                if "arm_extension" in shot:
                    arm_extensions.append(shot["arm_extension"])
                if "arm_path" in shot:
                    arm_paths.append(shot["arm_path"])
    
    if valid_shots:
        avg_load = np.mean([s["load_angle"] for s in valid_shots])
        avg_release = np.mean([s["release_angle"] for s in valid_shots])
        avg_explosion = avg_release - avg_load
        
        # Score each aspect
        if avg_load <= shot_standards["load_angle"]["elite"]:
            form_score += 15
        elif avg_load <= shot_standards["load_angle"]["good"]:
            form_score += 10
        elif avg_load <= shot_standards["load_angle"]["average"]:
            form_score += 5
        else:
            deductions.append("load_too_wide")
            form_score -= 5
        
        if avg_release >= shot_standards["release_angle"]["elite"]:
            form_score += 20
        elif avg_release >= shot_standards["release_angle"]["good"]:
            form_score += 15
        elif avg_release >= shot_standards["release_angle"]["average"]:
            form_score += 10
        else:
            deductions.append("release_incomplete")
            form_score -= 10
        
        if avg_explosion >= shot_standards["shot_explosion"]["elite"]:
            form_score += 15
        elif avg_explosion >= shot_standards["shot_explosion"]["good"]:
            form_score += 10
        elif avg_explosion >= shot_standards["shot_explosion"]["average"]:
            form_score += 5
        else:
            deductions.append("low_power")
            form_score -= 5
        
        if len(valid_shots) >= 2:
            shot_variance = np.std([s["release_angle"] for s in valid_shots])
            if shot_variance < 8:
                form_score += 10
            elif shot_variance < 15:
                form_score += 5
            else:
                deductions.append("inconsistent")
                form_score -= 5
    
    form_score = max(0, min(100, form_score))
    
    # Determine rating
    if len(valid_shots) == 0:
        rating = "NO SHOTS DETECTED"
        rating_color = "gray"
        comparison = "No clear shots detected in this video"
        main_feedback = "Make sure you're taking a full shooting motion with proper extension"
    elif form_score >= 90:
        rating = "ELITE LEVEL"
        rating_color = "purple"
        comparison = f"🔥 Elite {shot_standards['name']} form! Your mechanics are exceptional."
        main_feedback = f"Outstanding {shot_standards['name'].lower()} mechanics!"
    elif form_score >= 80:
        rating = "ADVANCED"
        rating_color = "blue"
        comparison = f"Strong {shot_standards['name']} form - approaching elite level."
        main_feedback = f"Great {shot_standards['name'].lower()} mechanics!"
    elif form_score >= 70:
        rating = "INTERMEDIATE+"
        rating_color = "green"
        comparison = f"Good {shot_standards['name']} fundamentals with room to grow."
        main_feedback = f"Solid {shot_standards['name'].lower()} form!"
    elif form_score >= 60:
        rating = "INTERMEDIATE"
        rating_color = "yellow"
        comparison = f"Developing {shot_standards['name']} form - keep working!"
        main_feedback = f"You have the basics of {shot_standards['name'].lower()}"
    elif form_score >= 50:
        rating = "DEVELOPING"
        rating_color = "orange"
        comparison = f"Building {shot_standards['name']} fundamentals - stay consistent."
        main_feedback = f"Good start with {shot_standards['name'].lower()}"
    else:
        rating = "FOUNDATIONAL"
        rating_color = "red"
        comparison = f"Focus on {shot_standards['name']} basics - everyone starts here!"
        main_feedback = f"Building your {shot_standards['name'].lower()} foundation"
    
    # Generate drills
    drills = []
    
    if len(valid_shots) == 0:
        drills.append({
            "name": "Full Motion Drill",
            "description": "Practice the complete shooting motion from load to follow-through",
            "coaching_point": "Focus on full extension at release",
            "reps": "3 sets of 10 shots"
        })
    
    if "load_too_wide" in deductions:
        drills.append({
            "name": "Compact Load Drill",
            "description": "Keep your elbow tucked during windup",
            "coaching_point": "Think 'elbow to body' as you load",
            "reps": "3 sets of 10 shots"
        })
    
    if "release_incomplete" in deductions:
        drills.append({
            "name": "Full Extension Drill",
            "description": "Snap through to full extension at release",
            "coaching_point": "Arm should be nearly straight at release",
            "reps": "3 sets of 8 shots"
        })
    
    if "low_power" in deductions:
        drills.append({
            "name": "Explosion Drill",
            "description": "Quick transfer from load to release",
            "coaching_point": "Generate power from your core, not just your arm",
            "reps": "2 sets of 5 explosive shots"
        })
    
    if "inconsistent" in deductions:
        drills.append({
            "name": "Consistency Drill",
            "description": "Same motion, same spot, same result",
            "coaching_point": "Focus on repeating identical form",
            "reps": "30 shots, aim for 80% consistency"
        })
    
    # Add shot-type specific drills
    if shot_style == "riser":
        drills.append({
            "name": "Riser Load Drill",
            "description": "Start with the stick low by your hips",
            "coaching_point": "Really focus on that low load position - it's what defines the riser",
            "reps": "3 sets of 8 risers, emphasizing the low start"
        })
    
    # Add arm extension drills based on analysis
    if arm_extensions:
        latest_extension = arm_extensions[-1]
        if latest_extension.get("extension_style") == "t_rex_arm":
            drills.append({
                "name": "Charlotte North Extension Drill",
                "description": "Your arm is too close to your body (T-Rex style). Focus on reaching OUT, not just back.",
                "coaching_point": "Imagine you're trying to hit someone standing to your side - reach away from your body",
                "reps": "3 sets of 10 'reach' shots"
            })
        elif latest_extension.get("extension_style") == "charlotte_north":
            drills.append({
                "name": "Charlotte North Power Drill",
                "description": "You're already showing elite arm extension! Now focus on maintaining that arc while adding speed.",
                "coaching_point": "Keep that arm AWAY from your body - the distance creates the whip",
                "reps": "3 sets of 5 max-power shots"
            })
    
    # Add arm path drills
    if arm_paths:
        latest_path = arm_paths[-1]
        if latest_path.get("path_type") == "straight_line":
            drills.append({
                "name": "Rainbow Arc Drill",
                "description": "Your arm moves in a straight line. Create more of an arc for maximum whip.",
                "coaching_point": "Paint a rainbow with your stick - focus on the curve, not the straight line",
                "reps": "3 sets of 10 'rainbow' shots"
            })
    
    # Add layback drills
    if layback_analysis and layback_analysis.get("quality") == "needs_work":
        drills.append({
            "name": "Forearm Layback Drill",
            "description": "Let your wrist 'lay back' more during windup to create whip.",
            "coaching_point": "Think of your forearm as a spring - load it by letting the stick fall back",
            "reps": "3 sets of 10 slow-motion laybacks"
        })
    
    # Add H2SS drills
    if h2ss_analysis and h2ss_analysis.get("quality") == "needs_work":
        drills.append({
            "name": "Hip-Shoulder Separation Drill",
            "description": "Create more torque by rotating hips while keeping shoulders back.",
            "coaching_point": "Feel the stretch in your core - that's stored energy",
            "reps": "3 sets of 8 separation holds"
        })
    
    # Add hand acceleration drills
    if hand_accel_analysis and hand_accel_analysis.get("timing") == "early":
        drills.append({
            "name": "Late Release Drill",
            "description": "Delay your wrist snap until the very last moment.",
            "coaching_point": "Accelerate THROUGH the ball, not AT the ball",
            "reps": "3 sets of 10 'delayed snap' shots"
        })
    
    # Compile all feedback
    all_feedback = []
    if arm_extensions:
        all_feedback.extend(arm_extensions[-1].get("feedback", []))
    if arm_paths:
        all_feedback.extend(arm_paths[-1].get("feedback", []))
    if layback_analysis:
        all_feedback.extend(layback_analysis.get("feedback", []))
    if h2ss_analysis:
        all_feedback.extend(h2ss_analysis.get("feedback", []))
    if hand_accel_analysis:
        all_feedback.extend(hand_accel_analysis.get("feedback", []))
    if trunk_analysis:
        all_feedback.extend(trunk_analysis.get("feedback", []))
    
    metrics = {
        "avg_angle": round(avg_angle, 1),
        "max_angle": round(max_angle, 1),
        "min_angle": round(min_angle, 1),
        "angle_range": round(angle_range, 1),
        "shots_detected": len(valid_shots),
        "form_score": round(form_score, 1),
        "avg_load_angle": round(avg_load, 1) if valid_shots else 0,
        "avg_release_angle": round(avg_release, 1) if valid_shots else 0,
        "avg_explosion": round(avg_explosion, 1) if valid_shots else 0,
        "arm_extension_style": arm_extensions[-1].get("extension_style", "unknown") if arm_extensions else "unknown",
        "arm_path_type": arm_paths[-1].get("path_type", "unknown") if arm_paths else "unknown",
        "max_layback": round(layback_analysis.get("max_layback", 0), 1) if layback_analysis else 0,
        "max_h2ss": round(h2ss_analysis.get("max_h2ss", 0), 1) if h2ss_analysis else 0,
        "max_hand_velocity": round(hand_accel_analysis.get("max_velocity", 0), 1) if hand_accel_analysis else 0,
        "trunk_rotation": round(trunk_analysis.get("max_rotation", 0), 1) if trunk_analysis else 0
    }
    
    return {
        "rating": rating,
        "rating_color": rating_color,
        "score": form_score,
        "comparison": comparison,
        "main_feedback": main_feedback,
        "shot_type": shot_standards['name'] if valid_shots else "Unknown",
        "drills": drills,
        "feedback": all_feedback,
        "metrics": metrics,
        "shots_analyzed": len(valid_shots),
        "standards_used": "Charlotte North"
    }

@app.post("/api/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        content = await video.read()
        tmp.write(content)
        temp_input_path = tmp.name

    temp_output_path = temp_input_path.replace('.mp4', '_out.mp4')
    
    # Read video
    cap = cv2.VideoCapture(temp_input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Tracking variables
    primary_person_landmarks = None
    angles_over_time = []
    angle_history = deque(maxlen=45)
    shoulder_history = deque(maxlen=45)
    elbow_history = deque(maxlen=45)
    wrist_history = deque(maxlen=45)
    
    # Advanced metrics tracking
    layback_history = deque(maxlen=45)
    h2ss_history = deque(maxlen=45)
    trunk_rotation_history = deque(maxlen=45)
    wrist_positions_pixels = deque(maxlen=60)
    
    shots = []
    shot_counter = 0
    last_shot_frame = -100
    primary_person_center_history = []
    frame_count = 0
    dominant_shot_type = "unknown"
    in_shot_motion = False
    current_shot_frames = []
    
    # NEW: Player breakdown
    player = PlayerBreakdown("player_1")
    player.video_duration = frame_count / fps if fps > 0 else 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Find primary person
        current_person = get_primary_person_from_frame(frame, results) if results.pose_landmarks else None
        
        if current_person:
            if primary_person_center_history:
                last_center = primary_person_center_history[-1]
                current_center = current_person["center"]
                movement = abs(current_center[0] - last_center[0]) + abs(current_center[1] - last_center[1])
                
                if movement < 0.3:
                    primary_person_landmarks = current_person["landmarks"]
                    primary_person_center_history.append(current_center)
                else:
                    current_person = None
            else:
                primary_person_landmarks = current_person["landmarks"]
                primary_person_center_history.append(current_person["center"])
        
        if primary_person_landmarks:
            landmarks = primary_person_landmarks
            
            # Get right arm points
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Calculate angle
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angles_over_time.append(angle)
            angle_history.append(angle)
            
            # Store positions for motion analysis
            shoulder_history.append({"x": right_shoulder.x, "y": right_shoulder.y})
            elbow_history.append({"x": right_elbow.x, "y": right_elbow.y})
            wrist_history.append({"x": right_wrist.x, "y": right_wrist.y})
            
            # Calculate advanced metrics
            layback = calculate_forearm_layback(right_shoulder, right_elbow, right_wrist)
            layback_history.append(layback)
            
            h2ss = calculate_h2ss(landmarks)
            h2ss_history.append(h2ss)
            
            trunk = calculate_trunk_rotation(landmarks)
            trunk_rotation_history.append(trunk)
            
            # Store pixel positions for hand acceleration
            h, w, _ = frame.shape
            wrist_px = (int(right_wrist.x * w), int(right_wrist.y * h))
            wrist_positions_pixels.append(wrist_px)
            
            # Add frame metrics to player breakdown
            player.add_frame_metrics({
                "elbow_angle": angle,
                "layback": layback,
                "h2ss": h2ss,
                "trunk_rotation": trunk["trunk_rotation"],
                "arm_extension": abs(right_wrist.x - right_shoulder.x)
            })
            
            # Detect shots
            if len(angle_history) >= 30 and (shot_counter == 0 or frame_count - last_shot_frame > fps * 3):
                shot_result = detect_lacrosse_shot(angle_history, shoulder_history, elbow_history, wrist_history, fps)
                if shot_result["shot_detected"]:
                    if shots:
                        last_shot = shots[-1]
                        time_since_last = (frame_count - last_shot_frame) / fps
                        angle_similarity = abs(shot_result["release_angle"] - last_shot["release_angle"])
                        
                        if time_since_last < 2.0 or angle_similarity < 15:
                            pass
                        else:
                            shot_counter += 1
                            
                            # Enhance shot result with advanced metrics
                            shot_result["max_layback"] = max(layback_history) if layback_history else 0
                            shot_result["max_h2ss"] = max(h2ss_history) if h2ss_history else 0
                            shot_result["max_hand_velocity"] = max(calculate_hand_velocity(wrist_positions_pixels, fps)) if wrist_positions_pixels else 0
                            shot_result["trunk_rotation"] = trunk["trunk_rotation"]
                            shot_result["form_score"] = 70  # Placeholder - would calculate properly
                            shot_result["timestamp"] = frame_count / fps
                            
                            shots.append(shot_result)
                            player.add_shot(shot_result)
                            
                            last_shot_frame = frame_count
                            in_shot_motion = True
                            current_shot_frames = [shot_result["shot_start_frame"], shot_result["shot_end_frame"]]
                            
                            if dominant_shot_type == "unknown":
                                dominant_shot_type = shot_result["shot_type"]
                    else:
                        shot_counter += 1
                        
                        # Enhance first shot
                        shot_result["max_layback"] = max(layback_history) if layback_history else 0
                        shot_result["max_h2ss"] = max(h2ss_history) if h2ss_history else 0
                        shot_result["max_hand_velocity"] = max(calculate_hand_velocity(wrist_positions_pixels, fps)) if wrist_positions_pixels else 0
                        shot_result["trunk_rotation"] = trunk["trunk_rotation"]
                        shot_result["form_score"] = 70
                        shot_result["timestamp"] = frame_count / fps
                        
                        shots.append(shot_result)
                        player.add_shot(shot_result)
                        
                        last_shot_frame = frame_count
                        in_shot_motion = True
                        current_shot_frames = [shot_result["shot_start_frame"], shot_result["shot_end_frame"]]
                        dominant_shot_type = shot_result["shot_type"]
            
            # Check if we're still in the shot motion
            if in_shot_motion and current_shot_frames:
                # Turn off after shot is complete (about 1 second after release)
                if frame_count - last_shot_frame > fps:
                    in_shot_motion = False
                    current_shot_frames = []
            
            # ===== DRAW EVERYTHING =====
            h, w, _ = frame.shape
            
            # Draw skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw bounding box during shot motion
            if in_shot_motion or (frame_count - last_shot_frame < fps):
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                x_min = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w) - 30
                x_max = int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w) + 30
                y_min = int(min(left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y) * h) - 50
                y_max = int(max(left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y) * h) + 30
                
                if not in_shot_motion and frame_count - last_shot_frame < fps:
                    opacity = 1.0 - ((frame_count - last_shot_frame) / fps)
                    color = (0, int(255 * opacity), 0)
                    thickness = max(1, int(2 * opacity))
                else:
                    color = (0, 255, 0)
                    thickness = 2
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
                
                if in_shot_motion:
                    cv2.putText(frame, "🎯 You", (x_min, y_min - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw shot type during motion
            if in_shot_motion:
                temp_shot_type = determine_shot_type_at_release(right_shoulder, right_elbow, right_wrist, angle)
                if temp_shot_type != "unknown":
                    cv2.putText(frame, f"{temp_shot_type.upper()} (analyzing...)", (10, y_max + 25 if 'y_max' in locals() else 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw angle at elbow
            elbow_pos = (int(right_elbow.x * w), int(right_elbow.y * h))
            cv2.rectangle(frame, 
                         (elbow_pos[0] - 40, elbow_pos[1] - 25),
                         (elbow_pos[0] + 40, elbow_pos[1] + 5),
                         (0, 0, 0), -1)
            cv2.putText(frame, f"{int(angle)}°", 
                       (elbow_pos[0] - 30, elbow_pos[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw metrics during motion
            if in_shot_motion:
                # Extension indicator
                horiz_dist = abs(right_wrist.x - right_shoulder.x)
                if horiz_dist > 0.3:
                    ext_color = (0, 255, 0)
                    ext_text = "EXTENDED"
                elif horiz_dist > 0.2:
                    ext_color = (0, 255, 255)
                    ext_text = "MODERATE"
                else:
                    ext_color = (0, 0, 255)
                    ext_text = "T-REX"
                
                cv2.putText(frame, ext_text, (w - 200, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ext_color, 2)
                
                # Layback indicator
                if layback_history:
                    latest_layback = layback_history[-1]
                    if latest_layback > 100:
                        lb_color = (0, 255, 0)
                    elif latest_layback > 80:
                        lb_color = (0, 255, 255)
                    else:
                        lb_color = (0, 0, 255)
                    cv2.putText(frame, f"LAYBACK: {latest_layback:.0f}°", (w - 200, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, lb_color, 2)
                
                # H2SS indicator
                if h2ss_history:
                    latest_h2ss = h2ss_history[-1]
                    cv2.putText(frame, f"H2SS: {latest_h2ss:.0f}°", (w - 200, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Shot phase
                dx = right_wrist.x - right_shoulder.x
                dy = right_wrist.y - right_shoulder.y
                arm_extension = math.sqrt(dx*dx + dy*dy)
                is_cocked = arm_extension < 0.25
                
                if angle < 45 and is_cocked:
                    phase = "LOADING"
                    phase_color = (255, 255, 0)
                elif angle < 70 and is_cocked:
                    phase = "WIND UP"
                    phase_color = (0, 255, 255)
                elif angle < 120 and not is_cocked:
                    phase = "POWER ZONE"
                    phase_color = (0, 255, 0)
                elif angle < 150 and not is_cocked:
                    phase = "RELEASE"
                    phase_color = (0, 165, 255)
                elif angle >= 150 and not is_cocked:
                    phase = "FOLLOW THRU"
                    phase_color = (255, 0, 255)
                else:
                    phase = None
                
                if phase:
                    cv2.rectangle(frame, (10, 10), (200, 45), (0, 0, 0), -1)
                    cv2.putText(frame, phase, (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)
            
            # Draw shot count
            if shot_counter > 0:
                cv2.putText(frame, f"Shots: {shot_counter}", (w - 150, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Update player video duration
    player.video_duration = frame_count / fps if fps > 0 else 0
    player.frames_processed = frame_count
    
    # Analyze advanced metrics
    layback_analysis = analyze_layback(layback_history) if layback_history else None
    h2ss_analysis = analyze_h2ss(h2ss_history, is_female=True) if h2ss_history else None
    hand_accel_analysis = analyze_hand_acceleration(
        wrist_positions_pixels, fps, 
        current_shot_frames[0] if current_shot_frames else 0,
        current_shot_frames[1] if current_shot_frames else len(wrist_positions_pixels)-1
    ) if wrist_positions_pixels and current_shot_frames else None
    trunk_analysis = analyze_trunk_rotation(trunk_rotation_history) if trunk_rotation_history else None
    
    # Analyze form
    analysis = analyze_lacrosse_form(
        angles_over_time, shots, dominant_shot_type, fps,
        layback_analysis, h2ss_analysis, hand_accel_analysis, trunk_analysis
    )
    analysis["person_detected"] = len(primary_person_center_history) > 0
    
    # NEW: Add player breakdown to response
    player_breakdown = player.get_breakdown()
    
    # Read processed video
    with open(temp_output_path, 'rb') as f:
        video_content = f.read()
    
    # Clean up
    os.unlink(temp_input_path)
    os.unlink(temp_output_path)
    
    # Return response with player breakdown
    response_data = {
        "analysis": analysis,
        "player_breakdown": player_breakdown
    }
    
    response = Response(
        content=video_content,
        media_type="video/mp4",
        headers={
            "X-Analysis-Results": json.dumps(response_data),
            "Access-Control-Expose-Headers": "X-Analysis-Results"
        }
    )
    
    return response

@app.get("/api/test")
async def test():
    return {
        "message": "Lacrosse analysis API running!",
        "features": [
            "Full motion analysis for accurate shot type detection",
            "Arm extension analysis (Charlotte North vs T-Rex)",
            "Arm path analysis (wide arc vs straight line)",
            "🔥 FOREARM LAYBACK analysis (up to 120° elite)",
            "🔥 HIP-TO-SHOULDER SEPARATION (H2SS) analysis",
            "🔥 HAND ACCELERATION tracking",
            "🔥 TRUNK ROTATION analysis",
            "📊 PLAYER BREAKDOWN with strengths/weaknesses",
            "📊 Shot-by-shot detailed breakdown",
            "Clean UI with overlays only during shots"
        ],
        "shot_detection": "70° minimum change, analyzes entire motion",
        "endpoints": {
            "analyze": "POST /api/analyze-video"
        }
    }