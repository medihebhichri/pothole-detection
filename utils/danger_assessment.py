import numpy as np
import cv2


class DangerAssessment:
    def __init__(self):
        pass

    def calculate_dimensions(self, mask, pixel_to_cm=1.0):
        """
        Calculate pothole dimensions
        Args:
            mask: binary mask (0=background, 1=pothole)
            pixel_to_cm: conversion factor (pixels to cm)
        """
        pothole_pixels = np.sum(mask == 1)

        if pothole_pixels == 0:
            return None

        # Get bounding box
        rows, cols = np.where(mask == 1)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        width_px = max_col - min_col
        height_px = max_row - min_row

        # Convert to cm (approximate)
        width_cm = width_px * pixel_to_cm
        height_cm = height_px * pixel_to_cm
        area_cm2 = pothole_pixels * (pixel_to_cm ** 2)

        # Calculate percentage of image
        total_pixels = mask.size
        coverage_percent = (pothole_pixels / total_pixels) * 100

        return {
            'width_px': width_px,
            'height_px': height_px,
            'width_cm': width_cm,
            'height_cm': height_cm,
            'area_px': pothole_pixels,
            'area_cm2': area_cm2,
            'coverage_percent': coverage_percent,
            'bbox': (min_col, min_row, max_col, max_row)
        }

    def assess_danger(self, dimensions):
        """Assess danger level based on dimensions"""
        if dimensions is None:
            return {'level': 'NONE', 'score': 0, 'action': 'No pothole detected'}

        score = 0
        reasons = []

        # Size-based scoring
        area_pct = dimensions['coverage_percent']
        if area_pct > 15:
            score += 40
            reasons.append(f"Very large pothole ({area_pct:.1f}% coverage)")
        elif area_pct > 8:
            score += 25
            reasons.append(f"Large pothole ({area_pct:.1f}% coverage)")
        elif area_pct > 3:
            score += 15
            reasons.append(f"Medium pothole ({area_pct:.1f}% coverage)")
        else:
            score += 5
            reasons.append(f"Small pothole ({area_pct:.1f}% coverage)")

        # Width-based scoring
        width_px = dimensions['width_px']
        if width_px > 150:
            score += 30
            reasons.append("Wide pothole - significant lane obstruction")
        elif width_px > 80:
            score += 15
            reasons.append("Moderate width")

        # Shape irregularity
        aspect_ratio = dimensions['width_px'] / max(dimensions['height_px'], 1)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            score += 15
            reasons.append("Irregular shape - potential depth hazard")

        # Determine level
        if score >= 60:
            level = 'CRITICAL'
        elif score >= 40:
            level = 'HIGH'
        elif score >= 20:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        from config import Config
        action = Config.DANGER_LEVELS[level]['action']

        return {
            'level': level,
            'score': score,
            'reasons': reasons,
            'action': action
        }