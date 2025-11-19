# src/core/lighting.py
import cv2
import numpy as np

def detect_lighting(frame):
    """
    Estimate lighting temperature and brightness.
    Returns (light_type, brightness) where light_type is 'warm'|'cool'|'neutral'
    and brightness is normalized 0..1.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    avg_l = np.mean(lab[:, :, 0])
    
    if avg_b - avg_a > 10:
        light_type = "warm"
    elif avg_a - avg_b > 10:
        light_type = "cool"
    else:
        light_type = "neutral"
    
    return light_type, avg_l / 255.0
