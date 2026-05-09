import numpy as np


LEFT_EYE_POINTS = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_POINTS = [362, 263, 387, 386, 385, 373, 374, 380]

LEFT_EAR_POINTS = [33, 133, 160, 144, 158, 153, 159, 145]
RIGHT_EAR_POINTS = [362, 263, 385, 380, 387, 373, 386, 374]

MOUTH_POINTS = [61, 291, 13, 14, 78, 308, 82, 312, 87, 317]
MOUTH_MAR_POINTS = [61, 81, 13, 291, 311, 14]

HEAD_POSE_POINTS = [1, 152, 33, 263, 61, 291]
HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26.0),  # Left eye outer corner
        (43.3, 32.7, -26.0),  # Right eye outer corner
        (-28.9, -28.9, -24.1),  # Left mouth corner
        (28.9, -28.9, -24.1),  # Right mouth corner
    ],
    dtype=np.float64,
)
