from collections import deque
import math
import sys
import time

import cv2
import mediapipe as mp
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from config import (
    ADAPTIVE_EAR_WINDOW,
    CAMERA_INDEX,
    EAR_SMOOTHING_WINDOW,
    EYE_CLOSED_EAR_THRESHOLD,
    MODEL_PATH,
)
from face_landmarks import create_face_landmarker, open_camera, print_camera_resolution
from live_interface import analyze_face, unavailable_ml_result, update_no_face_telemetry
from ml_predictor import load_ml_model
from telemetry_store import reset_telemetry
from temporal_analysis import TemporalFatigueAnalyzer


APP_NAME = "SafeDrive AI"
ANALYSIS_INTERVAL_SECONDS = 0.12

BG_TOP = "#07111F"
BG_BOTTOM = "#020711"
PANEL = "#0E1A2A"
PANEL_SOFT = "#102033"
BORDER = "#1E3148"
TEXT = "#F8FAFC"
MUTED = "#A8B3C7"
DIM = "#64748B"
GREEN = "#4ADE80"
YELLOW = "#FBBF24"
ORANGE = "#F59E0B"
RED = "#F87171"


def frame_to_qimage(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    image = QImage(
        rgb_frame.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    return image.copy()


def percent_text(probability):
    if probability is None:
        return "--"

    return str(int(round(probability * 100)))


def format_timer(seconds):
    seconds = max(0, int(seconds))
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"


def risk_profile(probability, face_detected=True):
    if not face_detected:
        return {
            "label": "FACE NOT DETECTED",
            "status": "Face not detected",
            "message": "Adjust camera position.",
            "advice": "Keep your face visible for an accurate estimate.",
            "color": QColor(MUTED),
        }

    if probability is None:
        return {
            "label": "UNKNOWN",
            "status": "Initializing",
            "message": "Waiting for the first estimate.",
            "advice": "Keep looking forward while the system starts.",
            "color": QColor(MUTED),
        }

    if probability >= 0.80:
        return {
            "label": "CRITICAL",
            "status": "Critical fatigue",
            "message": "Stop safely as soon as possible.",
            "advice": "Pull over safely and take a break.",
            "color": QColor(RED),
        }

    if probability >= 0.65:
        return {
            "label": "HIGH",
            "status": "Fatigue detected",
            "message": "Stay focused on the road.",
            "advice": "Rest at the next safe stop.",
            "color": QColor(YELLOW),
        }

    if probability >= 0.40:
        return {
            "label": "MODERATE",
            "status": "Attention needed",
            "message": "Maintain a stable posture.",
            "advice": "Keep your eyes open and posture stable.",
            "color": QColor(ORANGE),
        }

    return {
        "label": "LOW",
        "status": "Driver alert",
        "message": "Monitoring active.",
        "advice": "Continue driving normally.",
        "color": QColor(GREEN),
    }


def card_stylesheet(radius=18):
    return f"""
        QFrame {{
            background-color: {PANEL};
            border: 1px solid {BORDER};
            border-radius: {radius}px;
        }}
    """


class GradientBackground(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(BG_TOP))
        gradient.setColorAt(1.0, QColor(BG_BOTTOM))
        painter.fillRect(self.rect(), QBrush(gradient))
        super().paintEvent(event)


class BrandHeader(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(54)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        icon_center = QPointF(20, 27)
        painter.setPen(QPen(QColor(GREEN), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(icon_center, 15, 15)
        painter.setBrush(QBrush(QColor(GREEN)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(16, 23), 3, 3)
        painter.setPen(QPen(QColor(GREEN), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(20, 39), QPointF(29, 16))

        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Medium))
        painter.drawText(QRectF(45, 0, 220, self.height()), Qt.AlignmentFlag.AlignVCenter, APP_NAME)


class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(142)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        card = QRectF(0, 0, self.width(), self.height())
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(QColor(PANEL))
        painter.drawRoundedRect(card, 18, 18)

        active = QRectF(16, 38, self.width() - 32, 158)
        active_gradient = QLinearGradient(active.topLeft(), active.bottomRight())
        active_gradient.setColorAt(0.0, QColor("#143E34"))
        active_gradient.setColorAt(1.0, QColor("#0D2D26"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(active_gradient))
        painter.drawRoundedRect(active, 16, 16)

        self._draw_camera_icon(painter, self.width() / 2, 94, QColor(GREEN))
        self._draw_center_text(painter, "Live Monitor", 0, 140, self.width(), QColor(GREEN), 10)

        items = [
            ("History", 300, self._draw_bars_icon),
            ("Settings", 485, self._draw_settings_icon),
            ("About", 720, self._draw_info_icon),
        ]
        for label, y, icon in items:
            icon(painter, self.width() / 2, y, QColor(MUTED))
            self._draw_center_text(painter, label, 0, y + 40, self.width(), QColor(MUTED), 10)

    def _draw_center_text(self, painter, text, x, y, width, color, size):
        painter.setPen(color)
        painter.setFont(QFont("Segoe UI", size))
        painter.drawText(QRectF(x, y, width, 28), Qt.AlignmentFlag.AlignCenter, text)

    def _draw_camera_icon(self, painter, x, y, color):
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(x - 20, y - 12, 30, 24), 4, 4)
        path = QPainterPath()
        path.moveTo(x + 10, y - 8)
        path.lineTo(x + 26, y - 16)
        path.lineTo(x + 26, y + 16)
        path.lineTo(x + 10, y + 8)
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(QBrush(QColor("#0D2D26")))
        painter.drawEllipse(QPointF(x - 8, y), 4, 4)

    def _draw_bars_icon(self, painter, x, y, color):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        for index, height in enumerate((14, 26, 38)):
            painter.drawRoundedRect(QRectF(x - 17 + index * 11, y + 18 - height, 6, height), 2, 2)

    def _draw_settings_icon(self, painter, x, y, color):
        painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(x, y), 14, 14)
        painter.drawEllipse(QPointF(x, y), 5, 5)
        for angle in range(0, 360, 60):
            radians = math.radians(angle)
            x1 = x + math.cos(radians) * 18
            y1 = y + math.sin(radians) * 18
            x2 = x + math.cos(radians) * 24
            y2 = y + math.sin(radians) * 24
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _draw_info_icon(self, painter, x, y, color):
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(x, y), 15, 15)
        painter.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        painter.drawText(QRectF(x - 12, y - 14, 24, 28), Qt.AlignmentFlag.AlignCenter, "i")


class CameraView(QLabel):
    def __init__(self):
        super().__init__()
        self.current_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(720, 430)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("Starting camera...")
        self.setStyleSheet(
            """
            QLabel {
                background-color: #050B13;
                color: #A8B3C7;
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                padding: 0;
            }
            """
        )

    def set_frame(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self._update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self.current_pixmap is None:
            return

        scaled_pixmap = self.current_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if scaled_pixmap.width() > self.width() or scaled_pixmap.height() > self.height():
            x = max(0, (scaled_pixmap.width() - self.width()) // 2)
            y = max(0, (scaled_pixmap.height() - self.height()) // 2)
            scaled_pixmap = scaled_pixmap.copy(x, y, self.width(), self.height())
        self.setPixmap(scaled_pixmap)


class LiveBadge(QLabel):
    def __init__(self):
        super().__init__("LIVE")
        self.setFixedSize(96, 42)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.setStyleSheet(
            f"""
            QLabel {{
                color: {TEXT};
                background-color: rgba(15, 23, 42, 175);
                border: 1px solid #64748B;
                border-radius: 13px;
            }}
            """
        )


class RiskOverview(QWidget):
    def __init__(self):
        super().__init__()
        self.probability = None
        self.raw_probability = None
        self.rule_score = None
        self.face_detected = False
        self.setMinimumHeight(165)

    def update_data(self, probability, face_detected, raw_probability=None, rule_score=None):
        self.probability = probability
        self.raw_probability = raw_probability
        self.rule_score = rule_score
        self.face_detected = face_detected
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        profile = risk_profile(self.probability, self.face_detected)
        color = profile["color"]
        percent = percent_text(self.probability)
        width = self.width()
        height = self.height()

        painter.fillRect(self.rect(), QColor(PANEL))
        painter.setPen(QPen(QColor("#18283B"), 1))
        painter.drawLine(0, 0, width, 0)

        left_x = 46
        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(QRectF(left_x, 34, 260, 24), "Fatigue level")

        painter.setPen(color)
        painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        painter.drawText(QRectF(left_x, 62, 260, 38), profile["label"])

        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(QRectF(left_x, 112, 330, 28), "Keep your eyes open and take breaks.")

        self._draw_waveform(painter, QRectF(270, 52, 165, 62), color)

        center_x = width / 2
        center_y = height / 2 + 4
        radius = 63
        painter.setPen(QPen(QColor("#1E3148"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)
        painter.drawEllipse(QPointF(center_x, center_y), radius - 16, radius - 16)

        arc_rect = QRectF(center_x - radius + 8, center_y - radius + 8, (radius - 8) * 2, (radius - 8) * 2)
        painter.setPen(QPen(QColor("#24364D"), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(arc_rect, -210 * 16, 300 * 16)

        if self.probability is not None:
            painter.setPen(QPen(color, 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawArc(arc_rect, -210 * 16, int(300 * self.probability) * 16)

        painter.setPen(color)
        painter.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
        painter.drawText(QRectF(center_x - 80, center_y - 30, 160, 50), Qt.AlignmentFlag.AlignCenter, percent)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(center_x - 80, center_y + 22, 160, 24),
            Qt.AlignmentFlag.AlignCenter,
            "FINAL RISK",
        )

        raw_text = percent_text(self.raw_probability)
        rule_text = "--" if self.rule_score is None else str(int(round(self.rule_score)))
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(center_x - 115, center_y + 44, 230, 22),
            Qt.AlignmentFlag.AlignCenter,
            f"Raw ML: {raw_text}  Rule: {rule_text}",
        )

        advice_x = width - 265
        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(QRectF(advice_x, 35, 220, 24), "Advice")
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(
            QRectF(advice_x, 76, 220, 62),
            Qt.TextFlag.TextWordWrap,
            profile["advice"],
        )
        self._draw_waveform(painter, QRectF(width - 375, 52, 125, 62), color)

    def _draw_waveform(self, painter, rect, color):
        painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        center_y = rect.center().y()
        previous = None
        for index in range(int(rect.width())):
            ratio = index / max(1, rect.width() - 1)
            value = math.sin(ratio * math.pi * 4) * 0.55 + math.sin(ratio * math.pi * 12) * 0.25
            point = QPointF(rect.left() + index, center_y - value * rect.height() * 0.36)
            if previous is not None and index % 6 < 4:
                painter.drawLine(previous, point)
            previous = point


class CameraPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(card_stylesheet(18))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_wrap = QWidget()
        self.video_wrap.setMinimumHeight(480)
        self.video_layout = QGridLayout(self.video_wrap)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(0)

        self.camera_view = CameraView()
        self.video_layout.addWidget(self.camera_view, 0, 0)
        self.video_layout.setContentsMargins(0, 0, 0, 0)

        self.overview = RiskOverview()

        layout.addWidget(self.video_wrap, stretch=5)
        layout.addWidget(self.overview, stretch=0)

    def set_frame(self, image):
        self.camera_view.set_frame(image)

    def update_data(self, probability, face_detected, raw_probability=None, rule_score=None):
        self.overview.update_data(probability, face_detected, raw_probability, rule_score)


class Card(QFrame):
    def __init__(self, fixed_height=None):
        super().__init__()
        self.setStyleSheet(card_stylesheet(18))
        if fixed_height is not None:
            self.setFixedHeight(fixed_height)


class StatusCard(Card):
    def __init__(self):
        super().__init__(fixed_height=170)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 18, 28, 16)
        layout.setSpacing(5)

        self.title = QLabel("Current status")
        self.title.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        self.title.setFont(QFont("Segoe UI", 13))

        self.icon = QLabel("")
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.setFixedSize(44, 44)
        self.icon.setStyleSheet(
            f"background-color: {GREEN}; border: none; border-radius: 22px;"
        )

        self.status = QLabel("Initializing")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.status.setStyleSheet(f"color: {MUTED}; border: none; background: transparent;")

        self.message = QLabel("Waiting for the first estimate.")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setFont(QFont("Segoe UI", 8))
        self.message.setWordWrap(True)
        self.message.setStyleSheet(f"color: {MUTED}; border: none; background: transparent;")

        layout.addWidget(self.title)
        layout.addSpacing(2)
        layout.addWidget(self.icon, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        layout.addWidget(self.message)

    def update_data(self, probability, face_detected):
        profile = risk_profile(probability, face_detected)
        color = profile["color"].name()
        self.icon.setStyleSheet(
            f"background-color: {color}; border: none; border-radius: 22px;"
        )
        self.status.setText(profile["status"])
        self.status.setStyleSheet(f"color: {color}; border: none; background: transparent;")
        self.message.setText(profile["message"])


class TimerCard(Card):
    def __init__(self, title, icon_text, footer):
        super().__init__(fixed_height=110)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 14, 28, 14)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 12))
        title_label.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")

        row = QHBoxLayout()
        row.setSpacing(16)

        icon = QLabel(icon_text)
        icon.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedWidth(54)
        icon.setStyleSheet(f"color: {GREEN}; border: none; background: transparent;")

        values = QVBoxLayout()
        values.setSpacing(0)
        self.value_label = QLabel("--:--")
        self.value_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Medium))
        self.value_label.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        footer_label = QLabel(footer)
        footer_label.setFont(QFont("Segoe UI", 9))
        footer_label.setStyleSheet(f"color: {MUTED}; border: none; background: transparent;")
        values.addWidget(self.value_label)
        values.addWidget(footer_label)

        row.addWidget(icon)
        row.addLayout(values)
        row.addStretch(1)

        layout.addWidget(title_label)
        layout.addLayout(row)


class TrendChart(QWidget):
    def __init__(self):
        super().__init__()
        self.history = deque(maxlen=36)
        self.color = QColor(YELLOW)
        self.setMinimumHeight(72)

    def update_data(self, probability):
        if probability is not None:
            self.history.append(probability)
        self.color = risk_profile(probability)["color"]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#1F2D42"), 1))
        painter.drawLine(0, self.height() - 18, self.width(), self.height() - 18)

        if len(self.history) < 2:
            return

        values = list(self.history)
        points = []
        for index, value in enumerate(values):
            x = index / (len(values) - 1) * self.width()
            y = (1 - value) * (self.height() - 28) + 4
            points.append(QPointF(x, y))

        painter.setPen(QPen(self.color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for start, end in zip(points, points[1:]):
            painter.drawLine(start, end)
        painter.setBrush(QBrush(self.color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(points[-1], 5, 5)


class SessionCard(Card):
    def __init__(self):
        super().__init__(fixed_height=120)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 14, 28, 12)
        layout.setSpacing(5)

        row = QHBoxLayout()
        title = QLabel("Current session")
        title.setFont(QFont("Segoe UI", 12))
        title.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        self.level = QLabel("Unknown")
        self.level.setFont(QFont("Segoe UI", 9))
        self.level.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.level.setStyleSheet(f"color: {MUTED}; border: none; background: transparent;")
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(self.level)

        self.chart = TrendChart()
        footer = QLabel("Low")
        footer.setFont(QFont("Segoe UI", 9))
        footer.setStyleSheet(f"color: {MUTED}; border: none; background: transparent;")

        layout.addLayout(row)
        layout.addWidget(self.chart)
        layout.addWidget(footer)

    def update_data(self, probability):
        self.level.setText(risk_profile(probability)["label"].title())
        self.chart.update_data(probability)


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    metrics_ready = Signal(dict)
    error_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.should_run = True
        self.last_timestamp_ms = 0

    def stop(self):
        self.should_run = False
        self.wait(3000)

    def _next_timestamp_ms(self):
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def run(self):
        if not MODEL_PATH.exists():
            self.error_ready.emit(f"Missing model: {MODEL_PATH}")
            return

        cap = open_camera()
        if not cap.isOpened():
            self.error_ready.emit(f"Camera {CAMERA_INDEX} unavailable.")
            return

        print_camera_resolution(cap)
        reset_telemetry()

        landmarker = None
        try:
            landmarker = create_face_landmarker()
            ml_model = load_ml_model()

            closed_start_time = None
            heavy_start_time = None
            last_sound_time = 0.0
            ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)
            adaptive_ear_history = deque(maxlen=ADAPTIVE_EAR_WINDOW)
            ear_threshold = EYE_CLOSED_EAR_THRESHOLD
            session_start_time = time.time()
            last_telemetry_sample_time = 0.0
            ml_result = unavailable_ml_result(ml_model)
            last_analysis_time = 0.0
            face_detected = False
            temporal_analyzer = TemporalFatigueAnalyzer()

            while self.should_run:
                success, frame = cap.read()
                if not success:
                    self.error_ready.emit("Frame read failed.")
                    break

                frame = cv2.flip(frame, 1)
                now = time.time()

                if now - last_analysis_time >= ANALYSIS_INTERVAL_SECONDS:
                    last_analysis_time = now
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
                        self._next_timestamp_ms(),
                    )

                    face_detected = bool(result.face_landmarks)
                    face_landmarks = result.face_landmarks[0] if face_detected else None
                    if face_detected:
                        (
                            ml_result,
                            closed_start_time,
                            heavy_start_time,
                            last_sound_time,
                            ear_threshold,
                            last_telemetry_sample_time,
                        ) = analyze_face(
                            frame,
                            face_landmarks,
                            closed_start_time,
                            heavy_start_time,
                            last_sound_time,
                            ear_history,
                            adaptive_ear_history,
                            ear_threshold,
                            session_start_time,
                            last_telemetry_sample_time,
                            ml_model,
                            temporal_analyzer,
                        )
                    else:
                        closed_start_time = None
                        heavy_start_time = None
                        ml_result = unavailable_ml_result(ml_model)
                        elapsed = now - session_start_time
                        last_telemetry_sample_time = update_no_face_telemetry(
                            elapsed,
                            ear_threshold,
                            ml_model,
                            last_telemetry_sample_time,
                            temporal_analyzer,
                        )

                    self.metrics_ready.emit(
                        {
                            "elapsed": now - session_start_time,
                            "face_detected": face_detected,
                            "ml_prediction": ml_result.get("ml_prediction", "Unavailable"),
                            "ml_drowsy_probability": ml_result.get("ml_drowsy_probability"),
                            "final_drowsiness_probability": ml_result.get(
                                "final_drowsiness_probability",
                                ml_result.get("ml_drowsy_probability"),
                            ),
                            "ml_raw_drowsy_probability": ml_result.get(
                                "ml_raw_drowsy_probability"
                            ),
                            "rule_based_score": ml_result.get("rule_based_score"),
                            "rule_based_status": ml_result.get("rule_based_status"),
                        }
                    )

                self.frame_ready.emit(frame_to_qimage(frame))
        except Exception as error:
            self.error_ready.emit(str(error))
        finally:
            if landmarker is not None:
                landmarker.close()
            cap.release()


class SafeDriveWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = CameraWorker()
        self.current_probability = None
        self.current_raw_probability = None
        self.current_rule_score = None
        self.face_detected = False

        self.setWindowTitle(APP_NAME)
        self.resize(1536, 960)
        self.setMinimumSize(1280, 760)

        root = GradientBackground()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(36, 34, 36, 34)
        outer.setSpacing(22)

        self.header = BrandHeader()
        outer.addWidget(self.header)

        body = QHBoxLayout()
        body.setSpacing(24)

        self.sidebar = Sidebar()
        self.camera_panel = CameraPanel()

        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(18)
        self.status_card = StatusCard()
        self.drive_card = TimerCard("Driving time", "TIME", "mm:ss")
        self.break_card = TimerCard("Last break", "REST", "not recorded")
        self.session_card = SessionCard()
        self.break_card.value_label.setText("--:--")

        right_column.addWidget(self.status_card)
        right_column.addWidget(self.drive_card)
        right_column.addWidget(self.break_card)
        right_column.addWidget(self.session_card)
        right_column.addStretch(1)

        right_widget = QWidget()
        right_widget.setFixedWidth(288)
        right_widget.setMinimumHeight(600)
        right_widget.setLayout(right_column)

        body.addWidget(self.sidebar)
        body.addWidget(self.camera_panel, stretch=1)
        body.addWidget(right_widget)
        outer.addLayout(body, stretch=1)

        self.worker.frame_ready.connect(self.camera_panel.set_frame)
        self.worker.metrics_ready.connect(self.update_metrics)
        self.worker.error_ready.connect(self.show_error)
        self.worker.start()

    def update_metrics(self, metrics):
        self.current_probability = metrics.get(
            "final_drowsiness_probability",
            metrics.get("ml_drowsy_probability"),
        )
        self.current_raw_probability = metrics.get("ml_raw_drowsy_probability")
        self.current_rule_score = metrics.get("rule_based_score")
        self.face_detected = bool(metrics.get("face_detected"))
        elapsed = metrics.get("elapsed", 0.0)

        self.camera_panel.update_data(
            self.current_probability,
            self.face_detected,
            self.current_raw_probability,
            self.current_rule_score,
        )
        self.status_card.update_data(self.current_probability, self.face_detected)
        self.session_card.update_data(self.current_probability)
        self.drive_card.value_label.setText(format_timer(elapsed))

    def show_error(self, message):
        self.status_card.message.setText(message)
        self.status_card.status.setText("Camera error")
        self.status_card.status.setStyleSheet(f"color: {RED}; border: none; background: transparent;")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


def run_qt_interface():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setStyle("Fusion")
    window = SafeDriveWindow()
    window.show()
    return app.exec()
