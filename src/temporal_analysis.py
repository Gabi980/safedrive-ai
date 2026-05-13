from collections import deque
from math import sqrt

from config import (
    LONG_YAWN_SECONDS_THRESHOLD,
    MICROSLEEP_RECENT_SECONDS,
    MICROSLEEP_SECONDS_THRESHOLD,
    SLOW_BLINK_SECONDS_THRESHOLD,
    TEMPORAL_MIN_SAMPLES,
    TEMPORAL_WINDOWS_SECONDS,
    YAWN_COUNT_WINDOWS_SECONDS,
)


CLOSED_EYE_STATUSES = {"Eyes closed", "Drowsiness warning"}
HEAVY_EYE_STATUSES = {"Eyes heavy"}
PERCLOS_EYE_STATUSES = CLOSED_EYE_STATUSES | HEAVY_EYE_STATUSES
YAWN_STATUSES = {"Yawning detected"}
HEAD_DOWN_STATUSES = {"Head down"}
SIDE_LOOK_STATUSES = {"Looking sideways"}
HEAD_ABNORMAL_STATUSES = {"Head down", "Looking sideways", "Head tilted"}


class TemporalFatigueAnalyzer:
    def __init__(self, windows_seconds=TEMPORAL_WINDOWS_SECONDS):
        self.windows_seconds = tuple(sorted(windows_seconds))
        self.yawn_windows_seconds = tuple(sorted(set(YAWN_COUNT_WINDOWS_SECONDS)))
        self.max_window_seconds = max(
            max(self.windows_seconds),
            max(self.yawn_windows_seconds),
        )
        self.samples = deque()

    def add_sample(
        self,
        timestamp,
        face_detected,
        ear=None,
        raw_ear=None,
        left_ear=None,
        right_ear=None,
        mar=None,
        pitch=None,
        yaw=None,
        roll=None,
        eye_status="Unavailable",
        mouth_status="Unavailable",
        head_status="Unavailable",
    ):
        self.samples.append(
            {
                "time": timestamp,
                "face_detected": bool(face_detected),
                "ear": ear,
                "raw_ear": raw_ear,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "mar": mar,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "eye_status": eye_status,
                "mouth_status": mouth_status,
                "head_status": head_status,
            }
        )
        self._prune(timestamp)
        return self.get_features(timestamp)

    def get_features(self, now):
        features = {}
        for window_seconds in self.windows_seconds:
            window_samples = self._window_samples(now, window_seconds)
            prefix = f"temporal_{window_seconds}s"
            features.update(self._calculate_window_features(prefix, window_samples, now, window_seconds))
        features.update(self._primary_window_aliases(features))
        features.update(self._calculate_yawn_features(now))
        return features

    def _primary_window_aliases(self, features):
        primary_window_seconds = max(self.windows_seconds)
        prefix = f"temporal_{primary_window_seconds}s"
        return {
            "head_down_duration": features.get(f"{prefix}_head_down_duration"),
            "side_look_duration": features.get(f"{prefix}_side_look_duration"),
            "head_abnormal_duration": features.get(f"{prefix}_head_abnormal_duration"),
            "head_pose_instability": features.get(f"{prefix}_head_pose_instability"),
            "microsleep_active": features.get(f"{prefix}_microsleep_active"),
            "microsleep_recent": features.get(f"{prefix}_microsleep_recent"),
            "seconds_since_last_microsleep": features.get(
                f"{prefix}_seconds_since_last_microsleep"
            ),
        }

    def _calculate_yawn_features(self, now):
        features = {}

        for window_seconds in self.yawn_windows_seconds:
            window_samples = self._window_samples(now, window_seconds)
            features.update(
                self._calculate_yawn_window_features(
                    window_samples,
                    now,
                    window_seconds,
                )
            )

        primary_window_seconds = max(self.yawn_windows_seconds)
        features["avg_yawn_duration"] = features.get(
            f"avg_yawn_duration_{primary_window_seconds}s"
        )
        features["max_yawn_duration"] = features.get(
            f"max_yawn_duration_{primary_window_seconds}s"
        )
        features["long_yawn_count"] = features.get(
            f"long_yawn_count_{primary_window_seconds}s"
        )
        seconds_since_last_yawn = self._seconds_since_last_event(
            now,
            lambda sample: sample["mouth_status"] in YAWN_STATUSES,
        )
        features["seconds_since_last_yawn"] = seconds_since_last_yawn
        features["time_since_last_yawn"] = seconds_since_last_yawn

        return features

    def _calculate_yawn_window_features(self, samples, now, window_seconds):
        suffix = f"{window_seconds}s"
        if len(samples) < TEMPORAL_MIN_SAMPLES:
            return self._empty_yawn_window(suffix)

        yawn_durations = self._event_durations(
            samples,
            now,
            lambda sample: sample["mouth_status"] in YAWN_STATUSES,
        )

        return {
            f"yawn_count_{suffix}": len(yawn_durations),
            f"avg_yawn_duration_{suffix}": self._mean_list(yawn_durations),
            f"max_yawn_duration_{suffix}": self._max_list(yawn_durations),
            f"long_yawn_count_{suffix}": self._threshold_count(
                yawn_durations,
                LONG_YAWN_SECONDS_THRESHOLD,
            ),
        }

    def _empty_yawn_window(self, suffix):
        return {
            f"yawn_count_{suffix}": None,
            f"avg_yawn_duration_{suffix}": None,
            f"max_yawn_duration_{suffix}": None,
            f"long_yawn_count_{suffix}": None,
        }

    def _prune(self, now):
        oldest_allowed = now - self.max_window_seconds
        while self.samples and self.samples[0]["time"] < oldest_allowed:
            self.samples.popleft()

    def _window_samples(self, now, window_seconds):
        start = now - window_seconds
        return [sample for sample in self.samples if sample["time"] >= start]

    def _calculate_window_features(self, prefix, samples, now, window_seconds):
        if len(samples) < TEMPORAL_MIN_SAMPLES:
            return self._empty_window(prefix)

        eye_closure_events = self._event_intervals(
            samples,
            now,
            lambda sample: sample["eye_status"] in CLOSED_EYE_STATUSES,
        )
        eye_closure_durations = [
            event["duration"]
            for event in eye_closure_events
        ]
        blink_count = len(eye_closure_durations)
        longest_eye_closure = self._max_list(eye_closure_durations)
        microsleep_events = [
            event
            for event in eye_closure_events
            if event["duration"] >= MICROSLEEP_SECONDS_THRESHOLD
        ]
        latest_microsleep = microsleep_events[-1] if microsleep_events else None
        microsleep_active = bool(
            latest_microsleep is not None
            and latest_microsleep["active"]
        )
        seconds_since_last_microsleep = self._seconds_since_event(latest_microsleep, now)
        microsleep_recent = (
            seconds_since_last_microsleep is not None
            and seconds_since_last_microsleep <= MICROSLEEP_RECENT_SECONDS
        )
        face_visible_duration = self._denominator_duration(
            samples,
            now,
            window_seconds,
            lambda sample: sample["face_detected"],
        )
        perclos = self._duration_ratio(
            samples,
            now,
            window_seconds,
            lambda sample: sample["eye_status"] in PERCLOS_EYE_STATUSES,
            lambda sample: sample["face_detected"],
        )
        head_down_duration = self._positive_duration(
            samples,
            now,
            window_seconds,
            lambda sample: sample["head_status"] in HEAD_DOWN_STATUSES,
            lambda sample: sample["face_detected"],
        )
        side_look_duration = self._positive_duration(
            samples,
            now,
            window_seconds,
            lambda sample: sample["head_status"] in SIDE_LOOK_STATUSES,
            lambda sample: sample["face_detected"],
        )
        head_abnormal_duration = self._positive_duration(
            samples,
            now,
            window_seconds,
            lambda sample: sample["head_status"] in HEAD_ABNORMAL_STATUSES,
            lambda sample: sample["face_detected"],
        )
        head_down_durations = self._event_durations(
            samples,
            now,
            lambda sample: sample["head_status"] in HEAD_DOWN_STATUSES,
        )
        side_look_durations = self._event_durations(
            samples,
            now,
            lambda sample: sample["head_status"] in SIDE_LOOK_STATUSES,
        )
        pitch_std = self._std_value(samples, "pitch")
        yaw_std = self._std_value(samples, "yaw")
        roll_std = self._std_value(samples, "roll")

        return {
            f"{prefix}_sample_count": len(samples),
            f"{prefix}_duration": self._duration(samples, now, window_seconds),
            f"{prefix}_face_visible_ratio": self._duration_ratio(
                samples,
                now,
                window_seconds,
                lambda sample: sample["face_detected"],
            ),
            f"{prefix}_face_visible_duration": face_visible_duration,
            f"{prefix}_ear_mean": self._mean_value(samples, "ear"),
            f"{prefix}_ear_min": self._min_value(samples, "ear"),
            f"{prefix}_ear_std": self._std_value(samples, "ear"),
            f"{prefix}_mar_mean": self._mean_value(samples, "mar"),
            f"{prefix}_mar_max": self._max_value(samples, "mar"),
            f"{prefix}_mar_std": self._std_value(samples, "mar"),
            f"{prefix}_pitch_mean": self._mean_value(samples, "pitch"),
            f"{prefix}_pitch_std": pitch_std,
            f"{prefix}_yaw_mean": self._mean_value(samples, "yaw"),
            f"{prefix}_yaw_std": yaw_std,
            f"{prefix}_roll_mean": self._mean_value(samples, "roll"),
            f"{prefix}_roll_std": roll_std,
            f"{prefix}_head_pose_instability": self._head_pose_instability(
                pitch_std,
                yaw_std,
                roll_std,
            ),
            f"{prefix}_perclos": perclos,
            f"{prefix}_perclos_percent": self._percent(perclos),
            f"{prefix}_perclos_duration": self._positive_duration(
                samples,
                now,
                window_seconds,
                lambda sample: sample["eye_status"] in PERCLOS_EYE_STATUSES,
                lambda sample: sample["face_detected"],
            ),
            f"{prefix}_closed_eye_ratio": self._duration_ratio(
                samples,
                now,
                window_seconds,
                lambda sample: sample["eye_status"] in CLOSED_EYE_STATUSES,
                lambda sample: sample["face_detected"],
            ),
            f"{prefix}_heavy_eye_ratio": self._duration_ratio(
                samples,
                now,
                window_seconds,
                lambda sample: sample["eye_status"] in HEAVY_EYE_STATUSES,
                lambda sample: sample["face_detected"],
            ),
            f"{prefix}_yawn_ratio": self._duration_ratio(
                samples,
                now,
                window_seconds,
                lambda sample: sample["mouth_status"] in YAWN_STATUSES,
            ),
            f"{prefix}_yawn_event_count": self._event_count(
                samples,
                lambda sample: sample["mouth_status"] in YAWN_STATUSES,
            ),
            f"{prefix}_head_down_ratio": self._duration_ratio(
                samples,
                now,
                window_seconds,
                lambda sample: sample["head_status"] in HEAD_DOWN_STATUSES,
                lambda sample: sample["face_detected"],
            ),
            f"{prefix}_head_down_duration": head_down_duration,
            f"{prefix}_head_down_event_count": len(head_down_durations),
            f"{prefix}_longest_head_down_duration": self._max_list(head_down_durations),
            f"{prefix}_side_look_ratio": self._duration_ratio(
                samples,
                now,
                window_seconds,
                lambda sample: sample["head_status"] in SIDE_LOOK_STATUSES,
                lambda sample: sample["face_detected"],
            ),
            f"{prefix}_side_look_duration": side_look_duration,
            f"{prefix}_side_look_event_count": len(side_look_durations),
            f"{prefix}_longest_side_look_duration": self._max_list(side_look_durations),
            f"{prefix}_head_abnormal_ratio": self._duration_ratio(
                samples,
                now,
                window_seconds,
                lambda sample: sample["head_status"] in HEAD_ABNORMAL_STATUSES,
                lambda sample: sample["face_detected"],
            ),
            f"{prefix}_head_abnormal_duration": head_abnormal_duration,
            f"{prefix}_eye_closure_count": blink_count,
            f"{prefix}_blink_count": blink_count,
            f"{prefix}_blink_rate_per_minute": self._events_per_minute(
                blink_count,
                face_visible_duration,
            ),
            f"{prefix}_avg_blink_duration": self._mean_list(eye_closure_durations),
            f"{prefix}_max_blink_duration": longest_eye_closure,
            f"{prefix}_longest_eye_closure": longest_eye_closure,
            f"{prefix}_slow_blink_count": self._threshold_count(
                eye_closure_durations,
                SLOW_BLINK_SECONDS_THRESHOLD,
            ),
            f"{prefix}_microsleep_event_count": len(microsleep_events),
            f"{prefix}_microsleep_active": microsleep_active,
            f"{prefix}_microsleep_recent": microsleep_recent,
            f"{prefix}_seconds_since_last_microsleep": seconds_since_last_microsleep,
            f"{prefix}_microsleep_detected": microsleep_active or microsleep_recent,
        }

    def _empty_window(self, prefix):
        names = [
            "sample_count",
            "duration",
            "face_visible_ratio",
            "face_visible_duration",
            "ear_mean",
            "ear_min",
            "ear_std",
            "mar_mean",
            "mar_max",
            "mar_std",
            "pitch_mean",
            "pitch_std",
            "yaw_mean",
            "yaw_std",
            "roll_mean",
            "roll_std",
            "head_pose_instability",
            "perclos",
            "perclos_percent",
            "perclos_duration",
            "closed_eye_ratio",
            "heavy_eye_ratio",
            "yawn_ratio",
            "yawn_event_count",
            "head_down_ratio",
            "head_down_duration",
            "head_down_event_count",
            "longest_head_down_duration",
            "side_look_ratio",
            "side_look_duration",
            "side_look_event_count",
            "longest_side_look_duration",
            "head_abnormal_ratio",
            "head_abnormal_duration",
            "eye_closure_count",
            "blink_count",
            "blink_rate_per_minute",
            "avg_blink_duration",
            "max_blink_duration",
            "longest_eye_closure",
            "slow_blink_count",
            "microsleep_event_count",
            "microsleep_active",
            "microsleep_recent",
            "seconds_since_last_microsleep",
            "microsleep_detected",
        ]
        return {f"{prefix}_{name}": None for name in names}

    def _duration(self, samples, now, window_seconds):
        if not samples:
            return 0.0

        start = max(now - window_seconds, samples[0]["time"])
        end = max(start, samples[-1]["time"])
        return end - start

    def _values(self, samples, key):
        return [
            float(sample[key])
            for sample in samples
            if sample.get(key) is not None
        ]

    def _mean_value(self, samples, key):
        values = self._values(samples, key)
        if not values:
            return None
        return sum(values) / len(values)

    def _min_value(self, samples, key):
        values = self._values(samples, key)
        if not values:
            return None
        return min(values)

    def _max_value(self, samples, key):
        values = self._values(samples, key)
        if not values:
            return None
        return max(values)

    def _std_value(self, samples, key):
        values = self._values(samples, key)
        if len(values) < 2:
            return 0.0 if values else None

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return sqrt(variance)

    def _head_pose_instability(self, pitch_std, yaw_std, roll_std):
        values = [
            value
            for value in (pitch_std, yaw_std, roll_std)
            if value is not None
        ]
        if not values:
            return None

        return sqrt(sum(value ** 2 for value in values) / len(values))

    def _duration_ratio(self, samples, now, window_seconds, predicate, denominator_predicate=None):
        positive_duration = self._positive_duration(
            samples,
            now,
            window_seconds,
            predicate,
            denominator_predicate,
        )
        total_duration = self._denominator_duration(
            samples,
            now,
            window_seconds,
            denominator_predicate,
        )

        if total_duration <= 0:
            return None

        return positive_duration / total_duration

    def _positive_duration(self, samples, now, window_seconds, predicate, denominator_predicate=None):
        positive_duration = 0.0
        window_start = now - window_seconds
        window_end = now

        for index, sample in enumerate(samples):
            start = max(sample["time"], window_start)
            if index + 1 < len(samples):
                end = min(samples[index + 1]["time"], window_end)
            else:
                end = window_end

            duration = max(0.0, end - start)
            if denominator_predicate is not None and not denominator_predicate(sample):
                continue

            if predicate(sample):
                positive_duration += duration

        return positive_duration

    def _denominator_duration(self, samples, now, window_seconds, denominator_predicate=None):
        total_duration = 0.0
        window_start = now - window_seconds
        window_end = now

        for index, sample in enumerate(samples):
            start = max(sample["time"], window_start)
            if index + 1 < len(samples):
                end = min(samples[index + 1]["time"], window_end)
            else:
                end = window_end

            duration = max(0.0, end - start)
            if denominator_predicate is None or denominator_predicate(sample):
                total_duration += duration

        return total_duration

    def _percent(self, ratio):
        if ratio is None:
            return None

        return ratio * 100.0

    def _event_count(self, samples, predicate):
        count = 0
        was_active = False

        for sample in samples:
            is_active = predicate(sample)
            if is_active and not was_active:
                count += 1
            was_active = is_active

        return count

    def _event_durations(self, samples, now, predicate):
        return [
            event["duration"]
            for event in self._event_intervals(samples, now, predicate)
        ]

    def _event_intervals(self, samples, now, predicate):
        events = []
        current_start = None

        for sample in samples:
            if predicate(sample):
                if current_start is None:
                    current_start = sample["time"]
            elif current_start is not None:
                end_time = sample["time"]
                events.append(
                    {
                        "start": current_start,
                        "end": end_time,
                        "duration": max(0.0, end_time - current_start),
                        "active": False,
                    }
                )
                current_start = None

        if current_start is not None:
            events.append(
                {
                    "start": current_start,
                    "end": now,
                    "duration": max(0.0, now - current_start),
                    "active": True,
                }
            )

        return events

    def _longest_event_duration(self, samples, now, predicate):
        return self._max_list(self._event_durations(samples, now, predicate))

    def _seconds_since_event(self, event, now):
        if event is None:
            return None

        if event["active"]:
            return 0.0

        return max(0.0, now - event["end"])

    def _seconds_since_last_event(self, now, predicate):
        last_end_time = None
        was_active = False

        for sample in self.samples:
            is_active = predicate(sample)
            if was_active and not is_active:
                last_end_time = sample["time"]
            was_active = is_active

        if was_active:
            return 0.0

        if last_end_time is None:
            return None

        return max(0.0, now - last_end_time)

    def _events_per_minute(self, event_count, duration_seconds):
        if duration_seconds is None or duration_seconds <= 0:
            return None

        return event_count / duration_seconds * 60.0

    def _mean_list(self, values):
        if not values:
            return 0.0

        return sum(values) / len(values)

    def _max_list(self, values):
        if not values:
            return 0.0

        return max(values)

    def _threshold_count(self, values, threshold):
        return sum(1 for value in values if value >= threshold)
