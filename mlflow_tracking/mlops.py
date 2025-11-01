import mlflow
import mlflow.pytorch
import json
from datetime import datetime
import os


class MLflowTracker:
    def __init__(self, experiment_name="Pothole_Detection"):
        mlflow.set_experiment(experiment_name)
        self.run = None
        self.run_active = False
        self.detection_count = 0

    def start_run(self, run_name=None):
        """Start MLflow tracking run"""
        try:
            if run_name is None:
                run_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.run = mlflow.start_run(run_name=run_name)
            self.run_active = True
            print(f"✅ MLflow run started: {run_name}")
            return self.run
        except Exception as e:
            print(f"❌ MLflow start_run failed: {e}")
            self.run_active = False
            return None

    def log_param(self, key, value):
        """Safely log parameter"""
        if self.run_active:
            try:
                mlflow.log_param(key, str(value))
                print(f"📝 Logged param: {key}={value}")
            except Exception as e:
                print(f"❌ log_param failed: {e}")

    def log_metric(self, key, value, step=None):
        """Safely log metric"""
        if self.run_active:
            try:
                if step is not None:
                    mlflow.log_metric(key, float(value), step=step)
                else:
                    mlflow.log_metric(key, float(value))
                print(f"📊 Logged metric: {key}={value}")
            except Exception as e:
                print(f"❌ log_metric failed: {e}")

    def log_detection(self, image_path, dimensions, assessment):
        """Log individual detection results"""
        if not self.run_active:
            return

        self.detection_count += 1

        try:
            # Always log detection attempt
            self.log_metric("total_detections", self.detection_count, step=self.detection_count)

            if dimensions:
                # Log all dimensions as metrics with step for tracking over time
                self.log_metric("pothole_area_px", float(dimensions['area_px']), step=self.detection_count)
                self.log_metric("pothole_width_px", float(dimensions['width_px']), step=self.detection_count)
                self.log_metric("pothole_height_px", float(dimensions['height_px']), step=self.detection_count)
                self.log_metric("pothole_area_cm2", float(dimensions['area_cm2']), step=self.detection_count)
                self.log_metric("coverage_percent", float(dimensions['coverage_percent']), step=self.detection_count)
                self.log_metric("danger_score", float(assessment['score']), step=self.detection_count)

                # Create danger level mapping for metrics
                danger_levels = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0}
                self.log_metric("danger_level_numeric", danger_levels.get(assessment['level'], 0),
                                step=self.detection_count)

                print(f"✅ Detection #{self.detection_count}: {assessment['level']} - {assessment['score']}/100")
            else:
                # Log when no pothole detected
                self.log_metric("no_pothole_detected", 1, step=self.detection_count)
                self.log_metric("danger_level_numeric", 0, step=self.detection_count)
                print(f"✅ Detection #{self.detection_count}: No pothole")

        except Exception as e:
            print(f"❌ log_detection failed: {e}")

    def log_stats(self, stats):
        """Log statistics summary"""
        if not self.run_active:
            return

        try:
            # Log current stats
            self.log_metric("cumulative_total", stats['total'])
            self.log_metric("cumulative_critical", stats['critical'])
            self.log_metric("cumulative_high", stats['high'])
            self.log_metric("cumulative_medium", stats['medium'])
            self.log_metric("cumulative_low", stats['low'])

            # Calculate percentages
            if stats['total'] > 0:
                self.log_metric("critical_percentage", (stats['critical'] / stats['total']) * 100)
                self.log_metric("high_percentage", (stats['high'] / stats['total']) * 100)
                self.log_metric("medium_percentage", (stats['medium'] / stats['total']) * 100)
                self.log_metric("low_percentage", (stats['low'] / stats['total']) * 100)

        except Exception as e:
            print(f"❌ log_stats failed: {e}")

    def end_run(self):
        """End MLflow run"""
        if self.run_active:
            try:
                mlflow.end_run()
                print("✅ MLflow run ended")
            except:
                pass
            self.run = None
            self.run_active = False