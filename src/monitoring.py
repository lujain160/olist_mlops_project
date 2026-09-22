import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "monitoring.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ServiceMonitoring:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.prediction_logs_path = log_dir / "prediction_history.csv"

    def track_request(self, latency: float, is_error: bool = False):
        """Track request metrics: count, latency, error rate."""
        self.request_count += 1
        self.total_latency += latency
        if is_error:
            self.error_count += 1

        error_rate = (self.error_count / self.request_count) * 100
        avg_latency = self.total_latency / self.request_count

        logger.info(
            f"[Metrics] Requests: {self.request_count} | "
            f"Avg Latency: {avg_latency:.2f}ms | Error Rate: {error_rate:.2f}%"
        )

        if error_rate > 10.0:
            logger.warning(f"ALERT! High error rate detected: {error_rate:.2f}%")
        if avg_latency > 500.0:
            logger.warning(f"ALERT! High latency detected: {avg_latency:.2f}ms")

    def log_predictions_for_evaluation(
        self, input_data: dict, prediction: int, probability: float
    ):
        """Log predictions with timestamp for later evaluation."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **input_data,
            "predicted_is_late": prediction,
            "prediction_probability": probability,
            "actual_is_late": None,
        }
        df_new = pd.DataFrame([log_entry])
        if self.prediction_logs_path.exists():
            df_new.to_csv(
                self.prediction_logs_path, mode="a", header=False, index=False
            )
        else:
            df_new.to_csv(self.prediction_logs_path, index=False)
        logger.info("Prediction logged successfully for future evaluation.")

    def check_data_drift(
        self, recent_data: pd.DataFrame, reference_mean_hour: float = 12.0
    ):
        """Check for data drift in recent predictions."""
        if "purchase_hour" in recent_data.columns:
            current_mean_hour = recent_data["purchase_hour"].mean()
            drift_diff = abs(current_mean_hour - reference_mean_hour)
            logger.info(
                f"[Drift Check] Current mean purchase hour: {current_mean_hour:.2f} "
                f"(Diff from baseline: {drift_diff:.2f})"
            )
            if drift_diff > 4.0:
                logger.warning(
                    "ALERT! Data Drift detected in purchase hours compared to baseline!"
                )


# Global monitor instance
monitor = ServiceMonitoring()
