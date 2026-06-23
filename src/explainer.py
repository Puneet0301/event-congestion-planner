import numpy as np
from sklearn.pipeline import Pipeline

FEATURE_DISPLAY: dict[str, str] = {
    "event_cause":            "Event cause",
    "event_type":             "Event type",
    "corridor":               "Corridor",
    "zone":                   "Zone",
    "police_station":         "Police station",
    "hour_band":              "Time of day band",
    "priority":               "Priority level",
    "junction":               "Junction",
    "hour_of_day":            "Hour of day",
    "day_of_week":            "Day of week",
    "requires_road_closure":  "Road closure required",
    "month":                  "Month",
    "is_weekend":             "Weekend event",
    "desc_traffic_slow":      "Congestion keywords in description",
    "desc_breakdown":         "Breakdown keywords in description",
    "is_holiday":             "Public holiday",
    "holiday_risk_tier":      "Holiday / festival severity tier",
    "estimated_attendance":   "Expected attendance",
    "has_vip":                "VIP presence",
    "is_route_event":         "Route-based event",
    "corridor_high_rate":     "Historical HIGH-severity rate on corridor",
    "corridor_event_count":   "Historical event volume on corridor",
    "corridor_auth_rate":     "Authenticated incident rate on corridor",
    "corridor_closure_rate":  "Road closure frequency on corridor",
}


def _importance_dict(clf, feature_names: list | None = None) -> dict:
    names = feature_names or list(clf.feature_name_)
    imps  = clf.feature_importances_.astype(float)
    return {"names": names, "importances": imps}


def build_explainers(severity_pipeline: Pipeline, risk_models: dict) -> dict:
    """Return LightGBM feature-importance dicts — no shap dependency required."""
    return {
        "severity":   _importance_dict(severity_pipeline.named_steps["lgbm"]),
        "congestion": _importance_dict(risk_models["congestion"].named_steps["clf"]),
        "law_order":  _importance_dict(risk_models["law_order"].named_steps["clf"]),
    }


def _top5_drivers(names: list, importances: np.ndarray) -> list[dict]:
    total = float(np.sum(importances)) or 1.0
    drivers = [
        {
            "feature":   name,
            "display":   FEATURE_DISPLAY.get(name, name.replace("_", " ").title()),
            "shap":      float(val),
            "direction": "+",
            "pct":       round(float(val) / total * 100, 1),
        }
        for name, val in zip(names, importances)
    ]
    return sorted(drivers, key=lambda d: d["shap"], reverse=True)[:5]


def explain_severity(
    explainer: dict,
    severity_pipeline: Pipeline,  # noqa: ARG001
    features: dict,               # noqa: ARG001
    predicted_class: str,         # noqa: ARG001
) -> list[dict]:
    return _top5_drivers(explainer["names"], explainer["importances"])


def explain_risk(
    explainer: dict,
    risk_pipeline: Pipeline,  # noqa: ARG001
    features: dict,           # noqa: ARG001
) -> list[dict]:
    return _top5_drivers(explainer["names"], explainer["importances"])
