
def detect_anomaly(actual, predicted, threshold):

    residual = actual - predicted
    absolute_error = abs(residual)

    is_anomaly = absolute_error > threshold

    if is_anomaly:
        if residual > 0:
            direction = "Higher than expected"
        else:
            direction = "Lower than expected"
    else:
        direction = "Normal"

    return {
        "actual": actual,
        "predicted": predicted,
        "residual": residual,
        "absolute_error": absolute_error,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "direction": direction
    }
