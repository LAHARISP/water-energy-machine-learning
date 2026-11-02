def investigate_cause(resource, anomaly_result, context):

    causes = []

    recommendations = []

    if not anomaly_result["anomaly_detected"]:

        return {

            "status": "NORMAL",

            "possible_causes": [
                "No significant prediction error detected."
            ],

            "recommendations": [
                "Continue monitoring consumption."
            ]

        }

    actual = anomaly_result["actual"]

    predicted = anomaly_result["predicted"]

    residual = anomaly_result["residual"]

    print("\nInvestigating unusual consumption...")

    if residual > 0:

        # Unusually high water consumption.
        if resource == "water":

            causes.append(
                "Possible water leakage or unusual water usage."
            )

            recommendations.append(
                "Inspect plumbing, taps, toilets, and water meters."
            )

            recommendations.append(
                "Check recent water usage and maintenance records."
            )

        # Unusually high electricity consumption.
        elif resource == "electricity":

            if context["operating_hours"] > 14:

                causes.append(
                    "Equipment operating hours are unusually high."
                )

                recommendations.append(
                    "Review equipment operating schedules."
                )

            if context["temperature"] > 30:

                causes.append(
                    "Higher temperature may have increased "
                    "cooling demand."
                )

                recommendations.append(
                    "Review air-conditioning consumption and "
                    "weather conditions."
                )

            causes.append(
                "Possible equipment malfunction or unusual "
                "appliance usage."
            )

            recommendations.append(
                "Inspect high-consumption equipment."
            )

    else:

        causes.append(
            "Consumption is significantly lower than predicted."
        )

        recommendations.append(
            "Check meter readings, resource supply, and "
            "equipment operating status."
        )

    return {

        "status": "ANOMALY DETECTED",

        "actual_consumption": actual,

        "expected_consumption": predicted,

        "possible_causes": causes,

        "recommendations": recommendations,

        "confidence": (
            "Potential causes are inferred from available "
            "features. They are not confirmed diagnoses."
        )

    }