from decimal import Decimal, ROUND_HALF_UP

import pandas as pd


def score_roster(records, model, features):
    frame = pd.DataFrame(records)
    predictions = model.predict(frame[features]).clip(0, 100)
    results = frame[["Student name", "USN"]].copy()
    results["Predicted score"] = predictions.round(1)
    mean = results["Predicted score"].mean()
    average = float(Decimal(str(mean)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    results["Marks from average"] = (results["Predicted score"] - average).round(1)
    return results.sort_values(["Predicted score", "USN"]), average
