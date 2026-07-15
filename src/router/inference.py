from pathlib import Path
import numpy as np
from setfit import SetFitModel

BASE_DIR = Path(__file__).parent

model = SetFitModel.from_pretrained(
    BASE_DIR.parent.parent / "models" / "router"
)

LABELS = {
    0: "chat",
    1: "retrieval"
}

THRESHOLD = 0.65


def route_query(query: str) -> str:
    probs = model.predict_proba([query])[0]

    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])

    print(f"Prediction: {LABELS[prediction]}")
    print(f"Confidence: {confidence:.3f}")

    if confidence < THRESHOLD:
        print("Confidence too low -> using retrieval")
        return "retrieval"

    return LABELS[prediction]