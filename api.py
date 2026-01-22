"""
FastAPI endpoint for habitat classification predictions.

Usage:
    python api.py

The server will start on http://0.0.0.0:4321
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import predict
from utils import decode_patch

HOST = "0.0.0.0"
PORT = 4321


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    patch: str  # base64-encoded numpy array (15, 35, 35) float32


class PredictResponse(BaseModel):
    """Response body from /predict endpoint."""
    prediction: int  # Class index 0-70


app = FastAPI(
    title="Habitat Classification API",
    description="Classify Icelandic satellite image patches into 71 habitat types",
    version="1.0.0"
)


@app.get("/")
def index():
    """Health check endpoint."""
    return {"status": "running", "message": "Habitat Classification API"}


@app.get("/api")
def api_info():
    """API information endpoint."""
    return {
        "service": "habitat-classification",
        "version": "1.0.0",
        "endpoints": {
            "/": "Health check",
            "/api": "API information",
            "/predict": "POST - Classify a patch"
        }
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Classify a satellite image patch.

    The patch should be base64-encoded numpy array of shape (15, 35, 35) with dtype float32.

    Returns the predicted habitat class index (0-70).
    """
    # Decode base64 to numpy array
    patch = decode_patch(request.patch)

    # Get prediction from model
    prediction = predict(patch)

    return PredictResponse(prediction=int(prediction))


if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    uvicorn.run("api:app", host=HOST, port=PORT, reload=False)
