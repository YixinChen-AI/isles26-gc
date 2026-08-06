"""Grand Challenge invoke API for the ISLES 2026 algorithm."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Response, status
from uvicorn.config import LOGGING_CONFIG

import inference


MODEL = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    MODEL = inference.init_model()
    yield
    MODEL = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    code = (
        status.HTTP_200_OK
        if MODEL is not None
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return Response(status_code=code)


@app.post("/invoke")
async def invoke():
    inference.run(MODEL)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    config = LOGGING_CONFIG.copy()
    config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=config)
