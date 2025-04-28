from fastapi import FastAPI
from src.ocr_bank_docs.api.routers import router as ocr_router



app = FastAPI()


app.include_router(ocr_router)

