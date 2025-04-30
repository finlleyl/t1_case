from fastapi import FastAPI
from src.ocr_bank_docs.api.routers import router as ocr_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.include_router(ocr_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все домены
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)
