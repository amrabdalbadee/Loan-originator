import uvicorn
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from src.core.config import settings

app = FastAPI(
    title="Loan Originator API",
    description="AI-powered document verification and loan assessment platform",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/applications")
async def submit_application(request: Request):
    # TODO: Logic for submitting application and triggering stage 0 (Intake)
    return JSONResponse(content={"message": "Application submitted", "application_id": "uuid"}, status_code=202)

@app.get("/applications/{application_id}")
async def get_application_status(application_id: str):
    # TODO: Logic to retrieve status & metrics
    return {"id": application_id, "status": "PENDING"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
