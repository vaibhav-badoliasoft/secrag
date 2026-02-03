from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "SecRAG backend is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "size_in_bytes": len(contents)
    }