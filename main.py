from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class BeautifyTextResponse(BaseModel):
    location: str
    start_time: str
    content_md: str


@app.get("/beautify-text", response_model=BeautifyTextResponse)
async def beautify_text(
        text: str = Query(..., description="The input text to be beautified."),
        docType: str = Query(..., description="The document type."),
):
    # Here, you can process the input 'text' and 'docType' as needed to generate your JSON response.
    # For demonstration purposes, I'll create a simple response with placeholders.

    location = "Sample Location"
    start_time = "Sample Start Time"
    content_md = "Sample Markdown Content"

    response_data = {
        "location": location,
        "start_time": start_time,
        "content_md": content_md,
    }

    return response_data
