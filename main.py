from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import Optional, List
from typing import Union

load_dotenv()
chat_model = ChatOpenAI()
app = FastAPI()


class EventResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="Keywords summarizing the event's content, aiding in search and categorization.")
    title: str = Field(..., description="The title of the event.")
    location: Optional[str] = Field(..., description="The location of the event.")
    start_time: Optional[str] = Field(..., description="The time event begin.")
    end_time: Optional[str] = Field(..., description="The time event finish.")
    content_md: str = Field(...,
                            description="The content of the event in Markdown format. sprinkled with emojis for "
                                        "enhanced readability.")


class NoteResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="Keywords summarizing the note's content, aiding in search and categorization.")
    title: str = Field(..., description="The subject or topic of the notes.")
    content_md: str = Field(..., description="The content of the note in Markdown format. sprinkled with emojis for "
                                             "enhanced readability.")


class Item(BaseModel):
    name: str = Field(..., description="The name of the item.")
    quantity: float = Field(..., description="Quantity of the item purchased.")
    total_price: float = Field(..., description="Total price of the purchased item.")


class ReceiptResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="Keywords summarizing the receipt's content, aiding in search and"
                                                      "categorization.")
    title: str = Field(..., description="A short name to identify the receipt.")
    store_name: Optional[str] = Field(..., description="Name of the store where the purchase was made.")
    items: Optional[List[Item]] = Field(..., description="List of items purchased.")
    total_price: Optional[float] = Field(..., description="Total price of all items purchased.")
    content_md: str = Field(..., description="A summary or description of the purchase, in Markdown format. "
                                             "sprinkled with emojis for enhanced readability.")


class OthersResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="Keywords summarizing the document's content, aiding in search and categorization.")
    title: str = Field(..., description="A short name or title to identify the document.")
    content_md: str = Field(..., description="Content or description of the document, in Markdown format. sprinkled "
                                             "with emojis for enhanced readability.")


DocumentsDict = {
    "event": {"type_description": "an event or activity", "response_model": EventResponse},
    "note": {"type_description": "a lecture note or a piece of information", "response_model": NoteResponse},
    "receipt": {"type_description": "a shopping receipt or invoice", "response_model": ReceiptResponse},
    "others": {"type_description": "a document of any type", "response_model": OthersResponse},
}

system_msg = "You are a professional organizer whose goal is to convert unstructured data into a formatted structure " \
             "and extract valuable information."

main_human_prompt = PromptTemplate(
    template="You are presented with a disorganized document containing information about "
             "{type_description}. Your task is to extract crucial details from the document."
             "\n{format_instructions}\n---\ndocument:\n{doc}",
    input_variables=['doc', 'type_description', 'format_instructions'])


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/beautify-text", response_model=Union[EventResponse, NoteResponse, ReceiptResponse, OthersResponse])
async def beautify_text(
        text: str = Query(..., description="The input text to be beautified."),
        docType: str = Query(..., description="The document type."),
):
    docType = docType.lower()
    assert docType in ["event", "note", "receipt", "others"], "Invalid document type."

    # form chat prompt
    type_description = DocumentsDict[docType]["type_description"]
    response_model_cls = DocumentsDict[docType]["response_model"]
    parser = PydanticOutputParser(pydantic_object=response_model_cls)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=main_human_prompt.format_prompt(
            doc=text,
            type_description=type_description,
            format_instructions=parser.get_format_instructions()
        ).to_string()),
    ]

    # get response from chat model
    llm_response = chat_model.predict_messages(messages)

    # parse response
    parsed_response = parser.parse(llm_response.content)

    return parsed_response
