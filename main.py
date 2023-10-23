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
import asyncio

load_dotenv()
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
app = FastAPI()


class EventMetaDataResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="List 3 most important keywords which summarizing the event's content, aiding in search and categorization.")
    title: str = Field(..., description="A short title for the event; it should be limited to 20 characters.")
    location: Optional[str] = Field(..., description="The location of the event.")
    start_time: Optional[str] = Field(..., description="The time event begin in Iso8601 format. The default year is 2023.")
    end_time: Optional[str] = Field(..., description="The time event finish in Iso8601 format. The default year is 2023.")


class NoteMetaDataResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="List 3 most important keywords which summarizing the note's content, aiding in search and categorization.")
    title: str = Field(..., description="A short subject or topic of the notes; it should be limited to 20 characters.")


class Item(BaseModel):
    name: str = Field(..., description="The name of the item.")
    quantity: float = Field(..., description="Quantity of the item purchased.")
    total_price: float = Field(..., description="Total price of the purchased item.")


class ReceiptMetaDataResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="List 3 most important keywords which summarizing the receipt's content, aiding in search and categorization.")
    title: str = Field(..., description="A short name to identify the receipt; it should be limited to 20 characters.")
    store_name: Optional[str] = Field(..., description="Name of the store where the purchase was made.")
    items: List[Item] = Field(..., description="List of items purchased.")
    total_price: Optional[float] = Field(..., description="Total price of all items purchased.")


class OthersMetaDataResponse(BaseModel):
    keywords: Optional[List[str]] = Field(...,
                                          description="List 3 most important keywords which summarizing the document's content, aiding in search and categorization.")
    title: str = Field(..., description="A short name or title to identify the document; it should be limited to 20 characters.")


class EventResponse(EventMetaDataResponse):
    content_md: str = Field(...,
                            description="The summarization of the event in Markdown format.")


class NoteResponse(NoteMetaDataResponse):
    content_md: str = Field(...,
                            description="The content of the note in Markdown format.")


class ReceiptResponse(ReceiptMetaDataResponse):
    content_md: str = Field(...,
                            description="A summary or description of the purchase, in Markdown format.")


class OthersResponse(OthersMetaDataResponse):
    content_md: str = Field(...,
                            description="Content or description of the document, in Markdown format.")


DocumentsDict = {
    "event": {"type_description": "an event or activity", "response_model": EventMetaDataResponse,
              "content_md_prompt": "Write a short summary of the event in Markdown format, sprinkled with emojis, and incorporate various levels of headlines to improve readability."},
    "note": {"type_description": "a lecture note or a piece of information", "response_model": NoteMetaDataResponse,
             "content_md_prompt": "Write a short summary of the note in Markdown format, sprinkled with emojis, and incorporate various levels of headlines to improve readability."},
    "receipt": {"type_description": "a shopping receipt or invoice", "response_model": ReceiptMetaDataResponse,
                "content_md_prompt": "Write a short summary of the receipt in Markdown format, sprinkled with emojis, and incorporate various levels of headlines to improve readability."},
    "others": {"type_description": "a document of any type", "response_model": OthersMetaDataResponse,
               "content_md_prompt": "Write a short summary of the document in Markdown format, sprinkled with emojis, and incorporate various levels of headlines to improve readability."},
}

system_msg = "You are a professional organizer whose goal is to convert unstructured data into a formatted structure " \
             "and extract valuable information."

meta_data_prompt = PromptTemplate(
    template="You are presented with a disorganized document containing information about "
             "{type_description}. Your task is to extract crucial details from the document."
             "\n{format_instructions}\n\nDocument:\n{doc}",
    input_variables=['doc', 'type_description', 'format_instructions'])

content_md_prompt = PromptTemplate(
    template="You are presented with a disorganized document containing information about "
             "{type_description}. Your task is to extract crucial details from the document.\n"
             "{instruction}\n\nDocument:\n{doc}",
    input_variables=['doc', 'type_description', 'instruction']
)


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
    meta_data_messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=meta_data_prompt.format_prompt(
            doc=text,
            type_description=type_description,
            format_instructions=parser.get_format_instructions()
        ).to_string()),
    ]
    content_md_message = [
        SystemMessage(content=system_msg),
        HumanMessage(content=content_md_prompt.format_prompt(
            doc=text,
            type_description=type_description,
            instruction=DocumentsDict[docType]["content_md_prompt"]
        ).to_string()),
    ]

    # get response from chat model
    meta_data_response, content_md_response = await asyncio.gather(
        asyncio.to_thread(chat_model.predict_messages, meta_data_messages),
        asyncio.to_thread(chat_model.predict_messages, content_md_message)
    )

    # parse response
    meta_data_response = parser.parse(meta_data_response.content)
    meta_data_response = dict(meta_data_response)
    meta_data_response['content_md'] = content_md_response.content

    return meta_data_response
