import azure.functions as func
import logging
import os
import shutil
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma 
from langchain.prompts import ChatPromptTemplate  # for Chat Prompts
from get_embedding_function import get_embedding_function_hf # local python file
 
# [Start consuming Azure cloud serverless deployed llama2 model]
# from langchain.schema import SystemMessage
from langchain_community.chat_models.azureml_endpoint import (
                        AzureMLChatOnlineEndpoint,
                        AzureMLEndpointApiType,
                        LlamaChatContentFormatter,
                        )

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

CHROMA_PATH = ".\chroma"
DATA_PATH = "data"
 
PROMPT_TEMPLATE = """
                    Answer the question based only on the following context:
 
                    {context}
 
                    ---
 
                    Answer the question based on the above context: {question}
                """
 
def main_rag(query):
    start_time = time.time()
    result  = query_rag(query)
    end_time = time.time()
    # print("Total Time Taken: ", end_time - start_time, "seconds")
    return {"response": result}
   
def query_rag(query_text: str):
    logging.info(".....Clearing Database")
    clear_database()
    # # Create (or update) the data store.
    start_time = time.time()
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    end_time = time.time()
    doc_load_time = end_time - start_time
    logging.info(f"Document loaded in database in {doc_load_time} seconds")  
   
    embedding_function = get_embedding_function_hf()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # Search the DB.
    # results = db.similarity_search_with_score(query_text, k=2)
    results = db.similarity_search_with_score("Contact number of the person", k = 2)
    logging.info("......results from similarity search: \n{}".format(results[0]))
 
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
   
    chat_model = AzureMLChatOnlineEndpoint(
        endpoint_url="https://Lxn-Llama-2-7b-chat-tyhga.eastus2.models.ai.azure.com/v1/chat/completions",
        endpoint_api_type=AzureMLEndpointApiType.serverless,
        endpoint_api_key="endpoint_api_key_here",
        content_formatter=LlamaChatContentFormatter(),
        # prompt = prompt
        )
   
    logging.info("........ Azure Llama 2 chat model created.")
    response_text = chat_model.invoke(prompt)
    logging.info("........ Type of response content: {}".format(type(response_text.content)))
    logging.info("........ Received response content: \n{}".format(response_text.content))
    # response_text = response_text['message']['content']
 
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    logging.info("{}".format(formatted_response))
   
    return response_text.content
 
 
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()
 
 
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
 
 
def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function_hf()
                )
    logging.info("........Embedding done through HuggingFace Model.")
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
 
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logging.info(f".....Number of existing documents in DB: {len(existing_ids)}")
 
    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
 
    if len(new_chunks):
        logging.info(f".....Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        logging.info("......No new documents to add")
       
def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
 
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
 
        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
 
        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
 
        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
 
    return chunks
 
 
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


@app.route(route="skillset_extract_resume")
def skillset_extract_resume(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        result = main_rag(name)
        return func.HttpResponse(f"Hello, {result}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )