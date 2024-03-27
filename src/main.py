import os
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
import database
import ingest.main as ingest
import traceback
import uuid
from typing import Optional
from log import log
from contextlib import asynccontextmanager
import logging
from database.db import migrate
from config import settings


@asynccontextmanager
async def lifespan(a: FastAPI):
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.cache_dir, exist_ok=True)
    # DB Initialization & Migrations
    try:
        log.info("Running database migrations")
        await migrate()
    except Exception as e:
        logging.error(f"Database migration failed: {e}\n{traceback.format_exc()}")
    log.info("Database migrations completed")
    yield
    # Shutdown


app = FastAPI(title="Rubra - Knowledge Retrieval API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="src/static"), name="static")


#
# Pydantic Data Models
#
class Dataset(BaseModel):
    name: str
    embed_dim: Optional[int] = Field(1536, description="Embedding Dimension")


class Query(BaseModel):
    prompt: str
    topk: Optional[int] = Field(5, description="Number of results to return")


class Ingest(BaseModel):
    filename: Optional[str] = Field(None, description="Filename")
    file_id: Optional[str] = Field(None, description="File ID")
    content: str  # Base64 encoded data


#
# Exception Handler
#
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    log.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


#
# Endpoints
#
@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    return HTMLResponse(
        content="""
        <html>
            <body>
                <p align="center">
                    <img src="/static/img/icon.png" />
                </p>
                <h1 align="center">Rubra - Knowledge Retrieval API</h1>
                <p align="center">Visit <a href="/docs">/docs</a> for the API documentation</p>
            </body>
        </html>
        """
    )


@app.get("/docs", include_in_schema=False)
async def swagger_ui_html():
    return get_swagger_ui_html(
        title=app.title + "- Swagger UI",
        openapi_url="/openapi.json",
        swagger_favicon_url="/static/img/favicon.ico",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )


@app.get("/favicon.io", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(path="/static/img/favicon.ico")


@app.post("/datasets/create")
async def create_dataset(dataset: Dataset) -> Dataset:
    """Endpoint to create a new dataset in the VectorDB.

    Args:
        dataset (str): Name of the dataset to create

    Raises:
        HTTPException: 409 Conflict if the dataset already exists
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        Dataset: Resulting dataset object
    """
    try:
        log.info(f"Creating dataset '{dataset.name}'")
        dataset.name = dataset.name.lower()
        database.create_dataset(dataset.name)  # TODO: take embed_dim as input
        res = Dataset(name=dataset.name, embed_dim=1536)
        log.info(f"Created dataset '{dataset.name}'")
        return res
    except database.DatasetExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.error(
            f"Error creating dataset '{dataset.name}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """Deletes a dataset from the VectorDB.

    Args:
        name (str): Name of the dataset to delete

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Success message on successful deletion
    """
    # Delete dataset from the VectorDB
    try:
        name = name.lower()
        database.delete_dataset(name)
        return {"message": f"Dataset '{name}' deleted successfully"}
    except database.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Error deleting dataset '{name}': {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/{name}/query")
async def query(name: str, q: Query):
    """Query the VectorDB with a user-given prompt.

    Args:
        name (str): Target Dataset Name
        query (str): Prompt to query the dataset with

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Top-k results from the query
    """
    if q.prompt is None or q.prompt.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="You didn't provide a query prompt, so I cannot retrieve potential answers to your question.",
        )
    try:
        name = name.lower()
        log.info(f"Querying dataset '{name}' with prompt: '{q.prompt}'")
        results = database.query(prompt=q.prompt, dataset=name, topk=q.topk or 5)
        return {"results": results}
    except database.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(
            f"Error querying dataset '{name}' with prompt '{q.prompt}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/{name}/ingest")
async def ingest_data(name: str, input_data: Ingest):
    """Ingest new data into the VectorDB.

    Args:
        input (Ingest): Base64 encoded data to ingest, with an optional filename.

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Success message on successful ingestion
    """
    # Ingest new data into the VectorDB
    try:
        name = name.lower()
        # Convert base64 data to appropriate format and ingest into pgvector
        # You would need to implement the logic here based on your requirements
        # This could involve using OpenAI API for embedding and then storing in pgvector
        file_id = input_data.file_id if input_data.file_id else str(uuid.uuid4())

        ingested = await ingest.ingest_file(
            dataset=name,
            filename=input_data.filename,
            file_id=file_id,
            content=input_data.content,
        )
        return {
            "message": (
                "Successfully ingested data" + f" from file '{input_data.filename}'"
                if input_data.filename
                else ""
            ),
            "num_ingested_docs": ingested["num_ingested_docs"],
            "documents": ingested["documents"],
            "file_id": file_id,
        }
    except database.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except database.DocumentExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.error(
            f"Error ingesting data [file_id='{input_data.file_id}', file_name='{input_data.filename}'] into dataset '{name}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/{name}/documents/{document_id}")
def remove_document(name: str, document_id: str):
    """Remove a document from a dataset.

    Args:
        name (str): Name of the target dataset.
        document_id (str): ID of the Document to remove

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Success message on successful document removal
    """
    try:
        name = name.lower()
        database.remove_document(name, document_id)
        return {"message": f"Document '{document_id}' removed successfully"}
    except (database.DatasetDoesNotExistError, database.DocumentDoesNotExistError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(
            f"Error removing document '{document_id}' from dataset '{name}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/{name}/files/{file_id}")
def remove_file(name: str, file_id: str):
    """Remove a file from a dataset.

    Args:
        name (str): Name of the target dataset.
        file_id (str): ID of the File to remove

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Success message on successful document removal
    """
    try:
        name = name.lower()
        database.remove_file(name, file_id)
        return {"message": f"File '{file_id}' removed successfully"}
    except (
        database.DatasetDoesNotExistError,
        database.DocumentDoesNotExistError,
        database.FileDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(
            f"Error removing file '{file_id}' from dataset '{name}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
def get_datasets():
    """Get all datasets in the VectorDB.

    Returns:
        JSONResponse: List of all datasets in the VectorDB
    """
    return {"datasets": database.list_datasets()}


@app.get("/datasets/{name}")
def get_dataset(name: str):
    """Get a dataset from the VectorDB.

    Args:
        name (str): Name of the target dataset.

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist

    Returns:
        JSONResponse: Dataset details
    """
    try:
        name = name.lower()
        return database.get_dataset(name)
    except database.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=os.path.join(os.path.dirname(__file__), "log_conf.yaml"),
    )
