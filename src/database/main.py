import os
import time
from log import log
from database.db import get_session as dbconn
from database import models
from typing import List
from config import settings
from database.errors import (
    DatasetExistsError,
    DatasetDoesNotExistError,
    FileDoesNotExistError,
)
from llama_index.vector_stores.chroma import ChromaVectorStore


from llama_index.core import Document

from llama_index.core.text_splitter import TokenTextSplitter

from llama_index.core.ingestion import IngestionPipeline


from llama_index.core.indices import VectorStoreIndex
from llama_index.core import set_global_service_context, ServiceContext

from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion.cache import IngestionCache
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore


def dataset_exists(name: str) -> bool:
    c = dbconn()
    name = name.lower()
    # Check if the Datasets table has the dataset
    x = c.get(models.Dataset, name)
    if not x:
        return False
    return True


def get_vector_store(name: str, embed_dim: int = 1536) -> ChromaVectorStore:
    """Get a VectorStore for the given dataset name and embedding dimension

    Args:
        name (str): Name of the dataset
        embed_dim (int, optional): Embedding Dimension. Defaults to 1536, which is the OpenAI embedding dimension.

    Returns:
        PGVectorStore: VectorStore for the given dataset name and embedding dimension
    """
    return ChromaVectorStore.from_params(
        collection_name=name,
        embed_dim=embed_dim,
        persist_dir=settings.vector_store_dir,
    )


def create_dataset(name: str, embed_dim: int = 1536):
    """Creates and initializes a new VectorDB Dataset with the given name and embedding dimension but errors if the dataset already exists.

    Args:
        name (str): Name of the dataset
        embed_dim (int, optional): Embedding Dimension. Defaults to 1536, which is the OpenAI embedding dimension.

    Raises:
        Exception: Error if the dataset already exists
    """

    # Raise Exception if dataset already exists
    if dataset_exists(name):
        raise DatasetExistsError(name)

    # Initialize VectorStore for the dataset
    _ = get_vector_store(name, embed_dim=embed_dim)

    # Create the Dataset in the Datasets table
    session = dbconn()
    session.add(models.Dataset(name=name))
    session.commit()
    session.close()

    log.info(f"Created dataset {name} with embedding dimension {embed_dim}")


def delete_dataset(name: str):
    """Deletes a VectorDB Dataset with the given name but errors if the dataset does not exist.

    Args:
        name (str): Name of the dataset

    Raises:
        Exception: Error if the dataset does not exist
    """

    # Raise Exception if dataset does not exist
    if not dataset_exists(name):
        raise DatasetDoesNotExistError(name)

    session = dbconn()

    # Drop Files and Documents
    session.query(models.FileIndex).filter(models.FileIndex.dataset == name).delete()
    session.query(models.DocumentIndex).filter(
        models.DocumentIndex.dataset == name
    ).delete()
    session.query(models.Dataset).filter(models.Dataset.name == name).delete()
    session.commit()
    session.close()


async def ingest_documents(
    dataset: str,
    documents: List[Document],
    embed_model_name: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
):
    if not dataset_exists(dataset):
        raise DatasetDoesNotExistError(dataset)

    vector_store = get_vector_store(dataset)

    embed_model = OpenAIEmbedding(
        model=embed_model_name,
        api_base=settings.api_base,
        # additional_kwargs={"encoding_format": "float"},
        # dimensions=vector_store.embed_dim, # FIXME: set dimensions only for models that support it
        # TODO: Set API parameters and allow for other embed_models to make it work with Rubra
        # - also allows specifying different modes for text search (current) or similarity search
    )

    vector_store_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    llm = OpenAI(
        api_base=settings.api_base,
    )
    service_context = ServiceContext.from_defaults(llm=llm)
    set_global_service_context(
        service_context
    )  # helper to avoid passing around the service_context manually

    # Run transformations to split input data into chunks and extract useful metadata.
    # This uses LLM calls!

    transformations = [
        TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128),
        # TitleExtractor(
        #     nodes=5, llm=llm
        # ),  # assuming the title is located within the first 5 nodes ("pages")
        # SummaryExtractor(
        #     summaries=["prev", "self"], llm=llm
        # ),  # extract summaries for previous and current node
        # KeywordExtractor(keywords=10, llm=llm),  # extract 10 keywords for each node
    ]

    # Sanitization of documents - remove NUL (0x00) characters which are not allowed in Postgres/pgvector
    for doc in documents:
        doc.text = doc.text.replace("\x00", "")

    if os.path.exists(settings.cache_path):
        kvstore = SimpleKVStore.from_persist_path(settings.cache_path)
    else:
        kvstore = SimpleKVStore()

    cache = IngestionCache(cache=kvstore)

    pipeline = IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store,
        docstore=vector_store_index.docstore,
        cache=cache,
    )

    pipeline_start_time = time.time()
    nodes = await pipeline.arun(documents=documents, show_progress=False, num_workers=1)
    pipeline_duration = time.time() - pipeline_start_time
    log.info(
        f"[dataset={dataset}] Transformations took {pipeline_duration:.2f} seconds for {len(nodes)} nodes from {len(documents)} documents"
    )

    start = time.time()
    log.info(f"using api base {settings.api_base}")
    log.info(f"[dataset={dataset}] Inserting {len(nodes)} nodes into the VectorStore")
    vector_store_index.insert_nodes(nodes)
    end = time.time() - start
    log.info(
        f"[dataset={dataset}] Inserted {len(nodes)} nodes from {len(documents)} documents took {end:.2f} seconds"
    )

    cache.persist(persist_path=settings.cache_path)

    return nodes


def remove_document(dataset: str, document_id: str, session=None):
    """Remove a document from the Index and underlying Docstore

    Args:
        dataset (str): Name of the target dataset
        document_id (str): ID of the document to remove

    Raises:
        DatasetDoesNotExistError: Error if the dataset does not exist

    NOTE: The Postgres VectorStore does not have any option for us to know if that document exists in the store or not, so we cannot raise an error if the document does not exist.
    """
    vector_store = get_vector_store(dataset)
    if vector_store is None:
        raise DatasetDoesNotExistError(dataset)

    vector_store_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    vector_store_index.delete_ref_doc(document_id, delete_from_docstore=True)

    dbsess = dbconn() if session is None else session
    log.info(f"Removing document {dataset}/{document_id}")
    dbsess.query(models.DocumentIndex).filter(
        models.DocumentIndex.document_id == document_id,
        models.DocumentIndex.dataset == dataset,
    ).delete()

    # If an existing session was passed in, then the caller is responsible for committing and closing the session
    if session is None:
        dbsess.commit()
        dbsess.close()


def remove_file(dataset: str, file_id: str) -> list[str] | None:
    """Remove a file from the Index and underlying Docstore

    Args:
        dataset (str): Name of the target dataset
        file_id (str): ID of the file to remove

    Raises:
        DatasetDoesNotExistError: Error if the dataset does not exist

    NOTE: The Postgres VectorStore does not have any option for us to know if that document exists in the store or not, so we cannot raise an error if the document does not exist.
    """

    session = dbconn()

    file = (
        session.query(models.FileIndex)
        .filter(
            models.FileIndex.file_id == file_id, models.FileIndex.dataset == dataset
        )
        .one_or_none()
    )

    if file is None:
        raise FileDoesNotExistError(dataset, file_id)

    for document in file.documents:
        log.info(f"Removing document {dataset}/{document.document_id}")
        remove_document(dataset, document.document_id, session=session)

    log.info(f"Removing file {dataset}/{file_id}")

    session.refresh(file)
    session.delete(file)
    session.commit()
    session.close()


def get_dataset(dataset: str) -> dict[str, any]:
    """Get information about the dataset

    Args:
        dataset (str): Name of the target dataset

    Returns:
        dict[str, any]: Information about the dataset
    """
    if not dataset_exists(dataset):
        raise DatasetDoesNotExistError(dataset)

    response = {
        "name": dataset,
        "num_files": 0,
        "files": [],
    }

    session = dbconn()
    files = (
        session.query(models.FileIndex)
        .filter(models.FileIndex.dataset == dataset)
        .all()
    )

    response["num_files"] = len(files)

    for file in files:
        response["files"].append(
            {
                "file_id": file.file_id,
                "num_documents": len(file.documents),
                "documents": [doc.document_id for doc in file.documents],
            }
        )

    session.close()

    return response


def list_datasets() -> list[str]:
    """Get a list of all available datasets

    Returns:
        list[str]: List of all available datasets
    """
    session = dbconn()
    datasets = session.query(models.Dataset).all()
    session.close()
    return [dataset.name for dataset in datasets]
