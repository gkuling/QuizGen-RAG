'''
Build (or load) the llama-index vector store for QuizGen-RAG.

Reads every PDF in a folder, splits them into token-based chunks, embeds
them with the configured Azure OpenAI embedding deployment, and persists
the resulting vector store to disk. If a vector store already exists at
the target path, it is loaded instead of rebuilt, unless --force-rebuild
is given.

Nothing executes at import time: Azure credentials are only loaded inside
main(), after argument parsing, and no client or index is constructed
until main() (or build_or_load_vec_store()) actually runs.
'''

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from azure_config import MissingCredentialsError, configure_llama_settings, load_azure_settings


def build_or_load_vec_store(
    pdf_folder: str | Path,
    vec_store: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    force_rebuild: bool,
) -> Any:
    '''
    Build a vector store from the PDFs in pdf_folder, or load it from
    vec_store if a valid one is already persisted there.

    Requires llama-index's global Settings.llm/embed_model to already be
    configured (e.g. via azure_config.configure_llama_settings), since
    building or loading an index both depend on the embedding model.

    :param pdf_folder: folder containing the PDFs to ingest. Only read
        when a new vector store actually needs to be built.
    :param vec_store: directory the vector store is persisted to (or
        loaded from).
    :param chunk_size: token chunk size used by the text splitter when
        building a new index. Ignored when loading an existing store.
    :param chunk_overlap: token chunk overlap used by the text splitter
        when building a new index. Ignored when loading an existing
        store.
    :param force_rebuild: if True, any existing vector store at
        vec_store is deleted and rebuilt from the PDFs, even if a valid
        one is already present.
    :return: the loaded or newly built llama-index VectorStoreIndex.
    '''
    from llama_index.core import (
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.core.node_parser import TokenTextSplitter

    vec_store_path = Path(vec_store)
    docstore_path = vec_store_path / "docstore.json"

    if force_rebuild and vec_store_path.exists():
        shutil.rmtree(vec_store_path)

    if not docstore_path.exists():
        vec_store_path.mkdir(parents=True, exist_ok=True)

        text_splitter = TokenTextSplitter(
            separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        print("Beginning document parsing...")
        documents = SimpleDirectoryReader(str(pdf_folder)).load_data()
        print("Finished document parsing.")

        print("Indexing data...")
        index = VectorStoreIndex.from_documents(
            documents, show_progress=True, transformations=[text_splitter]
        )
        print("Finished indexing data.")

        index.storage_context.persist(persist_dir=str(vec_store_path))
    else:
        print("Loading index from vec_store...")
        storage_context = StorageContext.from_defaults(persist_dir=str(vec_store_path))
        index = load_index_from_storage(storage_context)
        print("Finished loading index from vec_store.")

    return index


def main(argv: list[str] | None = None) -> int:
    '''
    Command-line entry point for building or loading the vector store.

    :param argv: argument list to parse (excluding the program name); if
        None, sys.argv[1:] is used.
    :return: process exit code: 0 on success, 1 on a clean error (missing
        PDF folder or missing Azure credentials); argparse itself exits
        with a nonzero code on bad arguments.
    '''
    parser = argparse.ArgumentParser(
        description="Build (or load) the llama-index vector store for a folder of PDFs."
    )
    parser.add_argument(
        "--pdf-folder", type=str, required=True,
        help="Path to the folder containing PDFs to parse.",
    )
    parser.add_argument(
        "--vec-store", type=str, default=None,
        help="Directory to persist the vector store to. Defaults to "
             "<pdf-folder>/vec_store.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024,
        help="Token chunk size used by the text splitter (default: 1024).",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=128,
        help="Token chunk overlap used by the text splitter (default: 128).",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Delete and rebuild the vector store even if one already exists.",
    )
    parser.add_argument(
        "--smoke-test-query", type=str, default=None, metavar="TEXT",
        help="If given, run this query against the built index after "
             "indexing and print the response, as a smoke test. No query "
             "is run if this is not given.",
    )
    args = parser.parse_args(argv)

    pdf_folder = Path(args.pdf_folder)
    vec_store = Path(args.vec_store) if args.vec_store else pdf_folder / "vec_store"

    if not pdf_folder.is_dir():
        print(f"Error: PDF folder not found: {pdf_folder}", file=sys.stderr)
        return 1

    # Credentials are loaded only now, after argument and path validation
    # have succeeded, so that --help, bad arguments, and a missing PDF
    # folder never require Azure OpenAI environment variables to be set.
    try:
        azure_settings = load_azure_settings()
    except MissingCredentialsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    configure_llama_settings(azure_settings)

    index = build_or_load_vec_store(
        pdf_folder, vec_store, args.chunk_size, args.chunk_overlap, args.force_rebuild
    )

    if args.smoke_test_query:
        query_engine = index.as_query_engine()
        response = query_engine.query(args.smoke_test_query)
        print("Query:", args.smoke_test_query)
        print("-" * 50)
        print("Response:", response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
