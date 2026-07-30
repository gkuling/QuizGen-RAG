'''
Azure OpenAI configuration loading for QuizGen-RAG.

This module reads Azure OpenAI credentials and deployment settings from the
environment (optionally populated from a .env file) into a single frozen
AzureSettings object, and can wire that object into llama-index's global
Settings singleton.

Importing this module has no side effects: nothing is read from the
environment and no client is constructed until load_azure_settings() or
configure_llama_settings() is actually called. llama-index itself is only
imported lazily, inside configure_llama_settings(), so this module can be
imported even in environments where llama-index is not installed.

There is a single environment variable for the API key, AZURE_OPENAI_API_KEY.
The old AZURE_OPENAI_API_KEY_Z (generate_quiz.py) and AZURE_OPENAI_API_KEY_C
(MedEd round table) variants are gone.
'''

import os
from dataclasses import dataclass

import dotenv


class MissingCredentialsError(RuntimeError):
    '''Raised when one or more required Azure OpenAI environment variables are unset.'''


@dataclass(frozen=True)
class AzureSettings:
    '''
    Azure OpenAI connection settings for both the chat and embedding
    deployments.

    :ivar api_key: the shared Azure OpenAI API key.
    :ivar chat_endpoint: the Azure endpoint hosting the chat deployment.
    :ivar embedding_endpoint: the Azure endpoint hosting the embedding
        deployment.
    :ivar chat_deployment: the chat model deployment name.
    :ivar embedding_deployment: the embedding model deployment name.
    :ivar chat_api_version: the Azure OpenAI API version used for chat
        calls.
    :ivar embedding_api_version: the Azure OpenAI API version used for
        embedding calls.
    '''

    api_key: str
    chat_endpoint: str
    embedding_endpoint: str
    chat_deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-small"
    chat_api_version: str = "2024-05-01-preview"
    embedding_api_version: str = "2023-05-15"


def load_azure_settings(env_file: str = ".env") -> AzureSettings:
    '''
    Load Azure OpenAI settings from the environment, optionally populating
    the environment first from a .env file.

    :param env_file: path to a .env file to load with python-dotenv before
        reading the environment. Ignored (no error) if the file does not
        exist, so this works fine when credentials are supplied directly
        as real environment variables (e.g. in CI).
    :return: an AzureSettings instance built from the environment, with
        AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_EMBEDDING_API_VERSION
        used as optional overrides of the AzureSettings defaults.
    :raises MissingCredentialsError: if any of AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT, or AZURE_OPENAI_EMBEDDING_ENDPOINT are unset.
        The error message names every missing variable at once.
    '''
    if os.path.exists(env_file):
        dotenv.load_dotenv(env_file)

    required = {
        "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_EMBEDDING_ENDPOINT": os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise MissingCredentialsError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    defaults = AzureSettings(
        api_key=required["AZURE_OPENAI_API_KEY"],
        chat_endpoint=required["AZURE_OPENAI_ENDPOINT"],
        embedding_endpoint=required["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
    )

    return AzureSettings(
        api_key=defaults.api_key,
        chat_endpoint=defaults.chat_endpoint,
        embedding_endpoint=defaults.embedding_endpoint,
        chat_deployment=os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT", defaults.chat_deployment
        ),
        embedding_deployment=os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", defaults.embedding_deployment
        ),
        chat_api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", defaults.chat_api_version
        ),
        embedding_api_version=os.environ.get(
            "AZURE_OPENAI_EMBEDDING_API_VERSION", defaults.embedding_api_version
        ),
    )


def configure_llama_settings(settings: AzureSettings) -> None:
    '''
    Wire the given AzureSettings into llama-index's global Settings
    singleton, configuring both the chat LLM and the embedding model.

    llama-index is imported lazily inside this function specifically so
    that importing azure_config never requires llama-index to be
    installed.

    :param settings: the AzureSettings to apply.
    :return: None. This function mutates llama_index.core.Settings.
    '''
    from llama_index.core import Settings
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
    from llama_index.llms.azure_openai import AzureOpenAI

    Settings.llm = AzureOpenAI(
        engine=settings.chat_deployment,
        api_key=settings.api_key,
        azure_endpoint=settings.chat_endpoint,
        api_version=settings.chat_api_version,
    )

    # Azure routes by deployment name, not `model` (metadata only); both come from
    # embedding_deployment -- the old ada-002/3-small mismatch was silent since both are 1536-dim.
    Settings.embed_model = AzureOpenAIEmbedding(
        model=settings.embedding_deployment,
        deployment_name=settings.embedding_deployment,
        api_key=settings.api_key,
        azure_endpoint=settings.embedding_endpoint,
        api_version=settings.embedding_api_version,
    )
