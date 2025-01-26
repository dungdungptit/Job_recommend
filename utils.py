from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import torch
import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uuid

logger = get_logger("Langchain-Chatbot")
from dotenv import load_dotenv

load_dotenv()


# Function for computing embeddings based on the selected embedding model. These could be CV embeddings, skill embeddings, or job embeddings
def get_embedding(texts: List[str], model: str) -> torch.Tensor:
    if model.startswith("text-embedding-"):
        from openai import OpenAI

        client = OpenAI(api_key=openai.api_key)
        response = client.embeddings.create(input=texts, model=model)
        embeddings = [torch.tensor(item.embedding) for item in response.data]
    elif model == "BAAI/bge-small-en-v1.5":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_model = AutoModel.from_pretrained(model).to(device)

        embeddings = []
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                outputs = hf_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze())
    else:
        raise ValueError(f"Unsupported embedding model: {model}")

    return torch.stack(embeddings)


def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="OPENAI_API_KEY_SECRET",
    )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info(
            "Obtain your key from this link: https://platform.openai.com/account/api-keys"
        )
        st.stop()

    model = "gpt-4o-mini"

    return model, openai_api_key


def configure_llm():
    available_llms = [
        "gpt-4o-mini",
        "llama3.2:3b",
        "use your openai api key",
    ]
    llm_opt = st.sidebar.radio(label="LLM", options=available_llms, key="SELECTED_LLM")

    if llm_opt == "llama3.2":
        llm = ChatOllama(model="llama3.2", base_url=st.secrets["OLLAMA_ENDPOINT"])
    elif llm_opt == "gpt-4o-mini":
        llm = ChatOpenAI(
            model_name=llm_opt,
            temperature=0,
            streaming=True,
            api_key=st.secrets["OPENAI_API_KEY"],
        )
    else:
        model, openai_api_key = choose_custom_openai_key()
        llm = ChatOpenAI(
            model_name=model, temperature=0, streaming=True, api_key=openai_api_key
        )
    return llm


@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model


def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
