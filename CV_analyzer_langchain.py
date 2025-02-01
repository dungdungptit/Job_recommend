import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

import shutil
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, root_validator, field_validator
import os
from dotenv import load_dotenv

load_dotenv()
import openai
import numpy as np
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from uuid import uuid4

from langchain_core.documents import Document
from utils import get_embedding

# Set your API keys in a secret.toml file
# openai.api_key = st.secrets["OPENAI_API_KEY"]


# Pydantic model for extracting education
class Education(BaseModel):
    institution: Optional[str] = Field(
        None, description="The name of the educational institution"
    )
    degree: Optional[str] = Field(
        None, description="The degree or qualification earned"
    )
    graduation_date: Optional[str] = Field(
        None, description="The graduation date (e.g., 'YYYY-MM')"
    )
    details: Optional[List[str]] = Field(
        None,
        description="Additional details about the education (e.g., coursework, achievements)",
    )

    @field_validator("details", mode="before")
    def validate_details(cls, v):
        if isinstance(v, str) and v.lower() == "n/a":
            return []
        elif not isinstance(v, list):
            return []
        return v


# Pydantic model for extracting experience
class Experience(BaseModel):
    company: Optional[str] = Field(
        None, description="The name of the company or organization"
    )
    location: Optional[str] = Field(
        None, description="The location of the company or organization"
    )
    role: Optional[str] = Field(
        None, description="The role or job title held by the candidate"
    )
    start_date: Optional[str] = Field(
        None, description="The start date of the job (e.g., 'YYYY-MM')"
    )
    end_date: Optional[str] = Field(
        None,
        description="The end date of the job or 'Present' if ongoing (e.g., 'MM-YYYY')",
    )
    responsibilities: Optional[List[str]] = Field(
        None, description="A list of responsibilities and tasks handled during the job"
    )

    @field_validator("responsibilities", mode="before")
    def validate_responsibilities(cls, v):
        if isinstance(v, str) and v.lower() == "n/a":
            return []
        elif not isinstance(v, list):
            return []
        return v


# Main Pydantic class ensapsulating education and epxerience classes with other information
class Candidate(BaseModel):
    name: Optional[str] = Field(None, description="The full name of the candidate")
    email: Optional[EmailStr] = Field(None, description="The email of the candidate")
    age: Optional[int] = Field(None, description="The age of the candidate.")
    skills: Optional[List[str]] = Field(
        None, description="A list of high-level skills possessed by the candidate."
    )
    experience: Optional[List[Experience]] = Field(
        None,
        description="A list of experiences detailing previous jobs, roles, and responsibilities",
    )
    education: Optional[List[Education]] = Field(
        None,
        description="A list of educational qualifications of the candidate including degrees, institutions studied in, and dates of start and end.",
    )

    @root_validator(pre=True)
    def handle_invalid_values(cls, values):
        for key, value in values.items():
            if isinstance(value, str) and value.lower() in {"n/a", "none", ""}:
                values[key] = None
        return values


# Class for analyzing the CV contents
class CvAnalyzer:
    def __init__(self, file_path, llm_option, embedding_option):
        self.file_path = file_path
        self.llm_option = llm_option
        self.embedding_option = embedding_option
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._resume_content = None
        self.llm = None
        self._configure_settings()

    def extract_candidate_data(self) -> object:
        """
        Extracts candidate data from the resume.
        """
        print(f"Extracting CV data. LLM: {self.llm_option}")
        output_schema = Candidate.model_json_schema()

        if self.llm_option == "llama3.2":
            self.llm = OllamaLLM(model="llama3.2", temperature=0.0)
        else:
            self.llm = ChatOpenAI(model=self.llm_option, temperature=0.0)

        # Load resume
        loader = PyPDFLoader(self.file_path)
        documents = []
        for page in loader.load():
            documents.append(page)

        # Store the pre-extracted content
        self._resume_content = "\n".join([doc.page_content for doc in documents])
        parser = JsonOutputParser(pydantic_object=Candidate)

        prompt = PromptTemplate(
            template=""" You are an expert in analyzing resumes. Use the following JSON schema to extract relevant information.
            Answer the user resume.\n{format_instructions}\n{resume}\n""",
            input_variables=["resume"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        response = self._resume_content
        is_error = False
        errors = None
        chain = prompt | self.llm | parser
        try:
            response = chain.invoke({"resume": self._resume_content})
            if not response or not response:
                raise ValueError("Failed to get a response from LLM.")
            """ 
            response = {
                "name": "Dang Quang Dung",
                "email": "anhdungvk01@gmail.com",
                "age": None,
                "skills": [
                    "English",
                    "Vietnamese (native speaker)",
                    "Python",
                    "Linux",
                    "SQL",
                    "FastAPI",
                    "Docker",
                    "MySQL",
                    "MongoDB",
                    "FAISS",
                    "Chroma",
                    "HTML",
                    "CSS",
                    "JavaScript",
                    "ReactJS",
                    "Ant Design",
                    "MUI",
                    "Transformers (Hugging Face)",
                    "spaCy",
                    "scikit-learn",
                    "Gensim",
                    "NLTK",
                    "BERTopic",
                    "Rasa",
                    "Botpress",
                    "Langchain",
                    "LlamaIndex",
                    "TensorFlow",
                    "Keras",
                    "PyTorch",
                    "Research Skills",
                    "Problem Solving",
                    "Report Writing",
                    "Teamwork",
                    "Data Structures and Algorithms",
                    "Data Visualization (Tableau, Excel)",
                ],
                "experience": [
                    {
                        "company": "Centre for Development of Advanced Computing (C-DAC)",
                        "location": "India",
                        "role": "Master Trainer",
                        "start_date": "2024-09",
                        "end_date": "Present",
                        "responsibilities": [
                            "Participate in Artificial Intelligence & Data Science: PG-DBDA - Post Graduate Diploma in Big Data Analytics"
                        ],
                    },
                    {
                        "company": "Research Institute of Posts and Telecommunications (RIPT)-PTIT",
                        "location": None,
                        "role": "AI Researcher",
                        "start_date": "2024-08",
                        "end_date": "Present",
                        "responsibilities": [
                            "Research and writing paper",
                            "Build a chatbot system using RAG techniques, Function Calling with LLMs",
                        ],
                    },
                    {
                        "company": "A.I-Soft JSC",
                        "location": None,
                        "role": "Software AI Engineer",
                        "start_date": "2023-02",
                        "end_date": "2024-08",
                        "responsibilities": [
                            "Build websites for research topics",
                            "Build a chatbot system to support enrollment (End-to-End) using Rasa and Botpress Framework",
                            "Build a crawl data module, data analysis using BERTopic for social network listening system",
                            "Research paper about AI and Natural Language Processing and apply to used software products",
                            "Build a chatbot system to support enrollment (End-to-End) with LLMs using RAG techniques with LangChain and LlamaIndex Framework",
                        ],
                    },
                ],
                "education": [
                    {
                        "institution": "Posts and Telecommunications Institute of Technology",
                        "degree": "Information Technology",
                        "graduation_date": "2024-05",
                        "details": ["GPA 3.31/4.0"],
                    }
                ],
            }
             """
            # parsed_data = json.loads(response.text)
            response = Candidate.model_validate(response)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            errors = e

            raise ValueError(
                "Failed to extract insights. Please ensure the resume and query engine are properly configured."
            )
        if is_error:
            prompt = PromptTemplate(
                template=""" You are an expert in repair invalid JSON by errors. Use the JSON formart correct to repair invalid json.
                \n{format_instructions}\n, Error: {error} and json invalid: {json_invalid}\n
                Answer the JSON format correct\n""",
                input_variables=["json_invalid", "error"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            chain = prompt | self.llm | parser
            res_repair = chain.invoke({"json_invalid": response, "error": str(errors)})
            # response = Candidate.model_validate(res_repair)
            response = res_repair
        return response

    # Compute skill scores based on their semantic similarity (Cosine similarity) with the CV contents
    def compute_skill_scores(self, skills: list[str]) -> dict:
        """
        Compute semantic weightage scores for each skill based on the resume content

        Parameters:
        - skills (list of str): A list of skills to evaluate.

        Returns:
        - dict: A dictionary mapping each skill to a score
        """
        # Extract resume content and compute its embedding
        resume_content = self._extract_resume_content()

        # Compute embeddings for all skills at once
        skill_embeddings = get_embedding(skills, model=self.embedding_model.model_name)

        # Compute raw similarity scores and semantic frequency for each skill
        raw_scores = {}
        # frequency_scores = {}
        for skill, skill_embedding in zip(skills, skill_embeddings):
            # Compute semantic similarity with the entire resume
            similarity = self._cosine_similarity(
                get_embedding([resume_content], model=self.embedding_model.model_name)[
                    0
                ],
                skill_embedding,
            )
            raw_scores[skill] = similarity
        return raw_scores

    # Extract all the contents from a CV
    def _extract_resume_content(self) -> str:
        """
        Extracts and returns the full text of the resume.
        """
        if self._resume_content:
            return self._resume_content  # Use the pre-stored content
        else:
            raise ValueError(
                "Resume content not available. Ensure `extract_candidate_data` is called first."
            )

    # Function to compute the Cosine similarity of skills with the CV contents
    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two vectors.

        Parameters:
        - vec1 (np.ndarray): First vector.
        - vec2 (np.ndarray): Second vector.

        Returns:
        - float: Cosine similarity score.
        """
        vec1, vec2 = vec1.to(self.device), vec2.to(self.device)
        return (torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))).item()

    # Function to configure model settings
    def _configure_settings(self):
        """
        Configure the LLM and embedding model based on user selections.
        """
        # Determine the device based on CUDA availability
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA is available. Using GPU.")
        else:
            device = "cpu"
            print("CUDA is not available. Using CPU.")

        # Configure the LLM
        if self.llm_option == "gpt-4o-mini":
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        elif self.llm_option == "llama3.2":
            llm = OllamaLLM(
                model="llama3.2",
            )
        else:
            raise ValueError(f"Unsupported LLM option: {self.llm_option}")

        # Configure the embedding model
        if self.embedding_option.startswith("text-embedding-"):
            embed_model = OpenAIEmbeddings(model=self.embedding_option)
        elif self.embedding_option == "BAAI/bge-small-en-v1.5":
            embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_option}")

        # Set the models in Settings
        self.llm = llm
        self.embedding_model = embed_model

    # Function to create an existing job vector dataset or create a new job vector dataset
    def create_or_load_job_index(
        self,
        json_file: str,
        index_folder: str = "job_index_3k",
        recreate: bool = False,
    ):
        """
        Create or load a vector database for jobs using LlamaIndex.

        Args:
        - json_file: Path to job dataset JSON file.
        - index_folder: Folder to save/load the vector index.
        - recreate: Boolean flag indicating whether to recreate the index.

        Returns:
        - VectorStoreIndex: The job vector index.
        """
        if recreate and os.path.exists(index_folder):
            # Delete the existing job index storage
            print(f"Deleting the existing job dataset: {index_folder}...")
            shutil.rmtree(index_folder)
        if not os.path.exists(index_folder):
            print(
                f"Creating new job vector index with {self.embedding_model.model_name} model..."
            )
            with open(json_file, "r") as f:
                job_data = json.load(f)
            # Convert job descriptions to Document objects by serializing all fields dynamically
            documents = []
            for job in job_data["jobs"]:
                job_text = "\n".join(
                    [f"{key.capitalize()}: {value}" for key, value in job.items()]
                )
                document_tmp = Document(
                    page_content=job_text,
                    metadata={"source": "job"},
                )
                documents.append(document_tmp)

            # Create the vector index directly from documents
            index = FAISS.from_documents(documents, self.embedding_model)
            # Save index to disk
            index.save_local(folder_path=index_folder)
            return index
        else:
            print(f"Loading existing job index from {index_folder}...")
            index = FAISS.load_local(
                folder_path=index_folder,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            return index

    # Function to query job dataset to fetch the top k matching jobs according to the given education, skills, and experience.
    def query_jobs(self, education, skills, experience, index, top_k=10):
        """
        Query the vector database for jobs matching the resume.

        Args:
        - education: List of educational qualifications.
        - skills: List of skills.
        - experience: List of experiences.
        - index: Job vector database index.
        - top_k: Number of top results to return.

        Returns:
        - List of job matches.
        """
        print(
            f"Fetching job suggestions.(LLM: {self.llm_option}, embed_model: {self.embedding_option})"
        )
        query = f"Education: {', '.join(education)}; Skills: {', '.join(skills)}; Experience: {', '.join(experience)}"
        # Use retriever with appropriate model
        retriever = index.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": 0.1, "fetch_k": 15},
        )
        matches = retriever.invoke(query)
        matches = [doc for doc in matches if "score" in doc.metadata]
        matches = sorted(matches, key=lambda x: x.metadata["score"], reverse=True)
        return matches


if __name__ == "__main__":
    pass
