import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from typing import Union
import streamlit as st
import tempfile
import random
import os
from dotenv import load_dotenv

load_dotenv()

# from CV_analyzer import CvAnalyzer
from CV_analyzer_langchain import CvAnalyzer, Candidate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# class RAGStringQueryEngine(BaseModel):
#     """
#     Custom Query Engine for Retrieval-Augmented Generation (fetching matching job recommendations).
#     """

#     retriever: BaseRetriever
#     llm: Union[ChatOpenAI, OllamaLLM]
#     qa_prompt: PromptTemplate

#     # Allow arbitrary types
#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def custom_query(self, candidate_details: str, retrieved_jobs: str):
#         query_str = self.qa_prompt.format(
#             query_str=candidate_details, context_str=retrieved_jobs
#         )
#         print("CUSTOM QUERY")
#         output_parser = StrOutputParser()
#         llm_chain = self.qa_prompt | self.llm | output_parser
#         if isinstance(self.llm, ChatOpenAI):
#             # OpenAI-specific query handling
#             response = llm_chain.invoke(
#                 {"query_str": candidate_details, "context_str": retrieved_jobs}
#             )
#         elif isinstance(self.llm, OllamaLLM):
#             # Ollama-specific query handling
#             response = llm_chain.invoke(
#                 {"query_str": candidate_details, "context_str": retrieved_jobs}
#             )
#         else:
#             raise ValueError("Unsupported LLM type. Please use OpenAI or Ollama.")

#         return str(response)


def custom_query(
    llm, qa_prompt: PromptTemplate, candidate_details: str, retrieved_jobs: str
):
    query_str = qa_prompt.format(
        query_str=candidate_details, context_str=retrieved_jobs
    )

    output_parser = StrOutputParser()
    print("CUSTOM QUERY LLM", llm)
    try:
        llm_chain = qa_prompt | llm | output_parser
        response = llm_chain.invoke(
            {"query_str": candidate_details, "context_str": retrieved_jobs}
        )
    except Exception as e:
        raise ValueError(
            "Unsupported LLM type. Please use OpenAI or Ollama or ChatGoogleGenerativeAI",
            e,
        )

    return str(response)


def main():
    st.set_page_config(page_title="CV Analyzer & Job Recommender", page_icon="🔍")
    st.title("CV Analyzer & Job Recommender")
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Selection")
        llm_option = st.selectbox(
            "Select an LLM:",
            options=[
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "llama3.2",
                "gpt-4o-mini",
            ],
        )
        embedding_option = st.selectbox(
            "Select an embedding model:",
            options=[
                "BAAI/bge-small-en-v1.5",
                "text-embedding-3-small",
            ],
        )
        recreate_index = st.checkbox("Create new job embeddings")

    st.write("Upload a CV to extract key information.")
    uploaded_file = st.file_uploader(
        "Select Your CV (PDF)", type="pdf", help="Choose a PDF file up to 5MB"
    )

    if uploaded_file is not None:
        if st.button("Analyze"):
            with st.spinner("Parsing CV... This may take a moment."):
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    # Initialize CvAnalyzer with selected models
                    analyzer = CvAnalyzer(temp_file_path, llm_option, embedding_option)
                    print("Resume extractor initialized.")
                    # Extract insights from the resume
                    insights = analyzer.extract_candidate_data()

                    print("Candidate data extracted.")
                    # st.write(str(insights))
                    # Load or create job vector index
                    job_index = analyzer.create_or_load_job_index(
                        json_file="sample_jobs.json",
                        index_folder="job_index_3k",
                        recreate=recreate_index,
                    )
                    # Query jobs based on resume data
                    education = (
                        [edu.degree for edu in insights.education]
                        if insights.education
                        else []
                    )
                    skills = insights.skills or []
                    experience = (
                        [exp.role for exp in insights.experience]
                        if insights.experience
                        else []
                    )
                    matching_jobs = analyzer.query_jobs(
                        education, skills, experience, job_index
                    )

                    # Send retrieved nodes to LLM for final output
                    retrieved_context = "\n\n".join(
                        [match.page_content for match in matching_jobs]
                    )
                    candidate_details = f"Education: {', '.join(education)};\nSkills: {', '.join(skills)};\nExperience: {', '.join(experience)}"
                    st.subheader("Candidate details")
                    st.write(str(candidate_details))
                    # st.subheader("Retrieved Context")
                    # st.write(str(retrieved_context))
                    # Check for selected LLM and use the appropriate class
                    # if llm_option == "gemini-1.5-flash":
                    #     llm = ChatGoogleGenerativeAI(
                    #         model="gemini-1.5-flash",
                    #         temperature=0,
                    #         max_tokens=None,
                    #         timeout=None,
                    #         max_retries=2,
                    #         api_key=os.getenv("GOOGLE_API_KEY"),
                    #         # other params...
                    #     )
                    # elif llm_option == "gemini-1.5-pro":
                    #     llm = ChatGoogleGenerativeAI(
                    #         model="gemini-1.5-pro",
                    #         temperature=0,
                    #         max_tokens=None,
                    #         timeout=None,
                    #         max_retries=2,
                    #         api_key=os.getenv("GOOGLE_API_KEY"),
                    #         # other params...
                    #     )
                    # elif llm_option == "llama3.2":
                    #     llm = OllamaLLM(model="llama3.2", temperature=0.0)
                    # else:
                    #     llm = ChatOpenAI(model=llm_option, temperature=0.0)

                    # rag_engine = RAGStringQueryEngine(
                    #     retriever=job_index.as_retriever(),
                    #     llm=analyzer.llm,  # This can be OpenAI or Ollama
                    #     # llm=llm,  # This can be OpenAI or Ollama
                    #     qa_prompt=PromptTemplate(
                    #         input_variables=["query_str", "context_str"],
                    #         template="""\
                    #         You are expert in analyzing resumes, based on the following candidate details and job descriptions:
                    #         Candidate Details:
                    #         ---------------------
                    #         {query_str}
                    #         ---------------------
                    #         Job Descriptions:
                    #         ---------------------
                    #         {context_str}
                    #         ---------------------
                    #         Provide two concise list of the matching jobs: IT related and Non-IT related jobs.Specify if it is IT related job or not in heading. For each matching job, mention job-related details such as
                    #         job matching score, company, brief job description, location, employment type, salary range, URL for each suggestion, and a brief explanation of why the job matches the candidate's profile.
                    #         Be critical in matching profile with the jobs. Thoroughly analyze education, skills, and experience to match jobs.
                    #         Do not explain why the candidate's profile does not match with the other jobs. Do not include any summary. Order the jobs based on their relevance.
                    #         Note: give more preference to hard skills( python java c c++ etc)than soft skills (teamwork, communication, etc). Each jobs should be separated by a new line and sorted decending based on job matching score.
                    #         Answer:
                    #         """,
                    #     ),
                    # )

                    # This llm can be OpenAI or Ollama or ChatGoogleGenerativeAI

                    qa_prompt = PromptTemplate(
                        input_variables=["query_str", "context_str"],
                        template="""\
                        You are expert in analyzing resumes, based on the following candidate details and job descriptions:
                        Candidate Details:
                        ---------------------
                        {query_str}
                        ---------------------
                        Job Descriptions:
                        ---------------------
                        {context_str}
                        ---------------------
                        Provide two concise list of the matching jobs: IT related and Non-IT related jobs.Specify if it is IT related job or not in heading. For each matching job, mention job-related details such as 
                        job matching score, company, brief job description, location, employment type, salary range, URL for each suggestion, and a brief explanation of why the job matches the candidate's profile.
                        Be critical in matching profile with the jobs. Thoroughly analyze education, skills, and experience to match jobs.  
                        Do not explain why the candidate's profile does not match with the other jobs. Do not include any summary. Order the jobs based on their relevance. 
                        Note: give more preference to hard skills( python java c c++ etc)than soft skills (teamwork, communication, etc). Each jobs should be separated by a new line and sorted decending based on job matching score.
                        Answer: 
                        """,
                    )

                    # llm_response = rag_engine.custom_query(
                    #     candidate_details=candidate_details,
                    #     retrieved_jobs=retrieved_context,
                    # )
                    llm_response = custom_query(
                        llm=analyzer.llm,
                        qa_prompt=qa_prompt,
                        candidate_details=candidate_details,
                        retrieved_jobs=retrieved_context,
                    )
                    # llm_response = "1. **Job Title:** Human Resources Recruiter  \n   **Company:** OfficeTeam  \n   **Job Description:** Responsible for gathering job description details, posting jobs, screening resumes, actively recruiting, phone screening candidates, scheduling interviews, generating offer letters, and assisting with onboarding.  \n   **Location:** Milwaukee, WI  \n   **Employment Type:** Seasonal/Temp  \n   **Salary Range:** $16.15 to $18.70 per hour  \n   **URL:** [Apply Here](mailto:OfficeTeam@414-271-4003)  \n   **Explanation:** The candidate has strong communication skills and experience in teamwork and problem-solving, which are essential for recruiting. Their research skills and ability to work with various technologies could aid in sourcing candidates effectively.\n\n2. **Job Title:** Receptionist  \n   **Company:** OfficeTeam  \n   **Job Description:** First point of contact for a private school, responsible for greeting parents, managing enrollment papers, calendar management, and answering phones.  \n   **Location:** San Bruno, CA  \n   **Employment Type:** Seasonal/Temp  \n   **Salary Range:** $14.00 to $15.00 per hour  \n   **URL:** [Apply Here](mailto:OfficeTeam@ClickHereToEmailYourResumé)  \n   **Explanation:** The candidate's strong verbal and written communication skills, along with their ability to multitask, align well with the requirements of a receptionist role. Their experience in customer service and teamwork would be beneficial in this position.\n\n3. **Job Title:** Receptionist  \n   **Company:** Mind of Beauty Day Spa  \n   **Job Description:** Act as a receptionist and support staff as needed.  \n   **Location:** Los Altos, CA  \n   **Employment Type:** Part-Time  \n   **Salary Range:** Not specified  \n   **URL:** [Apply Here](mailto:MindOfBeautyDaySpa@Apply)  \n   **Explanation:** The candidate's communication skills and ability to work in a team environment make them suitable for a receptionist role. Although the job requires fluency in Mandarin, the candidate's bilingual skills in Vietnamese and English may still be advantageous in a diverse workplace."

                    # Display extracted information
                    st.subheader("Extracted Information")
                    st.write(f"**Name:** {insights.name}")
                    st.write(f"**Email:** {insights.email}")
                    st.write(f"**Age:** {insights.age}")
                    display_education(insights.education or [])
                    with st.spinner("Extracting skills..."):
                        display_skills(insights.skills or [], analyzer)
                    display_experience(insights.experience or [])
                    st.subheader("Top Matching Jobs with Explanation")
                    st.markdown(llm_response)
                    print("Done.")
                except Exception as e:
                    st.error(f"Failed to analyze the resume: {str(e)}")


def display_skills(skills: list[str], analyzer):
    """
    Display skills with their computed scores as large golden stars with partial coverage.
    """
    if not skills:
        st.warning("No skills found to display.")
        return
    st.subheader("Skills")
    # Custom CSS for large golden stars
    st.markdown(
        """
        <style>
        .star-container {
            display: inline-block;
            position: relative;
            font-size: 1.5rem;
            color: lightgray;
        }
        .star-container .filled {
            position: absolute;
            top: 0;
            left: 0;
            color: gold;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Compute scores for all skills
    skill_scores = analyzer.compute_skill_scores(skills)
    # Display each skill with a star rating
    for skill in skills:
        score = skill_scores.get(skill, 0)  # Get the raw score
        max_score = (
            max(skill_scores.values()) if skill_scores else 1
        )  # Avoid division by zero
        # Normalize the score to a 5-star scale
        normalized_score = (score / max_score) * 5 if max_score > 0 else 0
        # Split into full stars and partial star percentage
        full_stars = int(normalized_score)
        if (normalized_score - full_stars) <= 0.40:
            partial_star_percentage = 0
        elif (normalized_score - full_stars) > 0.40 and (
            normalized_score - full_stars
        ) <= 70:
            partial_star_percentage = 50
        else:
            partial_star_percentage = 100

        # Generate the star display
        stars_html = ""
        for i in range(5):
            if i < full_stars:
                # Fully filled star
                stars_html += (
                    '<span class="star-container"><span class="filled">★</span>★</span>'
                )
            elif i == full_stars:
                # Partially filled star
                stars_html += f'<span class="star-container"><span class="filled" style="width: {partial_star_percentage}%">★</span>★</span>'
            else:
                # Empty star
                stars_html += '<span class="star-container">★</span>'

        # Display skill name and star rating
        st.markdown(f"**{skill}**: {stars_html}", unsafe_allow_html=True)


def display_education(education_list):
    """
    Display a list of educational qualifications.
    """
    if education_list:
        st.subheader("Education")
        for education in education_list:
            institution = (
                education.institution if education.institution else "Not found"
            )
            degree = education.degree if education.degree else "Not found"
            year = (
                education.graduation_date if education.graduation_date else "Not found"
            )
            details = education.details if education.details else []
            formatted_details = (
                ". ".join(details) if details else "No additional details provided."
            )
            st.markdown(f"**{degree}**, {institution} ({year})")
            st.markdown(f"_Details_: {formatted_details}")


def display_experience(experience_list):
    """
    Display a single-level bulleted list of experiences.
    """
    if experience_list:
        st.subheader("Experience")
        for experience in experience_list:
            job_title = experience.role if experience.role else "Not found"
            company_name = experience.company if experience.company else "Not found"
            location = experience.location if experience.location else "Not found"
            start_date = experience.start_date if experience.start_date else "Not found"
            end_date = experience.end_date if experience.end_date else "Not found"
            responsibilities = (
                experience.responsibilities
                if experience.responsibilities
                else ["Not found"]
            )
            brief_responsibilities = ", ".join(responsibilities)
            st.markdown(
                f"- Worked as **{job_title}** from {start_date} to {end_date} in *{company_name}*, {location}, "
                f"where responsibilities include {brief_responsibilities}."
            )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
