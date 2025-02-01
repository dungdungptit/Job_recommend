FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY . /app/

EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["python", "-m", "streamlit", "run", "job_recommender.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection", "false"]