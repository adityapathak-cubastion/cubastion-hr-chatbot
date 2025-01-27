FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# streamlit runs on port 8501
EXPOSE 8501

CMD ["streamlit", "run", "streamlitFrontend.py"]