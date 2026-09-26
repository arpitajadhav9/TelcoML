FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY app ./app
COPY models ./models
COPY .streamlit ./.streamlit

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"]
