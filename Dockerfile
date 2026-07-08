FROM python:3.12-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY dp_dashboard.py .
COPY results/ results/

EXPOSE 8501

CMD ["streamlit", "run", "dp_dashboard.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]
