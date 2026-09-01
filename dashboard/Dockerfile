 FROM python:3.10-slim

  WORKDIR /app

  COPY . .

  RUN pip install --no-cache-dir -r dashboard/requirements.txt

  WORKDIR /app/dashboard

  EXPOSE 8501

  ENV PYTHONUNBUFFERED=1

  CMD ["streamlit", "run", "app.py", \
       "--server.port=8501", \
       "--server.address=0.0.0.0", \
       "--server.headless=true"]
