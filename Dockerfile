FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# app.py imports pandas; include it so switching APP_FILE works
RUN pip install --no-cache-dir -r requirements.txt pandas

COPY . /app
COPY images ./images

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

# Default entrypoint: new guided wizard UX
ENV APP_FILE=app_ui_wizard.py

EXPOSE 8501

CMD ["sh", "-c", "streamlit run ${APP_FILE} --server.address=0.0.0.0 --server.port=8501"]
