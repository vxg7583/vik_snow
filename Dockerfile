FROM python:3.9-slim-buster
WORKDIR /app
RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/vxg7583/vik_snow.git .
CMD ["python", "pipeline.py"]
