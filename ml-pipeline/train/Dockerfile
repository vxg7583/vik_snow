FROM python:3.10-slim
WORKDIR /app

# Install dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the script
COPY train.py .
CMD ["python", "train.py"]
