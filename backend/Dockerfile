# Backend Dockerfile (backend/Dockerfile)

# Gebruik een Python image
FROM python:3.10-slim

# Stel de werkdirectory in
WORKDIR /app

# Kopieer requirements en installeer afhankelijkheden
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Kopieer de rest van de backend code
COPY . /app

# Expose de poort waarop Flask draait
EXPOSE 5000

# Start het backend script
CMD ["python", "run.py"]
