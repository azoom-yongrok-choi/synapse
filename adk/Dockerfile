FROM python:3.11-slim

# Node.js 20 LTS 
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

CMD ["adk", "web", "--port=8000", "--host=0.0.0.0"]