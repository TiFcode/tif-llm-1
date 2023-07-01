FROM python:3.8

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the model
RUN transformers-cli download bert-base-uncased
RUN transformers-cli download bert-large-uncased-whole-word-masking-finetuned-squad

COPY . .
