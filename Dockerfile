# 1. Start from a small official image that already has Python 3.13
FROM python:3.13-slim

# 2. Work inside a folder called /app in the container
WORKDIR /app

# 3. Copy ONLY requirements first and install them.
#    (Docker caches this step, so rebuilding after a code change is fast.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project (code + data)
COPY . .

# 5. Cloud Run tells the app which port to use through the PORT variable
ENV PORT=8080

# 6. The command that starts the server.
#    --host 0.0.0.0 = accept connections from outside the container
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]