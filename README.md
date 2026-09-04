# Single Stage Hybrid Recommendation System

A unified, real-time recommendation system built with FastAPI and PostgreSQL (`pgvector`). This system handles both the fetching of product recommendations (via fast KNN similarity search performed directly in PostgreSQL) and the background training pipeline (upserting user embeddings based on interaction events).

## Prerequisites
- **Python:** 3.10+ (Tested with Python 3.12)
- **Database:** PostgreSQL with the `pgvector` extension installed.

## Quick Start Setup

Follow these steps to get the project running on your local machine.

### 1. Create a Virtual Environment
It is highly recommended to use a virtual environment to isolate project dependencies. Run the following commands in your terminal (PowerShell for Windows):

```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\activate

# Install all required dependencies
pip install -r requirements.txt
```
*(If you are on macOS/Linux, activate the environment using `source .venv/bin/activate` instead).*

### 2. Configure the Databases
Before starting the application, you need to point it to your live databases. 

Open `constants.py` and update the placeholders with your actual credentials:
- **EVENTS_DB**: Update `EVENTS_DB_PATH` with your SQLite or Postgres connection string for the user interaction events.
- **PGVECTOR_DB**: Update `PGVECTOR_DB_PATH`, `USERNAME`, and `PASSWORD` with your PostgreSQL database URI that has `pgvector` enabled.
- Ensure the table names (`PGVECTOR_PRODUCT_EMBEDDINGS_TABLE` and `PGVECTOR_USER_EMBEDDINGS_TABLE`) match the tables created in your database.

#### 2.1 Create Database Tables
Run the following SQL commands in your PostgreSQL database to set up the necessary tables and enable the `pgvector` extension:

```sql
-- 1. Enable the pgvector extension (required)
CREATE EXTENSION IF NOT EXISTS vector;
```

```sql
-- 2. Create the Product Embeddings table
-- Replace '3' with your actual embedding dimensionality (e.g., 128, 768)
CREATE TABLE IF NOT EXISTS product_embeddings (
    product_id VARCHAR(255) PRIMARY KEY,
    embedding vector(3) 
);
```

```sql
-- 3. Create the User Embeddings table
CREATE TABLE IF NOT EXISTS user_embeddings (
    user_id VARCHAR(255) PRIMARY KEY, 
    embedding vector(3) 
);
```

```sql
-- 4. Create the Events table (if using Postgres for events)
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 3. Run the Application
Once the dependencies are installed and the constants are configured, you can start the FastAPI server using Uvicorn:

```powershell
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

---

## 📡 API Endpoints

Once the server is running, you can access the interactive API documentation at: **`http://127.0.0.1:8000/docs`**

### 1. Get Recommendations (Inference)
- **Endpoint:** `GET /api/v1/recommendations/{user_id}`
- **Description:** Fetches the top-N product recommendations for a specific user. The system uses pgvector to calculate cosine similarity (`<=>`) directly within the database for maximum performance. If it's a new user (cold start), a random embedding is generated and stored on the fly.

### 2. Trigger Training (Pipeline)
- **Endpoint:** `POST /api/v1/train`
- **Description:** Triggers the recommendation training pipeline as a background task. It reads recent interaction events from the events database, calculates new user embeddings (using the mean vector of interacted products), and upserts them into the PostgreSQL user embeddings table.

---

## Optional: Deploying as a Docker Image

If you prefer to run the application in a containerized environment, you can build and run a Docker image.

### 1. Create a `Dockerfile`

If you haven't already, create a file named `Dockerfile` in the root directory of the project:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Build the Docker Image

Run the following command in the same directory as your `Dockerfile`:

```bash
docker build -t hybrid-recommendation-system .
```

### 3. Run the Docker Container

Make sure your database connection strings in `constants.py` are pointing to addresses accessible from inside the Docker container (e.g., using your machine's IP address or `host.docker.internal` instead of `127.0.0.1` or `localhost`).

```bash
docker run -d -p desired_unique_port:8000 hybrid-recommendation-system
```

The application will be accessible at `http://localhost:desired_unique_port`.
