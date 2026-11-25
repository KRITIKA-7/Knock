# Knock

A web app that helps overlooked builders get noticed by startups.

## Quickstart

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Set your `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` in `.env`.

### 4. Run the development server

From the project root:

```bash
uvicorn backend.app.main:app --reload
```

Or from the `backend/` directory:

```bash
cd backend
uvicorn app.main:app --reload
```

### 5. Check health endpoint

Verify the server is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok"}
```
