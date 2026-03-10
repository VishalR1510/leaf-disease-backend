# LeafGuard AI — Backend

Production-grade FastAPI backend for AI-powered leaf disease detection.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
#    Edit .env and add your Gemini API key
#    GEMINI_API_KEY=your_key_here

# 4. Run the server
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/analyze-leaf` | Upload & analyse a leaf image |

## Interactive Docs

Once the server is running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required.** Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3-flash` | LLM model name |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_TOP_P` | `0.8` | Nucleus sampling |
| `LLM_MAX_TOKENS` | `2048` | Max response tokens |

| `MAX_FILE_SIZE_MB` | `10` | Max upload size (MB) |

## Project Structure

```
backend/
├── app/
│   ├── main.py                            # App factory & entrypoint
│   ├── core/
│   │   ├── config.py                      # Settings (env vars)
│   │   ├── logging.py                     # Loguru setup
│   │   └── security.py                    # CORS config
│   ├── api/v1/endpoints/
│   │   └── leaf_analysis.py               # POST /analyze-leaf
│   ├── services/
│   │   ├── gemini_service.py               # Google Gemini AI integration
│   │   ├── image_validator.py             # Upload validation
│   │   └── leaf_detector.py               # Leaf heuristics
│   ├── models/
│   │   ├── request_models.py
│   │   └── response_models.py
│   ├── utils/
│   │   └── image_processing.py            # Resize / base64 / normalise
│   └── dependencies/
│       └── ai_dependencies.py             # DI factories
├── tests/
├── requirements.txt
├── .env
└── README.md
```
