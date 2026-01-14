# Research Intelligence Platform

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![React 19](https://img.shields.io/badge/react-19-61DAFB.svg?logo=react)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.30+-orange.svg)

## Overview

Research Intelligence Platform is an AI-driven research assistant that transforms complex queries into comprehensive reports. Built with a **ReAct-style LangGraph architecture**, it features a **chat-first UI** with streaming responses and **interactive mindmap visualization**.

### Key Features

- **ReAct Agent Architecture**: 5-node workflow (planner/searcher/fetcher/ranker/writer)
- **Chat-First Interface**: Natural language interaction with streaming responses
- **Interactive Mindmap**: ReactFlow visualization of sub-questions
- **Real-Time Updates**: SSE streaming for live progress
- **Citation Management**: Automatic extraction and formatting
- **Statistical Analysis**: Pandas/NumPy-based data analysis
- **Visualizations**: Matplotlib charts and tables

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Frontend (React 19 + TypeScript + @xyflow/react)│
├─────────────────────────────────────────────────┤
│  ChatInterface │ ResearchMindmap │ ReportPanel │
├─────────────────────────────────────────────────┤
│         SSE Streaming + REST API                │
├────────────────────┬────────────────────────────┤
│ Backend (Flask + Celery)                        │
│  /api/chat  │  /api/stream  │  /api/health      │
├─────────────────────────────────────────────────┤
│ ReAct Agent (LangGraph)                         │
│  Planner → Search → Fetch → Rank → Write        │
├─────────────────────────────────────────────────┤
│ Tools (crawl4ai + statistical + visualization)  │
├─────────────────────────────────────────────────┤
│ SQLite (connection pooling) + Redis (Celery)    │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- pnpm
- Docker (for code execution sandbox)
- Redis (optional, for async tasks)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Start the server
python app_research.py
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

Visit `http://localhost:5173` to access the application.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/start_research` | Start new research project |
| GET | `/api/research_status/<id>` | Get research status |
| POST | `/api/chat/<id>` | Send chat message (streaming) |
| GET | `/api/stream/<id>` | SSE stream for real-time updates |
| GET | `/api/projects` | List all projects |
| GET | `/api/health` | Health check endpoint |

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `DEBUG` | Enable debug mode | No |
| `LOG_LEVEL` | Logging level (INFO, DEBUG, etc.) | No |
| `REDIS_URL` | Redis broker URL (Celery) | No |

## Tech Stack

### Backend

- **Python 3.12+** - Core language
- **Flask** - Web framework
- **LangGraph** - Agent orchestration
- **crawl4ai** - Web scraping
- **Pandas/NumPy** - Data processing
- **Celery** - Async task queue
- **Pydantic** - Validation

### Frontend

- **React 19** - UI framework
- **Material-UI 7** - Component library
- **@xyflow/react** - Mindmap visualization
- **Vite** - Build tool
- **Chart.js** - Charts

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Type checking
mypy backend/ --strict
```

### Docker Services

```bash
# Start code executor (sandboxed Python execution)
docker-compose up code-executor

# Start all services
docker-compose up --build
```

## Security

- Settings validation via Pydantic models
- Docker resource limits (CPU, memory)
- Health check endpoints
- Input sanitization on all endpoints

## License

MIT License - see [LICENSE](LICENSE) file for details.
