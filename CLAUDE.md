# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research Intelligence Platform - an AI-driven research assistant that transforms complex queries into comprehensive reports using a multi-stage LangGraph architecture.

## Commands

### Development
```bash
# Run both frontend and backend
./run_dev.sh

# Backend only (port 5001)
cd backend && source env/bin/activate && python3 app_research.py

# Frontend only (port 5173)
cd frontend && pnpm run dev

# Frontend build
cd frontend && pnpm build

# Frontend lint
cd frontend && pnpm lint

# Type checking (mypy)
mypy backend/ --strict

# Run tests
cd backend && pytest tests/ -v
```

### Docker
```bash
# Build and run all services
docker-compose up --build

# Code executor container (sandbox for Python execution)
docker-compose up code-executor
```

## Architecture

### Backend (Flask + LangGraph)
- **Entry**: `backend/app_research.py` - Flask server on port 5001
- **State**: `backend/agent/state.py` - TypedDict definitions for ResearchState
- **Tools**: `backend/tools/langgraph_tools.py` - LangChain-style tools
- **Legacy Agent**: `backend/agent_definition.py` - Original 11-node StateGraph

**ReAct Agent Structure (new):**
- State: messages + sub_questions + sources + final_report
- Tools: search_web, browse_url, browse_urls, analyze_quantitative_data, calculate_correlation, generate_bar_chart, generate_line_chart

**Legacy Agent Structure (existing):**
- 11 sequential nodes: planning → search → scraping → citation → synthesis → extraction → validation → statistics → compare → visualization → report

### Data Persistence
- **Project State**: `research_projects/{project_id}/state.json`
- **Reports**: `research_projects/{project_id}/final_research_report.md`
- **Charts**: `research_projects/{project_id}/charts/*.png`
- **Database**: `backend/research_projects.db` (SQLite with connection pooling)

### Frontend (React 19 + Vite + MUI)
- **Entry**: `frontend/src/App.jsx`
- **Components**: `frontend/src/components/` - Dashboard, ResearchForm, ReportDisplay, ProjectHistory, **ChatInterface**, **ResearchMindmap**
- **API Client**: `frontend/src/services/api.js`

### Key Improvements Made
| Category | Change |
|----------|--------|
| Security | Pydantic validation for settings endpoint |
| Security | Docker resource limits + health checks |
| Architecture | Database connection pooling |
| Code Quality | Shared synthetic data utility |
| Frontend | ChatInterface + ResearchMindmap components |
| Testing | Pytest configuration + tests |

## Environment

Required in `backend/.env`:
- `GEMINI_API_KEY` - Google Gemini API key (mandatory)
- `DEBUG=True` - Enable debug logging
- `OLLAMA_BASE_URL=http://localhost:11434` - Optional: local Ollama for LLM

## Key Conventions

- Project IDs are UUIDs: `research_{uuid.uuid4()}`
- Agent state uses TypedDict `ResearchState` (see `agent/state.py:18`)
- All agent nodes return updated state dicts
- Frontend proxies assets from `backend/research_projects/`
- Docker code executor lives at `backend/docker/` (isolated Python sandbox)

## Ollama Local Models (if using)

Your local Ollama has these models available:
- `deepseek-ocr:3b` - OCR processing
- `nomic-embed-text:latest` - Embeddings
- `ministral-3:3b` - Fast reasoning
- `ibm/granite-docling:latest` - Document processing

Set `OLLAMA_BASE_URL` and configure in `app_research.py` to use local models.
