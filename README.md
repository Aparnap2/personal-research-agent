# Research Intelligence Platform

![Research Intelligence Platform](docs/images/platform-banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18.0+-61DAFB.svg?logo=react)](https://reactjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.30+-orange.svg)](https://github.com/langchain-ai/langgraph)

## 🚀 Overview

The Research Intelligence Platform is a powerful, AI-driven research assistant that transforms complex research queries into comprehensive, data-driven reports. Built with a sophisticated LangGraph architecture, this platform automates the entire research workflow from planning and data collection to statistical analysis and visualization.

**[Live Demo](https://research-intelligence-platform.demo.com)** | **[Documentation](https://github.com/yourusername/personal-research-agent/wiki)**

![Dashboard Screenshot](docs/images/dashboard-screenshot.png)

## ✨ Features

- **Advanced Research Automation**: Transform natural language queries into comprehensive research reports
- **Multi-stage LangGraph Architecture**: Sophisticated agent workflow with specialized nodes for each research phase
- **Data Validation & Quality Assessment**: Automatic validation of extracted data with quality scoring
- **Citation Management**: Automatic extraction and formatting of citations in multiple academic styles
- **Comparative Analysis**: Segment data across multiple dimensions for insightful comparisons
- **Interactive Visualizations**: Generate charts, graphs, and tables to represent research findings
- **Modern React UI**: Professional, responsive interface with dashboard analytics
- **Project History**: Track and revisit previous research projects

## 🔍 How It Works

The platform uses a multi-stage LangGraph architecture to process research queries:

1. **Research Planning**: Analyzes the query and creates a structured research plan
2. **Web Search & Content Scraping**: Gathers relevant information from authoritative sources
3. **Citation Management**: Extracts and formats citations from all sources
4. **Content Synthesis**: Processes and synthesizes information into coherent insights
5. **Quantitative Extraction**: Identifies and extracts numerical data from sources
6. **Data Validation**: Validates extracted data for quality and consistency
7. **Statistical Analysis**: Performs statistical analysis on validated data
8. **Comparative Analysis**: Compares data across different segments and categories
9. **Visualization**: Generates charts, graphs, and tables to represent findings
10. **Report Compilation**: Creates a comprehensive, professional research report

![Architecture Diagram](docs/images/architecture-diagram.png)

## 🛠️ Technology Stack

- **Backend**:
  - Python 3.9+
  - LangGraph for agent orchestration
  - FastAPI for API endpoints
  - Pandas & NumPy for data processing
  - Matplotlib for chart generation

- **Frontend**:
  - React 18+
  - Material-UI for component library
  - React Router for navigation
  - Chart.js for interactive visualizations

## 📋 Installation

### Prerequisites

- Python 3.9+
- Node.js 16+
- API key for LLM service (Google Gemini, OpenAI, etc.)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/personal-research-agent.git
cd personal-research-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start the backend server
cd backend
uvicorn main:app --reload
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

Visit `http://localhost:3000` to access the application.

## 🧪 Example Usage

1. Enter a research query like "Market size and growth projections for AI in healthcare diagnostics in North America for 2024-2026"
2. The system will process the query through its LangGraph workflow
3. Monitor progress in real-time as the agent works through each research phase
4. Receive a comprehensive report with:
   - Executive summary
   - Detailed analysis
   - Statistical findings
   - Comparative insights
   - Data visualizations
   - Citations and references

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For freelance inquiries or collaboration opportunities:

- Email: your.email@example.com
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Portfolio: [yourportfolio.com](https://yourportfolio.com)

---

<p align="center">
  <i>Built with ❤️ by Your Name</i>
</p>