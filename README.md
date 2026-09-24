# AI-ATS: AI-Powered Applicant Tracking System

An intelligent, ethical, and explainable Applicant Tracking System designed for modern recruitment workflows. Built with offline-first architecture for enterprise deployment.

## Features

### Core Capabilities

- **Smart Resume Parsing**: Automated extraction of skills, experience, education, and contact information
- **Intelligent Matching**: AI-powered candidate-job matching using semantic understanding
- **Ranking Engine**: Multi-factor scoring with customizable weights

### AI & ML Components

- **NLP Pipeline**: Text preprocessing, Named Entity Recognition, and feature extraction
- **Embedding Engine**: Document chunking, vectorization, and RAG-based contextual retrieval
- **Fine-tuned Models**: Domain-specific transformer models for recruitment context

### Ethical AI & Explainability

- **Bias Detection**: Automated detection of potential biases in scoring
- **Fairness Metrics**: Quantitative fairness measurements across demographics
- **LIME & SHAP Explanations**: Transparent, interpretable AI decisions
- **Audit Logging**: Complete decision trail for compliance

### User Interface

- **Recruiter Dashboard**: Intuitive visualization of ranked candidates
- **Analytics & Reports**: Comprehensive hiring metrics and insights
- **Manual Override**: Human-in-the-loop control for final decisions

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI-ATS Architecture                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    │
│  │   Data       │───▶│   NLP & ML      │───▶│   Embedding &    │    │
│  │   Ingestion  │    │   Pipeline      │    │   Similarity     │    │
│  └──────────────┘    └─────────────────┘    └──────────────────┘    │
│         │                    │                       │              │
│         ▼                    ▼                       ▼              │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    │
│  │   Storage    │    │  Explainability │    │   Ranking &      │    │
│  │   (SQL/NoSQL)│    │  (LIME/SHAP)    │    │   Scoring        │    │
│  └──────────────┘    └─────────────────┘    └──────────────────┘    │
│                              │                       │              │
│                              ▼                       ▼              │
│                      ┌─────────────────┐    ┌──────────────────┐    │
│                      │   Ethical AI    │───▶│   Recruiter      │    │
│                      │   Subsystem     │    │   Dashboard      │    │
│                      └─────────────────┘    └──────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component      | Technology                                 |
| -------------- | ------------------------------------------ |
| Language       | Python 3.10+                               |
| GUI Framework  | PyQt6                                      |
| Database       | MongoDB (documents), ChromaDB (vectors)    |
| NLP            | spaCy, Transformers, Sentence-Transformers |
| ML Framework   | PyTorch, scikit-learn                      |
| Explainability | SHAP, LIME                                 |
| Fairness       | Fairlearn, AIF360                          |

## Installation

### Prerequisites

- Python 3.10 or higher
- MongoDB (local installation or Docker)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ai-ats.git
cd ai-ats
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -e .
```

4. For development:

```bash
pip install -e ".[dev]"
```

5. For model training (on powerful machines):

```bash
pip install -e ".[training]"
```

6. Download required models:

```bash
python -m spacy download en_core_web_trf
```

## Project Structure

```
ai-ats/
├── src/
│   ├── core/           # Core business logic
│   │   ├── candidate/  # Candidate management
│   │   ├── job/        # Job posting management
│   │   ├── matching/   # Matching engine
│   │   └── ranking/    # Ranking module
│   │
│   ├── ml/             # Machine Learning
│   │   ├── nlp/        # NLP pipeline
│   │   ├── embeddings/ # Embedding engine
│   │   ├── explainability/  # LIME/SHAP
│   │   └── ethics/     # Bias & fairness
│   │
│   ├── data/           # Data layer
│   │   ├── models/     # Data schemas
│   │   └── repositories/  # DB operations
│   │
│   ├── ui/             # PyQt6 GUI
│   │   ├── views/      # Screen views
│   │   └── widgets/    # UI components
│   │
│   ├── services/       # Business services
│   └── utils/          # Utilities
│
├── tests/              # Test suite
├── data/               # Data directory
├── docs/               # Documentation
├── configs/            # Configuration files
└── resources/          # UI resources
```

## Usage

### Running the Application

```bash
ai-ats-gui
```

### CLI Interface

```bash
ai-ats-cli --help
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

Proprietary - All Rights Reserved
