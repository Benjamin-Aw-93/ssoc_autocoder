# SSOC Autocoder

Predicts **Singapore Standard Occupational Classification (SSOC) 2020** codes from job descriptions on [MyCareersFuture](https://www.mycareersfuture.gov.sg/). Paste a job listing URL, get the top 10 predicted SSOC codes with confidence scores.

## Architecture

```
React SPA  -->  Heroku CORS Proxy  -->  AWS API Gateway  -->  FastAPI / Lambda
                                                                    |
                                                          MCF API + DistilBERT
                                                          Hierarchical Classifier
```

## Project Structure

| Folder | Description |
|--------|-------------|
| `ssoc_autocoder/` | Core Python package — data processing, model training, prediction |
| `Deployments/` | Backend API, frontend React app, Lambda functions, data processing scripts |
| `Notebooks/` | Jupyter notebooks for exploration, analysis, and development |
| `Tests/` | Unit and integration tests |

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Run frontend
cd Deployments/frontend
npm install
npm start
```

## Tech Stack

- **ML:** PyTorch, DistilBERT (HuggingFace Transformers), spaCy
- **Backend:** FastAPI, AWS Lambda
- **Frontend:** React 17, MUI v5
- **Infra:** AWS Amplify, Docker, Heroku CORS proxy

## Team

Shaun Khoo, Benjamin Aw — started Aug 2021.

## Docs

See [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) for the full reverse-engineering specification.
