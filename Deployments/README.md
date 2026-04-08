# Deployments

All deployment configurations and scripts.

## Subdirectories

| Folder | Description |
|--------|-------------|
| `backend/` | AWS Amplify backend — FastAPI app in Docker container |
| `frontend/` | React SPA (Create React App) |
| `lambda/` | AWS Lambda functions (model-predict + dummy-api) |
| `feature/` | Data processing scripts for MCF data ingestion |

## Infrastructure

- **AWS Account:** `project-finch`
- **API Gateway:** `e81tvuwky6.execute-api.us-east-1.amazonaws.com`
- **CORS Proxy:** Heroku app `evening-plateau-95803`
