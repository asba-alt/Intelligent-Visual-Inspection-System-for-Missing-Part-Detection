# Intelligent Visual Inspection System - Migration Guide

## Migration from Flask + HTML to FastAPI + Streamlit

The application has been successfully migrated to use FastAPI for the backend and Streamlit for the frontend.

### New Architecture

#### Backend: FastAPI (`app_fastapi.py`)
- **Port**: 8000
- **Features**:
  - RESTful API with automatic documentation
  - File upload endpoint: `/predict`
  - Health check endpoint: `/health`
  - CORS enabled for frontend access
  - Static file serving
  - Model loaded once at startup for better performance

#### Frontend: Streamlit (`app_streamlit.py`)
- **Port**: 8501 (default)
- **Features**:
  - Interactive UI with file upload
  - Real-time image preview
  - Side-by-side comparison (Original vs Grad-CAM)
  - Metrics display
  - API health monitoring
  - Responsive layout

### Running the Application

#### 1. Start FastAPI Backend
```powershell
# Option 1: Direct Python
python app_fastapi.py

# Option 2: Using Uvicorn (recommended for production)
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload

# With TensorFlow warnings suppressed
$env:TF_ENABLE_ONEDNN_OPTS='0'; $env:TF_CPP_MIN_LOG_LEVEL='2'; python app_fastapi.py
```

#### 2. Start Streamlit Frontend
```powershell
# In a separate terminal
streamlit run app_streamlit.py

# With custom port
streamlit run app_streamlit.py --server.port 8501
```

### API Documentation

Once FastAPI is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

- `GET /` - API information
- `GET /health` - Health check and model status
- `POST /predict` - Upload image and get prediction
- `GET /image/{folder}/{filename}` - Serve images

### Dependencies Added

```
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
python-multipart>=0.0.6
requests>=2.31.0
```

### Key Differences

| Feature | Old (Flask + HTML) | New (FastAPI + Streamlit) |
|---------|-------------------|---------------------------|
| Backend | Flask | FastAPI |
| Frontend | Jinja2 Templates | Streamlit |
| API Docs | Manual | Auto-generated (Swagger/ReDoc) |
| Port | 5000 | Backend: 8000, Frontend: 8501 |
| Performance | Model loads per request | Model loaded once at startup |
| UI Updates | Page reload | Real-time |

### Benefits

1. **Better Performance**: Model loaded once at startup
2. **API Documentation**: Automatic interactive API docs
3. **Modern Stack**: FastAPI (async) + Streamlit (reactive)
4. **Better UX**: Real-time updates, no page reloads
5. **Separation of Concerns**: Clean backend/frontend split
6. **Type Safety**: FastAPI provides automatic validation

### Legacy Files

The old Flask application files are preserved:
- `app.py` - Original Flask application
- `templates/` - HTML templates
- `static/css/` - CSS files

You can delete these if you no longer need them.
