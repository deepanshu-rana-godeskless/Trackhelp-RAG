# RAG Document Processing API - Project Structure

## 📁 Recommended Directory Structure

```
TRACKHELP-RAG/
├── main.py                     # Main application entry point
├── config.py                   # Configuration management
├── schemas.py                  # Pydantic models for validation
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables
├── api/
│   ├── __init__.py
│   └── routes.py              # API route handlers
├── services/
│   ├── __init__.py
│   ├── vector_store.py        # Vector store operations
│   ├── llm.py                 # LLM service
│   └── document_processor.py   # Document processing
├── uploads/                   # Temporary file uploads (auto-created)
├── chroma_store/             # Vector database storage (auto-created)
└── tests/                    # Unit tests (optional)
    ├── __init__.py
    ├── test_routes.py
    └── test_services.py
```

## 🚀 Quick Setup

1. **Create the directory structure:**
```bash
mkdir TRACKHELP-RAG
cd TRACKHELP-RAG
mkdir api services tests
touch api/__init__.py services/__init__.py tests/__init__.py
```

2. **Create environment file (.env):**
```bash
CHROMA_PERSIST_DIR=./chroma_store
EMBED_MODEL=nomic-embed-text
LLM_MODEL=llama3.2:3b
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🏗️ Architecture Overview

### **Separation of Concerns:**
- **`main.py`**: Application bootstrap and middleware setup
- **`config.py`**: Centralized configuration management
- **`schemas.py`**: Request/response validation models
- **`api/routes.py`**: HTTP route handlers and endpoint logic
- **`services/`**: Business logic and external service integrations

### **Service Layer:**
- **`VectorStoreService`**: Manages Chroma vector database operations
- **`LLMService`**: Handles LLM interactions and prompt engineering
- **`DocumentProcessorService`**: PDF processing and text chunking

### **Key Improvements:**
1. **Type Safety**: Full typing with Pydantic models
2. **Error Handling**: Comprehensive exception management
3. **Async Support**: Proper async/await for file operations
4. **Resource Management**: Automatic cleanup of temporary files
5. **Modularity**: Clear separation of business logic
6. **Configuration**: Environment-based settings management
7. **Documentation**: Comprehensive docstrings and API docs

## 📊 API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /upload/` - Upload PDF files
- `GET /files/` - List processed files
- `GET /files/{file_name}/details` - Get file chunks
- `DELETE /files/{file_name}` - Delete file vectors
- `POST /ask/` - RAG-based Q&A
- `POST /insights/` - General LLM queries

## 🔧 Configuration Options

All settings are managed through environment variables and can be customized in the `.env` file or through the `Settings` class in `config.py`.

## 🧪 Testing

The modular structure makes unit testing straightforward:
- Mock services independently
- Test business logic separate from API layer
- Validate schemas and error handling
