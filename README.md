# Adhyatmik Intelligence AI - Multilingual RAG Chatbot

A sophisticated multilingual RAG (Retrieval-Augmented Generation) chatbot designed for Gujarati session transcripts with support for English, Hindi, and Hinglish queries.

## 🌟 Features

- **Multilingual Support**: Query in English, Hindi, Gujarati, or Hinglish
- **Cross-Language Search**: Find Gujarati content with English queries and vice versa
- **Vector Database**: ChromaDB for semantic similarity search
- **Modern UI**: Clean, professional chat interface inspired by enterprise tools
- **Real-time Processing**: Fast embedding generation and retrieval
- **Session Management**: Conversation history and context management

## 🏗️ Architecture

### Backend (FastAPI)
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: Multilingual sentence transformers
- **API**: RESTful endpoints for document management and search
- **Language Processing**: Semantic search across languages

### Frontend (React + TypeScript)
- **Framework**: Vite + React 18 + TypeScript
- **Styling**: Tailwind CSS with custom design system
- **UI Components**: Radix UI for accessibility
- **State Management**: React hooks with proper error handling

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- Node.js 20+
- Git

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip3 install fastapi uvicorn pydantic chromadb sentence-transformers
   ```

3. **Start the backend server**:
   ```bash
   python3 main.py
   ```

   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173`

## 📊 API Endpoints

### Core Endpoints
- `GET /health` - Health check and system status
- `GET /stats` - System statistics and model information
- `GET /languages` - Supported languages list

### Document Management
- `POST /documents/add` - Add documents to vector database
- `POST /documents/search` - Semantic search across documents
- `GET /documents/stats` - Document collection statistics

### Chat Interface
- `POST /chat` - Main chat endpoint (connects to RAG system)

## 🧪 Testing the System

### Add a Sample Document
```bash
curl -X POST "http://localhost:8000/documents/add" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "આજે આપણે ધ્યાન અને આધ્યાત્મિકતા વિશે વાત કરીશું. ધ્યાન એ મનની શાંતિ માટે ખૂબ જ મહત્વપૂર્ણ છે.",
    "document_id": "gujarati_session_001",
    "language": "gujarati",
    "metadata": {
      "speaker": "Teacher",
      "topic": "Meditation and Spirituality",
      "session_date": "2024-01-15"
    }
  }'
```

### Search Across Languages
```bash
# English query finding Gujarati content
curl -X POST "http://localhost:8000/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was discussed about meditation?",
    "max_results": 3
  }'

# Gujarati query
curl -X POST "http://localhost:8000/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ધ્યાન વિશે શું કહેવાયું?",
    "max_results": 3
  }'
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Multilingual text embeddings
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server

### Frontend
- **React 18**: Modern React with hooks
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible UI components
- **Axios**: HTTP client for API communication

## 📁 Project Structure

```
AI-Chatbot/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── config.py              # Configuration management
│   └── services/
│       ├── vector_db.py       # ChromaDB integration
│       ├── embedding_service.py # Text embedding service
│       └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main React component
│   │   ├── lib/
│   │   │   └── utils.ts       # Utility functions
│   │   └── services/
│   │       └── api.ts         # API client
│   ├── package.json           # Node.js dependencies
│   └── tailwind.config.js     # Tailwind configuration
└── README.md
```

## 🔮 Roadmap

### Phase 1: Core RAG System ✅
- [x] Vector database integration
- [x] Multilingual embeddings
- [x] Document storage and search
- [x] Basic API endpoints

### Phase 2: Enhanced Processing (In Progress)
- [ ] Document preprocessing pipeline
- [ ] Text chunking for long transcripts
- [ ] Translation services integration
- [ ] Language detection

### Phase 3: Advanced Features (Planned)
- [ ] Conversation memory
- [ ] Query suggestions
- [ ] Document upload interface
- [ ] Analytics and monitoring

### Phase 4: Production Ready (Future)
- [ ] Authentication and authorization
- [ ] Rate limiting
- [ ] Caching layer
- [ ] Deployment configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** for multilingual embeddings
- **ChromaDB** for vector database capabilities
- **FastAPI** for the excellent Python web framework
- **React** and **Vite** for the modern frontend stack

---

**Built with ❤️ for multilingual AI conversations**
