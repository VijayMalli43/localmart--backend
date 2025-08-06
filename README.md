"# 🛒 LocalMartAI - Smart Local Marketplace with AI

An intelligent local marketplace platform powered by AI for optimal vendor recommendations, sentiment analysis, and smart order fulfillment.

## 🚀 Features

### Core Functionality
- **Multi-vendor marketplace** with smart product management
- **AI-powered order fulfillment** optimization
- **Real-time sentiment analysis** on customer reviews
- **Dynamic vendor scoring** and recommendations
- **Smart cart management** with automated vendor selection

### AI Capabilities
- **Sentiment Analysis**: NLP-powered review sentiment scoring
- **Vendor Recommendations**: ML-based vendor scoring system
- **Order Optimization**: AI-driven fulfillment planning
- **Dynamic Pricing**: Smart vendor selection based on multiple factors

## 📁 Project Structure

```
LocalMartAI_Project/
├── manage.py                 # Django management script
├── README.md                # Project documentation
├── .gitignore              # Git ignore rules
├── api_tester.html         # Web-based API testing tool
├── api_tester.py           # Python API testing script
│
├── LocalMartAI/            # Main Django project
│   ├── __init__.py
│   ├── settings.py         # Django settings
│   ├── urls.py            # Main URL configuration
│   ├── wsgi.py            # WSGI application
│   └── asgi.py            # ASGI application
│
└── core/                   # Main application
    ├── __init__.py
    ├── admin.py           # Django admin configuration
    ├── apps.py            # App configuration
    ├── models.py          # Database models
    ├── serializers.py     # DRF serializers
    ├── views.py           # API views and business logic
    ├── urls.py            # App URL patterns
    ├── ml_engine.py       # AI/ML integration engine
    ├── tests.py           # Unit tests
    │
    ├── migrations/        # Database migrations
    │   ├── __init__.py
    │   └── 0001_initial.py
    │
    └── ai_models/         # AI/ML model files
        ├── dynamic_ml/    # ML models for recommendations
        │   ├── vendor_recommender.pkl
        │   └── feature_scaler.pkl
        └── sentiment_model/  # NLP sentiment analysis
            ├── config.json
            ├── pytorch_model.bin
            ├── tokenizer.json
            ├── vocab.txt
            └── ...
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- MySQL 8.0+
- Virtual Environment

### Important Note about AI Models
⚠️ **AI model files are not included in this repository due to size limitations.**
The following directories are excluded from the repository:
- `core/ai_models/dynamic_ml/` (ML recommendation models)
- `core/ai_models/sentiment_model/` (NLP sentiment analysis model)

To fully enable AI features, you'll need to:
1. Train your own models or obtain pre-trained models
2. Place them in the appropriate directories as shown in the project structure
3. Ensure model file names match those expected by `ml_engine.py`

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/VijayMalli43/localmart--backend.git
cd localmart--backend
```

2. **Create and activate virtual environment**
```bash
python -m venv venv_new
venv_new\Scripts\activate  # Windows
# source venv_new/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure database**
- Update `LocalMartAI/settings.py` with your MySQL credentials
- Create database: `local_mart_ai`

5. **Run migrations**
```bash
python manage.py makemigrations
python manage.py migrate
```

6. **Start development server**
```bash
python manage.py runserver
```

## 🧪 API Testing

### Option 1: Web Interface (Recommended)
Open `api_tester.html` in your browser for a user-friendly testing interface.

### Option 2: Python Script
```bash
python api_tester.py
```

## 📚 API Endpoints

### Authentication
- `POST /api/auth/register/` - User registration
- `POST /api/auth/login/` - User login (returns token)
- `POST /api/auth/logout/` - User logout

### Core Resources
- `GET/POST /api/vendors/` - Vendor management
- `GET/POST /api/products/` - Product catalog
- `GET/POST /api/reviews/` - Customer reviews (with AI sentiment)
- `GET/POST /api/cart/` - Shopping cart
- `GET/POST /api/orders/` - Order management

### AI Features
- `POST /api/calculate-fulfillment/` - AI order optimization
- Reviews automatically get sentiment analysis
- Vendor scores updated based on AI algorithms

## 🤖 AI Models

### Sentiment Analysis
- **Model**: Fine-tuned BERT-based transformer
- **Purpose**: Analyze customer review sentiment
- **Output**: Sentiment score (-1.0 to 1.0)

### Vendor Recommendation
- **Algorithm**: Machine Learning ensemble
- **Features**: Delivery time, ratings, stock levels, location
- **Purpose**: Optimize vendor selection for orders

---

**LocalMartAI** - Revolutionizing local commerce with artificial intelligence! 🚀" 
"# grocery-adv-features" 
"# back--end" 
"# localmart--backend" 
