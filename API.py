# MRWAI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path
from datetime import datetime
import hashlib
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize FastAPI app
app = FastAPI(
    title="Movie Review AI API",
    description="API for analyzing movie reviews with AI",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache setup
CACHE_DIR = "review_cache"
Path(CACHE_DIR).mkdir(exist_ok=True)

# Load AI models (lazy loading)
class AIModels:
    def __init__(self):
        self._sentiment_model = None
        self._summarization_model = None
        self._spam_model = None
        self._spam_vectorizer = None
    
    @property
    def sentiment_model(self):
        if self._sentiment_model is None:
            print("Loading sentiment model...")
            self._sentiment_model = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        return self._sentiment_model
    
    @property
    def summarization_model(self):
        if self._summarization_model is None:
            print("Loading summarization model...")
            self._summarization_model = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn"
            )
        return self._summarization_model
    
    @property
    def spam_model(self):
        if self._spam_model is None:
            print("Loading spam detection model...")
            # This would normally be loaded from a trained model file
            # For demo, we'll use a simple logistic regression
            self._spam_model, self._spam_vectorizer = self._train_demo_spam_model()
        return self._spam_model, self._spam_vectorizer
    
    def _train_demo_spam_model(self):
        # This is just for demo - in production you'd load a pre-trained model
        texts = [
            "Great movie, loved it!",
            "Terrible film, waste of time",
            "Click here to win $1000!",
            "Free tickets available now!!!",
            "The acting was superb",
            "Make money fast with this one trick"
        ]
        labels = [0, 0, 1, 1, 0, 1]  # 1 = spam
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression()
        model.fit(X, labels)
        
        return model, vectorizer

models = AIModels()

# Request models
class ReviewAnalysisRequest(BaseModel):
    review_text: str
    user_id: Optional[str] = None
    movie_id: Optional[str] = None

class BatchReviewRequest(BaseModel):
    reviews: list[ReviewAnalysisRequest]

# Response models
class SentimentResult(BaseModel):
    label: str
    score: float

class SummaryResult(BaseModel):
    summary: str
    length_reduction: float

class SpamResult(BaseModel):
    is_spam: bool
    confidence: float

class RatingPrediction(BaseModel):
    predicted_rating: float  # 1-5 scale
    confidence: float

class ReviewAnalysisResult(BaseModel):
    sentiment: SentimentResult
    summary: Optional[SummaryResult] = None
    spam: Optional[SpamResult] = None
    rating: Optional[RatingPrediction] = None
    review_hash: str
    processing_time_ms: float

# Helper functions
def get_review_hash(review_text: str) -> str:
    return hashlib.md5(review_text.encode()).hexdigest()

def save_to_cache(review_hash: str, result: dict):
    cache_file = Path(CACHE_DIR) / f"{review_hash}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

def load_from_cache(review_hash: str) -> Optional[dict]:
    cache_file = Path(CACHE_DIR) / f"{review_hash}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def predict_rating(sentiment_score: float) -> RatingPrediction:
    """Convert sentiment score (0-1) to 1-5 star rating"""
    rating = 1 + (sentiment_score * 4)  # Scale to 1-5
    return RatingPrediction(
        predicted_rating=round(rating, 1),
        confidence=min(sentiment_score, 1 - sentiment_score) * 2
    )

# API Endpoints
@app.post("/analyze", response_model=ReviewAnalysisResult)
async def analyze_review(request: ReviewAnalysisRequest):
    """Analyze a single movie review"""
    start_time = datetime.now()
    
    # Check cache first
    review_hash = get_review_hash(request.review_text)
    cached_result = load_from_cache(review_hash)
    if cached_result:
        return {**cached_result, "cached": True}
    
    try:
        # Sentiment analysis
        sentiment_result = models.sentiment_model(request.review_text)[0]
        
        # Spam detection
        spam_model, spam_vectorizer = models.spam_model
        features = spam_vectorizer.transform([request.review_text])
        spam_prob = spam_model.predict_proba(features)[0][1]
        
        # Rating prediction
        rating_prediction = predict_rating(
            sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' 
            else 1 - sentiment_result['score']
        )
        
        # Build result
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result = ReviewAnalysisResult(
            sentiment=SentimentResult(
                label=sentiment_result['label'],
                score=float(sentiment_result['score'])
            ),
            spam=SpamResult(
                is_spam=spam_prob > 0.7,
                confidence=float(spam_prob)
            ),
            rating=rating_prediction,
            review_hash=review_hash,
            processing_time_ms=processing_time
        )
        
        # Save to cache
        save_to_cache(review_hash, result.dict())
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_batch", response_model=list[ReviewAnalysisResult])
async def analyze_batch_reviews(request: BatchReviewRequest):
    """Analyze multiple reviews at once"""
    return [await analyze_review(review) for review in request.reviews]

@app.post("/summarize", response_model=SummaryResult)
async def summarize_review(review_text: str = Form(...)):
    """Generate a summary of a movie review"""
    try:
        start_time = datetime.now()
        
        summary = models.summarization_model(
            review_text,
            max_length=100,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SummaryResult(
            summary=summary,
            length_reduction=1 - (len(summary) / len(review_text))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Service health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
