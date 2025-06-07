# MRWAI
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status, Request, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr, validator
from passlib.context import CryptContext
from jose import JWTError, jwt
import requests
import logging
from logging.handlers import RotatingFileHandler
import uuid
import aiofiles
from typing import Annotated
from enum import Enum
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncpg
from asyncpg.pool import Pool
import uvicorn
from config import settings

# --------------------------
# Setup and Configuration
# --------------------------

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("api.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Review AI Analysis API",
    description="Advanced API for analyzing movie reviews with DeepSeek AI integration",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Security and Authentication
# --------------------------

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: Optional[bool] = False

class UserInDB(User):
    hashed_password: str

class UserCreate(User):
    password: str

    @validator('password')
    def password_complexity(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

# --------------------------
# Database Models
# --------------------------

class ReviewSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class MovieReview(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    movie_id: str
    movie_title: str
    review_text: str
    rating: int = Field(..., ge=1, le=5)
    sentiment: Optional[ReviewSentiment] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MovieReviewCreate(BaseModel):
    movie_id: str
    movie_title: str
    review_text: str
    rating: int = Field(..., ge=1, le=5)

class MovieReviewUpdate(BaseModel):
    review_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)

class MovieAnalysisResult(BaseModel):
    movie_id: str
    movie_title: str
    average_rating: float
    sentiment_distribution: Dict[ReviewSentiment, int]
    common_themes: List[str]
    similar_movies: List[str]

# --------------------------
# Database Connection
# --------------------------

async def get_db_pool() -> Pool:
    return await asyncpg.create_pool(
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        min_size=5,
        max_size=20
    )

@app.on_event("startup")
async def startup_event():
    app.state.db_pool = await get_db_pool()
    logger.info("Database connection pool created")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.db_pool.close()
    logger.info("Database connection pool closed")

# --------------------------
# Authentication Utilities
# --------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

async def authenticate_user(db_pool: Pool, username: str, password: str):
    async with db_pool.acquire() as conn:
        user_record = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1", username
        )
        if not user_record:
            return False
        if not verify_password(password, user_record["hashed_password"]):
            return False
        return UserInDB(**user_record)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db_pool: Pool = Depends(get_db_pool)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    async with db_pool.acquire() as conn:
        user_record = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1", token_data.username
        )
        if user_record is None:
            raise credentials_exception
        return UserInDB(**user_record)

# --------------------------
# DeepSeek AI Integration
# --------------------------

class DeepSeekAnalysisRequest(BaseModel):
    review_text: str
    analyze_sentiment: bool = True
    extract_key_phrases: bool = True
    suggest_improvements: bool = False
    compare_with_similar: bool = False

async def analyze_with_deepseek(review_text: str, additional_context: str = "") -> Dict[str, Any]:
    """
    Send review text to DeepSeek AI for comprehensive analysis
    """
    try:
        headers = {
            "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Perform a comprehensive analysis of this movie review:
        
        Review Text:
        {review_text}
        
        {additional_context}
        
        Provide analysis in JSON format with these fields:
        - sentiment (positive/negative/neutral)
        - sentiment_score (0-1)
        - key_phrases (list of important phrases)
        - themes (list of themes detected)
        - rating_suggestion (1-5)
        - constructive_criticism (if any)
        - notable_aspects (what stood out)
        """
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": { "type": "json_object" }
        }
        
        response = requests.post(
            settings.DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", {})
        
    except Exception as e:
        logger.error(f"DeepSeek API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis service is currently unavailable"
        )

# --------------------------
# API Endpoints
# --------------------------

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db_pool: Pool = Depends(get_db_pool)
):
    user = await authenticate_user(db_pool, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate, db_pool: Pool = Depends(get_db_pool)):
    async with db_pool.acquire() as conn:
        existing_user = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1 OR email = $2", 
            user.username, 
            user.email
        )
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
        
        hashed_password = get_password_hash(user.password)
        new_user = await conn.fetchrow(
            """
            INSERT INTO users (username, email, full_name, hashed_password, disabled)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            user.username,
            user.email,
            user.full_name,
            hashed_password,
            user.disabled or False
        )
        return UserInDB(**new_user)

@app.post("/reviews/", response_model=MovieReview)
async def create_review(
    review: MovieReviewCreate,
    current_user: User = Depends(get_current_user),
    db_pool: Pool = Depends(get_db_pool)
):
    # Get AI analysis
    ai_analysis = await analyze_with_deepseek(review.review_text)
    
    # Determine sentiment from AI analysis or fallback to rating
    sentiment = (
        ReviewSentiment(ai_analysis.get("sentiment", "neutral"))
        if ai_analysis and "sentiment" in ai_analysis
        else (
            ReviewSentiment.POSITIVE if review.rating >= 4 
            else ReviewSentiment.NEGATIVE if review.rating <= 2 
            else ReviewSentiment.NEUTRAL
        )
    )
    
    async with db_pool.acquire() as conn:
        new_review = await conn.fetchrow(
            """
            INSERT INTO reviews (
                id, user_id, movie_id, movie_title, 
                review_text, rating, sentiment, ai_analysis
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            """,
            str(uuid.uuid4()),
            current_user.username,
            review.movie_id,
            review.movie_title,
            review.review_text,
            review.rating,
            sentiment.value,
            ai_analysis
        )
        return MovieReview(**new_review)

@app.get("/reviews/{review_id}", response_model=MovieReview)
async def read_review(
    review_id: str,
    current_user: User = Depends(get_current_user),
    db_pool: Pool = Depends(get_db_pool)
):
    async with db_pool.acquire() as conn:
        review = await conn.fetchrow(
            "SELECT * FROM reviews WHERE id = $1 AND user_id = $2",
            review_id,
            current_user.username
        )
        if not review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review not found"
            )
        return MovieReview(**review)

@app.get("/reviews/", response_model=List[MovieReview])
async def read_user_reviews(
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
    db_pool: Pool = Depends(get_db_pool)
):
    async with db_pool.acquire() as conn:
        reviews = await conn.fetch(
            "SELECT * FROM reviews WHERE user_id = $1 ORDER BY created_at DESC OFFSET $2 LIMIT $3",
            current_user.username,
            skip,
            limit
        )
        return [MovieReview(**review) for review in reviews]

@app.put("/reviews/{review_id}", response_model=MovieReview)
async def update_review(
    review_id: str,
    review_update: MovieReviewUpdate,
    current_user: User = Depends(get_current_user),
    db_pool: Pool = Depends(get_db_pool)
):
    async with db_pool.acquire() as conn:
        # Get existing review
        existing_review = await conn.fetchrow(
            "SELECT * FROM reviews WHERE id = $1 AND user_id = $2",
            review_id,
            current_user.username
        )
        if not existing_review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review not found"
            )
        
        # Update fields if provided
        updated_fields = {}
        if review_update.review_text is not None:
            updated_fields["review_text"] = review_update.review_text
            # Re-analyze with DeepSeek if text changed
            ai_analysis = await analyze_with_deepseek(review_update.review_text)
            updated_fields["ai_analysis"] = ai_analysis
            updated_fields["sentiment"] = ReviewSentiment(ai_analysis.get("sentiment", "neutral"))
        
        if review_update.rating is not None:
            updated_fields["rating"] = review_update.rating
        
        if not updated_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        updated_fields["updated_at"] = datetime.utcnow()
        
        # Build and execute update query
        set_clause = ", ".join([f"{k} = ${i+2}" for i, k in enumerate(updated_fields.keys())])
        values = list(updated_fields.values())
        
        updated_review = await conn.fetchrow(
            f"""
            UPDATE reviews
            SET {set_clause}
            WHERE id = $1
            RETURNING *
            """,
            review_id,
            *values
        )
        
        return MovieReview(**updated_review)

@app.delete("/reviews/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_review(
    review_id: str,
    current_user: User = Depends(get_current_user),
    db_pool: Pool = Depends(get_db_pool)
):
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM reviews WHERE id = $1 AND user_id = $2",
            review_id,
            current_user.username
        )
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review not found"
            )

@app.post("/reviews/analyze", response_model=Dict[str, Any])
async def analyze_review_text(
    request: DeepSeekAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Standalone endpoint for analyzing review text without saving
    """
    return await analyze_with_deepseek(
        request.review_text,
        additional_context=(
            "Please provide detailed analysis including suggestions for improvement "
            "if requested by the user."
        )
    )

@app.get("/movies/{movie_id}/analysis", response_model=MovieAnalysisResult)
async def analyze_movie_reviews(
    movie_id: str,
    current_user: User = Depends(get_current_user),
    db_pool: Pool = Depends(get_db_pool)
):
    """
    Aggregate analysis of all reviews for a specific movie
    """
    async with db_pool.acquire() as conn:
        # Get all reviews for the movie
        reviews = await conn.fetch(
            "SELECT * FROM reviews WHERE movie_id = $1",
            movie_id
        )
        
        if not reviews:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No reviews found for this movie"
            )
        
        reviews = [MovieReview(**review) for review in reviews]
        
        # Calculate average rating
        avg_rating = sum(review.rating for review in reviews) / len(reviews)
        
        # Sentiment distribution
        sentiment_counts = {
            ReviewSentiment.POSITIVE: 0,
            ReviewSentiment.NEGATIVE: 0,
            ReviewSentiment.NEUTRAL: 0
        }
        for review in reviews:
            sentiment_counts[review.sentiment] += 1
        
        # Extract common themes using TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', max_features=20)
        tfidf_matrix = tfidf.fit_transform([r.review_text for r in reviews])
        feature_names = tfidf.get_feature_names_out()
        dense = tfidf_matrix.todense()
        importance = np.array(dense).sum(axis=0)
        top_themes = [feature_names[i] for i in importance.argsort()[0][-5:][::-1]]
        
        # Find similar movies (simplified example)
        similar_movies = ["movie123", "movie456"]  # In real app, implement similarity logic
        
        return MovieAnalysisResult(
            movie_id=movie_id,
            movie_title=reviews[0].movie_title,
            average_rating=round(avg_rating, 1),
            sentiment_distribution=sentiment_counts,
            common_themes=top_themes,
            similar_movies=similar_movies
        )

@app.post("/reviews/batch_upload")
async def upload_reviews_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db_pool: Pool = Depends(get_db_pool)
):
    """
    Upload a CSV file with multiple reviews for batch processing
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported"
        )
    
    try:
        # Save uploaded file temporarily
        file_path = f"temp_{file.filename}"
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Process CSV
        df = pd.read_csv(file_path)
        required_columns = {'movie_id', 'movie_title', 'review_text', 'rating'}
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV must contain columns: {required_columns}"
            )
        
        # Process each review
        results = []
        async with db_pool.acquire() as conn:
            for _, row in df.iterrows():
                try:
                    # Get AI analysis
                    ai_analysis = await analyze_with_deepseek(row['review_text'])
                    
                    # Determine sentiment
                    sentiment = (
                        ReviewSentiment(ai_analysis.get("sentiment", "neutral"))
                        if ai_analysis and "sentiment" in ai_analysis
                        else (
                            ReviewSentiment.POSITIVE if row['rating'] >= 4 
                            else ReviewSentiment.NEGATIVE if row['rating'] <= 2 
                            else ReviewSentiment.NEUTRAL
                        )
                    )
                    
                    # Insert review
                    new_review = await conn.fetchrow(
                        """
                        INSERT INTO reviews (
                            id, user_id, movie_id, movie_title, 
                            review_text, rating, sentiment, ai_analysis
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        RETURNING id
                        """,
                        str(uuid.uuid4()),
                        current_user.username,
                        row['movie_id'],
                        row['movie_title'],
                        row['review_text'],
                        int(row['rating']),
                        sentiment.value,
                        ai_analysis
                    )
                    results.append({"status": "success", "review_id": new_review["id"]})
                except Exception as e:
                    results.append({
                        "status": "error",
                        "error": str(e),
                        "data": row.to_dict()
                    })
        
        # Clean up
        os.remove(file_path)
        
        return {
            "processed": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Batch upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch upload"
        )

# --------------------------
# Error Handlers
# --------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder({
            "detail": exc.detail,
            "path": request.url.path
        }),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# --------------------------
# Main Execution
# --------------------------

if _name_ == "_main_":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
