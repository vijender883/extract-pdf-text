from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import json
from typing import List, Dict, Any
import PyPDF2
import google.generativeai as genai
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="PDF Processing API",
    description="API for extracting text from PDFs and generating job-related multiple-choice questions",
    version="1.0.0"
)

# Configuration
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf'}

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Pydantic models for responses
class TextExtractionResponse(BaseModel):
    success: bool
    text: str
    character_count: int
    word_count: int

class Question(BaseModel):
    question: str
    options: List[str]
    correct_option: str  # Will be "A", "B", "C", or "D"

class QuestionGenerationResponse(BaseModel):
    success: bool
    extracted_text: str
    questions: List[Question]
    total_questions: int
    text_stats: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    message: str

class ErrorResponse(BaseModel):
    error: str

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 16MB")

async def extract_text_from_pdf(file: UploadFile) -> str:
    """Extract text content from PDF file."""
    try:
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        extracted_text = text.strip()
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF or PDF is empty")
        
        return extracted_text
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def generate_questions_with_gemini(resume_text: str) -> List[Dict[str, Any]]:
    """Generate multiple-choice questions using Gemini 2.0 Flash."""
    
    prompt = f"""
    Based on the following resume, generate exactly 25 multiple-choice questions focused on the technical skills, technologies, frameworks, tools, and programming concepts mentioned in the resume.

    RESUME CONTENT:
    {resume_text}

    INSTRUCTIONS:
    - Focus ONLY on technical content from the resume (programming languages, frameworks, databases, tools, methodologies, etc.)
    - Create questions about specific technologies, their features, best practices, and technical concepts
    - Include questions about experience levels with mentioned technologies
    - Each question should have exactly 4 options
    - The correct_option should be specified as "A", "B", "C", or "D" (corresponding to the position in the options array)
    - Make questions challenging but fair for someone with the experience level shown in the resume
    - Focus on practical technical knowledge and implementation details
    - If the resume mentions specific projects, create questions about the technical aspects of those projects

    EXAMPLE QUESTION TYPES:
    - "Which Python framework mentioned in the resume is primarily used for web development?"
    - "What is the main advantage of using Docker containers as mentioned in the candidate's experience?"
    - "Based on the resume, which database technology was used in the e-commerce project?"
    - "What testing framework does the candidate have experience with according to the resume?"

    REQUIRED OUTPUT FORMAT (respond with ONLY valid JSON, no additional text):
    [
        {{
            "question": "Which programming language mentioned in the resume is object-oriented?",
            "options": ["Python", "HTML", "CSS", "SQL"],
            "correct_option": "A"
        }},
        {{
            "question": "What JavaScript framework is mentioned in the candidate's project experience?",
            "options": ["Angular", "React", "Vue.js", "Svelte"],
            "correct_option": "B"
        }}
    ]

    IMPORTANT: The correct_option must be "A", "B", "C", or "D" where:
    - "A" corresponds to options[0]
    - "B" corresponds to options[1]
    - "C" corresponds to options[2]
    - "D" corresponds to options[3]

    Generate exactly 25 technical questions based on the resume content.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        # Remove any markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse the JSON response
        questions = json.loads(response_text)
        
        # Validate the structure
        if not isinstance(questions, list) or len(questions) != 25:
            raise ValueError("Response must be a list of exactly 25 questions")
        
        valid_options = {'A', 'B', 'C', 'D'}
        
        for i, q in enumerate(questions):
            if not all(key in q for key in ['question', 'options', 'correct_option']):
                raise ValueError(f"Question {i+1} is missing required fields")
            if len(q['options']) != 4:
                raise ValueError(f"Question {i+1} must have exactly 4 options")
            if q['correct_option'] not in valid_options:
                raise ValueError(f"Question {i+1} correct_option must be A, B, C, or D")
            
            # Additional validation: ensure the correct_option corresponds to a valid index
            option_index = ord(q['correct_option']) - ord('A')  # Convert A->0, B->1, C->2, D->3
            if option_index < 0 or option_index >= len(q['options']):
                raise ValueError(f"Question {i+1} correct_option index is out of range")
        
        return questions
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions with Gemini: {str(e)}")

@app.post("/extracttext", response_model=TextExtractionResponse)
async def extract_text(file: UploadFile = File(...)):
    """Extract text from PDF file."""
    
    # Validate file
    validate_file(file)
    
    # Extract text from PDF
    resume_text = await extract_text_from_pdf(file)
    
    return TextExtractionResponse(
        success=True,
        text=resume_text,
        character_count=len(resume_text),
        word_count=len(resume_text.split())
    )

@app.post("/generatequestions", response_model=QuestionGenerationResponse)
async def generate_questions(file: UploadFile = File(...)):
    """Extract text from PDF and generate technical multiple-choice questions."""
    
    # Validate file
    validate_file(file)
    
    # Extract text from PDF
    resume_text = await extract_text_from_pdf(file)
    
    # Generate questions using Gemini
    questions_data = generate_questions_with_gemini(resume_text)
    
    # Convert to Pydantic models
    questions = [Question(**q) for q in questions_data]
    
    return QuestionGenerationResponse(
        success=True,
        extracted_text=resume_text,
        questions=questions,
        total_questions=len(questions),
        text_stats={
            "character_count": len(resume_text),
            "word_count": len(resume_text.split())
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="PDF processing service is running"
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PDF Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/extracttext": "POST - Extract text from PDF",
            "/generatequestions": "POST - Extract text and generate technical questions",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

# Exception handlers
@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"error": "File too large. Maximum size is 16MB."}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check if Gemini API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable is not set!")
        print("Please set it before running the application:")
        print("export GEMINI_API_KEY='your_api_key_here'")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)