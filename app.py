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
import math

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
    difficulty: str  # Will be "easy", "medium", or "hard"

class QuestionGenerationResponse(BaseModel):
    success: bool
    extracted_text: str
    questions: List[Question]
    total_questions: int
    text_stats: Dict[str, int]
    difficulty_breakdown: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    message: str

class ErrorResponse(BaseModel):
    error: str

class DetailedQuestionRequest(BaseModel):
    years_of_experience: int
    area_of_interest: str

class QuestionAnswer(BaseModel):
    question: str
    userAnswer: str
    correctAnswer: str

class EvaluationRequest(BaseModel):
    experience: int
    interests: List[str]
    questionArray: List[QuestionAnswer]
    textExtracted: str

class EvaluationSummaryResponse(BaseModel):
    success: bool
    summary: str

def calculate_difficulty_distribution(total_questions: int) -> Dict[str, int]:
    """Calculate the distribution of questions by difficulty level."""
    easy_count = math.ceil(total_questions * 0.2)  # 20% easy
    hard_count = math.ceil(total_questions * 0.2)  # 20% hard
    medium_count = total_questions - easy_count - hard_count  # Remaining 60% medium
    
    return {
        "easy": easy_count,
        "medium": medium_count,
        "hard": hard_count
    }

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

def generate_questions_with_gemini(resume_text: str, num_questions: int = 20) -> List[Dict[str, Any]]:
    """Generate multiple-choice questions using Gemini 2.0 Flash."""
    
    difficulty_dist = calculate_difficulty_distribution(num_questions)
    
    prompt = f"""
    Based on the following resume, generate exactly {num_questions} multiple-choice questions focused on the technical skills, technologies, frameworks, tools, and programming concepts mentioned in the resume.

    DIFFICULTY DISTRIBUTION REQUIREMENTS:
    - {difficulty_dist['easy']} EASY questions (fundamental concepts, basic syntax, definitions)
    - {difficulty_dist['medium']} MEDIUM questions (practical application, best practices, common scenarios)
    - {difficulty_dist['hard']} HARD questions (advanced concepts, optimization, complex scenarios, edge cases)

    RESUME CONTENT:
    {resume_text}

    INSTRUCTIONS:
    - Focus ONLY on technical content from the resume (programming languages, frameworks, databases, tools, methodologies, etc.)
    - Create questions about specific technologies, their features, best practices, and technical concepts
    - Include questions about experience levels with mentioned technologies
    - Each question should have exactly 4 options
    - The correct_option should be specified as "A", "B", "C", or "D" (corresponding to the position in the options array)
    - Each question must include a "difficulty" field with value "easy", "medium", or "hard"
    - Make questions challenging but fair for someone with the experience level shown in the resume
    - Focus on practical technical knowledge and implementation details
    - If the resume mentions specific projects, create questions about the technical aspects of those projects

    DIFFICULTY GUIDELINES:
    - EASY: Basic definitions, syntax, fundamental concepts that any developer should know
    - MEDIUM: Practical application, common use cases, best practices, standard implementation
    - HARD: Advanced optimization, edge cases, complex scenarios, architectural decisions

    EXAMPLE QUESTIONS BY DIFFICULTY:
    
    EASY EXAMPLE:
    {{
        "question": "Which of the following is a Python web framework mentioned in the resume?",
        "options": ["Django", "Spring", "Laravel", "Ruby on Rails"],
        "correct_option": "A",
        "difficulty": "easy"
    }}

    MEDIUM EXAMPLE:
    {{
        "question": "What is the primary advantage of using Docker containers in the deployment process?",
        "options": ["Faster compilation", "Environment consistency", "Code minification", "Database optimization"],
        "correct_option": "B",
        "difficulty": "medium"
    }}

    HARD EXAMPLE:
    {{
        "question": "In a microservices architecture with high traffic, which caching strategy would be most effective for session management?",
        "options": ["Browser localStorage", "Database-level caching", "Distributed Redis cache", "File system cache"],
        "correct_option": "C",
        "difficulty": "hard"
    }}

    REQUIRED OUTPUT FORMAT (respond with ONLY valid JSON, no additional text):
    [
        {{
            "question": "Which programming language mentioned in the resume is object-oriented?",
            "options": ["Python", "HTML", "CSS", "SQL"],
            "correct_option": "A",
            "difficulty": "easy"
        }},
        {{
            "question": "What JavaScript framework is mentioned in the candidate's project experience?",
            "options": ["Angular", "React", "Vue.js", "Svelte"],
            "correct_option": "B",
            "difficulty": "medium"
        }}
    ]

    IMPORTANT: 
    - The correct_option must be "A", "B", "C", or "D" where A=options[0], B=options[1], C=options[2], D=options[3]
    - Each question must have a "difficulty" field with exactly one of: "easy", "medium", "hard"
    - Generate exactly {difficulty_dist['easy']} easy, {difficulty_dist['medium']} medium, and {difficulty_dist['hard']} hard questions
    - Total questions must be exactly {num_questions}

    Generate exactly {num_questions} technical questions based on the resume content with the specified difficulty distribution.
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
        if not isinstance(questions, list) or len(questions) != num_questions:
            raise ValueError(f"Response must be a list of exactly {num_questions} questions")
        
        valid_options = {'A', 'B', 'C', 'D'}
        valid_difficulties = {'easy', 'medium', 'hard'}
        
        for i, q in enumerate(questions):
            if not all(key in q for key in ['question', 'options', 'correct_option', 'difficulty']):
                raise ValueError(f"Question {i+1} is missing required fields")
            if len(q['options']) != 4:
                raise ValueError(f"Question {i+1} must have exactly 4 options")
            if q['correct_option'] not in valid_options:
                raise ValueError(f"Question {i+1} correct_option must be A, B, C, or D")
            if q['difficulty'] not in valid_difficulties:
                raise ValueError(f"Question {i+1} difficulty must be easy, medium, or hard")
            
            # Additional validation: ensure the correct_option corresponds to a valid index
            option_index = ord(q['correct_option']) - ord('A')  # Convert A->0, B->1, C->2, D->3
            if option_index < 0 or option_index >= len(q['options']):
                raise ValueError(f"Question {i+1} correct_option index is out of range")
        
        return questions
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions with Gemini: {str(e)}")

def generate_detailed_questions_with_gemini(resume_text: str, years_of_experience: int, area_of_interest: str, num_questions: int = 20) -> List[Dict[str, Any]]:
    """Generate detailed multiple-choice questions using Gemini 2.0 Flash based on experience and area of interest."""
    
    difficulty_dist = calculate_difficulty_distribution(num_questions)
    
    # Define experience level based on years
    if years_of_experience <= 2:
        experience_level = "Junior"
        complexity = "fundamental concepts and basic implementation"
    elif years_of_experience <= 5:
        experience_level = "Mid-level"
        complexity = "intermediate concepts, best practices, and problem-solving"
    elif years_of_experience <= 10:
        experience_level = "Senior"
        complexity = "advanced concepts, architecture decisions, and optimization"
    else:
        experience_level = "Expert/Lead"
        complexity = "expert-level concepts, system design, and leadership aspects"
    
    prompt = f"""
    Based on the following resume and the specified professional profile, generate exactly {num_questions} multiple-choice questions tailored for a {experience_level} professional with {years_of_experience} years of experience in {area_of_interest}.

    DIFFICULTY DISTRIBUTION REQUIREMENTS:
    - {difficulty_dist['easy']} EASY questions (fundamental concepts, basic definitions)
    - {difficulty_dist['medium']} MEDIUM questions (practical application, best practices)
    - {difficulty_dist['hard']} HARD questions (advanced concepts, optimization, complex scenarios)

    PROFESSIONAL PROFILE:
    - Years of Experience: {years_of_experience}
    - Experience Level: {experience_level}
    - Area of Interest: {area_of_interest}

    RESUME CONTENT:
    {resume_text}

    INSTRUCTIONS:
    - Focus on {area_of_interest}-related technologies, concepts, and practices mentioned in the resume
    - Tailor question difficulty to {experience_level} level focusing on {complexity}
    - Include industry-specific scenarios and challenges relevant to {area_of_interest}
    - Consider the candidate's {years_of_experience} years of experience when crafting questions
    - Each question should have exactly 4 options
    - The correct_option should be specified as "A", "B", "C", or "D"
    - Each question must include a "difficulty" field with value "easy", "medium", or "hard"
    - Make questions practical and relevant to real-world {area_of_interest} scenarios
    - Include questions about scalability, performance, security, and best practices relevant to {area_of_interest}

    DIFFICULTY GUIDELINES FOR {experience_level} LEVEL:
    - EASY: Fundamental {area_of_interest} concepts, basic tools, standard definitions
    - MEDIUM: Common {area_of_interest} practices, implementation scenarios, troubleshooting
    - HARD: Advanced {area_of_interest} optimization, complex architecture, edge cases

    QUESTION FOCUS AREAS for {area_of_interest}:
    - Technical skills and frameworks mentioned in the resume relevant to {area_of_interest}
    - Industry best practices and methodologies
    - Problem-solving scenarios typical for {experience_level} professionals
    - Architecture and design patterns appropriate for the experience level
    - Performance optimization and scalability considerations
    - Security practices and compliance relevant to {area_of_interest}
    - Team collaboration and project management (for senior+ levels)

    REQUIRED OUTPUT FORMAT (respond with ONLY valid JSON, no additional text):
    [
        {{
            "question": "In a {area_of_interest} project with {years_of_experience} years of experience, which approach would be most suitable for...",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_option": "A",
            "difficulty": "medium"
        }}
    ]

    IMPORTANT: 
    - The correct_option must be "A", "B", "C", or "D" where A=options[0], B=options[1], C=options[2], D=options[3]
    - Please donot annotate the options with any additional text or explanations[no 'A) Option A' format]
    - Each question must have a "difficulty" field with exactly one of: "easy", "medium", "hard"
    - Generate exactly {difficulty_dist['easy']} easy, {difficulty_dist['medium']} medium, and {difficulty_dist['hard']} hard questions
    - Total questions must be exactly {num_questions}

    Generate exactly {num_questions} technical questions based on the resume content, experience level, and area of interest with the specified difficulty distribution.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse the JSON response
        questions = json.loads(response_text)
        
        # Validate the structure
        if not isinstance(questions, list) or len(questions) != num_questions:
            raise ValueError(f"Response must be a list of exactly {num_questions} questions")
        
        valid_options = {'A', 'B', 'C', 'D'}
        valid_difficulties = {'easy', 'medium', 'hard'}
        
        for i, q in enumerate(questions):
            if not all(key in q for key in ['question', 'options', 'correct_option', 'difficulty']):
                raise ValueError(f"Question {i+1} is missing required fields")
            if len(q['options']) != 4:
                raise ValueError(f"Question {i+1} must have exactly 4 options")
            if q['correct_option'] not in valid_options:
                raise ValueError(f"Question {i+1} correct_option must be A, B, C, or D")
            if q['difficulty'] not in valid_difficulties:
                raise ValueError(f"Question {i+1} difficulty must be easy, medium, or hard")
            
            option_index = ord(q['correct_option']) - ord('A')
            if option_index < 0 or option_index >= len(q['options']):
                raise ValueError(f"Question {i+1} correct_option index is out of range")
        
        return questions
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating detailed questions with Gemini: {str(e)}")

def generate_evaluation_summary(
    experience: int, 
    interests: List[str], 
    question_answers: List[Dict[str, str]], 
    resume_text: str
) -> str:
    """Generate a simple text summary of candidate evaluation and job recommendations."""
    
    # Calculate basic metrics
    total_questions = len(question_answers)
    correct_answers = sum(1 for qa in question_answers if qa['userAnswer'].lower().strip() == qa['correctAnswer'].lower().strip())
    score_percentage = int((correct_answers / total_questions) * 100) if total_questions > 0 else 0
    
    interests_str = ", ".join(interests)
    
    prompt = f"""
    Based on this candidate's performance, write a comprehensive evaluation summary as plain text.

    CANDIDATE INFO:
    - Current Experience: {experience} years
    - Areas of Interest: {interests_str}
    - Assessment Score: {score_percentage}% ({correct_answers}/{total_questions} correct)
    - Resume Summary: {resume_text[:500]}...

    Write a detailed evaluation summary that includes:
    1. Performance assessment based on the {score_percentage}% score
    2. Recommended job titles they should apply for
    3. Suggested experience level for job applications (Junior/Mid-level/Senior)
    4. Specific advice on what types of roles to target
    5. Any areas they should focus on improving

    Write this as a flowing text summary, not bullet points or structured format. Make it personalized and actionable.

    Keep the tone professional but encouraging. The summary should be 3-4 paragraphs long.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating evaluation summary: {str(e)}")

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
async def generate_questions(
    file: UploadFile = File(...),
    noOfQuestions: str = Form("20")
):
    """Extract text from PDF and generate technical multiple-choice questions."""
    
    # Validate file
    validate_file(file)
    
    # Validate and convert noOfQuestions
    try:
        num_questions = int(noOfQuestions)
        if num_questions < 5 or num_questions > 100:
            raise ValueError("Number of questions must be between 5 and 100")
    except ValueError:
        raise HTTPException(status_code=400, detail="noOfQuestions must be a valid number between 5 and 100")
    
    # Extract text from PDF
    resume_text = await extract_text_from_pdf(file)
    
    # Generate questions using Gemini
    questions_data = generate_questions_with_gemini(resume_text, num_questions)
    
    # Convert to Pydantic models
    questions = [Question(**q) for q in questions_data]
    
    # Calculate actual difficulty breakdown
    difficulty_breakdown = {
        "easy": sum(1 for q in questions if q.difficulty == "easy"),
        "medium": sum(1 for q in questions if q.difficulty == "medium"),
        "hard": sum(1 for q in questions if q.difficulty == "hard")
    }
    
    return QuestionGenerationResponse(
        success=True,
        extracted_text=resume_text,
        questions=questions,
        total_questions=len(questions),
        text_stats={
            "character_count": len(resume_text),
            "word_count": len(resume_text.split())
        },
        difficulty_breakdown=difficulty_breakdown
    )

@app.post("/detailedQuestionGeneration", response_model=QuestionGenerationResponse)
async def detailed_question_generation(
    file: UploadFile = File(...),
    years_of_experience: str = Form(...),
    area_of_interest: str = Form(...),
    noOfQuestions: str = Form("20")
):
    """Extract text from PDF and generate detailed technical multiple-choice questions based on experience and area of interest."""
    
    # Validate file
    validate_file(file)
    
    # Convert and validate years_of_experience
    try:
        years_exp = int(years_of_experience)
        if years_exp < 0:
            raise ValueError("Years of experience must be positive")
    except ValueError:
        raise HTTPException(status_code=400, detail="Years of experience must be a valid positive number")
    
    # Validate area_of_interest
    if not area_of_interest.strip():
        raise HTTPException(status_code=400, detail="Area of interest cannot be empty")
    
    # Validate and convert noOfQuestions
    try:
        num_questions = int(noOfQuestions)
        if num_questions < 5 or num_questions > 100:
            raise ValueError("Number of questions must be between 5 and 100")
    except ValueError:
        raise HTTPException(status_code=400, detail="noOfQuestions must be a valid number between 5 and 100")
    
    # Extract text from PDF
    resume_text = await extract_text_from_pdf(file)
    
    # Generate detailed questions using Gemini
    questions_data = generate_detailed_questions_with_gemini(resume_text, years_exp, area_of_interest.strip(), num_questions)
    
    # Convert to Pydantic models
    questions = [Question(**q) for q in questions_data]
    
    # Calculate actual difficulty breakdown
    difficulty_breakdown = {
        "easy": sum(1 for q in questions if q.difficulty == "easy"),
        "medium": sum(1 for q in questions if q.difficulty == "medium"),
        "hard": sum(1 for q in questions if q.difficulty == "hard")
    }
    
    return QuestionGenerationResponse(
        success=True,
        extracted_text=resume_text,
        questions=questions,
        total_questions=len(questions),
        text_stats={
            "character_count": len(resume_text),
            "word_count": len(resume_text.split())
        },
        difficulty_breakdown=difficulty_breakdown
    )

@app.post("/evaluateCandidate", response_model=EvaluationSummaryResponse)
async def evaluate_candidate(request: EvaluationRequest):
    """Evaluate candidate and provide a text summary with job recommendations."""
    
    # Basic validation
    if request.experience < 0:
        raise HTTPException(status_code=400, detail="Experience must be a positive number")
    
    if not request.interests:
        raise HTTPException(status_code=400, detail="At least one area of interest is required")
    
    if not request.questionArray:
        raise HTTPException(status_code=400, detail="Question array cannot be empty")
    
    # Convert to simple format
    question_answers = [
        {
            "question": qa.question,
            "userAnswer": qa.userAnswer,
            "correctAnswer": qa.correctAnswer
        }
        for qa in request.questionArray
    ]
    
    # Generate evaluation summary
    summary = generate_evaluation_summary(
        experience=request.experience,
        interests=request.interests,
        question_answers=question_answers,
        resume_text=request.textExtracted
    )
    
    return EvaluationSummaryResponse(
        success=True,
        summary=summary
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
            "/generatequestions": "POST - Extract text and generate technical questions (accepts noOfQuestions parameter, default: 20)",
            "/detailedQuestionGeneration": "POST - Extract text and generate detailed questions based on experience and area of interest (accepts noOfQuestions parameter, default: 20)",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
            "/evaluateCandidate": "POST - Evaluate candidate and provide job recommendations summary",
        },
        "features": {
            "dynamic_question_count": "Specify number of questions (5-100) using noOfQuestions parameter",
            "difficulty_distribution": "20% easy, 60% medium, 20% hard questions"
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