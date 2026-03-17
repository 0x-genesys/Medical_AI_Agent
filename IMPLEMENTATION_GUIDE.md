# Capstone Projects - Complete Implementation Guide

## Overview

This guide provides detailed implementation strategies for all 5 capstone projects, including architecture, tech stack, code examples, and best practices.

---

## Project 1: CV Creation using LLMs

### **Objective**
Automate CV creation from user information or documents using open-source LLMs, with ATS optimization.

### **Tech Stack**
- **LLM**: Gemma 3 1B (via Ollama) or Llama 3
- **Frameworks**: LangChain, LlamaIndex
- **Document Processing**: pdfplumber, python-docx, pypandoc
- **Backend**: FastAPI or Flask
- **Frontend**: Streamlit or React
- **Database**: SQLite or PostgreSQL

### **Architecture**

```
┌─────────────┐
│   User      │
│   Input     │ (PDF/Text/Form)
└──────┬──────┘
       │
       v
┌─────────────────────────┐
│  Document Parser        │
│  (PDF/DOCX → JSON)      │
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│  LLM Extraction         │
│  (Gemma/Llama)          │
│  - Personal Info        │
│  - Experience           │
│  - Skills               │
│  - Education            │
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│  Job Description        │
│  Analyzer               │
│  (Extract Keywords)     │
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│  CV Tailoring Engine    │
│  (LLM Rewrite)          │
│  - Keyword Optimization │
│  - ATS Friendly         │
└──────┬──────────────────┘
       │
       v
┌─────────────────────────┐
│  Template Engine        │
│  (DOCX/PDF Generation)  │
└──────┬──────────────────┘
       │
       v
┌─────────────┐
│  Final CV   │
└─────────────┘
```

### **Implementation Steps**

#### **Step 1: Project Structure**

```bash
cv-creator/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── models.py               # Data models
│   ├── parser.py               # Document parsing
│   ├── llm_service.py          # LLM interactions
│   ├── cv_generator.py         # CV generation
│   └── templates/              # CV templates
│       ├── modern.docx
│       └── classic.docx
├── tests/
│   ├── test_parser.py
│   └── test_llm_service.py
├── requirements.txt
├── .env
└── README.md
```

#### **Step 2: Document Parser**

```python
# parser.py
import pdfplumber
from docx import Document
import json

class CVParser:
    def parse_pdf(self, file_path):
        """Extract text from PDF CV"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    def parse_docx(self, file_path):
        """Extract text from DOCX CV"""
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    def extract_structured_data(self, text, llm_service):
        """Use LLM to extract structured data"""
        prompt = f"""
        Extract the following information from this CV and return as JSON:
        - name
        - email
        - phone
        - education (list of degrees)
        - experience (list of jobs with title, company, dates, description)
        - skills (list)
        - achievements (list)
        
        CV Text:
        {text}
        
        Return only valid JSON.
        """
        
        response = llm_service.generate(prompt)
        return json.loads(response)
```

#### **Step 3: LLM Service with Ollama**

```python
# llm_service.py
import ollama
import json

class LLMService:
    def __init__(self, model="gemma2:2b"):
        self.model = model
        
    def generate(self, prompt, temperature=0.1):
        """Generate response from LLM"""
        response = ollama.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': temperature,
                'num_predict': 2000
            }
        )
        return response['message']['content']
    
    def extract_cv_info(self, cv_text):
        """Extract structured info from CV"""
        prompt = f"""
        Extract structured information from this CV.
        Return as JSON with keys: name, email, phone, summary, education, 
        experience, skills, certifications, achievements.
        
        CV:
        {cv_text}
        """
        return self.generate(prompt)
    
    def analyze_job_description(self, jd_text):
        """Extract keywords and requirements from JD"""
        prompt = f"""
        Analyze this job description and extract:
        1. Required skills (list)
        2. Preferred skills (list)
        3. Responsibilities (list)
        4. Keywords for ATS (list)
        
        Return as JSON.
        
        Job Description:
        {jd_text}
        """
        return self.generate(prompt)
    
    def tailor_experience(self, experience, job_requirements):
        """Rewrite experience to match job requirements"""
        prompt = f"""
        Rewrite this work experience to better match the job requirements.
        Focus on relevant achievements and use keywords from requirements.
        Keep it truthful and ATS-friendly.
        
        Original Experience:
        {json.dumps(experience, indent=2)}
        
        Job Requirements:
        {json.dumps(job_requirements, indent=2)}
        
        Return rewritten experience as JSON with same structure.
        """
        return self.generate(prompt)
```

#### **Step 4: CV Generator**

```python
# cv_generator.py
from docx import Document
from docx.shared import Pt, RGBColor
import json

class CVGenerator:
    def __init__(self, template_path="templates/modern.docx"):
        self.template_path = template_path
    
    def generate_cv(self, cv_data, output_path):
        """Generate CV from structured data"""
        doc = Document()
        
        # Header - Name and Contact
        name = doc.add_heading(cv_data['name'], 0)
        name.alignment = 1  # Center
        
        contact = doc.add_paragraph()
        contact.add_run(f"{cv_data['email']} | {cv_data['phone']}")
        contact.alignment = 1
        
        # Summary
        if 'summary' in cv_data:
            doc.add_heading('Professional Summary', 1)
            doc.add_paragraph(cv_data['summary'])
        
        # Experience
        doc.add_heading('Experience', 1)
        for exp in cv_data.get('experience', []):
            title_para = doc.add_paragraph()
            title_para.add_run(exp['title']).bold = True
            title_para.add_run(f" | {exp['company']}")
            
            date_para = doc.add_paragraph()
            date_para.add_run(f"{exp['start_date']} - {exp['end_date']}").italic = True
            
            for bullet in exp.get('bullets', []):
                doc.add_paragraph(bullet, style='List Bullet')
        
        # Education
        doc.add_heading('Education', 1)
        for edu in cv_data.get('education', []):
            edu_para = doc.add_paragraph()
            edu_para.add_run(edu['degree']).bold = True
            edu_para.add_run(f" | {edu['institution']}")
            if 'year' in edu:
                edu_para.add_run(f" | {edu['year']}")
        
        # Skills
        doc.add_heading('Skills', 1)
        skills_text = ", ".join(cv_data.get('skills', []))
        doc.add_paragraph(skills_text)
        
        # Save
        doc.save(output_path)
        return output_path
```

#### **Step 5: FastAPI Application**

```python
# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import os
from parser import CVParser
from llm_service import LLMService
from cv_generator import CVGenerator

app = FastAPI(title="CV Creator API")

parser = CVParser()
llm_service = LLMService()
cv_generator = CVGenerator()

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    """Upload existing CV for parsing"""
    # Save uploaded file
    file_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Parse CV
    if file.filename.endswith('.pdf'):
        text = parser.parse_pdf(file_path)
    else:
        text = parser.parse_docx(file_path)
    
    # Extract structured data
    cv_data = parser.extract_structured_data(text, llm_service)
    
    return {"cv_data": cv_data}

@app.post("/generate-cv")
async def generate_cv(
    cv_data: dict,
    job_description: str = Form(None)
):
    """Generate tailored CV"""
    
    # If JD provided, tailor the CV
    if job_description:
        jd_analysis = llm_service.analyze_job_description(job_description)
        
        # Tailor experience
        tailored_experience = []
        for exp in cv_data.get('experience', []):
            tailored = llm_service.tailor_experience(exp, jd_analysis)
            tailored_experience.append(tailored)
        
        cv_data['experience'] = tailored_experience
    
    # Generate CV
    output_path = f"output/{cv_data['name']}_CV.docx"
    os.makedirs("output", exist_ok=True)
    
    cv_generator.generate_cv(cv_data, output_path)
    
    return FileResponse(
        output_path,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        filename=f"{cv_data['name']}_CV.docx"
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

#### **Step 6: Requirements**

```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pdfplumber==0.10.3
python-docx==1.1.0
ollama==0.1.6
python-dotenv==1.0.0
```

### **Testing Strategy**

```python
# tests/test_cv_generator.py
import pytest
from cv_generator import CVGenerator

def test_cv_generation():
    cv_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "+1234567890",
        "experience": [{
            "title": "Software Engineer",
            "company": "Tech Corp",
            "start_date": "Jan 2020",
            "end_date": "Present",
            "bullets": [
                "Developed scalable APIs",
                "Improved performance by 40%"
            ]
        }],
        "skills": ["Python", "FastAPI", "Docker"]
    }
    
    generator = CVGenerator()
    output = generator.generate_cv(cv_data, "test_output.docx")
    
    assert os.path.exists(output)
```

---

## Project 2: CV Sorting using LLMs

### **Objective**
Automated ranking of candidate resumes for specific job requirements using LLMs.

### **Tech Stack**
- **LLM**: Llama 3 8B via Ollama, Gemma
- **Frameworks**: LangChain, Resume Matcher
- **Parsers**: pyresparser, open-resume
- **Backend**: FastAPI
- **Database**: PostgreSQL with pgvector for embeddings
- **Frontend**: React + TailwindCSS

### **Architecture**

```
┌──────────────────┐
│  Bulk Upload     │
│  (CVs + JD)      │
└────────┬─────────┘
         │
         v
┌──────────────────────────┐
│  Resume Parser           │
│  (pyresparser)           │
│  → Structured JSON       │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Job Description Parser  │
│  (LLM Extraction)        │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Matching Engine         │
│  ┌────────────────────┐  │
│  │ Skill Match        │  │
│  │ Experience Match   │  │
│  │ Education Match    │  │
│  │ LLM Semantic Match │  │
│  └────────────────────┘  │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Scoring & Ranking       │
│  (Weighted Aggregation)  │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Ranked Candidate List   │
│  + Explanation           │
└──────────────────────────┘
```

### **Implementation**

```python
# resume_matcher.py
import ollama
import json
from typing import List, Dict

class ResumeMatching Engine:
    def __init__(self, model="llama3"):
        self.model = model
    
    def parse_resume(self, resume_text: str) -> Dict:
        """Extract structured data from resume"""
        prompt = f"""
        Extract from this resume and return as JSON:
        {{
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "experience_years": 0,
            "education": [{{"degree": "", "institution": "", "year": ""}}],
            "experience": [{{
                "title": "",
                "company": "",
                "duration_months": 0,
                "achievements": []
            }}]
        }}
        
        Resume:
        {resume_text}
        """
        
        response = ollama.chat(model=self.model, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return json.loads(response['message']['content'])
    
    def parse_job_description(self, jd_text: str) -> Dict:
        """Extract requirements from job description"""
        prompt = f"""
        Analyze this job description and extract:
        {{
            "required_skills": [],
            "preferred_skills": [],
            "min_experience_years": 0,
            "education_requirements": [],
            "responsibilities": [],
            "keywords": []
        }}
        
        Job Description:
        {jd_text}
        """
        
        response = ollama.chat(model=self.model, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return json.loads(response['message']['content'])
    
    def calculate_skill_match(self, candidate_skills: List[str], 
                             required_skills: List[str],
                             preferred_skills: List[str]) -> float:
        """Calculate skill match score"""
        candidate_set = set([s.lower() for s in candidate_skills])
        required_set = set([s.lower() for s in required_skills])
        preferred_set = set([s.lower() for s in preferred_skills])
        
        # Required skills match (weighted 70%)
        required_match = len(candidate_set & required_set) / len(required_set) if required_set else 0
        
        # Preferred skills match (weighted 30%)
        preferred_match = len(candidate_set & preferred_set) / len(preferred_set) if preferred_set else 0
        
        return (required_match * 0.7) + (preferred_match * 0.3)
    
    def calculate_experience_match(self, candidate_years: int, 
                                   required_years: int) -> float:
        """Calculate experience match score"""
        if candidate_years >= required_years:
            return 1.0
        else:
            return candidate_years / required_years if required_years > 0 else 0
    
    def semantic_match(self, candidate_data: Dict, jd_data: Dict) -> Dict:
        """Use LLM for semantic matching"""
        prompt = f"""
        Compare this candidate with job requirements and provide:
        1. Overall fit score (0-100)
        2. Strengths (list)
        3. Gaps (list)
        4. Recommendation (hire/interview/reject)
        5. Explanation (2-3 sentences)
        
        Return as JSON.
        
        Candidate:
        {json.dumps(candidate_data, indent=2)}
        
        Job Requirements:
        {json.dumps(jd_data, indent=2)}
        """
        
        response = ollama.chat(model=self.model, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return json.loads(response['message']['content'])
    
    def rank_candidates(self, candidates: List[Dict], jd_data: Dict) -> List[Dict]:
        """Rank all candidates"""
        results = []
        
        for candidate in candidates:
            # Calculate component scores
            skill_score = self.calculate_skill_match(
                candidate.get('skills', []),
                jd_data.get('required_skills', []),
                jd_data.get('preferred_skills', [])
            )
            
            experience_score = self.calculate_experience_match(
                candidate.get('experience_years', 0),
                jd_data.get('min_experience_years', 0)
            )
            
            # Semantic analysis
            semantic_result = self.semantic_match(candidate, jd_data)
            
            # Aggregate score (weighted)
            final_score = (
                skill_score * 0.4 +
                experience_score * 0.3 +
                (semantic_result['fit_score'] / 100) * 0.3
            )
            
            results.append({
                'candidate': candidate,
                'scores': {
                    'skill_match': skill_score,
                    'experience_match': experience_score,
                    'semantic_fit': semantic_result['fit_score'] / 100,
                    'overall': final_score
                },
                'analysis': semantic_result,
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by overall score
        results.sort(key=lambda x: x['scores']['overall'], reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        return results
```

```python
# api.py
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import PyPDF2
from resume_matcher import ResumeMatcher

app = FastAPI()
matcher = ResumeMatcher()

@app.post("/rank-candidates")
async def rank_candidates(
    resumes: List[UploadFile] = File(...),
    job_description: str = Form(...)
):
    """Rank multiple candidates against job description"""
    
    # Parse JD
    jd_data = matcher.parse_job_description(job_description)
    
    # Parse all resumes
    candidates = []
    for resume_file in resumes:
        # Extract text
        pdf_reader = PyPDF2.PdfReader(resume_file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Parse resume
        candidate_data = matcher.parse_resume(text)
        candidate_data['filename'] = resume_file.filename
        candidates.append(candidate_data)
    
    # Rank candidates
    results = matcher.rank_candidates(candidates, jd_data)
    
    return {
        "job_requirements": jd_data,
        "total_candidates": len(candidates),
        "ranked_candidates": results
    }
```

---

## Project 3: Advertisement Creation using Image Generation

### **Objective**
Automate creation of compelling ad visuals using AI image generation models.

### **Tech Stack**
- **Image Generation**: Stable Diffusion XL, FLUX.1
- **Backend**: FastAPI
- **Image Processing**: PIL, OpenCV
- **Frontend**: Streamlit
- **Storage**: AWS S3 or local filesystem

### **Architecture**

```
┌──────────────────┐
│  User Input      │
│  - Product Info  │
│  - Style/Theme   │
│  - Target Audience│
└────────┬─────────┘
         │
         v
┌──────────────────────────┐
│  Prompt Engineering      │
│  (Template + LLM)        │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Image Generation        │
│  (Stable Diffusion/FLUX) │
│  - Multiple candidates   │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Post-Processing         │
│  - Upscaling            │
│  - Background removal    │
│  - Text overlay          │
│  - Logo placement        │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Export & Optimize       │
│  (Multiple formats)      │
└──────────────────────────┘
```

### **Implementation**

```python
# ad_generator.py
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

class AdGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)
        
        # Optimize
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
    
    def generate_prompt(self, product_info: dict) -> str:
        """Generate optimized prompt for ad"""
        style_prompts = {
            "modern": "minimalist, clean, professional, high-end",
            "vibrant": "colorful, energetic, dynamic, bold",
            "elegant": "sophisticated, luxurious, premium, refined",
            "casual": "friendly, approachable, warm, inviting"
        }
        
        style = style_prompts.get(product_info.get('style', 'modern'), style_prompts['modern'])
        
        prompt = f"""
        Professional product advertisement photography, {product_info['product_name']},
        {product_info.get('description', '')},
        {style}, studio lighting, commercial photography,
        8k uhd, high quality, detailed, professional composition
        """
        
        negative_prompt = """
        low quality, blurry, amateur, watermark, text, signature,
        distorted, ugly, bad anatomy, worst quality
        """
        
        return prompt.strip(), negative_prompt.strip()
    
    def generate_image(self, prompt: str, negative_prompt: str,
                      width: int = 1024, height: int = 1024,
                      num_images: int = 4) -> List[Image.Image]:
        """Generate ad images"""
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=30,
            guidance_scale=7.5,
            num_images_per_prompt=num_images
        ).images
        
        return images
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple background removal using GrabCut
        mask = np.zeros(img_array.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        rect = (10, 10, img_array.shape[1]-10, img_array.shape[0]-10)
        cv2.grabCut(img_array, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = img_array * mask2[:, :, np.newaxis]
        
        # Convert back to PIL
        result_image = Image.fromarray(result)
        return result_image
    
    def add_text_overlay(self, image: Image.Image, text: str,
                         position: str = "bottom", 
                         font_size: int = 60) -> Image.Image:
        """Add text overlay to image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        img_width, img_height = image.size
        
        if position == "bottom":
            x = (img_width - text_width) // 2
            y = img_height - text_height - 50
        elif position == "top":
            x = (img_width - text_width) // 2
            y = 50
        else:  # center
            x = (img_width - text_width) // 2
            y = (img_height - text_height) // 2
        
        # Add shadow
        shadow_offset = 3
        draw.text((x + shadow_offset, y + shadow_offset), text, 
                 font=font, fill=(0, 0, 0, 180))
        
        # Add main text
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        return img_copy
    
    def create_ad_variants(self, product_info: dict) -> List[dict]:
        """Generate multiple ad variants"""
        prompt, negative_prompt = self.generate_prompt(product_info)
        
        # Generate base images
        images = self.generate_image(prompt, negative_prompt, 
                                     num_images=product_info.get('num_variants', 4))
        
        variants = []
        for i, img in enumerate(images):
            # Post-process
            if product_info.get('remove_bg', False):
                img = self.remove_background(img)
            
            if product_info.get('add_text', False):
                img = self.add_text_overlay(
                    img, 
                    product_info.get('tagline', ''),
                    position=product_info.get('text_position', 'bottom')
                )
            
            variants.append({
                'id': i,
                'image': img,
                'prompt_used': prompt
            })
        
        return variants
```

```python
# streamlit_app.py
import streamlit as st
from ad_generator import AdGenerator
import io

st.set_page_config(page_title="AI Ad Generator", layout="wide")

st.title("🎨 AI Advertisement Generator")
st.write("Create professional ad visuals using AI")

# Initialize generator
@st.cache_resource
def load_generator():
    return AdGenerator()

generator = load_generator()

# Input form
with st.sidebar:
    st.header("Ad Configuration")
    
    product_name = st.text_input("Product Name", "Premium Headphones")
    description = st.text_area("Product Description", 
                               "Wireless noise-cancelling headphones with premium sound quality")
    
    style = st.selectbox("Style", 
                        ["modern", "vibrant", "elegant", "casual"])
    
    tagline = st.text_input("Tagline (optional)", "Experience Pure Sound")
    
    num_variants = st.slider("Number of variants", 1, 4, 4)
    
    add_text = st.checkbox("Add text overlay", value=True)
    text_position = st.selectbox("Text position", ["bottom", "top", "center"])
    
    remove_bg = st.checkbox("Remove background", value=False)
    
    generate_btn = st.button("Generate Ads", type="primary")

# Main area
if generate_btn:
    with st.spinner("Generating your ads..."):
        product_info = {
            'product_name': product_name,
            'description': description,
            'style': style,
            'tagline': tagline,
            'num_variants': num_variants,
            'add_text': add_text,
            'text_position': text_position,
            'remove_bg': remove_bg
        }
        
        variants = generator.create_ad_variants(product_info)
        
        st.success(f"Generated {len(variants)} ad variants!")
        
        # Display in grid
        cols = st.columns(2)
        for i, variant in enumerate(variants):
            with cols[i % 2]:
                st.image(variant['image'], caption=f"Variant {i+1}", 
                        use_column_width=True)
                
                # Download button
                buf = io.BytesIO()
                variant['image'].save(buf, format='PNG')
                st.download_button(
                    label=f"Download Variant {i+1}",
                    data=buf.getvalue(),
                    file_name=f"{product_name}_ad_{i+1}.png",
                    mime="image/png"
                )
```

---

## Project 4: Multimodal Medical Assistant

### **Objective**
AI assistant for processing multimodal medical data (text, images, reports) to support clinical decision-making with both CLI and Web UI interfaces.

### **Tech Stack**
- **Text LLM**: Llama 3 via Ollama
- **Vision Model**: BiomedCLIP (MedCLIP)
- **Framework**: LangChain for orchestration
- **Vector Store**: FAISS for medical knowledge RAG
- **Image Processing**: OpenCV, pydicom, Pillow
- **Web UI**: Gradio with enhanced parsing and display
- **Session Management**: Cross-modal context preservation

### **Architecture**

```
┌─────────────────────────────────────────┐
│         main.py (Entry Point)           │
│  - Automated setup (venv, deps, Ollama) │
│  - User prompt: UI or CLI?              │
└──────────┬──────────────────────────────┘
           │
           ├────────────────┬─────────────────┐
           v                v                 v
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Web UI   │     │   CLI    │     │  Tests   │
    │ (Gradio) │     │ Interface│     │  (pytest)│
    └────┬─────┘     └────┬─────┘     └──────────┘
         │                │
         └────────────────┘
                │
                v
    ┌──────────────────────────────┐
    │ MedicalAssistantOrchestrator │
    │  (cli_main.py)               │
    │  - Shared session manager    │
    │  - Flow routing              │
    └──────────┬───────────────────┘
               │
    ┌──────────┼──────────┬──────────────┐
    v          v          v              v
┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
│  Text   │ │ Image  │ │Multimodal│ │ Session  │
│Processor│ │Processor│ │  Fusion  │ │ Manager  │
│(BioBERT)│ │(MedCLIP)│ │(LangChain)│ │          │
└────┬────┘ └────┬───┘ └────┬─────┘ └──────────┘
     │           │           │
     └───────────┴───────────┘
                 │
                 v
    ┌──────────────────────────┐
    │  Clinical Analysis       │
    │  - Report parsing        │
    │  - Image interpretation  │
    │  - Integrated assessment │
    │  - RAG with medical KB   │
    └──────────────────────────┘
```

### **New Flow Architecture**

#### **Dual Interface Design**

The application now supports two distinct user interfaces:

1. **Web UI Flow** (`ui-dashboard/medical_assistant_ui.py`)
   - Beautiful Gradio interface
   - File preview on upload
   - Structured output parsing with fallback
   - Color-coded, readable displays
   - Interactive components

2. **CLI Flow** (`cli_main.py`)
   - Terminal-based interface
   - Same underlying orchestrator
   - Session continuity
   - Professional formatted output

#### **Entry Point: main.py**

```bash
# Simplified usage
python main.py              # Prompts: UI or CLI?
python main.py --ui         # Launch Web UI directly
python main.py --cli        # Launch CLI directly
python main.py --test       # Run test suite
python main.py --setup      # Setup only
```

#### **Automated Setup Features**

`main.py` provides **zero-configuration deployment**:

✅ **Environment Setup**
- Detects Python version (requires 3.9+)
- Creates virtual environment automatically
- Installs all dependencies with progress tracking
- Cross-platform support (Windows/macOS/Linux)

✅ **Ollama Management**
- Detects Ollama installation
- Provides platform-specific install instructions
- Auto-starts Ollama service if not running
- Downloads llama3 model automatically

✅ **Error Handling**
- Colored terminal output (success/warning/error)
- Clear error messages with resolution steps
- Graceful failures with helpful guidance
- No manual configuration needed

### **UI Enhancements**

#### **File Preview Feature**
- Shows first 2000 characters of uploaded files
- Appears immediately on upload (before analysis)
- Helps users verify correct file selection
- Available in Report Analysis and Multimodal tabs

#### **Enhanced Output Parsing**
All flows now display structured, color-coded output:

**Report Analysis:**
- Clinical Summary
- Chief Complaints (with explicit dark text)
- Symptoms (color-coded badges)
- Medications (readable formatting)
- Laboratory Findings (proper text colors)

**Image Analysis:**
- Observations (fixed white-text issue)
- Potential Findings (black text on white background)
- Confidence Score (visual progress bar)

**Multimodal Fusion:**
- Clinical Summary
- Integrated Assessment
- Differential Diagnosis
- Recommended Workup
- Confidence Level (color-coded badge: High/Medium/Low)

**Raw LLM Output:**
- ✅ **No truncation** - full output always shown
- Character count indicator
- Scrollable container for long outputs
- Collapsible <details> element

#### **Error Handling with Fallback**
```python
try:
    # Parse structured output
    parse_medical_output(result, flow_type)
except Exception as e:
    # Show error warning
    # Always fallback to raw output
    display_raw_response(result['raw_response'])
```

### **Implementation**

```python
# medical_assistant.py
import ollama
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import pydicom
import numpy as np
from typing import Dict, List, Tuple

class MedicalAssistant:
    def __init__(self):
        # Text model
        self.text_model_name = "llama3"
        
        # Vision model (BiomedCLIP)
        self.vision_model = self.load_vision_model()
        
    def load_vision_model(self):
        """Load BiomedCLIP or similar medical vision model"""
        # Placeholder - would load actual BiomedCLIP
        # from biomedclip import BiomedCLIP
        # return BiomedCLIP.load_pretrained()
        pass
    
    def process_dicom(self, dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """Process DICOM medical image"""
        ds = pydicom.dcmread(dicom_path)
        
        # Extract image
        image_array = ds.pixel_array
        
        # Normalize
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        image_array = (image_array * 255).astype(np.uint8)
        
        # Extract metadata
        metadata = {
            'patient_id': ds.get('PatientID', 'Unknown'),
            'study_date': ds.get('StudyDate', 'Unknown'),
            'modality': ds.get('Modality', 'Unknown'),
            'body_part': ds.get('BodyPartExamined', 'Unknown'),
            'institution': ds.get('InstitutionName', 'Unknown')
        }
        
        return image_array, metadata
    
    def analyze_medical_image(self, image_path: str, 
                              clinical_context: str = "") -> Dict:
        """Analyze medical image with clinical context"""
        # Load and process image
        if image_path.endswith('.dcm'):
            image_array, metadata = self.process_dicom(image_path)
        else:
            image = Image.open(image_path)
            image_array = np.array(image)
            metadata = {}
        
        # Vision model inference (placeholder)
        # findings = self.vision_model.analyze(image_array)
        
        # Use LLM to interpret findings with context
        prompt = f"""
        As a medical AI assistant, analyze this medical image.
        
        Image Type: {metadata.get('modality', 'X-ray')}
        Body Part: {metadata.get('body_part', 'Chest')}
        
        Clinical Context:
        {clinical_context}
        
        Provide:
        1. Key observations
        2. Potential findings
        3. Differential diagnoses (if applicable)
        4. Recommendations for follow-up
        5. Confidence level
        
        Important: This is for educational purposes. Always consult qualified medical professionals.
        
        Return as structured JSON.
        """
        
        response = ollama.chat(model=self.text_model_name, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return {
            'metadata': metadata,
            'analysis': response['message']['content'],
            'image_processed': True
        }
    
    def analyze_clinical_text(self, text: str, question: str = None) -> Dict:
        """Analyze clinical text (EHR, notes, reports)"""
        if question:
            prompt = f"""
            Based on this clinical information, answer the following question:
            
            Clinical Information:
            {text}
            
            Question: {question}
            
            Provide a detailed, evidence-based answer with reasoning.
            """
        else:
            prompt = f"""
            Analyze this clinical text and extract:
            1. Chief complaints
            2. Relevant history
            3. Current symptoms
            4. Lab findings (if any)
            5. Medications
            6. Key clinical insights
            
            Clinical Text:
            {text}
            
            Return as structured JSON.
            """
        
        response = ollama.chat(model=self.text_model_name, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return {
            'analysis': response['message']['content']
        }
    
    def multimodal_analysis(self, text_data: str, 
                          image_path: str = None) -> Dict:
        """Combined analysis of text and image data"""
        # Analyze text
        text_analysis = self.analyze_clinical_text(text_data)
        
        # Analyze image if provided
        image_analysis = None
        if image_path:
            image_analysis = self.analyze_medical_image(
                image_path, 
                clinical_context=text_data
            )
        
        # Synthesize findings
        synthesis_prompt = f"""
        Synthesize these clinical findings into a comprehensive assessment:
        
        Clinical Text Analysis:
        {text_analysis['analysis']}
        
        {f"Image Analysis:\n{image_analysis['analysis']}" if image_analysis else ""}
        
        Provide:
        1. Integrated clinical picture
        2. Most likely diagnosis with differential
        3. Recommended diagnostic workup
        4. Treatment considerations
        5. Prognosis
        
        Format as clinical report.
        """
        
        response = ollama.chat(model=self.text_model_name, messages=[{
            'role': 'user',
            'content': synthesis_prompt
        }])
        
        return {
            'text_analysis': text_analysis,
            'image_analysis': image_analysis,
            'integrated_assessment': response['message']['content']
        }
    
    def query_medical_knowledge(self, query: str) -> str:
        """Answer medical knowledge questions"""
        prompt = f"""
        You are a medical knowledge assistant. Answer this question with:
        1. Direct answer
        2. Supporting evidence
        3. Clinical significance
        4. References (if applicable)
        
        Question: {query}
        
        Provide accurate, evidence-based information.
        """
        
        response = ollama.chat(model=self.text_model_name, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return response['message']['content']
```

```python
# api.py
from fastapi import FastAPI, UploadFile, File, Form
from medical_assistant import MedicalAssistant
import shutil

app = FastAPI(title="Medical Assistant API")
assistant = MedicalAssistant()

@app.post("/analyze/text")
async def analyze_text(
    clinical_text: str = Form(...),
    question: str = Form(None)
):
    """Analyze clinical text"""
    result = assistant.analyze_clinical_text(clinical_text, question)
    return result

@app.post("/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    clinical_context: str = Form("")
):
    """Analyze medical image"""
    # Save uploaded file
    image_path = f"temp/{image.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    result = assistant.analyze_medical_image(image_path, clinical_context)
    return result

@app.post("/analyze/multimodal")
async def multimodal_analysis(
    clinical_text: str = Form(...),
    image: UploadFile = File(None)
):
    """Combined text and image analysis"""
    image_path = None
    if image:
        image_path = f"temp/{image.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    
    result = assistant.multimodal_analysis(clinical_text, image_path)
    return result

@app.post("/query")
async def query_knowledge(query: str = Form(...)):
    """Answer medical knowledge questions"""
    answer = assistant.query_medical_knowledge(query)
    return {"query": query, "answer": answer}
```

---

## Project 5: Sports Commentator from Video

### **Objective**
Automated system for generating live or post-game commentary by analyzing sports video footage.

### **Tech Stack**
- **Video Processing**: OpenCV, PyTorchVideo
- **Video Understanding**: TimeSformer, Video Swin Transformer
- **LLM**: Llama 3, Gemma via Ollama
- **TTS**: Coqui TTS, Tacotron
- **Backend**: FastAPI + WebSocket for real-time
- **Frontend**: React + Video.js

### **Architecture**

```
┌──────────────────┐
│  Video Input     │
│  (Live/Recorded) │
└────────┬─────────┘
         │
         v
┌──────────────────────────┐
│  Frame Extraction        │
│  (Key moments)           │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Action Detection        │
│  - Player tracking       │
│  - Ball tracking         │
│  - Event detection       │
│  (Goal, foul, pass, etc.)│
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Scene Understanding     │
│  (Video Transformer)     │
│  - Game context          │
│  - Tactical analysis     │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Commentary Generation   │
│  (LLM + Game State)      │
│  - Exciting moments      │
│  - Play-by-play          │
│  - Analysis              │
└────────┬─────────────────┘
         │
         v
┌──────────────────────────┐
│  Text-to-Speech          │
│  (Natural voice)         │
└────────┬─────────────────┘
         │
         v
┌──────────────────┐
│  Audio Output    │
└──────────────────┘
```

### **Implementation**

```python
# sports_commentator.py
import cv2
import torch
import ollama
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class GameEvent:
    timestamp: float
    event_type: str  # goal, pass, shot, foul, etc.
    description: str
    players_involved: List[str]
    location: Tuple[float, float]
    confidence: float

class SportsCommentator:
    def __init__(self, sport="football"):
        self.sport = sport
        self.llm_model = "llama3"
        self.game_state = {
            'score': [0, 0],
            'time_elapsed': 0,
            'events': [],
            'possession': None
        }
    
    def extract_frames(self, video_path: str, 
                       sample_rate: int = 30) -> List[np.ndarray]:
        """Extract key frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_events(self, frames: List[np.ndarray]) -> List[GameEvent]:
        """Detect game events from frames"""
        events = []
        
        # Placeholder for actual computer vision model
        # In real implementation, use action recognition model
        # like SlowFast, TSN, or Video Swin Transformer
        
        # Simulated event detection
        for i, frame in enumerate(frames):
            # Process frame with CV model
            # detected_action = self.action_model.predict(frame)
            
            # For demonstration
            if i % 10 == 0:  # Simulate event every 10 frames
                event = GameEvent(
                    timestamp=i / 30.0,  # Assuming 30 fps
                    event_type="pass",
                    description="Player passes the ball",
                    players_involved=["Player 1", "Player 2"],
                    location=(frame.shape[1] // 2, frame.shape[0] // 2),
                    confidence=0.85
                )
                events.append(event)
        
        return events
    
    def generate_commentary(self, event: GameEvent, 
                          context: Dict = None) -> str:
        """Generate commentary for an event"""
        context_str = ""
        if context:
            context_str = f"""
            Current Score: {context['score'][0]} - {context['score'][1]}
            Time: {context['time_elapsed']} minutes
            Recent events: {context.get('recent_events', 'None')}
            """
        
        prompt = f"""
        You are an enthusiastic sports commentator for {self.sport}.
        Generate exciting commentary for this game event.
        
        Event: {event.event_type}
        Description: {event.description}
        Players: {', '.join(event.players_involved)}
        
        Game Context:
        {context_str}
        
        Generate 1-2 sentences of exciting commentary that a professional 
        sports commentator would say. Be energetic and engaging.
        """
        
        response = ollama.chat(model=self.llm_model, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        commentary = response['message']['content'].strip()
        return commentary
    
    def generate_analysis(self, events: List[GameEvent], 
                         timeframe: str = "half") -> str:
        """Generate tactical analysis commentary"""
        events_summary = "\n".join([
            f"{e.event_type} at {e.timestamp:.1f}s: {e.description}"
            for e in events[-10:]  # Last 10 events
        ])
        
        prompt = f"""
        You are a sports analyst providing tactical analysis.
        
        Analyze these recent game events and provide insightful commentary:
        
        {events_summary}
        
        Provide:
        1. Key observations about team strategy
        2. Player performance highlights
        3. Tactical patterns
        4. Predictions for next play
        
        Keep it concise and insightful (3-4 sentences).
        """
        
        response = ollama.chat(model=self.llm_model, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return response['message']['content'].strip()
    
    def process_video(self, video_path: str) -> List[Dict]:
        """Process entire video and generate commentary"""
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Detect events
        events = self.detect_events(frames)
        
        # Generate commentary for each event
        commentary_timeline = []
        
        for event in events:
            # Update game state
            self.game_state['time_elapsed'] = event.timestamp
            self.game_state['events'].append(event)
            
            # Generate commentary
            commentary = self.generate_commentary(
                event,
                context=self.game_state
            )
            
            commentary_timeline.append({
                'timestamp': event.timestamp,
                'event': event,
                'commentary': commentary
            })
            
            # Periodically generate analysis
            if len(self.game_state['events']) % 10 == 0:
                analysis = self.generate_analysis(self.game_state['events'])
                commentary_timeline.append({
                    'timestamp': event.timestamp,
                    'type': 'analysis',
                    'commentary': analysis
                })
        
        return commentary_timeline
    
    def text_to_speech(self, text: str, output_path: str):
        """Convert commentary text to speech"""
        # Using Coqui TTS or similar
        # from TTS.api import TTS
        # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        # tts.tts_to_file(text=text, file_path=output_path)
        pass
```

```python
# api.py
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import StreamingResponse
from sports_commentator import SportsCommentator
import asyncio

app = FastAPI(title="Sports Commentator API")

@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """Analyze video and generate commentary"""
    # Save video
    video_path = f"temp/{video.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())
    
    # Process video
    commentator = SportsCommentator()
    commentary_timeline = commentator.process_video(video_path)
    
    return {
        "total_events": len(commentary_timeline),
        "commentary": commentary_timeline
    }

@app.websocket("/live-commentary")
async def live_commentary(websocket: WebSocket):
    """WebSocket for live commentary"""
    await websocket.accept()
    
    commentator = SportsCommentator()
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            
            # Process frame (simplified)
            # In real implementation, accumulate frames and detect events
            
            # Send commentary back
            commentary = "Great pass! The team is building momentum..."
            await websocket.send_json({
                "commentary": commentary,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            await asyncio.sleep(0.1)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
```

---

## Common Setup & Best Practices

### **1. Environment Setup**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3
ollama pull gemma2:2b
```

### **2. Project Structure Template**

```
project-name/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── models/              # ML models
│   ├── services/            # Business logic
│   ├── api/                 # API routes
│   └── utils/               # Utilities
├── tests/
│   ├── unit/
│   └── integration/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/               # Jupyter notebooks for exploration
├── frontend/                # React/Streamlit UI
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### **3. Testing Framework**

```python
# tests/conftest.py
import pytest
from app.main import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_cv_text():
    return """
    John Doe
    Email: john@example.com
    ...
    """
```

### **4. Docker Deployment**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **5. Monitoring & Logging**

```python
# app/utils/logger.py
import logging
from pythonjsonlogger import jsonlogger

def setup_logger(name: str):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
```

---

## Evaluation Metrics

### **CV Creation & Sorting**
- ATS compatibility score
- Keyword matching accuracy
- User satisfaction rating
- Time saved vs manual process

### **Ad Generation**
- Image quality (FID score)
- Prompt adherence
- User preference testing
- A/B testing results

### **Medical Assistant**
- Diagnostic accuracy (on test cases)
- Response relevance
- Safety validation
- Compliance with medical standards

### **Sports Commentary**
- Event detection accuracy
- Commentary naturalness (human evaluation)
- Timing accuracy
- Engagement metrics

---

## Next Steps for Each Project

1. **Start with MVP**: Core functionality first
2. **Iterative Development**: Add features progressively  
3. **User Testing**: Get feedback early
4. **Optimize**: Performance and accuracy improvements
5. **Deploy**: Production-ready deployment
6. **Documentation**: Comprehensive docs for users

Would you like me to dive deeper into any specific project or create starter code for one of them?
