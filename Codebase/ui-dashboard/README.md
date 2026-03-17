# Medical Assistant UI Dashboard

Beautiful web-based interface for the Multimodal Medical Assistant.

## Features

✅ **Beautiful Medical Chat UI** - Modern, gradient-styled interface with medical theming  
✅ **Easy Option Selection** - Tab-based navigation for different workflows  
✅ **Session Context Display** - Real-time session info always visible  
✅ **File Upload Support** - Drag-and-drop for reports and medical images  
✅ **Intelligent Output Parsing** - Medical-styled sections for diagnosis, medications, findings  
✅ **Human-Consumable Results** - Structured data with color-coded sections, collapsible raw output  
✅ **Single File UI** - Lightweight `medical_assistant_ui.py` (no changes to codebase)  
✅ **Automated Setup** - Run script handles venv, dependencies, and Ollama

## Quick Start

### Prerequisites

Same as main project:
- Python 3.11+
- Ollama with Llama 3

### Launch UI

**macOS/Linux:**
```bash
cd ui-dashboard
./run_ui.sh
```

**Windows:**
```bash
cd ui-dashboard
run_ui.bat
```

The script automatically:
1. Creates virtual environment if needed
2. Installs dependencies (including Gradio)
3. Checks Ollama and pulls Llama 3 if needed
4. Starts Ollama service if not running
5. Launches UI at http://localhost:7860

### Manual Launch

If you prefer manual control:

```bash
# Activate venv
cd Codebase
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install Gradio (if not already)
pip install gradio==4.16.0

# Launch UI
cd ../ui-dashboard
python medical_assistant_ui.py
```

## UI Features

### Tab 1: Report Analysis
- Upload clinical reports (.txt, .doc, .docx)
- Displays structured sections:
  - 📋 Clinical Summary
  - 🔍 Chief Complaints
  - 🩺 Symptoms (as styled tags)
  - 💊 Medications
  - 🔬 Lab Findings
  - 🤖 Raw LLM Output (collapsible)

### Tab 2: Ask Medical Question
- Text input for queries
- RAG-powered semantic search
- Displays:
  - 💡 Answer with context
  - 🎯 Confidence score (visual progress bar)
  - 📚 References from knowledge base
  - 🤖 Raw LLM Output

### Tab 3: Image Analysis
- Upload medical images (jpg, png, dcm)
- Select modality (X-ray, MRI, CT, Ultrasound)
- Optional body part specification
- Displays:
  - 👁️ Observations
  - 🔍 Potential Findings
  - 📊 Confidence score
  - 🤖 Raw LLM Output

### Tab 4: Multimodal Fusion
- Upload both report AND image
- Integrated cross-modal analysis
- Displays:
  - 🔬 Integrated Clinical Assessment
  - 🩺 Differential Diagnosis (ranked)
  - 🔍 Recommended Workup
  - 🤖 Raw LLM Output

### Tab 5: Session Management
- Reset Current Session (clear history)
- Start New Session (new patient/case)
- Session info updates in real-time

## Session Context Display

Always visible at top:
```
📝 Session: Report: diabetes_case | ID: 30e7d3d3 | Interactions: 5
```

Shows:
- Session name (auto-generated from first interaction)
- Session ID (first 8 chars)
- Interaction count

## Output Styling

### Color-Coded Sections
- 🟢 Green border: Summaries, positive findings
- 🔴 Red border: Chief complaints, differential diagnosis
- 🟠 Orange border: Symptoms, findings
- 🟣 Purple border: Medications, integrated assessment
- 🔵 Blue border: Lab results, recommendations

### Confidence Visualization
- Visual progress bar with color coding:
  - Green: >80% confidence
  - Orange: 60-80% confidence
  - Red: <60% confidence

### Raw Output
- Collapsible details section
- Syntax-highlighted JSON/text
- Truncated to 1000 chars for readability

## Architecture

```
ui-dashboard/
├── medical_assistant_ui.py    # Single-file Gradio UI
├── run_ui.sh                   # Launch script (macOS/Linux)
├── run_ui.bat                  # Launch script (Windows)
└── README.md                   # This file

# Uses Codebase/ directly
../Codebase/
├── main.py                     # MedicalAssistantOrchestrator
├── text_processor.py           # BioBERT processing
├── image_processor.py          # MedCLIP processing
├── multimodal_fusion.py        # Cross-modal fusion
└── ... (no changes needed)
```

## Technical Details

**UI Framework**: Gradio 4.16.0  
**Theme**: Soft theme with purple/blue gradients  
**Port**: 7860 (default)  
**Integration**: Direct Python function calls (no API layer)

## Advantages

1. **Zero Codebase Changes**: UI imports from `Codebase/` without modifications
2. **Lightweight**: Single 500-line Python file
3. **Beautiful**: Professional medical styling with gradients and color coding
4. **Intelligent Parsing**: Structured output with medical sections
5. **Session Aware**: Real-time context display
6. **File Friendly**: Drag-and-drop for reports and images
7. **Automated Setup**: Run script handles everything

## Troubleshooting

**Gradio not installed:**
```bash
pip install gradio==4.16.0
```

**Port 7860 in use:**
Edit `medical_assistant_ui.py` line with `server_port=7860` to use different port

**Ollama not running:**
```bash
ollama serve
```

**Missing dependencies:**
```bash
cd Codebase
pip install -r requirements.txt
```

## Screenshots

Interface includes:
- Header with project branding
- Session info bar (purple gradient)
- 5 tabs for different workflows
- Color-coded output sections
- Collapsible raw LLM output
- Professional medical styling

---

**Ready to launch**: Just run `./run_ui.sh` and open http://localhost:7860
