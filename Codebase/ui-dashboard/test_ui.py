"""
Test script for Medical Assistant UI
Validates UI functionality without launching full interface
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Codebase"))

print("🧪 Testing Medical Assistant UI Components\n")
print("=" * 60)

# Test 1: Import UI module
print("\n1. Testing UI module import...")
try:
    from ui_dashboard import medical_assistant_ui
    print("   ✓ UI module imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import UI module: {e}")
    sys.exit(1)

# Test 2: Initialize orchestrator
print("\n2. Testing orchestrator initialization...")
try:
    orchestrator = medical_assistant_ui.initialize_orchestrator()
    print("   ✓ Orchestrator initialized")
    print(f"   Session ID: {orchestrator._session_id}")
except Exception as e:
    print(f"   ❌ Failed to initialize orchestrator: {e}")
    sys.exit(1)

# Test 3: Session info
print("\n3. Testing session info display...")
try:
    session_info = medical_assistant_ui.get_session_info()
    print(f"   ✓ Session info: {session_info}")
except Exception as e:
    print(f"   ❌ Failed to get session info: {e}")
    sys.exit(1)

# Test 4: Output parsing (mock data)
print("\n4. Testing medical output parsing...")
try:
    mock_report_result = {
        "summary": "Patient presents with Type 2 Diabetes",
        "chief_complaints": ["Increased thirst", "Frequent urination"],
        "symptoms": ["polyuria", "polydipsia", "fatigue"],
        "medications": ["Metformin 500mg", "Lisinopril 10mg"],
        "lab_findings": ["HbA1c: 8.9%", "FBG: 248 mg/dL"],
        "raw_response": '{"summary": "...", "chief_complaints": [...]}'
    }
    
    html_output = medical_assistant_ui.parse_medical_output(mock_report_result, "report_analysis")
    
    # Verify HTML contains key sections
    assert "Clinical Summary" in html_output
    assert "Chief Complaints" in html_output
    assert "Symptoms" in html_output
    assert "Medications" in html_output
    assert "Laboratory Findings" in html_output
    assert "Raw LLM Output" in html_output
    
    print("   ✓ Report analysis parsing works")
    print(f"   Generated HTML length: {len(html_output)} characters")
except Exception as e:
    print(f"   ❌ Failed output parsing: {e}")
    sys.exit(1)

# Test 5: Query output parsing
print("\n5. Testing query output parsing...")
try:
    mock_query_result = {
        "answer": "Type 2 Diabetes is characterized by insulin resistance...",
        "confidence": 0.85,
        "references": ["diabetes_overview.txt", "endocrine_disorders.txt"]
    }
    
    html_output = medical_assistant_ui.parse_medical_output(mock_query_result, "query")
    
    assert "Answer" in html_output
    assert "Confidence Score" in html_output
    assert "References" in html_output
    assert "85%" in html_output  # Confidence percentage
    
    print("   ✓ Query parsing works")
except Exception as e:
    print(f"   ❌ Failed query parsing: {e}")
    sys.exit(1)

# Test 6: Image analysis output parsing
print("\n6. Testing image analysis output parsing...")
try:
    mock_image_result = {
        "observations": ["Clear lung fields", "Normal cardiac silhouette"],
        "potential_findings": ["No acute abnormalities"],
        "confidence_score": 0.78
    }
    
    html_output = medical_assistant_ui.parse_medical_output(mock_image_result, "image_analysis")
    
    assert "Observations" in html_output
    assert "Potential Findings" in html_output
    assert "Confidence" in html_output
    
    print("   ✓ Image analysis parsing works")
except Exception as e:
    print(f"   ❌ Failed image parsing: {e}")
    sys.exit(1)

# Test 7: Multimodal fusion output parsing
print("\n7. Testing multimodal fusion output parsing...")
try:
    mock_mm_result = {
        "integrated_assessment": "Patient presents with clinical and radiographic findings consistent with pneumonia",
        "differential_diagnosis": ["Community-acquired pneumonia", "Viral pneumonia", "Aspiration pneumonia"],
        "recommended_workup": ["Sputum culture", "Blood cultures", "CBC with differential"],
        "confidence_level": "high"
    }
    
    html_output = medical_assistant_ui.parse_medical_output(mock_mm_result, "multimodal_fusion")
    
    assert "Integrated Clinical Assessment" in html_output
    assert "Differential Diagnosis" in html_output
    assert "Recommended Workup" in html_output
    
    print("   ✓ Multimodal fusion parsing works")
except Exception as e:
    print(f"   ❌ Failed multimodal parsing: {e}")
    sys.exit(1)

# Test 8: Session management
print("\n8. Testing session management...")
try:
    # Get initial session
    initial_session = orchestrator._session_id
    
    # Reset session
    msg, session_info = medical_assistant_ui.reset_session()
    assert "reset" in msg.lower()
    print(f"   ✓ Reset: {msg}")
    
    # New session
    msg, session_info = medical_assistant_ui.new_session()
    assert "new session" in msg.lower()
    print(f"   ✓ New session: {msg}")
    
except Exception as e:
    print(f"   ❌ Failed session management: {e}")
    sys.exit(1)

# Test 9: Gradio UI creation (without launching)
print("\n9. Testing Gradio UI creation...")
try:
    demo = medical_assistant_ui.create_ui()
    print("   ✓ Gradio UI created successfully")
    print(f"   UI components: {len(demo.blocks)} blocks")
except Exception as e:
    print(f"   ❌ Failed to create UI: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All UI tests passed successfully!")
print("\nReady to launch UI with:")
print("   ./run_ui.sh (macOS/Linux)")
print("   run_ui.bat (Windows)")
print("\nOr manually:")
print("   python medical_assistant_ui.py")
