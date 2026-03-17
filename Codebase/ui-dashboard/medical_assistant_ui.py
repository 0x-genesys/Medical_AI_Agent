"""
Multimodal Medical Assistant - Beautiful Web UI
Single-file Gradio interface with intelligent medical output parsing
"""
import gradio as gr
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json

# Add Codebase directory to path for imports
# ui-dashboard is inside Codebase, so parent.parent is project root, parent is Codebase
codebase_path = str(Path(__file__).parent.parent)
if codebase_path not in sys.path:
    sys.path.insert(0, codebase_path)

# Detect Google Colab
def _is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IS_COLAB = _is_colab()

# Import from Codebase directory
from cli_main import MedicalAssistantOrchestrator  # noqa: E402
from models import ImageModality  # noqa: E402

# Global orchestrator instance
orchestrator = None

def initialize_orchestrator():
    """Initialize the medical assistant orchestrator"""
    global orchestrator
    if orchestrator is None:
        orchestrator = MedicalAssistantOrchestrator()
    return orchestrator

def get_session_info() -> str:
    """Get current session information for display"""
    orch = initialize_orchestrator()
    if orch._session_id:
        return f"📝 Session: {orch._session_name} | ID: {orch._session_id[:8]} | Interactions: {orch._interaction_count}"
    else:
        return "🆕 No active session - will create on first interaction"

def show_file_preview(file_obj) -> str:
    """Display preview of uploaded file content"""
    if file_obj is None:
        return "<div style='padding: 20px; background: #f8f9fa; border-radius: 8px; color: #6c757d;'>📄 No file uploaded yet</div>"
    
    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        file_name = os.path.basename(file_path)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(2000)  # Limit preview to first 2000 characters
        
        # Truncate if needed
        is_truncated = len(content) >= 2000
        preview_text = content[:2000]
        
        return f"""
        <div style="background: white; padding: 20px; border-radius: 8px; border: 2px solid #667eea;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 10px 15px; border-radius: 6px; margin-bottom: 15px;">
                <strong style="color: white;">📄 {file_name}</strong>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 6px; 
                        font-family: monospace; font-size: 13px; line-height: 1.6; 
                        max-height: 400px; overflow-y: auto; color: #2c3e50; white-space: pre-wrap;">{preview_text}</div>
            {f'<p style="margin-top: 10px; color: #f39c12; font-size: 13px;">⚠️ Preview truncated - showing first 2000 characters</p>' if is_truncated else ''}
        </div>
        """
    except Exception as e:
        return f"<div style='padding: 20px; background: #fff3cd; border-radius: 8px; color: #856404;'>⚠️ Could not preview file: {str(e)}</div>"

def parse_medical_output(result: Dict[str, Any], flow_type: str) -> str:
    """
    Intelligently parse medical AI output into human-consumable HTML
    Shows structured data in styled sections, raw text separately
    With error handling and fallback to raw output
    """
    html_parts = []
    
    # Header with flow type
    html_parts.append(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 15px; border-radius: 10px 10px 0 0; margin-bottom: 0;">
        <h3 style="color: white; margin: 0; font-size: 18px;">
            🏥 {flow_type.replace('_', ' ').title()} Results
        </h3>
    </div>
    """)
    
    # Main content container
    html_parts.append('<div style="background: #f8f9fa; padding: 20px; border-radius: 0 0 10px 10px; border: 2px solid #667eea;">')
    
    # Track if we have any structured content
    has_structured_content = False
    
    try:
        if flow_type == "report_analysis":
            has_structured_content = True
            # Summary section - format with line breaks
            if result.get('summary'):
                summary_text = result['summary']
                formatted_summary = summary_text.replace('\n\n', '</p><p style="line-height: 1.8; color: #34495e; margin: 10px 0;">').replace('\n', '<br>')
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;">
                    <h4 style="color: #2c3e50; margin-top: 0;">📋 Clinical Summary</h4>
                    <p style="line-height: 1.8; color: #34495e; margin: 10px 0; white-space: pre-wrap;">{formatted_summary}</p>
                </div>
                """)
            
            # Chief Complaints
            if result.get('chief_complaints'):
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #e74c3c;">
                    <h4 style="color: #2c3e50; margin-top: 0;">🔍 Chief Complaints</h4>
                    <ul style="margin: 0; color: #34495e;">
                        {''.join(f'<li style="color: #34495e;">{item}</li>' for item in result['chief_complaints'])}
                    </ul>
                </div>
                """)
            
            # Symptoms
            if result.get('symptoms'):
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #f39c12;">
                    <h4 style="color: #2c3e50; margin-top: 0;">🩺 Symptoms</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        {''.join(f'<span style="background: #fff3cd; padding: 5px 12px; border-radius: 15px; font-size: 14px; color: #856404;">{item}</span>' for item in result['symptoms'])}
                    </div>
                </div>
                """)
            
            # Medications
            if result.get('medications'):
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;">
                    <h4 style="color: #2c3e50; margin-top: 0;">💊 Medications</h4>
                    <ul style="margin: 0; color: #34495e;">
                        {''.join(f'<li style="color: #34495e;"><strong style="color: #34495e;">{item}</strong></li>' for item in result['medications'])}
                    </ul>
                </div>
                """)
            
            # Lab Findings
            if result.get('lab_findings'):
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h4 style="color: #2c3e50; margin-top: 0;">🔬 Laboratory Findings</h4>
                    <ul style="margin: 0; color: #34495e;">
                        {''.join(f'<li style="color: #34495e;">{item}</li>' for item in result['lab_findings'])}
                    </ul>
                </div>
                """)
        
        elif flow_type == "query":
            has_structured_content = True
            # Answer - format text with line breaks for readability
            answer_text = result.get('answer', 'N/A')
            # Convert newlines to <br> tags and improve spacing
            formatted_answer = answer_text.replace('\n\n', '</p><p style="line-height: 1.8; color: #34495e; margin: 10px 0;">').replace('\n', '<br>')
            html_parts.append(f"""
            <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                <h4 style="color: #2c3e50; margin-top: 0;">💡 Answer</h4>
                <p style="line-height: 1.8; color: #34495e; margin: 10px 0; white-space: pre-wrap;">{formatted_answer}</p>
            </div>
            """)
            
            # Confidence
            confidence = result.get('confidence', 0)
            confidence_color = '#4CAF50' if confidence > 0.8 else '#f39c12' if confidence > 0.6 else '#e74c3c'
            html_parts.append(f"""
            <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
                <h4 style="color: #2c3e50; margin-top: 0;">🎯 Confidence Score</h4>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="flex: 1; background: #ecf0f1; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="width: {confidence*100}%; height: 100%; background: {confidence_color};"></div>
                    </div>
                    <strong style="color: {confidence_color};">{confidence:.2%}</strong>
                </div>
            </div>
            """)
            
            # References
            if result.get('references'):
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #95a5a6;">
                    <h4 style="color: #2c3e50; margin-top: 0;">📚 References</h4>
                    <ul style="margin: 0; color: #34495e; font-size: 13px;">
                        {''.join(f'<li style="color: #34495e;">{ref}</li>' for ref in result['references'][:5])}
                    </ul>
                </div>
                """)
        
        elif flow_type == "image_analysis":
            has_structured_content = True
            # Observations
            if result.get('observations'):
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h4 style="color: #2c3e50; margin-top: 0;">👁️ Observations ({len(result['observations'])})</h4>
                    <ul style="margin: 0; color: #34495e;">
                        {''.join(f'<li style="color: #34495e;">{obs}</li>' for obs in result['observations'])}
                    </ul>
                </div>
                """)
            
            # Potential Findings
            if result.get('potential_findings'):
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #f39c12;">
                    <h4 style="color: #2c3e50; margin-top: 0;">🔍 Potential Findings</h4>
                    <ul style="margin: 0; color: #34495e;">
                        {''.join(f'<li style="color: #34495e;">{finding}</li>' for finding in result['potential_findings'])}
                    </ul>
                </div>
                """)
            
            # Confidence
            confidence = result.get('confidence_score', 0)
            confidence_color = '#4CAF50' if confidence > 0.8 else '#f39c12' if confidence > 0.6 else '#e74c3c'
            html_parts.append(f"""
            <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
                <h4 style="color: #2c3e50; margin-top: 0;">📊 Confidence</h4>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="flex: 1; background: #ecf0f1; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="width: {confidence*100}%; height: 100%; background: {confidence_color};"></div>
                    </div>
                    <strong style="color: {confidence_color};">{confidence:.2%}</strong>
                </div>
            </div>
            """)
        
        elif flow_type == "multimodal_fusion":
            has_structured_content = True
            
            # Try to extract and parse JSON from raw_response if data is embedded in markdown
            parsed_data = None
            raw_response = result.get('raw_response', '')
            if raw_response and '```' in raw_response:
                try:
                    # Extract JSON from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', raw_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        parsed_data = json.loads(json_str)
                        # Merge parsed data into result
                        result = {**result, **parsed_data}
                except Exception as parse_error:
                    pass  # Continue with original result
            
            # Clinical Summary
            if result.get('clinical_summary'):
                summary_text = result['clinical_summary']
                if isinstance(summary_text, str):
                    formatted_summary = summary_text.replace('\n\n', '</p><p style="line-height: 1.8; color: #34495e; margin: 10px 0;">').replace('\n', '<br>')
                    html_parts.append(f"""
                    <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;">
                        <h4 style="color: #2c3e50; margin-top: 0;">📋 Clinical Summary</h4>
                        <p style="line-height: 1.8; color: #34495e; margin: 10px 0; white-space: pre-wrap;">{formatted_summary}</p>
                    </div>
                    """)
            
            # Integrated Assessment - handle both string and complex nested dict
            if result.get('integrated_assessment'):
                assessment = result['integrated_assessment']
                
                if isinstance(assessment, dict):
                    # Parse nested structure
                    html_parts.append('<div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;">')
                    html_parts.append('<h4 style="color: #2c3e50; margin-top: 0;">🔬 Integrated Clinical Assessment</h4>')
                    
                    # Chief Complaints
                    if assessment.get('chief_complaints'):
                        complaints = assessment['chief_complaints']
                        if isinstance(complaints, list):
                            html_parts.append('<h5 style="color: #7f8c8d; margin: 15px 0 8px 0;">Chief Complaints</h5>')
                            html_parts.append('<ul style="margin: 0 0 10px 0; color: #34495e;">')
                            for c in complaints:
                                html_parts.append(f'<li style="color: #34495e;">{c}</li>')
                            html_parts.append('</ul>')
                    
                    # Symptoms
                    if assessment.get('symptoms'):
                        symptoms = assessment['symptoms']
                        if isinstance(symptoms, list):
                            html_parts.append('<h5 style="color: #7f8c8d; margin: 15px 0 8px 0;">Symptoms</h5>')
                            html_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">')
                            for s in symptoms:
                                html_parts.append(f'<span style="background: #fff3cd; padding: 5px 12px; border-radius: 15px; font-size: 14px; color: #856404;">{s}</span>')
                            html_parts.append('</div>')
                    
                    # Medical History
                    if assessment.get('medical_history'):
                        history = assessment['medical_history']
                        if isinstance(history, list):
                            html_parts.append('<h5 style="color: #7f8c8d; margin: 15px 0 8px 0;">Medical History</h5>')
                            html_parts.append('<ul style="margin: 0 0 10px 0; color: #34495e;">')
                            for h in history:
                                html_parts.append(f'<li style="color: #34495e;">{h}</li>')
                            html_parts.append('</ul>')
                    
                    # Medications
                    if assessment.get('medications'):
                        meds = assessment['medications']
                        html_parts.append('<h5 style="color: #7f8c8d; margin: 15px 0 8px 0;">💊 Medications</h5>')
                        html_parts.append('<ul style="margin: 0 0 10px 0; color: #34495e;">')
                        for med in meds:
                            if isinstance(med, dict):
                                med_name = med.get('name', '')
                                med_dose = med.get('dose', '')
                                html_parts.append(f'<li style="color: #34495e;"><strong>{med_name}</strong> - {med_dose}</li>')
                            else:
                                html_parts.append(f'<li style="color: #34495e;">{med}</li>')
                        html_parts.append('</ul>')
                    
                    # Lab Findings
                    if assessment.get('lab_findings'):
                        labs = assessment['lab_findings']
                        html_parts.append('<h5 style="color: #7f8c8d; margin: 15px 0 8px 0;">🔬 Laboratory Findings</h5>')
                        html_parts.append('<div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">')
                        for lab in labs:
                            if isinstance(lab, dict):
                                test = lab.get('test', '')
                                result_val = lab.get('result', '')
                                html_parts.append(f'<div style="margin: 5px 0; color: #34495e;"><strong>{test}:</strong> {result_val}</div>')
                            else:
                                html_parts.append(f'<div style="margin: 5px 0; color: #34495e;">{lab}</div>')
                        html_parts.append('</div>')
                    
                    # Imaging Findings
                    if assessment.get('imaging_findings'):
                        imaging = assessment['imaging_findings']
                        html_parts.append('<h5 style="color: #7f8c8d; margin: 15px 0 8px 0;">🖼️ Imaging Findings</h5>')
                        html_parts.append('<ul style="margin: 0 0 10px 0; color: #34495e;">')
                        if isinstance(imaging, list):
                            for img in imaging:
                                html_parts.append(f'<li style="color: #34495e;">{img}</li>')
                        else:
                            html_parts.append(f'<li style="color: #34495e;">{imaging}</li>')
                        html_parts.append('</ul>')
                    
                    html_parts.append('</div>')
                    
                elif isinstance(assessment, str):
                    # Simple string format
                    formatted_assessment = assessment.replace('\n\n', '</p><p style="line-height: 1.8; color: #34495e; margin: 10px 0;">').replace('\n', '<br>')
                    html_parts.append(f"""
                    <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;">
                        <h4 style="color: #2c3e50; margin-top: 0;">🔬 Integrated Clinical Assessment</h4>
                        <p style="line-height: 1.8; color: #34495e; margin: 10px 0; white-space: pre-wrap;">{formatted_assessment}</p>
                    </div>
                    """)
            
            # Differential Diagnosis - handle both simple list and dict with probabilities
            if result.get('differential_diagnosis'):
                dx_list = result['differential_diagnosis']
                html_parts.append("""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #e74c3c;">
                    <h4 style="color: #2c3e50; margin-top: 0;">🩺 Differential Diagnosis</h4>
                    <ol style="margin: 0; color: #34495e;">
                """)
                for dx in dx_list:
                    if isinstance(dx, dict):
                        diagnosis = dx.get('diagnosis', '')
                        probability = dx.get('probability', 0)
                        prob_percent = int(probability * 100) if probability <= 1 else int(probability)
                        prob_color = '#4CAF50' if prob_percent > 70 else '#f39c12' if prob_percent > 40 else '#e74c3c'
                        html_parts.append(f"""
                        <li style="color: #34495e; margin: 8px 0;">
                            <strong style="color: #34495e;">{diagnosis}</strong>
                            <span style="margin-left: 10px; background: {prob_color}20; padding: 2px 8px; border-radius: 10px; color: {prob_color}; font-weight: bold; font-size: 12px;">
                                {prob_percent}%
                            </span>
                        </li>
                        """)
                    else:
                        html_parts.append(f'<li style="color: #34495e;"><strong style="color: #34495e;">{dx}</strong></li>')
                html_parts.append('</ol></div>')
            
            # Recommended Workup - handle both simple list and dict with test names
            if result.get('recommended_workup'):
                workup_list = result['recommended_workup']
                html_parts.append("""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h4 style="color: #2c3e50; margin-top: 0;">🔍 Recommended Workup</h4>
                    <ul style="margin: 0; color: #34495e;">
                """)
                for workup in workup_list:
                    if isinstance(workup, dict):
                        test = workup.get('test', workup.get('name', ''))
                        html_parts.append(f'<li style="color: #34495e;">{test}</li>')
                    else:
                        html_parts.append(f'<li style="color: #34495e;">{workup}</li>')
                html_parts.append('</ul></div>')
            
            # Confidence Level
            if result.get('confidence_level'):
                confidence_text = result['confidence_level']
                confidence_color = '#4CAF50' if confidence_text.lower() == 'high' else '#f39c12' if confidence_text.lower() == 'medium' else '#e74c3c'
                html_parts.append(f"""
                <div style="background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
                    <h4 style="color: #2c3e50; margin-top: 0;">🎯 Confidence Level</h4>
                    <div style="display: inline-block; background: {confidence_color}20; padding: 8px 16px; border-radius: 20px; border: 2px solid {confidence_color};">
                        <strong style="color: {confidence_color}; text-transform: uppercase;">{confidence_text}</strong>
                    </div>
                </div>
                """)
        
    except Exception as e:
        # If parsing fails, show error and fallback to raw output
        if not has_structured_content:
            html_parts.append(f"""
            <div style="background: #fff3cd; padding: 20px; border-radius: 8px; border-left: 4px solid #f39c12; margin-bottom: 15px;">
                <h4 style="color: #856404; margin-top: 0;">⚠️ Parsing Error</h4>
                <p style="color: #856404; margin: 0;">Could not parse structured output: {str(e)}</p>
                <p style="color: #856404; margin: 5px 0 0 0;">Showing raw response below.</p>
            </div>
            """)
    
    # Raw LLM Output (collapsible) - NO TRUNCATION
    if result.get('raw_response'):
        raw_text = result['raw_response']
        html_parts.append(f"""
        <details style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;">
            <summary style="cursor: pointer; color: #7f8c8d; font-weight: bold;">
                🤖 View Full Raw LLM Output ({len(raw_text)} characters)
            </summary>
            <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; 
                        overflow-x: auto; margin-top: 10px; font-size: 12px; max-height: 600px; overflow-y: auto;">{raw_text}</pre>
        </details>
        """)
    
    html_parts.append('</div>')  # Close main container
    
    return ''.join(html_parts)

def process_report(file_obj) -> Tuple[str, str]:
    """Process clinical report file"""
    orch = initialize_orchestrator()
    try:
        # Gradio file upload returns file object with .name attribute
        if file_obj is None:
            return "❌ Please upload a file", get_session_info()
        
        # Extract file path from Gradio file object
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        
        result = orch.analyze_report_flow(file_path)
        if 'error' in result:
            return f"❌ Error: {result['error']}", get_session_info()
        
        html_output = parse_medical_output(result, "report_analysis")
        return html_output, get_session_info()
    except Exception as e:
        return f"❌ Error processing report: {str(e)}", get_session_info()

def process_query(query: str) -> Tuple[str, str]:
    """Process medical query with RAG"""
    orch = initialize_orchestrator()
    try:
        result = orch.answer_query_flow(query)
        if 'error' in result:
            return f"❌ Error: {result['error']}", get_session_info()
        
        html_output = parse_medical_output(result, "query")
        return html_output, get_session_info()
    except Exception as e:
        return f"❌ Error processing query: {str(e)}", get_session_info()

def process_image(image_path: str, modality: str, body_part: str) -> Tuple[str, str]:
    """Process medical image"""
    orch = initialize_orchestrator()
    try:
        result = orch.analyze_image_flow(image_path, modality.lower(), body_part if body_part else None)
        if 'error' in result:
            return f"❌ Error: {result['error']}", get_session_info()
        
        html_output = parse_medical_output(result, "image_analysis")
        return html_output, get_session_info()
    except Exception as e:
        return f"❌ Error processing image: {str(e)}", get_session_info()

def process_multimodal(report_file, image_file, modality: str, body_part: str) -> Tuple[str, str]:
    """Process multimodal fusion (text + image)"""
    orch = initialize_orchestrator()
    try:
        if report_file is None or image_file is None:
            return "❌ Please upload both report and image files", get_session_info()
        
        # Extract file paths from Gradio file objects
        report_path = report_file.name if hasattr(report_file, 'name') else str(report_file)
        image_path = image_file.name if hasattr(image_file, 'name') else str(image_file)
        
        result = orch.multimodal_analysis_flow(
            user_input=None,
            modality=modality.lower(),
            body_part=body_part if body_part else None,
            text_file=report_path,
            image_file=image_path
        )
        if 'error' in result:
            return f"❌ Error: {result['error']}", get_session_info()
        
        html_output = parse_medical_output(result, "multimodal_fusion")
        return html_output, get_session_info()
    except Exception as e:
        return f"❌ Error processing multimodal: {str(e)}", get_session_info()

def reset_session() -> Tuple[str, str]:
    """Reset current session"""
    orch = initialize_orchestrator()
    orch.reset_session()
    return "✓ Session reset - context cleared", get_session_info()

def new_session() -> Tuple[str, str]:
    """Create new session"""
    orch = initialize_orchestrator()
    old_name = orch._session_name
    old_count = orch._interaction_count
    orch.reset_session()
    orch._session_id = None  # Force new session
    return f"✓ New session started (Previous: {old_name} with {old_count} interactions)", get_session_info()

# Create Gradio interface
def create_ui():
    """Create beautiful medical assistant UI"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
        title="Multimodal Medical Assistant",
        css="""
        .container { max-width: 1400px; margin: auto; }
        .session-info { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🏥 Multimodal Medical Assistant
        ### AI-Powered Clinical Decision Support with BioBERT, BiomedCLIP & LangChain
        """)
        
        # Session Info and Controls on Main Page
        with gr.Row():
            with gr.Column(scale=3):
                session_display = gr.Markdown(value=get_session_info(), elem_classes="session-info")
            with gr.Column(scale=1):
                with gr.Row():
                    restart_session_btn = gr.Button("🔄 Restart Session", variant="secondary", size="lg")
                    new_session_btn = gr.Button("➕ New Session", variant="primary", size="lg")
        
        session_action_msg = gr.Markdown(visible=False)
        
        # Wire up main page session controls
        restart_session_btn.click(
            fn=reset_session,
            outputs=[session_action_msg, session_display]
        )
        
        new_session_btn.click(
            fn=new_session,
            outputs=[session_action_msg, session_display]
        )
        
        # Main Tabs
        with gr.Tabs():
            
            # Tab 1: Report Analysis
            with gr.Tab("📋 Report Analysis"):
                gr.Markdown("Upload clinical report files (.txt, .doc) for structured analysis")
                with gr.Row():
                    with gr.Column(scale=1):
                        report_file = gr.File(label="Upload Report File", file_types=[".txt", ".doc", ".docx"])
                        report_preview = gr.HTML(label="File Preview")
                        report_btn = gr.Button("Analyze Report", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        report_output = gr.HTML(label="Analysis Results")
                
                # Show preview when file is uploaded
                report_file.change(
                    fn=show_file_preview,
                    inputs=[report_file],
                    outputs=[report_preview]
                )
                
                report_btn.click(
                    fn=process_report,
                    inputs=[report_file],
                    outputs=[report_output, session_display]
                )
            
            # Tab 2: Medical Query (RAG)
            with gr.Tab("💬 Ask Medical Question"):
                gr.Markdown("Ask questions with RAG-powered semantic search")
                query_input = gr.Textbox(
                    label="Your Medical Question",
                    placeholder="e.g., What are the symptoms of Type 2 Diabetes?",
                    lines=3
                )
                query_btn = gr.Button("Ask Question", variant="primary", size="lg")
                query_output = gr.HTML(label="Answer")
                
                query_btn.click(
                    fn=process_query,
                    inputs=[query_input],
                    outputs=[query_output, session_display]
                )
            
            # Tab 3: Image Analysis
            with gr.Tab("🖼️ Image Analysis"):
                gr.Markdown("Analyze medical images (X-rays, MRIs, CT scans)")
                with gr.Row():
                    with gr.Column(scale=1):
                        image_file = gr.Image(label="Upload Medical Image", type="filepath")
                        image_modality = gr.Dropdown(
                            choices=["xray", "mri", "ct", "ultrasound"],
                            value="xray",
                            label="Image Modality"
                        )
                        image_body_part = gr.Textbox(label="Body Part (optional)", placeholder="e.g., chest")
                        image_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        image_output = gr.HTML(label="Analysis Results")
                
                image_btn.click(
                    fn=process_image,
                    inputs=[image_file, image_modality, image_body_part],
                    outputs=[image_output, session_display]
                )
            
            # Tab 4: Multimodal Fusion
            with gr.Tab("🔬 Multimodal Fusion"):
                gr.Markdown("Integrated analysis of text report + medical image")
                with gr.Row():
                    with gr.Column(scale=1):
                        mm_report = gr.File(label="Clinical Report", file_types=[".txt", ".doc", ".docx"])
                        mm_report_preview = gr.HTML(label="Report Preview")
                        mm_image = gr.Image(label="Medical Image", type="filepath")
                        mm_modality = gr.Dropdown(
                            choices=["xray", "mri", "ct", "ultrasound"],
                            value="xray",
                            label="Image Modality"
                        )
                        mm_body_part = gr.Textbox(label="Body Part (optional)", placeholder="e.g., chest")
                        mm_btn = gr.Button("Fuse & Analyze", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        mm_output = gr.HTML(label="Integrated Assessment")
                
                # Show report preview when file is uploaded
                mm_report.change(
                    fn=show_file_preview,
                    inputs=[mm_report],
                    outputs=[mm_report_preview]
                )
                
                mm_btn.click(
                    fn=process_multimodal,
                    inputs=[mm_report, mm_image, mm_modality, mm_body_part],
                    outputs=[mm_output, session_display]
                )
            
            # Tab 5: Session Management
            with gr.Tab("⚙️ Session Management"):
                gr.Markdown("Manage conversation sessions and context")
                with gr.Row():
                    reset_btn = gr.Button("Reset Current Session", variant="secondary", size="lg")
                    new_btn = gr.Button("Start New Session", variant="secondary", size="lg")
                
                session_msg = gr.Markdown()
                
                reset_btn.click(
                    fn=reset_session,
                    outputs=[session_msg, session_display]
                )
                
                new_btn.click(
                    fn=new_session,
                    outputs=[session_msg, session_display]
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Disclaimer**: For educational purposes only. Not for clinical use without proper validation.
        
        **Features**: BioBERT embeddings | MedCLIP vision | RAG with FAISS | Session continuity | HIPAA compliant
        """)
    
    return demo

if __name__ == "__main__":
    print("🏥 Initializing Multimodal Medical Assistant UI...")
    demo = create_ui()
    
    # Enable public URL for Colab
    if IS_COLAB:
        print("🔗 Running in Google Colab - creating public URL...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Create public URL for Colab
            show_error=True
        )
    else:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
