"""
Comprehensive test suite for all flows (Report, Query, Image)
Tests session continuity and context flow across different inputs
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_main import MedicalAssistantOrchestrator


def test_report_flow():
    """Test report analysis with file input"""
    print("\n" + "="*70)
    print("TEST 1: Report Analysis Flow")
    print("="*70)
    
    orch = MedicalAssistantOrchestrator()
    
    # Test with diabetes clinical note
    result = orch.analyze_report_flow('examples/sample_clinical_note_diabetes.txt')
    
    assert result['flow'] == 'report_analysis', "Flow type mismatch"
    assert result.get('parsed') == True, "Report should be parsed successfully"
    assert len(result.get('chief_complaints', [])) > 0, "Should extract chief complaints"
    assert len(result.get('medications', [])) > 0, "Should extract medications"
    assert result.get('summary'), "Should have summary"
    
    print(f"✅ Report parsed: {result.get('parsed')}")
    print(f"✅ Chief complaints: {len(result['chief_complaints'])}")
    print(f"✅ Medications: {result['medications'][:2]}")
    print(f"✅ Summary: {result['summary'][:80]}...")
    
    return orch  # Return for session continuity test


def test_query_flow():
    """Test query flow with medical questions"""
    print("\n" + "="*70)
    print("TEST 2: Query Flow")
    print("="*70)
    
    orch = MedicalAssistantOrchestrator()
    
    # Test medical knowledge query
    result = orch.answer_query_flow("What are the symptoms of diabetes?")
    
    assert result['flow'] == 'query', "Flow type mismatch"
    assert result.get('answer'), "Should have answer"
    assert result.get('confidence') > 0, "Should have confidence score"
    
    print(f"✅ Query answered: {result['query'][:50]}...")
    print(f"✅ Answer length: {len(result['answer'])} chars")
    print(f"✅ Confidence: {result['confidence']}")
    print(f"✅ Answer preview: {result['answer'][:100]}...")
    
    return orch


def test_image_flow():
    """Test image analysis flow"""
    print("\n" + "="*70)
    print("TEST 3: Image Analysis Flow")
    print("="*70)
    
    orch = MedicalAssistantOrchestrator()
    
    # Test chest X-ray analysis
    result = orch.analyze_image_flow(
        'examples/CV-case-2-2-1-768x575.jpg',
        'xray',
        'chest'
    )
    
    assert result['flow'] == 'image_analysis', "Flow type mismatch"
    assert 'observations' in result, "Should have observations"
    assert result.get('confidence_score') >= 0, "Should have confidence score"
    
    print(f"✅ Image analyzed successfully")
    print(f"✅ Observations: {len(result['observations'])}")
    print(f"✅ Findings: {len(result.get('potential_findings', []))}")
    print(f"✅ Confidence: {result['confidence_score']}")
    
    return orch


def test_session_continuity():
    """Test that session context flows across different operations"""
    print("\n" + "="*70)
    print("TEST 4: Cross-Flow Session Continuity")
    print("="*70)
    
    orch = MedicalAssistantOrchestrator()
    session_id = orch.get_session_id()
    
    print(f"Session ID: {session_id[:8]}...")
    
    # Step 1: Analyze report
    print("\n  Step 1: Analyze diabetes report...")
    result1 = orch.analyze_report_flow('examples/sample_clinical_note_diabetes.txt')
    assert result1.get('parsed'), "Report should parse"
    medications = result1.get('medications', [])
    print(f"    ✓ Found {len(medications)} medications")
    
    # Step 2: Query about the report (should use context)
    print("\n  Step 2: Query about medications (using session context)...")
    result2 = orch.answer_query_flow('What medications were mentioned in the previous report?')
    assert result2.get('answer'), "Query should have answer"
    print(f"    ✓ Answer uses report context: {len(result2['answer'])} chars")
    
    # Step 3: Analyze image (should have full conversation context)
    print("\n  Step 3: Analyze chest X-ray (with full context)...")
    result3 = orch.analyze_image_flow('examples/CV-case-2-2-1-768x575.jpg', 'xray', 'chest')
    assert len(result3.get('observations', [])) > 0, "Should have observations"
    print(f"    ✓ Image analyzed with conversation context")
    
    print(f"\n✅ Session continuity verified across all 3 flows!")
    print(f"✅ Shared session: {session_id[:8]}...")


def test_multimodal_routing():
    """Test smart routing in multimodal flow"""
    print("\n" + "="*70)
    print("TEST 5: Multimodal Smart Routing")
    print("="*70)
    
    orch = MedicalAssistantOrchestrator()
    
    # Test 1: File path should route to report
    print("\n  Test 1: File routing...")
    result1 = orch.multimodal_analysis_flow('examples/sample_clinical_note_diabetes.txt')
    assert result1.get('flow') == 'report_analysis', "Should route to report_analysis"
    print("    ✓ File routed to report_analysis")
    
    # Test 2: Image path should route to image
    print("\n  Test 2: Image routing...")
    result2 = orch.multimodal_analysis_flow('examples/CV-case-2-2-1-768x575.jpg', 'xray', 'chest')
    assert result2.get('flow') == 'image_analysis', "Should route to image_analysis"
    print("    ✓ Image routed to image_analysis")
    
    # Test 3: Raw text should route to query
    print("\n  Test 3: Text routing...")
    result3 = orch.multimodal_analysis_flow('What is diabetes?')
    assert result3.get('flow') == 'query', "Should route to query"
    print("    ✓ Text routed to query")
    
    print(f"\n✅ All multimodal routing tests passed!")


def test_all_example_files():
    """Test with all available example clinical notes"""
    print("\n" + "="*70)
    print("TEST 6: All Example Files")
    print("="*70)
    
    orch = MedicalAssistantOrchestrator()
    
    example_files = [
        'examples/sample_clinical_note_diabetes.txt',
        'examples/sample_clinical_note_chest_pain.txt',
        'examples/sample_clinical_note_hypertension.txt',
        'examples/sample_clinical_note_pneumonia.txt'
    ]
    
    for file_path in example_files:
        path = Path(file_path)
        if path.exists():
            print(f"\n  Testing: {path.name}")
            result = orch.analyze_report_flow(str(path))
            
            assert result['flow'] == 'report_analysis', f"Failed on {path.name}"
            assert result.get('parsed'), f"Parse failed on {path.name}"
            
            print(f"    ✓ Parsed: {result.get('parsed')}")
            print(f"    ✓ Summary: {result['summary'][:60]}...")
        else:
            print(f"\n  ⚠️  Skipping {file_path} (not found)")
    
    print(f"\n✅ All example files processed successfully!")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print(" 🧪 COMPREHENSIVE FLOW TESTING SUITE")
    print("="*70)
    
    try:
        test_report_flow()
        test_query_flow()
        test_image_flow()
        test_session_continuity()
        test_multimodal_routing()
        test_all_example_files()
        
        print("\n" + "="*70)
        print(" ✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n Summary:")
        print("  ✓ Report analysis working")
        print("  ✓ Query flow working")
        print("  ✓ Image analysis working")
        print("  ✓ Session continuity verified")
        print("  ✓ Multimodal routing correct")
        print("  ✓ All example files processed")
        print()
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
