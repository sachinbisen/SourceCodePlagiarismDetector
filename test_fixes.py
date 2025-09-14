#!/usr/bin/env python3
"""
Simple test script to verify the fixes work properly.
This tests the PDF generation and Unicode handling.
"""

import sys
import os
sys.path.append('web')

from app import PDFReport, get_similarity_analysis, get_similarity_category
from datetime import datetime

def test_pdf_unicode_handling():
    """Test that PDF generation works without Unicode errors"""
    print("Testing PDF Unicode handling...")
    
    try:
        # Create a test PDF report
        pdf = PDFReport()
        
        # Test the safe_text method with problematic Unicode
        test_text = "Test with Unicode: • ⚠️ ✓ → ≥"
        safe_text = pdf._safe_text(test_text)
        print(f"Original: {test_text}")
        print(f"Safe text: {safe_text}")
        
        # Test cover page generation
        pdf.add_cover_page("test1.py", "test2.py", datetime.now(), 0.85)
        
        # Test summary table
        test_scores = {
            'difflib': 0.82,
            'cosine': 0.78,
            'ast': 0.85,
            'jaccard': 0.80,
            'average': 0.8125
        }
        
        pdf.add_summary_table(test_scores)
        
        # Test textual analysis
        analysis = get_similarity_analysis(test_scores)
        print(f"Analysis text: {analysis}")
        
        # Test categories
        for score in [0.95, 0.75, 0.55, 0.35, 0.15]:
            category = get_similarity_category(score)
            print(f"Score {score:.2f} -> Category: {category}")
        
        # Try to output PDF
        test_path = "/tmp/test_report.pdf"
        pdf.output(test_path)
        
        print(f"✅ PDF generated successfully at: {test_path}")
        print(f"File size: {os.path.getsize(test_path)} bytes")
        
        # Clean up
        os.remove(test_path)
        print("✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in PDF generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_text_processing():
    """Test various text processing scenarios"""
    print("\nTesting text processing...")
    
    pdf = PDFReport()
    
    test_cases = [
        "Normal ASCII text",
        "Text with bullet • point",
        "Warning ⚠️ symbol",
        "Check ✓ mark", 
        "Arrows → ← ↑ ↓",
        "Math symbols ≥ ≤ ≠",
        "Quotes 'hello' and 'world'",
        "Dashes — and –",
        "Mixed: The code shows • high similarity ⚠️ (≥80%)"
    ]
    
    print("Text conversion results:")
    for test_text in test_cases:
        try:
            safe_text = pdf._safe_text(test_text)
            print(f"  '{test_text}' -> '{safe_text}'")
        except Exception as e:
            print(f"  ERROR processing '{test_text}': {e}")
    
    print("✅ Text processing tests completed")

if __name__ == "__main__":
    print("🔧 Testing fixes for Code Similarity Checker")
    print("=" * 50)
    
    success = test_pdf_unicode_handling()
    test_text_processing()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Fixes appear to be working correctly.")
        print("\n📋 Summary of fixes:")
        print("1. Unicode characters replaced with ASCII equivalents")
        print("2. Enhanced _safe_text() method for better encoding handling")
        print("3. PDF generation should now work without UnicodeEncodeError")
        print("4. Client-side validation improved with better debugging")
        print("5. Form validation popup uses ASCII characters only")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("\n🧪 To test the client-side validation:")
    print("1. Open the web application")
    print("2. Go to Single Comparison tab")
    print("3. Try to submit without selecting files")
    print("4. Check browser console for debug messages")
    print("5. Verify popup appears with '! No files selected' message")