import os
import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A5, A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def process_arabic(text):
    """Helper function to reshape and align Arabic text"""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def create_failing_utility_bill():
    # Register Font (Update path if on Mac/Linux)
    font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dummy_policy", "Amiri-Regular.ttf")
    pdfmetrics.registerFont(TTFont('ArabicFont', font_path))
    
    doc = SimpleDocTemplate("Failing_Utility_Bill.pdf", pagesize=A5, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    arabic_style = ParagraphStyle('Arabic', fontName='ArabicFont', fontSize=12, leading=18, alignment=2) # 2 is Right alignment
    title_style = ParagraphStyle('Title', fontName='ArabicFont', fontSize=16, leading=22, alignment=1) # 1 is Center alignment
    
    story = []
    
    # Bill Content - INTENTIONALLY OLD DATE TO FAIL THE 90-DAY RULE
    content = [
        ("شركة جنوب القاهرة لتوزيع الكهرباء", title_style),
        ("إيصال استهلاك كهرباء", title_style),
        ("--------------------------------------------------", title_style),
        ("تاريخ الإصدار: 15 نوفمبر 2025", arabic_style), # FAILS HERE (> 90 days)
        ("رقم اللوحة: 456789123", arabic_style),
        ("اسم المشترك: أحمد محمود سالم", arabic_style),
        ("العنوان: ٤٥ شارع مصطفى النحاس، مدينة نصر، القاهرة", arabic_style),
        ("الاستهلاك: 350 كيلو وات", arabic_style),
        ("المبلغ المطلوب سداده: 450 جنيه مصري", arabic_style),
        ("--------------------------------------------------", title_style),
        ("تنبيه: يفصل التيار في حالة عدم السداد", arabic_style)
    ]
    
    for text, style in content:
        story.append(Paragraph(process_arabic(text), style))
        story.append(Spacer(1, 10))
        
    doc.build(story)
    print("Failing Utility Bill generated: Failing_Utility_Bill.pdf")

def create_failing_income_proof():
    font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dummy_policy", "Amiri-Regular.ttf")
    pdfmetrics.registerFont(TTFont('ArabicFont', font_path))
    
    doc = SimpleDocTemplate("Failing_Income_Proof.pdf", pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    arabic_style = ParagraphStyle('Arabic', fontName='ArabicFont', fontSize=14, leading=24, alignment=2)
    title_style = ParagraphStyle('Title', fontName='ArabicFont', fontSize=18, leading=28, alignment=1)
    
    story = []
    
    # HR Letter Content - INTENTIONALLY LOW INCOME TO FAIL DTI/DBR CALCULATION
    content = [
        ("شركة أوراكل للبرمجيات", title_style),
        ("إدارة الموارد البشرية", title_style),
        ("شهادة إثبات دخل (مفردات مرتب)", title_style),
        ("----------------------------------------------------------------", title_style),
        ("التاريخ: 05 أبريل 2026", arabic_style),
        ("تشهد الشركة بأن السيد / أحمد محمود سالم", arabic_style),
        ("يعمل لدينا بوظيفة: مهندس دعم فني", arabic_style),
        ("وذلك منذ تاريخ: 15 مايو 2019", arabic_style),
        ("وفيما يلي تفاصيل راتبه الشهري:", arabic_style),
        ("- الراتب الأساسي: 4,500 جنيه مصري", arabic_style),
        ("- البدلات والحوافز: 500 جنيه مصري", arabic_style),
        ("- الاستقطاعات والضرائب: 1,000 جنيه مصري", arabic_style),
        ("- صافي الدخل الشهري: 4,000 جنيه مصري", arabic_style), # FAILS HERE (Too low for DTI < 50%)
        ("----------------------------------------------------------------", title_style),
        ("وقد أعطيت له هذه الشهادة لتقديمها إلى البنك الأهلي المصري، دون أدنى مسؤولية على الشركة.", arabic_style),
        ("مدير شؤون العاملين:", arabic_style),
        ("التوقيع: ____________________", arabic_style),
        ("ختم الشركة:", arabic_style)
    ]
    
    for text, style in content:
        story.append(Paragraph(process_arabic(text), style))
        story.append(Spacer(1, 12))
        
    doc.build(story)
    print("Failing Income Proof generated: Failing_Income_Proof.pdf")

if __name__ == "__main__":
    create_failing_utility_bill()
    create_failing_income_proof()