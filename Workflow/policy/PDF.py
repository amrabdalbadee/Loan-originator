import os
import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_RIGHT

def create_expanded_arabic_pdf(output_filename):
    # 1. Register an Arabic-supporting font.
    # IMPORTANT: Update this path if you are not on Windows (e.g., to a Mac/Linux Arabic font path).
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Amiri-Regular.ttf")
    pdfmetrics.registerFont(TTFont('ArabicFont', font_path))

    # 2. Setup the document
    doc = SimpleDocTemplate(output_filename, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Create custom styles for RTL Arabic text
    arabic_style = ParagraphStyle(
        'Arabic',
        parent=styles['Normal'],
        fontName='ArabicFont',
        fontSize=10,
        leading=16, # Line spacing
        alignment=TA_RIGHT, # Align text to the right
        wordWrap='LTR'
    )
    
    subtitle_style = ParagraphStyle(
        'ArabicSubtitle',
        parent=arabic_style,
        fontSize=12,
        spaceAfter=5,
        spaceBefore=10,
        textColor='#003366' # A dark blue typical for banks
    )

    title_style = ParagraphStyle(
        'ArabicTitle',
        parent=arabic_style,
        fontSize=16,
        spaceAfter=15,
        alignment=TA_RIGHT,
        textColor='#003366'
    )

    story = []

    # 3. The Expanded Contract Content
    content = [
        ("NATIONAL BANK OF EGYPT - البنك الأهلي المصري", title_style),
        ("شروط وأحكام التمويل الشخصي النقدي", title_style),
        
        ("أولاً: المستندات المطلوبة للتقديم", subtitle_style),
        ("لكي يتم دراسة طلب التمويل الشخصي، يلتزم العميل بتقديم المستندات التالية:", arabic_style),
        ("- إثبات دخل حديث: شهادة مفردات مرتب معتمدة من جهة العمل، أو كشف حساب بنكي يوضح تحويل الراتب.", arabic_style),
        ("- إيصال مرافق حديث: إيصال كهرباء، غاز، أو مياه لم يمر عليه أكثر من 3 أشهر لإثبات محل الإقامة الحالي.", arabic_style),
        
        ("ثانياً: مبلغ التمويل ومدة السداد", subtitle_style),
        ("يمنح البنك تمويلا شخصيا للعميل بالمبلغ الذي سيتم الاتفاق عليه، على أن يتم إيداعه في حسابه الجاري الدائن طرف البنك. مدة التمويل تبدأ من تاريخ منح التمويل وتنتهي في تاريخ استحقاق آخر قسط. يختار العميل برنامج السداد الذي يناسبه من بين أنظمة السداد المتاحة ويلتزم بسداد أصل التمويل والعوائد والعمولات والمصروفات والملحقات وفقا لبرنامج السداد المختار.", arabic_style),
        
        ("ثالثاً: العوائد والعمولات وعوائد التأخير والمصروفات", subtitle_style),
        ("يسرى على مبلغ التمويل أو الجزء الباقي منه عائد سنويا وفقا والمعلن داخل البنك ويحتسب العائد على الرصيد المدين القائم. تسري غرامة تأخير شهريا بالإضافة إلى عوائد تأخير وتحتسب يوم بيوم من تاريخ التأخير وحتى تمام السداد وذلك على كل مبلغ يستحق ولا يسدد في موعده. يحق للبنك تعديل سعر العائد المحتسب في ضوء السياسة التي يحددها البنك.", arabic_style),

        ("رابعاً: الوفاء المعجل للتمويل أو جزء منه", subtitle_style),
        ("إذا ما رغب العميل في الوفاء المعجل لجزء من أو لكامل الرصيد المدين للتمويل، يسدد العميل للبنك العوائد المستحقة حتى تاريخ الوفاء المعجل بالإضافة لعمولة الوفاء المعجل وفقا والمعلن داخل البنك، وسوف يقوم البنك بتحميل العميل بأية تكاليف أخرى قد تستحق للبنك نتيجة الوفاء المعجل للتمويل.", arabic_style),
        
        ("خامساً: التزامات العميل وحق الخصم من الحساب", subtitle_style),
        ("تأمينا وضمانا لسداد كامل مبلغ التمويل، يقر العميل إقرارا نهائيا لا رجوع فيه بقبول استمرار تحويل الراتب الشهري / القسط الشهري للقرض خصما من راتبه طرف جهة عمله. ويفوض البنك بالخصم المباشر من حسابه أو من بطاقات الدفع الإلكترونية لسداد الأقساط. وفي حالة الخروج من العمل أو الاستقالة، يتم تحويل مكافأة نهاية الخدمة مباشرة للبنك لسداد الجزء المتبقي.", arabic_style),

        ("سادساً: التأمين", subtitle_style),
        ("يوافق العميل على قيام البنك بإصدار وثيقة تأمين على الحياة لصالح البنك وحده تستحق في حالة الوفاة أو العجز الكلى طبقا للشروط المطبقة بشركة التأمين التي يحددها البنك وتستخدم أية مستحقات لتسوية الرصيد المتبقي.", arabic_style),
        
        ("سابعاً: حالات الإخلال (I-Score)", subtitle_style),
        ("يصبح أصل التمويل وعوائده والعمولات مستحقة السداد فورا دون تنبيه أو حكم قضائي في حال عدم سداد أي قسط في مواعيد الاستحقاق المبينة بالعقد، أو إذا أخل العميل بأي شرط، أو إذا أشهر إفلاسه، أو فقد وظيفته. من المعلوم لدى العميل أنه سيتم إدراجه بالقوائم السلبية (I-SCORE) في حالة التوقف عن السداد لمدة تزيد عن ١٨٠ يوم بعد فترة السماح، أو اتخاذ إجراءات قضائية ضده.", arabic_style),

        ("ثامناً: دمج وتوحيد الحسابات والمقاصة", subtitle_style),
        ("تعتبر جميع الحسابات المفتوحة أو التي قد تفتح مستقبلا باسم العميل لدى البنك وفروعه وحدة واحدة لا تتجزأ ضمانا وتأمينا لسداد المديونية. ويحق للبنك في أي وقت يشاء أن يدمج أو يوحد هذه الحسابات كلها أو بعضها وإجراء المقاصة بينها لاستيفاء حقوقه.", arabic_style),

        ("تاسعاً: كشوف وسرية الحسابات وحماية العملاء", subtitle_style),
        ("يتعهد البنك بحماية بيانات ومعلومات العملاء حيث تعتبر معلومات سرية ولا يجوز مشاركتها مع الغير دون موافقة كتابية مسبقة، باستثناء ما ينص عليه القانون لغرض استيفاء مستحقات البنك. يقر العميل بأن عدم اعتراضه على كشوف الحسابات خلال ثلاثين يوما يعتبر موافقة نهائية منه.", arabic_style),

        ("عاشراً: القانون الواجب التطبيق والاختصاص القضائي", subtitle_style),
        ("يخضع عقد التمويل لأحكام القانون المصري وكل نزاع ينشأ بخصوص تنفيذ أو تطبيق أي شرط من شروط العقد يكون الفصل فيها من اختصاص محاكم القاهرة على اختلاف أنواعها ودرجاتها أو أية محكمة أخرى يختارها البنك.", arabic_style),

        ("حادي عشر: معايير قبول التمويل (السياسة الائتمانية)", subtitle_style),
        ("يتم دراسة وتحديد القرار الائتماني لطلبات التمويل بناءً على المعايير والسياسات التالية:", arabic_style),
        ("- الحد الأقصى لنسبة عبء الدين (DBR): يجب ألا يتجاوز إجمالي الأقساط الشهرية بعد المنح 50% من الدخل الشهري.", arabic_style),
        ("- الحد الأدنى لصافي الدخل: يجب أن يتبقى بحد أدنى 5000 جنيه مصري من الدخل بعد خصم العبء الشهري وكافة المصاريف الأساسية.", arabic_style),
        ("- الحد الأقصى لنسبة الدين للدخل (DTI): يجب ألا يتجاوز إجمالي الدين القائم 50% من إجمالي الدخل السنوي للعميل.", arabic_style),
        ("- الاستقرار الوظيفي: يشترط أن يكون الحد الأدنى لسنوات الخبرة أو مدة العمل سنة واحدة على الأقل.", arabic_style),
        ("- الحد الأقصى لمدة القرض: يجب ألا تتجاوز فترة سداد التمويل 84 شهراً كحد أقصى.", arabic_style),
    ]

    # 4. Process and format the Arabic text
    for text, style in content:
        # Reshape to connect Arabic letters properly
        reshaped_text = arabic_reshaper.reshape(text)
        # Apply BiDi algorithm to fix RTL direction
        bidi_text = get_display(reshaped_text)
        
        # Add to story with styling
        p = Paragraph(bidi_text, style)
        story.append(p)
        
        # Add spacing (less space after bullet points/regular text, more after subtitles)
        if style.name == 'ArabicSubtitle':
            story.append(Spacer(1, 4))
        else:
            story.append(Spacer(1, 8))

    # 5. Build the PDF
    doc.build(story)
    print(f"PDF successfully generated: {output_filename}")

if __name__ == "__main__":
    create_expanded_arabic_pdf("NBE_Expanded_Loan_Contract.pdf")