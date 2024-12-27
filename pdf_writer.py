from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.graphics.shapes import Drawing, Line
import re

def generate_pdf(data):
    """Generate a PDF file from the json essay data"""
    title = re.sub(r'[^a-zA-Z0-9]', '_', data['header'])
    filename = title + '.pdf'
    pdf = SimpleDocTemplate(filename, pagesize=A4, title=title)

    styles = getSampleStyleSheet()
    story = []
    line = Drawing(450, 1)
    line.add(Line(0, 0, 450, 0))

    header = Paragraph(data['header'], styles['Title'])
    story.append(header)
    story.append(Spacer(1, 6))

    entry = Paragraph(data['entry'], styles['BodyText'])
    story.append(entry)
    story.append(Spacer(1, 6))

    for para in data['paragraphs']:
        sub_header = Paragraph(para['sub_header'], styles['Heading2'])
        paragraph = Paragraph(para['paragraph'], styles['BodyText'])

        story.append(sub_header)
        story.append(Spacer(1, 3))
        story.append(paragraph)
        story.append(Spacer(1, 6))

    conclusion_title = Paragraph('Conclusion', styles['Heading2'])
    conclusion = Paragraph(data['conclusion'], styles['BodyText'])
    story.append(conclusion_title)
    story.append(conclusion)
    story.append(Spacer(1, 6))

    story.append(line)
    story.append(Spacer(1, 3))
    seo_keywords = Paragraph("Seo Keywords: "+", ".join( data['seo_keywords']), styles['BodyText'])
    story.append(seo_keywords)
    pdf.build(story)

    return filename