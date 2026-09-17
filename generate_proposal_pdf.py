#!/usr/bin/env python3
"""Generate a human-style HandsToVoice Project Proposal PDF."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 HRFlowable, PageBreak, Table, TableStyle)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

W, H = A4
NAVY   = colors.HexColor("#1A237E")
TEAL   = colors.HexColor("#00897B")
GREY   = colors.HexColor("#F5F5F5")
MGREY  = colors.HexColor("#9E9E9E")
BLACK  = colors.HexColor("#212121")
WHITE  = colors.white

def styles():
    return {
        "uni": ParagraphStyle("uni", fontSize=11, textColor=NAVY,
            fontName="Helvetica-Bold", alignment=TA_CENTER, leading=16),
        "dept": ParagraphStyle("dept", fontSize=10, textColor=NAVY,
            fontName="Helvetica", alignment=TA_CENTER, leading=14),
        "ptitle": ParagraphStyle("ptitle", fontSize=26, textColor=NAVY,
            fontName="Helvetica-Bold", alignment=TA_CENTER, leading=32, spaceAfter=6),
        "psub": ParagraphStyle("psub", fontSize=13, textColor=TEAL,
            fontName="Helvetica-BoldOblique", alignment=TA_CENTER, leading=18),
        "label": ParagraphStyle("label", fontSize=10, textColor=MGREY,
            fontName="Helvetica", alignment=TA_CENTER, leading=14),
        "value": ParagraphStyle("value", fontSize=11, textColor=BLACK,
            fontName="Helvetica-Bold", alignment=TA_CENTER, leading=15),
        "h1": ParagraphStyle("h1", fontSize=13, textColor=WHITE,
            fontName="Helvetica-Bold", leading=18),
        "h2": ParagraphStyle("h2", fontSize=11, textColor=NAVY,
            fontName="Helvetica-Bold", leading=16, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", fontSize=10.5, textColor=BLACK,
            fontName="Helvetica", leading=16, alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", fontSize=10.5, textColor=BLACK,
            fontName="Helvetica", leading=16, leftIndent=16, spaceAfter=4),
        "footer": ParagraphStyle("footer", fontSize=8.5, textColor=MGREY,
            fontName="Helvetica-Oblique", alignment=TA_CENTER, leading=12),
        "sig": ParagraphStyle("sig", fontSize=10, textColor=BLACK,
            fontName="Helvetica-Bold", alignment=TA_CENTER, leading=14),
    }

def sec(title, st):
    t = Table([[Paragraph(title, st["h1"])]], colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), TEAL),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ]))
    return t

def build(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="HandsToVoice Project Proposal",
        author="NTAWUKURIRYAYO Frederic & NYIRABACUMBITSI Melanie Celine")

    s = styles()
    story = []

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("University of Rwanda", s["uni"]),
        Paragraph("College of Science and Technology", s["uni"]),
        Paragraph("Department of Information and Communication Technology", s["dept"]),
        Spacer(1, 1*cm),
        HRFlowable(width="100%", thickness=2, color=NAVY),
        Spacer(1, 0.8*cm),
        Paragraph("HandsToVoice", s["ptitle"]),
        Paragraph("Real-time Kinyarwanda Sign Language Recognition &amp; Voice Conversion", s["psub"]),
        Spacer(1, 0.8*cm),
        HRFlowable(width="100%", thickness=2, color=NAVY),
        Spacer(1, 1.2*cm),
        Paragraph("Project Proposal", s["label"]),
        Spacer(1, 1.8*cm),
    ]

    # Team box
    team = Table([
        [Paragraph("Prepared by:", s["label"]), Paragraph("Submitted to:", s["label"])],
        [Paragraph("NTAWUKURIRYAYO Frederic", s["value"]),
         Paragraph("University of Rwanda", s["value"])],
        [Paragraph("ID: 26-BK-RE-076", s["label"]),
         Paragraph("College of Science and Technology", s["label"])],
        [Paragraph("NYIRABACUMBITSI Melanie Celine", s["value"]),
         Paragraph("Department of ICT", s["value"])],
        [Paragraph("ID: 26-BK-RE-055", s["label"]),
         Paragraph("April 30, 2026", s["label"])],
    ], colWidths=[8*cm, 8*cm])
    team.setStyle(TableStyle([
        ("BOX",        (0,0),(-1,-1), 1, NAVY),
        ("INNERGRID",  (0,0),(-1,-1), 0.5, colors.HexColor("#BDBDBD")),
        ("BACKGROUND", (0,0),(-1,-1), GREY),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
    ]))
    story.append(team)
    story.append(PageBreak())

    # ── 1. BASIC INFORMATION ─────────────────────────────────────────────────
    story += [sec("1.  Basic Information", s), Spacer(1,0.3*cm)]
    story.append(Paragraph(
        "The project described in this proposal is titled <b>HandsToVoice</b>. "
        "It is being developed as part of the undergraduate programme at the "
        "<b>University of Rwanda, College of Science and Technology</b>, "
        "within the <b>Department of Information and Communication Technology</b>. "
        "The project is a joint initiative by two students:", s["body"]))
    story += [
        Paragraph("• <b>NTAWUKURIRYAYO Frederic</b> — Team Lead &nbsp;&nbsp; (Student ID: 26-BK-RE-076)", s["bullet"]),
        Paragraph("• <b>NYIRABACUMBITSI Melanie Celine</b> — Team Member &nbsp;&nbsp; (Student ID: 26-BK-RE-055)", s["bullet"]),
        Spacer(1,0.3*cm),
    ]

    # ── 2. PROBLEM STATEMENT ─────────────────────────────────────────────────
    story += [sec("2.  Problem Statement", s), Spacer(1,0.3*cm)]
    story.append(Paragraph(
        "Communication is a fundamental human right, yet for many Rwandans living "
        "with hearing and speech disabilities, this right remains out of reach in "
        "everyday situations. According to the National Institute of Statistics of "
        "Rwanda (NISR, 2023), an estimated <b>72,000 people</b> in Rwanda have "
        "hearing or speech impairments. These individuals rely on "
        "<b>Kinyarwanda Sign Language (KSL)</b> as their primary means of expression. "
        "However, the general population — including teachers, doctors, police officers, "
        "and shopkeepers — has little to no knowledge of KSL, creating a daily barrier "
        "to communication.", s["body"]))

    story.append(Paragraph(
        "This communication gap has serious consequences across multiple sectors. "
        "In healthcare, deaf patients struggle to describe their symptoms to medical "
        "staff, which can lead to misdiagnosis and inadequate treatment. In education, "
        "students with hearing impairments are often excluded from mainstream classrooms "
        "because their teachers cannot understand their signed responses. In public life, "
        "accessing government services, employment, and even family conversations remains "
        "a daily challenge.", s["body"]))

    story.append(Paragraph(
        "The few professional KSL interpreters available in Rwanda are both scarce and "
        "expensive — fewer than 100 are certified nationwide. Furthermore, while "
        "sign language recognition technology does exist globally, all existing tools "
        "are built for American Sign Language (ASL) or British Sign Language (BSL) and "
        "are entirely unsuitable for the Rwandan context. There is currently "
        "<b>no software solution designed specifically for Kinyarwanda Sign Language</b>.",
        s["body"]))
    story.append(Spacer(1,0.3*cm))

    # ── 3. PROPOSED SOLUTION ─────────────────────────────────────────────────
    story += [sec("3.  Proposed Solution", s), Spacer(1,0.3*cm)]
    story.append(Paragraph(
        "Our proposed solution is <b>HandsToVoice</b> — a desktop software application "
        "that uses artificial intelligence and computer vision to recognise Kinyarwanda "
        "Sign Language in real time and convert it into spoken Kinyarwanda audio. "
        "The system requires nothing more than a standard laptop with a built-in or "
        "USB webcam, making it affordable and immediately accessible.", s["body"]))

    story.append(Paragraph(
        "The application works as follows: the user sits in front of a webcam and "
        "performs a sign. The system's hand-detection engine — powered by Google's "
        "MediaPipe framework — instantly identifies 21 key points on the hand and "
        "extracts 63 numerical features describing the shape and position of every "
        "finger joint. These features are then passed to a trained neural network "
        "that classifies the gesture into one of the known KSL signs. Once a sign is "
        "confirmed, the system looks up the corresponding Kinyarwanda word and speaks "
        "it aloud through the device's speakers using a text-to-speech engine. "
        "The entire process takes less than 100 milliseconds.", s["body"]))

    story.append(Paragraph(
        "The application has been built using Python 3.12, with TensorFlow/Keras "
        "for the neural network, OpenCV for camera handling, and PyQt5 for the "
        "graphical interface. It supports both an online mode (using Google TTS for "
        "high-quality voice output) and a fully offline mode (using pyttsx3), "
        "ensuring it works even without an internet connection.", s["body"]))
    story.append(Spacer(1,0.3*cm))

    # ── 4. KEY FEATURES ──────────────────────────────────────────────────────
    story += [sec("4.  Key Features", s), Spacer(1,0.3*cm)]
    features = [
        ("Real-Time Hand Detection",
         "The system uses MediaPipe Hand Landmarker to detect and track the user's "
         "hand at 30 frames per second. Detection is position-invariant, meaning the "
         "sign is recognised correctly regardless of where the hand appears on screen."),
        ("AI-Powered Sign Classification",
         "A deep neural network with three hidden layers and dropout regularisation "
         "classifies each hand pose. A consensus mechanism requires the same sign to "
         "appear in three consecutive frames before it is confirmed, preventing "
         "accidental triggers from brief, unintended gestures."),
        ("Kinyarwanda Voice Output",
         "Once a sign is recognised, the system immediately speaks the corresponding "
         "Kinyarwanda word aloud. Users can build up multi-word sentences by signing "
         "one word at a time and then press a button to hear the full sentence spoken."),
        ("Professional Graphical Interface",
         "The application features a clean, dark-themed interface showing the live "
         "camera feed, a confidence meter, the current detected sign, sentence history, "
         "and a status panel indicating whether the system is ready or currently "
         "processing a sign."),
        ("Expandable Training Pipeline",
         "The vocabulary is not fixed. A built-in data collection tool allows users "
         "to record new signs via webcam, and the model can be retrained with a single "
         "command. This means schools, hospitals, or communities can teach the system "
         "their own local signs over time."),
    ]
    for i, (title, body) in enumerate(features, 1):
        story.append(Paragraph(f"<b>{i}. {title}</b>", s["h2"]))
        story.append(Paragraph(body, s["body"]))
    story.append(Spacer(1,0.3*cm))

    # ── 5. VALUE PROPOSITION ─────────────────────────────────────────────────
    story += [sec("5.  Value Proposition", s), Spacer(1,0.3*cm)]
    story.append(Paragraph(
        "What sets HandsToVoice apart from anything else on the market is, "
        "first and foremost, the fact that it is <b>the only system in existence "
        "built specifically for Kinyarwanda Sign Language</b>. Every other sign "
        "language recognition tool in the world targets American, British, or "
        "European sign languages — none of which share gestures or vocabulary "
        "with KSL. We are building for Rwanda, from Rwanda.", s["body"]))
    story.append(Paragraph(
        "Beyond language, HandsToVoice stands out because it requires <b>no "
        "special hardware</b>. Unlike research-grade systems that use depth cameras "
        "or sensor-equipped gloves, our solution works on any ordinary laptop. "
        "This dramatically lowers the barrier to adoption, particularly in schools "
        "and clinics where budgets are limited.", s["body"]))
    story.append(Paragraph(
        "The system is also <b>fully functional offline</b>, which matters enormously "
        "in Rwanda where reliable internet connectivity is not yet universal. "
        "Finally, the modular and open design means that the vocabulary can grow "
        "continuously — driven by the community rather than constrained by a fixed "
        "product release.", s["body"]))
    story.append(Spacer(1,0.3*cm))

    # ── 6. BUSINESS MODEL ────────────────────────────────────────────────────
    story += [sec("6.  Business Model", s), Spacer(1,0.3*cm)]
    story.append(Paragraph(
        "HandsToVoice is primarily a social-impact product, but it is designed to "
        "be financially sustainable through several revenue streams.", s["body"]))
    biz = [
        ("Institutional Licensing",
         "Schools for the deaf, public hospitals, and government district offices "
         "would pay an annual software licence fee. Given that a single human "
         "interpreter costs several times more annually, the licence represents "
         "strong value for money."),
        ("NGO and Development Partner Funding",
         "Organisations such as Handicap International, NUDOR, and the Rwanda Union "
         "of the Deaf would be approached for subsidised deployment grants and "
         "partnership funding."),
        ("Government Contracts",
         "Integration into national accessibility programmes run by MINALOC, the "
         "Ministry of Health, and the Ministry of Education represents a significant "
         "long-term revenue opportunity."),
        ("Training and Capacity Building",
         "Workshops and onboarding services for institutions adopting the system "
         "provide a recurring service-based income stream."),
        ("Freemium Mobile Application",
         "A future Android version of HandsToVoice could be offered free at the "
         "basic level, with advanced features available through a small subscription, "
         "targeting individual deaf users and their families."),
    ]
    for title, body in biz:
        story.append(Paragraph(f"<b>{title}</b>", s["h2"]))
        story.append(Paragraph(body, s["body"]))
    story.append(Spacer(1,0.3*cm))

    # ── 7. SOCIAL IMPACT ─────────────────────────────────────────────────────
    story += [sec("7.  Social Impact", s), Spacer(1,0.3*cm)]
    story.append(Paragraph(
        "The impact of HandsToVoice, if deployed at scale, would be felt across "
        "multiple dimensions of Rwandan society. Most directly, it gives a voice "
        "to tens of thousands of people who currently struggle to be heard. "
        "A deaf patient can communicate their symptoms to a nurse. A hearing-impaired "
        "student can answer a teacher's question. A deaf job-seeker can participate "
        "in an interview. These are not small conveniences — they are transformative "
        "changes in a person's quality of life.", s["body"]))
    story.append(Paragraph(
        "Beyond individual interactions, the system contributes to building an "
        "inclusive society. When a hospital adopts HandsToVoice, it signals to "
        "the deaf community that they are valued patients. When a school uses it, "
        "it normalises the presence of sign language alongside spoken language. "
        "This visibility matters enormously for cultural acceptance and for the "
        "self-confidence of people with disabilities.", s["body"]))
    story.append(Paragraph(
        "At the national level, Rwanda has committed to inclusive development "
        "through Vision 2050 and the National Policy for Persons with Disabilities. "
        "HandsToVoice is a concrete, technology-driven step towards meeting these "
        "commitments. It also positions Rwanda as a pioneer in African assistive "
        "technology — a model that other countries with their own sign languages "
        "could replicate and adapt.", s["body"]))
    story.append(Paragraph(
        "The project aligns directly with three UN Sustainable Development Goals: "
        "<b>SDG 3</b> (Good Health and Well-Being), <b>SDG 4</b> (Quality Education), "
        "and <b>SDG 10</b> (Reduced Inequalities).", s["body"]))
    story.append(Spacer(1,0.3*cm))

    # ── 8. MVP ───────────────────────────────────────────────────────────────
    story += [sec("8.  Minimum Viable Product (MVP)", s), Spacer(1,0.3*cm)]
    story.append(Paragraph(
        "The MVP we have built is a fully working desktop application that "
        "demonstrates the complete pipeline from sign to speech. Every major "
        "component is implemented and operational:", s["body"]))
    mvp_items = [
        "Hand detection using MediaPipe at 30 FPS with real-time landmark overlay on the camera feed.",
        "A trained neural network that currently recognises <b>12 KSL signs</b> — greetings, common phrases, emergency vocabulary, an alphabet letter, and a body-part sign.",
        "Kinyarwanda text-to-speech output in both online and offline modes.",
        "A professional dark-themed graphical interface with live camera, confidence meter, sentence builder, and signing-state indicator.",
        "A data collection tool and model training pipeline for expanding the vocabulary.",
    ]
    for item in mvp_items:
        story.append(Paragraph(f"• {item}", s["bullet"]))

    story.append(Spacer(1,0.2*cm))
    story.append(Paragraph(
        "The 12 signs currently supported are: <b>Muraho</b> (Hello), "
        "<b>Amakuru</b> (How are you?), <b>Yego</b> (Yes), <b>Oya</b> (No), "
        "<b>Murakoze</b> (Thank you), <b>Mbabarira</b> (Sorry), "
        "<b>Ndagukunda</b> (I love you), <b>Amazi</b> (Water), "
        "<b>Ubufasha</b> (Help), <b>Muganga</b> (Doctor), "
        "<b>Letter C</b>, and <b>Amaso</b> (Eyes).", s["body"]))

    story.append(Paragraph("<b>What we will demonstrate:</b>", s["h2"]))
    story.append(Paragraph(
        "During the demo, one team member will sit in front of the webcam and "
        "perform a sequence of KSL signs. The audience will see the system "
        "identify each sign in real time on the screen, display the Kinyarwanda "
        "translation, and speak it aloud through the speakers. We will then "
        "show sentence building — signing several words in a row and pressing "
        "Speak to hear the full sentence. Finally, we will demonstrate the "
        "training pipeline by adding a brand-new sign live, showing that the "
        "system can learn and grow on the spot.", s["body"]))

    # ── SIGNATURE BLOCK ──────────────────────────────────────────────────────
    story += [
        Spacer(1, 1*cm),
        HRFlowable(width="100%", thickness=1, color=TEAL),
        Spacer(1, 0.3*cm),
        Paragraph(
            "Submitted by: &nbsp;&nbsp;"
            "<b>NTAWUKURIRYAYO Frederic</b> (26-BK-RE-076) &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>NYIRABACUMBITSI Melanie Celine</b> (26-BK-RE-055)", s["sig"]),
        Spacer(1, 0.1*cm),
        Paragraph(
            "University of Rwanda — College of Science and Technology &nbsp;|&nbsp; April 2026",
            s["footer"]),
    ]

    doc.build(story)
    print("✅  PDF saved →", path)

if __name__ == "__main__":
    build("HandsToVoice_Project_Proposal.pdf")
