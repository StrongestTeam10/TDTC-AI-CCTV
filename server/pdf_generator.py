"""
server/pdf_generator.py - ReportLab 기반 긴급 알람 PDF 명세서 자동 생성 모듈
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    reportlab_available = True
except ImportError:
    reportlab_available = False


def generate_emergency_pdf(
    output_pdf_path: str,
    zone_id: int,
    alert_type: str,
    cri_score: float,
    pedestrian_count: int,
    occupancy_rate: float,
    incident_summary: Optional[str] = None,
    stagnation_sec: float = 0.0,
    snapshot_image_path: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None
) -> str:
    """
    긴급 상황 발생 시 정형 사고 명세서 PDF 파일을 생성합니다.
    (ReportLab 서식화 PDF 또는 텍스트 fallback)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 기본 관제 명세서 텍스트 템플릿
    if not incident_summary:
        incident_summary = (
            f"망원시장 Zone {zone_id} 구역에서 인파 밀집도 지표(CRI: {cri_score:.1f}pt, 보행자: {pedestrian_count}명, "
            f"점유율: {occupancy_rate:.1f}%, 평균 정체: {stagnation_sec:.1f}초)가 위험 기준치를 초과하여 "
            f"현장 관제실 비상 출동 및 안전 유도 조치가 발령되었습니다."
        )

    if not reportlab_available:
        print("[PDF Generator] reportlab 미설치 - 텍스트 기반 fallback 문서 생성")
        with open(output_pdf_path, "w", encoding="utf-8") as f:
            f.write(f"=== TDTC SMART CCTV EMERGENCY REPORT ===\n")
            f.write(f"발생 일시: {now_str}\n")
            f.write(f"관제 영역: Zone {zone_id}\n")
            f.write(f"위험 유형: {alert_type}\n")
            f.write(f"CRI 위험도: {cri_score:.1f} pt\n")
            f.write(f"보행자 수: {pedestrian_count} 명\n")
            f.write(f"점유율: {occupancy_rate:.1f}%\n")
            f.write(f"정체 시간: {stagnation_sec:.1f} 초\n")
            f.write(f"상황 요약: {incident_summary}\n")
            f.write(f"========================================\n")
        return output_pdf_path

    try:
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#D32F2F'),
            spaceAfter=15
        )
        
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#212121')
        )
        
        summary_style = ParagraphStyle(
            'SummaryStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor('#0D47A1')
        )

        elements = []

        # 1. 헤더 타이틀
        elements.append(Paragraph("🚨 TDTC SMART CCTV EMERGENCY REPORT", title_style))
        elements.append(Paragraph(f"<b>Report Generation Time:</b> {now_str}", body_style))
        elements.append(Spacer(1, 15))

        # 2. 기본 정보 및 위험 지표 표
        table_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Zone ID", f"Zone {zone_id}", "Alert Type", str(alert_type)],
            ["CRI Score", f"{cri_score:.1f} pt", "Risk Level", "CRITICAL" if cri_score >= 70 else "WARNING"],
            ["Pedestrian Count", f"{pedestrian_count} persons", "Occupancy Rate", f"{occupancy_rate:.1f}%"],
            ["Avg Stagnation", f"{stagnation_sec:.1f} sec", "Action Status", "DISPATCH_CONFIRMED"]
        ]

        t = Table(table_data, colWidths=[120, 140, 120, 140])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474F')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#B0BEC5')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FAFAFA'), colors.HexColor('#ECEFF1')])
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

        # 3. 상황 요약 섹션
        elements.append(Paragraph("<b>[ Incident Description & Summary ]</b>", body_style))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(incident_summary, summary_style))
        elements.append(Spacer(1, 15))

        # 4. 캡처 이미지 (스냅샷이 있는 경우)
        if snapshot_image_path and os.path.exists(snapshot_image_path):
            try:
                elements.append(Paragraph("<b>[ CCTV Snapshot at Incident ]</b>", body_style))
                elements.append(Spacer(1, 5))
                img = RLImage(snapshot_image_path, width=480, height=270)
                elements.append(img)
                elements.append(Spacer(1, 10))
            except Exception as img_err:
                print(f"[PDF Generator Warning] 이미지 첨부 실패: {img_err}")

        # PDF 빌드 실행
        doc.build(elements)
        print(f"[PDF Generator] PDF 명세서 생성 완료: {output_pdf_path}")
        return output_pdf_path

    except Exception as e:
        print(f"[PDF Generator Error] PDF 생성 실패 ({e}) - 텍스트 파일로 fallback")
        with open(output_pdf_path, "w", encoding="utf-8") as f:
            f.write(f"=== TDTC SMART CCTV EMERGENCY REPORT ===\n")
            f.write(f"발생 일시: {now_str}\n")
            f.write(f"관제 영역: Zone {zone_id}\n")
            f.write(f"상황 요약: {incident_summary}\n")
        return output_pdf_path
