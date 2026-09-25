import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class PDFReportGenerator:
    """
    Generates academic & audit PDF reports using ReportLab.
    """
    @staticmethod
    def generate_job_report(
        job_id: int,
        project_name: str,
        video_name: str,
        analytics_summary: dict,
        violations: list,
        output_path: str
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.HexColor('#4F46E5'),
            spaceAfter=12
        )
        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=8,
            spaceBefore=12
        )
        normal_style = styles['Normal']

        # Header
        story.append(Paragraph("VisionRadar — Traffic Intelligence Report", title_style))
        story.append(Paragraph(f"<b>Project:</b> {project_name} | <b>Video:</b> {video_name} | <b>Job ID:</b> #{job_id}", normal_style))
        story.append(Spacer(1, 14))

        # Analytics KPI Table
        story.append(Paragraph("Traffic Analytics Summary", heading_style))
        kpi_data = [
            ["Metric", "Value"],
            ["Total Vehicles Processed", str(analytics_summary.get("total_vehicles", 0))],
            ["Mean Speed (km/h)", f"{analytics_summary.get('mean_speed_kmh', 0.0)} km/h"],
            ["Median Speed (km/h)", f"{analytics_summary.get('median_speed_kmh', 0.0)} km/h"],
            ["P85 Speed (85th Percentile)", f"{analytics_summary.get('p85_speed_kmh', 0.0)} km/h"],
            ["Max Speed Observed", f"{analytics_summary.get('max_speed_kmh', 0.0)} km/h"]
        ]
        t = Table(kpi_data, colWidths=[250, 250])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4F46E5')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0'))
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        # Violations Table
        story.append(Paragraph("Candidate Speed Violations", heading_style))
        viol_data = [["Track ID", "Class", "Est Speed", "Limit", "Uncertainty", "Status"]]
        for v in violations[:20]:
            viol_data.append([
                f"#{v.get('track_id')}",
                v.get('vehicle_class', 'Car'),
                f"{v.get('estimated_speed_kmh')} km/h",
                f"{v.get('speed_limit_kmh')} km/h",
                f"±{v.get('uncertainty_kmh')} km/h",
                v.get('review_status', 'PENDING')
            ])

        if len(viol_data) == 1:
            viol_data.append(["N/A", "N/A", "No violations recorded", "N/A", "N/A", "N/A"])

        t_v = Table(viol_data, colWidths=[60, 70, 90, 80, 100, 100])
        t_v.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1'))
        ]))
        story.append(t_v)

        doc.build(story)
        return output_path
