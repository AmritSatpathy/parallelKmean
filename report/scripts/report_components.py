"""Shared Word layout components for the canonical Project 2 report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

NUS_BLUE = "003D7C"
TEXT = "171717"
MUTED = "626A73"
PALE = "EFEFEF"
PALE_BLUE = "EAF1F5"


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    node = tc_pr.find(qn("w:shd"))
    if node is None:
        node = OxmlElement("w:shd")
        tc_pr.append(node)
    node.set(qn("w:fill"), fill)


def margins(cell, top=65, start=80, bottom=65, end=80) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def _no_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def set_widths(table, widths_cm) -> None:
    table.autofit = False
    total = int(sum(widths_cm) * 567)
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_cm:
        node = OxmlElement("w:gridCol")
        node.set(qn("w:w"), str(int(width * 567)))
        grid.append(node)
    for row in table.rows:
        for cell, width in zip(row.cells, widths_cm):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 567)))
            tc_w.set(qn("w:type"), "dxa")


def hyperlink(paragraph, text: str, url: str) -> None:
    rid = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "4F81BD")
    rpr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(underline)
    run.append(rpr)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    link.append(run)
    paragraph._p.append(link)


def _page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run("Project report  •  ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def table(doc, headers, rows, widths, size=8.0):
    result = doc.add_table(rows=1, cols=len(headers))
    result.style = "Table Grid"
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_widths(result, widths)
    _repeat_header(result.rows[0])
    for index, value in enumerate(headers):
        cell = result.rows[0].cells[index]
        shade(cell, PALE)
        margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(str(value))
        run.bold = True
        run.font.size = Pt(size)
    for values in rows:
        row = result.add_row()
        _no_split(row)
        for index, value in enumerate(values):
            cell = row.cells[index]
            margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.add_run(str(value)).font.size = Pt(size)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return result


def callout(doc, label: str, text: str) -> None:
    result = doc.add_table(rows=1, cols=1)
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    _no_split(result.rows[0])
    set_widths(result, [16.1])
    cell = result.cell(0, 0)
    shade(cell, PALE_BLUE)
    margins(cell, 95, 125, 95, 125)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(label + "  ")
    run.bold = True
    run.font.color.rgb = RGBColor.from_string(NUS_BLUE)
    paragraph.add_run(text)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(doc, path: Path, caption_text: str, width=6.55) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    caption = doc.add_paragraph(style="Caption")
    caption.add_run(caption_text)


def equation(path: Path, latex: str, width=10.0) -> None:
    chart = plt.figure(figsize=(width, 1.0), dpi=240)
    chart.text(0.5, 0.5, latex, ha="center", va="center", fontsize=16, color="#003D7C")
    chart.savefig(path, bbox_inches="tight", transparent=True, pad_inches=0.06)
    plt.close(chart)


def configure(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.3)
    section.right_margin = Cm(2.3)
    section.header_distance = Cm(0.75)
    section.footer_distance = Cm(0.75)
    section.different_first_page_header_footer = True
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(9.9)
    normal.font.color.rgb = RGBColor.from_string(TEXT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.05
    normal.paragraph_format.space_after = Pt(4)
    styles = (
        ("Title", 12, 0, 7, NUS_BLUE),
        ("Subtitle", 9.9, 0, 6, TEXT),
        ("Heading 1", 12, 10, 5, TEXT),
        ("Heading 2", 11.1, 8, 3, TEXT),
        ("Heading 3", 9.9, 6, 2, TEXT),
        ("Caption", 9, 1, 5, TEXT),
    )
    for name, size, before, after, color in styles:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        if name.startswith("Heading") or name == "Title":
            style.font.bold = True
        if name.startswith("Heading"):
            style.paragraph_format.keep_with_next = True
    doc.styles["Caption"].font.italic = True
    header = section.header.paragraphs[0]
    header.text = "CROSS-MARKET PRE-OPEN FORECASTING OF NIFTY IT STOCKS  |  DSA5205"
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header_run = header.runs[0]
    header_run.font.name = "Times New Roman"
    header_run.font.size = Pt(8.4)
    header_run.font.bold = True
    _page_field(section.footer.paragraphs[0])
    for run in section.footer.paragraphs[0].runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(8.1)
        run.font.color.rgb = RGBColor.from_string("4A4A4A")
    section.first_page_header.paragraphs[0].clear()
    section.first_page_footer.paragraphs[0].clear()
