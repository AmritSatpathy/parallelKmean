#!/usr/bin/env python3
"""Cache a linked Word contents field from actual rendered heading locations.

Page values are derived from the latest PDF rather than manually entered.
PAGEREF fields retain updateable Word bookmark references. Re-render and run
again if pagination changes. Only the generated publication DOCX is edited.
"""
import re
import sys
from pathlib import Path

import fitz
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.shared import Cm, Pt

root=Path(__file__).resolve().parents[2]
docpath=root/'report/Parallel_Kmeans_Project_Report.docx'
pdf=fitz.open(sys.argv[1])
doc=Document(docpath)


def normalize(text):
    return re.sub(r'\s+',' ',text).strip()


heads=[]
for par in doc.paragraphs:
    if par.style.name!='Heading 1' or par.text=='Contents':
        continue
    title=par.text
    starts=par._p.findall(qn('w:bookmarkStart'))
    bookmark=starts[0].get(qn('w:name'))
    found=[]
    for page_index,page in enumerate(pdf):
        # Match heading text only in the body, excluding cached contents on p2.
        if page_index<2:
            continue
        if normalize(title) in normalize(page.get_text()):
            found.append(page_index+1)
    if not found:
        raise ValueError('Heading missing from rendered PDF: '+title)
    heads.append((title,bookmark,found[0]))

body=doc._element.body
# Remove any previous contents cache, bounded by bookmarks rather than text.
start=None
for node in list(body):
    if node.tag==qn('w:p'):
        if node.find('.//'+qn('w:fldSimple')) is not None:
            fld=node.find('.//'+qn('w:fldSimple'))
            if fld.get(qn('w:instr'),'').startswith('TOC '):
                start=node
                break
        if any(x.get(qn('w:name'))=='contents_cache' for x in node.findall(qn('w:bookmarkStart'))):
            start=node
            break
if start is None:
    raise ValueError('Contents anchor missing')

# Existing cached TOC paragraphs are removed up to its matching end marker.
if any(x.get(qn('w:name'))=='contents_cache' for x in start.findall(qn('w:bookmarkStart'))):
    cursor=start.getnext()
    while cursor is not None:
        nxt=cursor.getnext()
        end=any(x.get(qn('w:id'))=='9000' for x in cursor.findall(qn('w:bookmarkEnd')))
        body.remove(cursor)
        if end:
            break
        cursor=nxt

start.clear()
mark=OxmlElement('w:bookmarkStart');mark.set(qn('w:id'),'9000');mark.set(qn('w:name'),'contents_cache');start.append(mark)
rr=OxmlElement('w:r');fld=OxmlElement('w:fldChar');fld.set(qn('w:fldCharType'),'begin');rr.append(fld);start.append(rr)
rr=OxmlElement('w:r');instr=OxmlElement('w:instrText');instr.set(qn('xml:space'),'preserve');instr.text=' TOC \\o "1-1" \\h \\z \\u ';rr.append(instr);start.append(rr)
rr=OxmlElement('w:r');fld=OxmlElement('w:fldChar');fld.set(qn('w:fldCharType'),'separate');rr.append(fld);start.append(rr)
previous=start
for title,bookmark,page in heads:
    par=doc.add_paragraph()
    par.paragraph_format.space_after=Pt(11)
    par.paragraph_format.tab_stops.add_tab_stop(Cm(16.1),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.SPACES)
    link=OxmlElement('w:hyperlink');link.set(qn('w:anchor'),bookmark)
    r=OxmlElement('w:r');pr=OxmlElement('w:rPr')
    underline=OxmlElement('w:u');underline.set(qn('w:val'),'none');pr.append(underline)
    color=OxmlElement('w:color');color.set(qn('w:val'),'171717');pr.append(color)
    r.append(pr);t=OxmlElement('w:t');t.text=title;r.append(t);link.append(r);par._p.append(link)
    par.add_run('\t')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),f'PAGEREF {bookmark} \\h')
    r=OxmlElement('w:r');pr=OxmlElement('w:rPr')
    underline=OxmlElement('w:u');underline.set(qn('w:val'),'none');pr.append(underline)
    color=OxmlElement('w:color');color.set(qn('w:val'),'003D7C');pr.append(color)
    r.append(pr);t=OxmlElement('w:t');t.text=str(page);r.append(t);field.append(r);par._p.append(field)
    previous.addnext(par._p);previous=par._p
par=doc.add_paragraph();par.paragraph_format.space_after=Pt(0);par.paragraph_format.line_spacing=Pt(1)
rr=OxmlElement('w:r');fld=OxmlElement('w:fldChar');fld.set(qn('w:fldCharType'),'end');rr.append(fld);par._p.append(rr)
end=OxmlElement('w:bookmarkEnd');end.set(qn('w:id'),'9000');par._p.append(end)
previous.addnext(par._p)
doc.save(docpath)
print('Automatic linked contents:',heads)
