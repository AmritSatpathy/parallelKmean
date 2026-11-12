"""Evidence-bound publication additions; no fitting or data downloading."""
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from report_components import table, figure, hyperlink
from report_release import load_registry, registry_id, registry_output

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH, RELEASE = load_registry(ROOT)
HYP_ID = os.environ.get('REPORT_HYPOTHESIS_ANALYSIS_ID', registry_id(RELEASE,'HYPOTHESIS_ANALYSIS_ID','hypotheses_20261101T030841Z_d494c2f238ca64a6'))
FIGURES = []


def publication_pack(root):
    folder = registry_output(root,RELEASE,'figure_pack','reports/publication_figure_pack_2026-11-01')
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    return {'folder': folder, 'manifest': manifest,
            'catalog': {x['id']: x for x in json.loads((folder/'figure_catalog.json').read_text())}}


def add_reviewed_figure(doc, pack, key, width=6.2):
    item = pack['catalog'][key]
    FIGURES.append(key)
    caption = f"Figure {len(FIGURES)}. {item['title']}. {item['caption']}"
    figure(doc, pack['folder']/item['png'], caption, min(width, 6.2 if key=='01_architecture' else 5.6))
    doc.paragraphs[-1].paragraph_format.keep_together = True


def add_contents(doc):
    p = doc.add_paragraph()
    begin = OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve')
    instr.text = ' TOC \\o "1-1" \\h \\z \\u '
    sep = OxmlElement('w:fldChar'); sep.set(qn('w:fldCharType'), 'separate')
    end = OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'), 'end')
    for element in [begin, instr, sep]: p.add_run()._r.append(element)
    p.add_run('Contents are generated from the document headings.')
    p.add_run()._r.append(end)


def hypotheses_intro(doc):
    doc.add_paragraph('Six evidence questions organize the final report. H1–H6 below follow the saved hypothesis-analysis protocol; they replace the earlier draft numbering. Peer information and feature complexity remain controlled modeling comparisons in Sections 4–5. These questions were refined after inspecting historical results and are exploratory, not prospectively preregistered tests.')
    table(doc, ['ID', 'Question / anticipated result', 'Evidence'], [
        ('H1', 'Positive U.S. services returns predict stronger subsequent Indian intraday returns.', 'Date-level conditional association and annual stability.'),
        ('H2', 'Frontier AI releases precede negative Indian IT intraday returns.', 'Event indicator with lagged Indian and market controls.'),
        ('H3', 'An initially negative AI-event effect becomes less negative after 2024.', 'Fixed 1 January 2025 interaction; adequate events on both sides.'),
        ('H4', 'Larger capability advances precede more negative Indian returns.', 'Retrospective frontier-increment regression; minimum 30 scored dates.'),
        ('H5', 'Signed forecasts generate robust positive returns after costs.', 'Frozen ledgers, benchmarks, HAC and date-block uncertainty.'),
        ('H6', 'IT services adapt through AI offerings and workforce change.', 'Company disclosures; descriptive evidence, not an AI treatment effect.'),
    ], [1,8.5,6.5], 8.1)


def hypothesis_results(doc, root, reviewed):
    folder=root/'artifacts/research_hypotheses'/HYP_ID
    tests=pd.read_csv(folder/'hypothesis_tests.csv')
    def row(h, period):
        return tests.loc[tests.hypothesis.eq(h)&tests.period.eq(period)].iloc[0]
    h1=row('H1_US_transmission','all'); h2=row('H2_negative_AI_release','all')
    h2_2026=row('H2_negative_AI_release','2026')
    h3=row('H3_event_attenuation','fixed_break_2025-01-01')
    doc.add_heading('7. Hypothesis evidence and regime diagnostics',level=1)
    doc.add_heading('7.1 Test definitions and multiplicity',level=2)
    doc.add_paragraph('The saved analysis aggregates observations by trading date rather than treating ten correlated stock rows as independent replications. H1 regresses Indian cohort intraday return on the available U.S. services return, lagged Indian return and lagged broad-market return. H2 replaces the U.S. term with the release indicator to avoid conditioning on a possible transmission channel. Newey–West HAC covariance uses five lags [19]. All estimable regression p-values in the saved family receive Holm step-down correction [20]; displayed confidence intervals are pointwise, not simultaneous. Circular date-block resampling uses 2,000 draws for H5 [21]. These controls do not erase prior researcher exposure to the sample.')
    table(doc,['Question','Observed evidence','Verdict'],[
        ('H1: U.S. association',f"β={h1.coefficient:.3f}; 95% CI [{h1.ci_low:.3f}, {h1.ci_high:.3f}]; Holm p={h1.pvalue_holm:.4f}; n={int(h1.n_dates):,} dates.",'Supported: positive conditional association. This test does not estimate trading alpha.'),
        ('H2: negative release',f"Pooled effect={100*h2.coefficient:+.3f} pp (Holm p={h2.pvalue_holm:.3f}; {int(h2.count_event)} sessions); 2026={100*h2_2026.coefficient:+.3f} pp (Holm p={h2_2026.pvalue_holm:.4f}; {int(h2_2026.count_event)} sessions).",'Inconclusive overall; supported as a negative association in the exploratory 2026 subgroup.'),
        ('H3: attenuation',f"Early effect count={int(h3.count_early_event)}; late count={int(h3.count_late_event)}; interaction={100*h3.coefficient:+.3f} pp, Holm p={h3.pvalue_holm:.3f}.",'Inconclusive: interval is too wide; the negative point estimate alone does not disprove attenuation.'),
        ('H4: capability shock','21 scored event dates versus a minimum of 30.','Inconclusive: insufficient scored event dates for the declared test.'),
        ('H5: economic robustness','Canonical model loses 4.04% net in 2025 and gains 15.01% in the 2026 decline; both Sharpe intervals span zero.','Inconclusive for repeatable positive alpha. The narrower claim of a gain in every evaluation year is contradicted by 2025.'),
        ('H6: adaptation','AI offerings are documented; headcount changes differ across firms.','Supported descriptively: firms offer AI services and adapt their workforce. A causal profit-margin effect is a separate question.'),
    ],[3.1,8.0,4.9],8.1)
    doc.add_paragraph('Supported means that the measured claim has evidence in this sample; contradicted means evidence conflicts with that claim; inconclusive means the data and design cannot decide. A nonsignificant p-value is not proof of no relationship. H1 and the descriptive H6 claim have support. H2 varies by period. H3–H5 retain uncertainty. More tests on the same inspected sample cannot establish causality: the company disclosures lack a comparable untreated group, and AI releases coincide with other market news.')
    doc.add_heading('7.2 Cross-market association and event uncertainty',level=2)
    reviewed('13_transmission_inference')
    doc.add_paragraph('Because both return variables are in decimal units, the pooled H1 coefficient implies approximately +0.098 percentage points of Indian intraday return per +1 percentage point of the available U.S. return, conditional on the included lags. The 2026 coefficient is 0.181 (Holm p=0.0240), but the post-2024 change interaction is inconclusive (Holm p=1.000). A significant coefficient in one year and an insignificant coefficient in another is not itself a significant difference between years.')
    doc.add_paragraph('The pooled 2023–2026 H2 estimate is −0.169 percentage points with an interval spanning zero. In 2026 alone, 22 marked sessions yield −0.695 percentage points (95% HAC interval −1.099 to −0.292; Holm p=0.0155). That year-specific association is exploratory because the calendar is retrospective, events are sparse, and no market-adjusted causal design is claimed. H3 does not show attenuation after 2024: its interaction is negative but does not survive multiplicity adjustment. H4 is withheld under its sample-size rule rather than promoted from the small capability sample.')
    doc.add_heading('7.3 Selected-specification record across protocol stages',level=2)
    reviewed('15_confirmation_diagnostic')
    doc.add_paragraph('The selected specification returns +8.31% in 2021, +0.11% in 2022, +6.26% in 2023, +9.93% in 2024, −4.04% in the 2025 historical replication and +15.01% in the January–September 2026 extension. These values form one protocol history, not six independent tests: 2021–2024 influenced screening and confirmation, 2025 is an estimator-OOS historical replication after researcher exposure, and 2026 uses one fit on the declared rolling pre-2026 window, 25 November 2020–30 December 2025. The 2025 and 2026 bootstrap Sharpe intervals are wide and overlap zero. The full record therefore supports regime sensitivity and disciplined staged validation, not a claim of stable annual alpha.')
    doc.add_heading('7.4 Sideways markets and trend reversals',level=2)
    reviewed('16_regime_distributions')
    monthly=pd.read_csv(pack_folder(root)/'16_monthly_regimes.csv')
    rows=[]
    for regime in ['Rising','Sideways','Falling']:
        g=monthly.loc[monthly.regime.eq(regime)]
        rows.append((regime,str(len(g)),f"{g['Model net'].mean():+.2%}",f"{g['Nifty IT ETF'].mean():+.2%}",f"{g['Nifty 50 ETF'].mean():+.2%}"))
    table(doc,['Realized regime','Months','Model mean','IT ETF mean','50 ETF mean'],rows,[4.0,2,3.3,3.3,3.4],8.1)
    doc.add_paragraph('These are averages of observed monthly returns, not annualized estimates or forecasts of the next regime. The distribution and reversal counts matter: a few favorable falling months cannot establish stable downside skill, and sparse upward-reversal observations cannot identify a response mechanism. The ±3% classification is ex post and never enters the trading signal. Full-sample development wealth is deliberately excluded from the headline evaluation because those dates influenced model selection.')


def company_discussion(doc):
    doc.add_heading('7.5 H6: adaptation, staffing and credible counterevidence',level=2)
    doc.add_paragraph('Commercial adaptation is observable without assuming that AI necessarily improves margins. Infosys announced its Topaz generative-AI services offering on 23 May 2023 [9]. The Bank of England/FCA 2024 survey reports human oversight in semi-autonomous financial-services applications and only 2% of use cases with fully autonomous decision-making [10]. These sources support the relevance of enterprise integration and governance. They do not establish that Indian clients retained particular vendors because of cybersecurity, or that AI caused an equity recovery.')
    doc.add_paragraph('CX Dive reported on 9 May 2025 that Klarna was piloting recruitment of human customer-service staff while retaining its chatbot [11]. This is evidence of a hybrid support arrangement, not proof that the company abandoned AI or rehired a specified number of software developers. Conversely, Reuters reported renewed Indian IT disruption concerns on 5 February 2026, alongside analysts who considered the selloff excessive [12]. This counterexample prevents a one-directional narrative of early fear followed by universal acceptance.')
    table(doc,['Company / interval','Headcount','Change','Source'],[
        ('TCS: Mar–Sep 2026','584,519 → 598,056','+13,537 / +2.32%','[13], [14]'),
        ('Infosys: Jun–Sep 2026','328,062 → 332,542','+4,480 / +1.37%','[15], employee metrics, p. 2'),
        ('Wipro: Jun–Sep 2026','243,044 → 236,328','−6,716 / −2.76%','[16], p. 1; [17]'),
    ],[5,4.3,3.9,2.8],8.2)
    doc.add_paragraph('Quarterly and six-month changes are labelled separately. The three firms do not show a uniform reduction in staff. TCS reports a March headline operating margin of 25.3% excluding one-offs and a September headline margin of 24.0%; those bases must not be interpreted as a clean AI productivity experiment [13,14]. Infosys reports 21.0% for September versus 21.1% for June [15]. Wipro reports IT-services margins of 16.5% versus 16.0% [16,17]. Different segment definitions and adjustment bases preclude a pooled margin test. Hiring, attrition, acquisitions, demand, currency and restructuring are alternative explanations.')
    doc.add_paragraph('September-quarter disclosures were published after the September trading endpoint and are used only for retrospective discussion. They are not historical forecasting inputs. The defensible conclusion is heterogeneous adaptation alongside continuing disruption risk, not that stronger models uniformly destroy or restore IT-services value.')


REFS = [
('Chinco, A., Clark-Joseph, A. D., and Ye, M. Sparse signals in the cross-section of returns. Journal of Finance 74(1), 449–492, 2019.', 'https://doi.org/10.1111/jofi.12733'),
('Kelly, B. T., Malamud, S., and Zhou, K. The virtue of complexity in return prediction. Journal of Finance 79(1), 459–503, 2024.', 'https://doi.org/10.1111/jofi.13298'),
('NSE Indices. Nifty IT: index description and constituent resources.', 'https://www.niftyindices.com/indices/equity/sectoral-indices/nifty-it'),
('Yahoo Finance historical prices, accessed through yfinance; software documentation.', 'https://ranaroussi.github.io/yfinance/'),
('Federal Reserve Bank of St. Louis. FRED/ALFRED API documentation.', 'https://fred.stlouisfed.org/docs/api/fred/'),
('Epoch AI. AI benchmarks and capabilities; current-vintage dataset.', 'https://epoch.ai/benchmarks'),
('NiftyHistory. Nifty Constituent Ledger; public reconstruction.', 'https://niftyhistory.in/'),
('fja05680. Current and historical lists of S&P 500 components; public GitHub repository.', 'https://github.com/fja05680/sp500'),
('Infosys. Launched today: Infosys Topaz—an AI-first offering for global enterprises. 23 May 2023.', 'https://www.infosys.com/newsroom/press-releases/2023/enterprises-generative-ai.html'),
('Bank of England and Financial Conduct Authority. Artificial intelligence in UK financial services—2024. Survey report.', 'https://www.bankofengland.co.uk/report/2024/artificial-intelligence-in-uk-financial-services-2024'),
('Doerer, K. Klarna changes its AI tune and again recruits humans for customer service. CX Dive, 9 May 2025.', 'https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/'),
('Reuters. Anthropic’s AI push raises analyst concerns over Indian IT services revenues. 5 February 2026; syndicated by Investing.com.', 'https://www.investing.com/news/stock-market-news/anthropics-ai-push-raises-analyst-concerns-over-it-services-revenues-4486681'),
('TCS. TCS closes FY26 with improving sequential growth momentum and strong deal wins. 9 April 2026.', 'https://www.tcs.com/who-we-are/newsroom/press-release/tcs-financial-results-q4-fy-2026'),
('TCS. International growth and strategic wins underpin TCS’ Q2. 8 October 2026.', 'https://www.tcs.com/who-we-are/newsroom/press-release/tcs-financial-results-q2-fy-2027'),
('Infosys. Fact sheet: second quarter, fiscal 2027. Employee metrics p. 2; financial metrics p. 1.', 'https://www.infosys.com/investors/reports-filings/quarterly-results/2026-2027/q2/documents/fact-sheet.pdf'),
('Wipro. Highlights for the quarter ended 30 June 2026. Data sheet, p. 1.', 'https://www.wipro.com/content/dam/nexus/en/investor/quarterly-results/2026-2027/q1fy27/datasheet-q1fy27.pdf'),
('Wipro. Highlights for the quarter ended 30 September 2026. SEC Exhibit 99.5.', 'https://www.sec.gov/Archives/edgar/data/1123799/000119312526426841/d188224dex995.htm'),
('Geurts, P., Ernst, D., and Wehenkel, L. Extremely randomized trees. Machine Learning 63, 3–42, 2006.', 'https://doi.org/10.1007/s10994-006-6226-1'),
('Newey, W. K., and West, K. D. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica 55(3), 703–708, 1987.', 'https://doi.org/10.2307/1913610'),
('Holm, S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6, 65–70, 1979.', 'https://doi.org/10.2307/4615733'),
('Politis, D. N., and Romano, J. P. A circular block-resampling procedure for stationary data. In Exploring the Limits of Bootstrap, 263–270, 1992.', 'https://mathweb.ucsd.edu/~politis/DPpublication.html'),
]


def references(doc):
    doc.add_paragraph('Web references were checked on 1 November 2026. Company statements are attributed disclosures, not independent causal findings. Sources [9]–[17] support the bounded discussion in Section 7.5; no article substitutes for an experiment result.')
    for i,(title,url) in enumerate(REFS,1):
        p=doc.add_paragraph(f'[{i}] {title} ')
        p.paragraph_format.left_indent=Inches(.22)
        p.paragraph_format.first_line_indent=Inches(-.22)
        p.paragraph_format.space_after=Pt(4)
        hyperlink(p,'Source / DOI',url)
        for r in p.runs:r.font.size=Pt(9)


def append_publication_diagnostics(doc, root, reviewed):
    doc.add_heading('Appendix D. Sensitivity and diagnostic evidence',level=1)
    doc.add_paragraph('The following figures qualify the main result rather than add independent confirmations. Cost sensitivity holds positions fixed; event counts expose missing coverage; permutation importance describes the existing fitted models.')
    reviewed('10_cost_sensitivity')
    doc.add_paragraph('Event coverage is 0 mapped sessions in 2021–2022, 5 in 2023, 12 in 2024, 7 in 2025 and 22 in January–September 2026. Sparse early coverage limits comparisons across periods. Full calibration, event-count and grouped-permutation plots remain available in the figure pack; they are omitted here because the main text already reports their decision-relevant findings.')
    doc.add_heading('Appendix E. Evidence addresses and reproduction',level=1)
    doc.add_paragraph('Hypothesis analysis: '+HYP_ID+'. The accompanying hypothesis_tests.csv contains estimates, sample sizes, covariance choices, raw and Holm p-values, and not-estimable statuses. Company_headcount_changes.csv retains source IDs and quarter definitions. The hypothesis manifest records the input and result hashes.')
    labels=[('Historical EDA','EDA_RUN_ID'),('2026 EDA','FORWARD_EDA_ID'),('2026 event study','EVENT_ANALYSIS_ID'),
            ('Capability analysis','CAPABILITY_ANALYSIS_ID'),('Epoch capability analysis','EPOCH_ANALYSIS_ID'),
            ('2025 model evidence','MODEL_EVIDENCE_ID'),('2026 model evidence','FORWARD_MODEL_EVIDENCE_ID'),
            ('Information ladder','LADDER_ANALYSIS_ID'),('2025 strategy comparison','STRATEGY_ANALYSIS_ID'),
            ('2026 strategy comparison','FORWARD_STRATEGY_ANALYSIS_ID'),('Paired strategy inference','PAIRED_STRATEGY_ANALYSIS_ID'),
            ('Strict-vintage sensitivity','STRICT_VINTAGE_ANALYSIS_ID')]
    rows=[(label,registry_id(RELEASE,key,'not-bound')) for label,key in labels]+[('Hypothesis family',HYP_ID)]
    table(doc,['Ref.','Evidence object','Immutable identifier'],[(f'E{i}',label,value) for i,(label,value) in enumerate(rows,1)],[1.0,4.2,10.8],7.1)
    doc.add_paragraph('The canonical screening, confirmation, freeze, 2025 replication and 2026 extension identifiers are listed in Appendix A. Main-text discussions refer to these evidence addresses through the appendices rather than embedding hashes in the narrative.')
    doc.add_paragraph('Figure evidence: '+str(registry_output(root,RELEASE,'figure_pack','reports/publication_figure_pack_2026-11-01').relative_to(root)/'manifest.json')+'. Its catalog provides figure captions, placement and changes; accompanying CSV files contain the exact cost, benchmark and regime summaries. The report builder verifies the figure file hashes before insertion.')
    doc.add_paragraph('README.md documents the complete figure, DOCX, contents, PDF, visual-QA and verification sequence. The release registry is retained beside this appendix.')


def finalize_report(doc, root, pack):
    pointers={
        'The fixed ridge experiment uses': ' Analysis record: Appendix E8.',
        'The saved analysis aggregates': ' Analysis record: Appendix E13.',
        'This is a post-selection robustness check': ' Analysis record: Appendix A8.',
        'The reporting view concatenates': ' Model and evaluation records: Appendix A5–A7.',
        'Development-only EDA shows': ' Dataset and exploratory-analysis records: Appendix A1 and E1–E2.',
    }
    for p in doc.paragraphs:
        for prefix,reference in pointers.items():
            if p.text.startswith(prefix):p.add_run(reference)
    # Submission prose uses appendix references; full machine identifiers stay there.
    for node in doc._element.body:
        content=''.join(node.itertext())
        if content.startswith('Appendix A. Reproducibility and provenance'):
            break
        if re.search(r'\b\w*\d{8}T\d{6}\w*|\b[a-f0-9]{12,64}\b',content):
            raise ValueError('Technical identifier outside appendices: '+content[:160])
    doc.styles['Caption'].font.bold = False
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
        fonts=doc.styles[name]._element.get_or_add_rPr().rFonts
        for attribute in list(fonts.attrib):
            if attribute.endswith('Theme'):del fonts.attrib[attribute]
        for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),'Times New Roman')
    # Numeric references are inserted at the factual claim, not merely listed at the end.
    additions={
        'Chinco, Clark-Joseph, and Ye (2019) motivate': ' [1,2]',
        'The tradable cohort is fixed': ' [3]',
        'AI context is represented': ' [6]',
    }
    replacements={
        'The selected depth-constrained Extra Trees model': 'The canonical depth-constrained Extra Trees model',
        'current-vintage macro fallbacks remain an identified limitation': 'current-vintage macro fallbacks remain an identified limitation',
        'Nifty 50 and Nifty IT are close-to-close index opportunity-cost references': 'Nifty 50 and Nifty IT ETF proxies are close-to-close opportunity-cost references',
        '5 bps commission plus 2.5 bps slippage': '5 bps fee/spread proxy plus 2.5 bps slippage',
        'it is displayed for context but was not used to tune the selected model.': 'it is displayed for context; prior researcher inspection prevents a pristine-holdout interpretation.',
        'Nifty 50 and Nifty IT are close-to-close index opportunity-cost references': 'Nifty 50 and Nifty IT ETF proxies are close-to-close opportunity-cost references',
    }
    for p in doc.paragraphs:
        # Preserve native equations and hyperlinks: restrict replacements to ordinary text runs.
        for r in p.runs:
            for a,b in replacements.items():
                if a in r.text:r.text=r.text.replace(a,b)
        for start,citation in additions.items():
            if p.text.startswith(start):p.add_run(citation)
    # Data citations immediately follow the source table.
    for p in doc.paragraphs:
        if p.text.startswith('Comparator contract'):
            break
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        for a,b in replacements.items():
                            if a in r.text:r.text=r.text.replace(a,b)
                        if r.text=='Yahoo Finance OHLCV':r.text+=' [4]'
                        if r.text=='FRED / ALFRED':r.text+=' [5]'
                        if r.text=='NiftyHistory / project reference':r.text+=' [7]'
                        if r.text=='S&P 500 historical components':r.text+=' [8]'
                        if r.text=='Epoch AI benchmarks':r.text+=' [6]'
    # Balance analytical pages by putting compact methodological context before charts.
    body=doc._element.body
    def move_before(nodes, target):
        for node in nodes:target.addprevious(node)
    paragraphs=doc.paragraphs
    def pstart(text):return next(p._p for p in paragraphs if p.text.startswith(text))
    # Explain the EDA responses before its market paths rather than splitting this table.
    diagnostic=next(t._tbl for t in doc.tables if t.cell(0,0).text=='Diagnostic')
    move_before([diagnostic],pstart('3.2 Market paths'))
    # Put interpretation ahead of the annual chart so prose can fill the previous page.
    annual_text=next(p._p for p in doc.paragraphs if p.text.startswith('The selected specification returns'))
    annual_heading=pstart('7.3 Selected-specification')
    annual_heading.addnext(annual_text)
    # Let explanatory prose fill space before an indivisible figure/caption pair.
    # Keep the figure order and all evidence intact; never move section headings.
    for p in list(doc.paragraphs):
        if not p.text.startswith('Figure '):
            continue
        image_node=p._p.getprevious()
        if image_node is None or not image_node.xpath('.//w:drawing'):
            continue
        following=p._p.getnext()
        if following is not None and following.tag.endswith('}p'):
            from docx.text.paragraph import Paragraph
            candidate=Paragraph(following,doc._body)
            if candidate.text and candidate.style.name=='Normal' and not candidate.text.startswith('Figure '):
                image_node.addprevious(following)
    # Keep short tables together without chaining entire sections to large figures.
    for t in doc.tables:
        if len(t.rows)<=4:
            for row in list(t.rows)[:-1]:
                for cell in row.cells:
                    for p in cell.paragraphs:p.paragraph_format.keep_with_next=True
    order=[]
    for p in doc.paragraphs:
        match=re.match(r'Figure (\d+)\.',p.text)
        if match:
            old_number=int(match.group(1));order.append(FIGURES[old_number-1])
            for r in p.runs:
                if r.text.startswith('Figure '):
                    r.text=re.sub(r'^Figure \d+\.',f'Figure {len(order)}.',r.text)
                    break
    FIGURES[:]=order
    # Generate real heading bookmarks and a cached TOC, populated after the first render.
    toc_path=registry_output(root,RELEASE,'toc_pages','reports/releases/_missing_toc_pages.json') if RELEASE else None
    pages=json.loads(toc_path.read_text()) if toc_path is not None and toc_path.is_file() else None
    make_contents(doc,pages=pages)
    # Native image alt text and simple explicit table geometry.
    shapes=list(doc.inline_shapes)
    if shapes:
        shapes[0]._inline.docPr.set('descr','National University of Singapore logo')
    for shape,key in zip(shapes[1:],FIGURES):
        shape._inline.docPr.set('descr',pack['catalog'][key]['title'])
    for t in doc.tables:
        first_row_properties=t.rows[0]._tr.get_or_add_trPr()
        if first_row_properties.find(qn('w:tblHeader')) is None:
            header=OxmlElement('w:tblHeader');header.set(qn('w:val'),'true');first_row_properties.append(header)
        grid=[int(x.get(qn('w:w'))) for x in t._tbl.tblGrid]
        t._tbl.tblPr.find(qn('w:tblW')).set(qn('w:w'),str(sum(grid)))
        for row in t.rows:
            for c,w in zip(row.cells,grid):
                c._tc.get_or_add_tcPr().find(qn('w:tcW')).set(qn('w:w'),str(w))
    evidence_path=registry_output(root,RELEASE,'publication_evidence','reports/report_publication_evidence.json')
    evidence_path.parent.mkdir(parents=True,exist_ok=True)
    evidence_path.write_text(json.dumps({
        'figure_pack_hash':pack['manifest']['content_hash'],'figures':FIGURES,
        'hypothesis_analysis':HYP_ID,'reference_check_date':'2026-11-01',
        'references':[{'number':i,'title':t,'url':u} for i,(t,u) in enumerate(REFS,1)],
        'note':'Source facts and experimental findings are separate; no model training performed.'
    },indent=2))


def pack_folder(root):
    return registry_output(root,RELEASE,'figure_pack','reports/publication_figure_pack_2026-11-01')


def make_contents(doc, pages=None):
    """Build a TOC field with cached PAGEREF entries; Word can refresh it normally."""
    from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
    headings=[p for p in doc.paragraphs if p.style.name=='Heading 1' and p.text!='Contents']
    anchor=next(p for p in doc.paragraphs if p.text=='Contents')
    # Remove only previous generated contents between heading and evidence callout.
    node=anchor._p.getnext()
    while node is not None and node.tag==qn('w:p'):
        nxt=node.getnext();node.getparent().remove(node);node=nxt
    cursor=anchor._p
    def insert():
        p=doc.add_paragraph();cursor.addnext(p._p)
        p.paragraph_format.space_after=Pt(6)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.3),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
        return p
    for i,h in enumerate(headings):
        name=f'report_section_{i}'
        if not h._p.findall(qn('w:bookmarkStart')):
            b=OxmlElement('w:bookmarkStart');b.set(qn('w:id'),str(100+i));b.set(qn('w:name'),name);h._p.insert(0,b)
            e=OxmlElement('w:bookmarkEnd');e.set(qn('w:id'),str(100+i));h._p.append(e)
        p=insert()
        if i==0:
            b=OxmlElement('w:fldChar');b.set(qn('w:fldCharType'),'begin');p.add_run()._r.append(b)
            ins=OxmlElement('w:instrText');ins.text=' TOC \\o "1-1" \\h \\z ';p.add_run()._r.append(ins)
            s=OxmlElement('w:fldChar');s.set(qn('w:fldCharType'),'separate');p.add_run()._r.append(s)
        p.add_run(h.text+'\t')
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),f' PAGEREF {name} \\h ')
        r=OxmlElement('w:r');t=OxmlElement('w:t');t.text=str((pages or {}).get(h.text,0));r.append(t);field.append(r);p._p.append(field)
        if i==len(headings)-1:
            e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'end');p.add_run()._r.append(e)
        cursor=p._p
