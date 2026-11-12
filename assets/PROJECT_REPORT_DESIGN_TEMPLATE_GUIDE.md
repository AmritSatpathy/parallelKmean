# NUS Academic Project Report Design Template

## Purpose

This document records the **visual language, typography, page architecture, and academic presentation rules** observed in `reports/Project2_report.pdf`. It is a reusable design specification for future project reports.

It deliberately does **not** reproduce the rainfall project’s subject matter, analysis, claims, or results. All examples below use placeholders so the template can be applied to another course or project.

## 1. Template identity

The reference is best described as a:

- formal NUS academic project report;
- compact, single-column technical paper;
- evidence-led report with restrained institutional branding;
- Word-first layout rendered to A4 PDF;
- report that places core results in the main body and implementation detail in appendices.

The design is polished through typography, alignment, whitespace, and consistent captions rather than decorative page furniture.

### 1.1 Governing visual principles

1. **Institutional, not promotional.** NUS branding is prominent on the cover but restrained inside the report.
2. **Dense but readable.** The report uses a compact academic rhythm without shrinking body text below a comfortable reading size.
3. **Single-column continuity.** Explanations, tables, and figures follow one clear vertical reading path.
4. **Evidence near interpretation.** A figure or table appears close to the paragraph that introduces and explains it.
5. **Consistent hierarchy.** Numbering and typography reveal section structure without oversized headings.
6. **Appendix separation.** Operational instructions and architecture atlases do not interrupt the main analytical narrative.

## 2. Page specification

| Property | Reference-template rule |
|---|---|
| Page size | A4 portrait, approximately 210 × 297 mm |
| Layout | Single column |
| Main text width | Approximately 160–165 mm |
| Suggested margins | 22–25 mm left/right; 18–22 mm top/bottom |
| Body alignment | Justified |
| Header | Present from the contents page onward |
| Footer | Centered report label and page number |
| Cover header/footer | Suppressed or substantially reduced |
| Page breaks | Used before contents and major appendices; otherwise content flows naturally |

The template avoids narrow journal-style columns. This is important because its figures, architecture diagrams, and technical tables depend on full-page width.

## 3. Typography system

The PDF embeds **Times New Roman** for almost all academic text. Arial appears only as a minor supporting font inside some graphic material. Cambria Math is used for equation glyphs and Consolas for code.

### 3.1 Measured type hierarchy

The following sizes were measured from the PDF and should be treated as the preferred baseline.

| Role | Font | Size | Weight/style | Color |
|---|---|---:|---|---|
| Cover project title | Times New Roman | 12 pt | Bold | NUS blue |
| Cover subtitle | Times New Roman | 9.9 pt | Italic | Near black |
| Cover course line | Times New Roman | 9.9 pt | Bold | Near black |
| Cover author details | Times New Roman | 9.9 pt | Regular/bold labels | Near black / muted gray |
| Contents title | Times New Roman | 12 pt | Bold | Black |
| Major numbered heading | Times New Roman | 12 pt | Bold | Black |
| Second-level heading | Times New Roman | 11.1 pt | Bold | Black |
| Body text | Times New Roman | 9.9 pt | Regular | Black |
| Figure caption | Times New Roman | 9.9 pt | Italic | Black |
| Table caption | Times New Roman | 9.9 pt | Bold label + regular title | Black |
| Running header | Times New Roman | 8.4 pt | Bold, uppercase | Black |
| Footer | Times New Roman | 8.1 pt | Regular | Dark gray |
| Appendix detail text | Times New Roman | 9–9.9 pt | Regular | Black |
| Code | Consolas | approximately 8.5–9 pt | Regular | Black |
| Equations | Cambria Math | sized to surrounding text | Regular | Black |

### 3.2 Typography rules

- Use Times New Roman consistently for cover text, body copy, headings, captions, references, and appendices.
- Use real heading styles rather than manual bold paragraphs.
- Use sentence case for section titles.
- Avoid all-caps headings in the body. Reserve uppercase for the running header and small cover labels such as `AUTHOR` and `STUDENT ID`.
- Use italics for subtitles, figure captions, source notes, and restrained emphasis.
- Use bold for hierarchy, table labels, and short lead-ins—not for full paragraphs.
- Use true subscripts and superscripts for equations and scientific notation.
- Use Cambria Math or native Word equations so fractions, hats, summations, and Greek symbols render correctly.
- Use Consolas only for executable commands, paths, and code fragments.

### 3.3 Paragraph rhythm

Recommended implementation values:

- body line spacing: 1.0–1.08;
- space after body paragraph: 3–5 pt;
- first-line indent: none;
- major heading space before: 10–14 pt;
- major heading space after: 4–6 pt;
- second-level heading space before: 7–10 pt;
- second-level heading space after: 3–4 pt;
- bullets: compact, with hanging indentation and 2–3 pt after each item;
- captions: keep with the associated visual.

This rhythm creates a compact report without making the pages feel crowded.

## 4. Color system

Color is used sparingly. The body remains primarily black and white.

### 4.1 Observed palette

| Token | Approximate value | Use |
|---|---|---|
| NUS blue | `#003D7C` | Cover title, thin rules, restrained institutional accent |
| Link blue | `#4F81BD` | Hyperlinks and contents-page repository note |
| Body black | `#000000` / `#171717` | Main text and headings |
| Muted gray | `#626A73` | Cover labels, metadata, source line |
| Footer gray | `#4A4A4A` | Page footer |
| Pale gray | `#EDEDED`–`#F3F3F3` | Table headers, note-box background, light separators |
| White | `#FFFFFF` | Page and figure background |

### 4.2 Color rules

- Do not color entire body sections.
- Do not use several unrelated accent colors in Word text.
- Allow analytical figures to use their own consistent palettes, but keep chart backgrounds white.
- Use NUS blue for lines and small anchors, not large filled blocks.
- Use pale gray for compact table headers and factual note boxes.
- Hyperlinks may use blue and underline, but long raw URLs should not dominate a paragraph.

## 5. Cover page specification

The cover is vertically centered in the upper half of the page and leaves a generous quiet area below. It does not use a conventional report header or footer.

### 5.1 Cover composition

The observed order is:

1. thin NUS-blue horizontal rule near the top of the content area;
2. project title in NUS blue, centered;
3. vertically stacked NUS logo, centered;
4. project subtitle in italic, centered;
5. course name and course code in bold, centered;
6. compact author/student-ID table;
7. dataset or project-source line in muted italic text;
8. month and year centered near the bottom of the title block.

### 5.2 Cover wireframe

```text
──────────────────────── thin NUS-blue rule ────────────────────────

                         [PROJECT TITLE]

                         [VERTICAL NUS LOGO]

                    [Descriptive project subtitle]

                       [Course Name COURSE1234]

                 AUTHOR                     STUDENT ID
                 [FULL NAME]                [ID]
                 [FULL NAME]                [ID]

        Dataset / repository / institutional source statement

                           [Month Year]
```

### 5.3 NUS logo treatment

- Prefer the official vertical/stacked NUS logo used in the reference cover.
- Preserve its original aspect ratio.
- Center it horizontally.
- Keep clear space around it; do not place text or rules directly against the crest.
- Suggested width: approximately 35–48 mm, depending on the exact logo asset.
- Do not recolor, crop, stretch, add shadows, or place it on a colored rectangle.
- Use a high-resolution image so the crest and wordmark remain sharp in PDF export.

### 5.4 Project title

- Keep the title concise enough for one or two centered lines.
- Use 12 pt bold Times New Roman in NUS blue to match the reference.
- Do not turn the title into an oversized poster headline.
- Use a descriptive italic subtitle to communicate the analytical scope.

### 5.5 Author block

Use a two-column table without visually heavy borders:

| AUTHOR | STUDENT ID |
|---|---|
| FULL NAME | A0000000X |
| FULL NAME | A0000000X |

Formatting rules:

- header row in pale gray;
- uppercase header labels;
- names and student IDs centered within their columns;
- authors sorted alphabetically unless the course specifies another order;
- include every contributor exactly once;
- keep names on one line where possible;
- use the same 9.9 pt Times New Roman as the surrounding cover metadata.

If a team name is required, place it between the course line and author table or immediately below the author table.

## 6. Contents page

The contents page begins on page 2 and introduces the running header/footer system.

### 6.1 Contents-page structure

1. uppercase running header at the top;
2. `Contents` as a 12 pt bold heading;
3. section entries with dot leaders and right-aligned page numbers;
4. main sections first, followed by references and appendices;
5. repository and data-access note centered near the lower half of the page;
6. centered footer: `Project report • [page number]`.

### 6.2 Contents formatting

- Use an automatic Word table of contents when possible.
- Include major sections and appendices; include second-level headings only if the contents remains readable.
- Use dot leaders and a right-aligned page-number tab.
- Keep line spacing compact and consistent.
- Do not type page numbers manually in the final document.
- Ensure appendix labels are unique and sequential. The reference contains a duplicated `Appendix A` entry in its contents; this should be corrected in future use.

### 6.3 Repository note

The reference places repository and data-access information below the contents list, separated by thin blue rules.

Recommended pattern:

```text
────────────────────────────────────────────────────────────
Source code: [clickable repository URL]
Data access: [short source/instruction]
────────────────────────────────────────────────────────────
```

Use 8.5–9.9 pt text. Keep instructions concise; detailed setup belongs in an appendix or README.

## 7. Running header and footer

### 7.1 Header

Observed pattern:

```text
[PROJECT TITLE IN UPPERCASE]  |  [COURSE CODE]
```

Rules:

- start on the contents page, not the cover;
- use 8.4 pt Times New Roman bold;
- align to the left margin;
- use black rather than a large color band;
- keep the text short enough to fit on one line;
- use a thin separator character, not a graphic banner.

### 7.2 Footer

Observed pattern:

```text
Project report • 7
```

Rules:

- centered;
- 8.1 pt Times New Roman;
- dark gray;
- use an automatic page-number field;
- retain the same baseline on every interior page.

## 8. Main-body architecture

The report follows a conventional research narrative.

### 8.1 Recommended section sequence

1. Abstract
2. Introduction and objectives
3. Dataset and data preparation
4. Computing or research architecture
5. Exploratory analysis
6. Feature engineering and prediction task
7. Models and experiments
8. Results and interpretation
9. Limitations and future work
10. Conclusion
11. AI usage declaration
12. References
13. Appendices

The exact names may change, but the logic should remain: **question → data → method → evidence → interpretation → limitations → reproducibility**.

### 8.2 Heading numbering

Use decimal section numbering:

```text
1. Major section
1.1 Second-level section
1.1.1 Third-level section
```

Appendices use letters:

```text
Appendix A. Reproducibility
A.1 Environment
A.2 Run commands
```

Figures inside appendices use appendix-specific numbering such as `Figure A1`, `Figure A2`, or a dedicated family such as `Figure M1` for a model atlas.

### 8.3 Abstract

- Begin on a new page after contents.
- Use an unnumbered 12 pt bold heading.
- Write one compact paragraph rather than multiple subsections.
- Cover the problem, data, method families, evaluation, principal result, and main limitation.
- Avoid citations, equations, tables, and figures unless essential.
- Target approximately 180–300 words.

### 8.4 Introduction

The introduction should establish:

- practical motivation;
- exact analytical objective;
- project tasks;
- scope and interpretation boundaries;
- a short contribution statement.

Use a short numbered or bulleted task list only when it materially improves scanning.

## 9. Tables

Tables are compact, full-width or near-full-width, and used for genuinely comparable records.

### 9.1 Table appearance

- caption above the table;
- caption begins with a bold label such as `Table 1.`;
- header row uses pale gray shading;
- body rows use white fill;
- borders are thin and gray;
- narrative columns are left-aligned;
- numerical columns are centered or decimal-aligned;
- cell text uses approximately 8.5–9.9 pt Times New Roman;
- column widths are chosen by content, not divided equally;
- rows expand naturally; fixed row heights are prohibited;
- repeat the header row if a table spans pages.

### 9.2 Table caption pattern

```text
Table 3. Experimental configurations and requested resources.
```

Keep captions factual. Do not place the full interpretation in the caption.

### 9.3 Table usage rules

Use tables for:

- dataset scope;
- cleaning rules and retained counts;
- model/configuration comparisons;
- experiment protocols;
- evaluation metrics;
- reproducibility inventories.

Do not use tables merely to box normal prose. If most cells contain long paragraphs, convert the material into subsections or bullets.

## 10. Figures and charts

The reference uses figures extensively but maintains a stable academic treatment.

### 10.1 Figure placement

- Center figures within the text column.
- Use the available width while preserving legible labels.
- Place the caption immediately below the visual.
- Keep the visual and caption on the same page.
- Introduce the figure before it appears.
- Follow the figure with interpretation, not a repetition of its title.
- Use one large figure or a deliberate two-panel composition; avoid a collage of tiny unreadable plots.

### 10.2 Figure captions

Observed style:

- Times New Roman italic;
- approximately 9.9 pt;
- begins with `Figure n.` or `Figure An.`;
- centered when short, left-aligned when long;
- contains enough context to understand the visual without becoming a full paragraph.

Example:

```text
Figure 4. Validation performance across model families under the common temporal split.
```

### 10.3 Chart styling

- white background;
- restrained grid lines;
- clear axis titles with units;
- readable tick labels at report size;
- colorblind-conscious, limited palette;
- consistent color meaning across figures;
- legend inside unused plot space or directly below the chart;
- avoid 3D effects, shadows, gradients, and decorative icons;
- export at sufficient resolution, preferably 200–300 dpi for raster charts;
- use SVG/EMF where Word compatibility permits.

### 10.4 Multi-panel figures

Use panel labels above or inside each subplot:

```text
(a) Training period        (b) Test period
```

The shared caption explains the comparison. Panel labels should be consistent in size and position.

### 10.5 Architecture diagrams

The reference appendices use a coherent diagram language:

- rounded rectangular nodes;
- pale blue for processing components;
- pale green for outputs;
- pale neutral gray for boundaries or infrastructure;
- thin dark connectors;
- dashed lines for control, scheduling, or instrumentation;
- short labels inside boxes;
- a one-sentence caption identifying what solid and dashed arrows mean.

Diagrams must explain architecture or workflow—not serve as decoration.

## 11. Notes and callouts

The reference uses restrained bordered note boxes for limitations and interpretation boundaries.

### 11.1 Callout pattern

```text
Data-quality note. [Short statement of the rule, caveat, or missing evidence.]
```

Recommended styling:

- pale gray fill;
- thin gray border;
- 9–9.9 pt body text;
- bold lead-in;
- modest internal padding;
- no icon required;
- one compact paragraph.

Appropriate uses include:

- data-quality caveats;
- interpretation guardrails;
- computational limitations;
- scope notes;
- reproducibility warnings.

Do not use callouts for ordinary narrative or to compensate for weak section structure.

## 12. Equations

- Create equations with Word’s equation editor or MathML-compatible tooling.
- Use Cambria Math.
- Center display equations.
- Keep equation numbers right-aligned only when the report refers back to them.
- Define every symbol immediately before or after the equation.
- Render subscripts, superscripts, hats, fractions, and inequalities natively.
- Never insert raw LaTeX syntax as body text.
- Avoid screenshots of equations unless required by the production system.

Example structure:

```text
The prediction target is defined as

                         [properly rendered equation]                 (1)

where ...
```

## 13. Code and command blocks

The reference keeps code mainly in the appendix.

Rules:

- use Consolas at approximately 8.5–9 pt;
- keep commands on separate lines;
- use indentation and continuation characters consistently;
- do not allow long paths to overflow the text column;
- explain inputs and outputs in prose immediately after the block;
- distinguish commands that were executed from illustrative commands;
- keep the main body focused on methodology and evidence.

## 14. References

### 14.1 Placement

References appear after the conclusion and AI declaration, before the appendices.

### 14.2 Style

The observed report uses a numbered technical reference style:

```text
[1] Author or organization. Title. Publisher or journal, year. URL or DOI.
```

Formatting rules:

- `References` is an unnumbered major heading;
- order references by first citation unless the course requires alphabetical order;
- use hanging indents;
- use complete titles and authoritative links;
- include access dates for unstable web pages when appropriate;
- prefer DOI links for academic papers;
- ensure every listed source is cited in the text;
- ensure every in-text citation resolves to one reference entry;
- use clickable hyperlinks but retain readable source names.

### 14.3 Source hierarchy

Prefer, in order:

1. peer-reviewed papers;
2. official university, government, exchange, or regulator material;
3. official software documentation;
4. first-party datasets and repositories;
5. secondary explanatory sources only when necessary.

## 15. AI usage declaration

The reference includes a dedicated, explicit AI declaration before the references.

It should state:

- tools used;
- specific purposes;
- which tasks were assisted;
- that the authors verified implementation, evidence, and wording;
- that the authors retain responsibility for the submission.

Keep it factual and proportional. Do not claim that AI independently validated empirical results.

## 16. Appendices

Appendices contain material needed for reproducibility or technical depth but not required for the core argument.

### 16.1 Recommended appendix organization

```text
Appendix A. Reproducibility and execution
  A.1 Environment
  A.2 Data placement
  A.3 Commands
  A.4 Saved artifacts

Appendix B. System or research architecture
  B.1 Data flow
  B.2 Training topology
  B.3 Evaluation workflow

Appendix C. Model architecture atlas
  C.1 Model family 1
  C.2 Model family 2
```

### 16.2 Appendix rules

- Begin a major appendix on a new page when practical.
- Introduce the appendix with a short paragraph explaining its purpose.
- Continue the same header, footer, typography, and margins as the main body.
- Use appendix-specific section and figure numbering.
- Keep diagrams large enough to read.
- Pair each architecture figure with a concise interpretation.
- Do not move essential conclusions or principal evidence into an appendix.
- Avoid duplicating the same figure in both the main body and appendix.

## 17. Page-density and whitespace rules

The reference is intentionally compact. Its whitespace comes from margins, heading rhythm, and figure separation—not large empty areas between paragraphs.

Use these controls:

- keep headings with the next paragraph;
- keep captions with figures and tables;
- prevent table rows from splitting where practical;
- allow long tables to continue with repeated headers;
- resize a figure modestly if it causes an otherwise nearly empty preceding page;
- do not force every subsection onto a new page;
- do not stack several large figures without explanatory text;
- accept a partially filled last appendix page when the alternative is shrinking diagrams below readable size.

## 18. Reusable page templates

### 18.1 Standard analytical page

```text
[RUNNING HEADER]

4. Major section
[Short framing paragraph]

4.1 Subsection
[Method or interpretation paragraph]

              [FULL-WIDTH FIGURE OR TABLE]
              Figure/Table caption

[Evidence-led interpretation]

                         Project report • n
```

### 18.2 Figure-led page

```text
[RUNNING HEADER]

5.2 Focused analysis
[One paragraph defining the question]

                   [LARGE READABLE FIGURE]
                   Figure caption

[One or two paragraphs explaining pattern, implication, and limitation]

                         Project report • n
```

### 18.3 Appendix architecture page

```text
[RUNNING HEADER]

Appendix B. Architecture
[Short scope paragraph]

                 [LARGE ARCHITECTURE DIAGRAM]
                 Figure B1 caption

                 [SECOND DIAGRAM, IF READABLE]
                 Figure B2 caption

                         Project report • n
```

## 19. Word style map

The final DOCX should define and consistently use these named styles:

| Style name | Implementation |
|---|---|
| Normal | Times New Roman 9.9 pt, justified, 1.0–1.08 spacing |
| Title | Times New Roman 12 pt bold, NUS blue, centered |
| Subtitle | Times New Roman 9.9 pt italic, centered |
| Heading 1 | Times New Roman 12 pt bold, black |
| Heading 2 | Times New Roman 11.1 pt bold, black |
| Heading 3 | Times New Roman 9.9 pt bold, black |
| Caption | Times New Roman 9.9 pt italic |
| Table text | Times New Roman 8.5–9.9 pt |
| Code | Consolas 8.5–9 pt |
| Header | Times New Roman 8.4 pt bold, uppercase |
| Footer | Times New Roman 8.1 pt, dark gray |
| Note box | Normal typography with bold lead-in and pale-gray container |

## 20. Quality checklist

### 20.1 Cover

- [ ] Official NUS logo is sharp, centered, undistorted, and has clear space.
- [ ] Project title is concise and uses NUS blue.
- [ ] Course name and code are correct.
- [ ] Every author and student ID is included once.
- [ ] Author order follows the agreed convention.
- [ ] Dataset/repository line is factual and concise.
- [ ] Submission month and year are correct.

### 20.2 Contents and navigation

- [ ] Contents page begins on page 2.
- [ ] Page numbers and dot leaders are automatic and aligned.
- [ ] Appendix letters are unique and sequential.
- [ ] Repository link is clickable.
- [ ] Running header begins after the cover.
- [ ] Footer page number is automatic.

### 20.3 Typography and layout

- [ ] Body is Times New Roman at approximately 9.9 pt.
- [ ] No body text is smaller merely to force a page fit.
- [ ] Headings use real Word styles and consistent numbering.
- [ ] Body paragraphs are justified without excessive word spacing.
- [ ] No headings are stranded at the bottom of a page.
- [ ] No large accidental blank areas appear.

### 20.4 Figures and tables

- [ ] Every figure and table is introduced in the text.
- [ ] Every visual has a numbered caption.
- [ ] Axis labels and legends remain readable at 100% page view.
- [ ] Figures do not contain tiny, redundant, or decorative elements.
- [ ] Table widths and cell padding are deliberate.
- [ ] Header rows repeat when tables span pages.
- [ ] Interpretive prose explains each important visual.

### 20.5 Academic completeness

- [ ] Abstract states problem, data, method, result, and limitation.
- [ ] Results distinguish evidence from interpretation.
- [ ] Limitations are explicit.
- [ ] AI usage is declared.
- [ ] References are complete and consistently formatted.
- [ ] Reproducibility commands and artifact inventories are in appendices.

### 20.6 Final production QA

- [ ] Update all Word fields and the table of contents.
- [ ] Export to PDF using embedded fonts.
- [ ] Inspect every rendered page at 100% zoom.
- [ ] Check for clipped text, missing glyphs, broken equations, and split captions.
- [ ] Check hyperlinks.
- [ ] Confirm figure resolution.
- [ ] Confirm that the DOCX and PDF show identical author and version information.
- [ ] Confirm that no temporary comments, tracked changes, or placeholder text remain.

## 21. Anti-patterns to avoid

- oversized marketing-style titles;
- multiple unrelated accent colors in body text;
- dark page backgrounds;
- decorative shapes around every section;
- tiny figures inserted only to increase figure count;
- unreadable correlation matrices with hundreds of labels;
- captions separated from visuals;
- repeated large blank spaces caused by unnecessary page breaks;
- manually typed contents page numbers;
- equations pasted as low-resolution screenshots;
- references consisting only of raw URLs;
- duplicate appendix labels;
- unsupported causal language beneath descriptive plots;
- placing the complete implementation manual in the main results narrative.

## 22. Canonical design summary

The reusable template can be reduced to the following rule set:

> Use an A4, single-column, Times New Roman academic layout. Place the official vertical NUS logo and a compact title/author block on the cover. Begin a clean automatic contents page on page 2. Use 12 pt major headings, 11.1 pt subheadings, 9.9 pt justified body text, understated NUS-blue accents, pale-gray tables and note boxes, italic captions beneath readable figures, a small uppercase running header, and a centered `Project report • page` footer. Keep the main body focused on the research argument; place code, execution details, and architecture atlases in lettered appendices. End with an explicit AI declaration, numbered references, and a visually verified appendix structure.

