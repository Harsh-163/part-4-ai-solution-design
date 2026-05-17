# Part 4: AI Solution Design

## 📋 Overview

This part contains **no ML training code**. Instead, it delivers a professional AI solution design for the **Healthcare domain** — specifically, an Automated Chest X-Ray Analysis system for pneumonia detection.

---

## 📁 Directory Structure

```
part-4-ai-solution-design/
│
├── README.md                       # This file
├── solution_report.md              # Full professional AI solution report (10 sections)
├── generate_diagram.py             # Python script that draws the architecture diagram
│
└── diagrams/
    └── solution_architecture.png   # Block diagram: Ingestion → Preprocess → CNN → HITL → Output
```

---

## 🏥 Domain: Healthcare — Chest X-Ray Pneumonia Detection

| Item | Detail |
|------|--------|
| **Problem** | Radiologist burnout; delayed pneumonia diagnosis |
| **Stakeholders** | Radiologists, Patients, Hospital Admin, Regulators |
| **AI Task** | Supervised image classification (CNN) |
| **Input Data** | Chest X-ray images (DICOM / JPEG) — unstructured |
| **Target** | Pneumonia / Normal |
| **Model** | ResNet-50 with Transfer Learning |
| **Key Metrics** | Recall, F1-Score, AUC-ROC |

---

## 📊 Architecture Diagram

See: `diagrams/solution_architecture.png`

**Five pipeline stages:**
1. **Data Ingestion** — Hospital PACS, audit logging, secure pipeline
2. **Preprocessing** — CLAHE normalisation, resize 224×224, augmentation
3. **CNN Model** — ResNet-50, transfer learning, Softmax, Grad-CAM
4. **Human-in-the-Loop** — Radiologist review, confidence flagging, override
5. **Clinical Output** — Structured report, EHR push, escalation alerts

---

## 📄 Solution Report Contents (`solution_report.md`)

1. Problem Statement (Radiologist Burnout)
2. Stakeholders
3. AI Task Definition
4. Data Strategy & Quality Risks
5. Model Architecture
6. Evaluation Metrics (Recall, F1, AUC-ROC)
7. Human-in-the-Loop Design
8. Responsible AI (Bias, HIPAA, Over-reliance)
9. Architecture Diagram Reference
10. **One-Page Executive Summary**

---

## ⚙️ Regenerate the Diagram in VS Code (Interactive — see output inline)

**Step 1:** Install the **Jupyter** extension in VS Code
- Press `Ctrl+Shift+X` → search **Jupyter** → Install (by Microsoft)

**Step 2:** Open `generate_diagram.py`
- You will see **"Run Cell"** buttons appear above each `# %%` section

**Step 3:** Click **"Run Cell"** on each section
- Output and the diagram appear in the **Interactive Window** panel on the right — just like Jupyter

```bash
# OR run from terminal (saves PNG to diagrams/ folder)
cd part-4-ai-solution-design
python3 generate_diagram.py
# Output: diagrams/solution_architecture.png
```

---

## 🔗 Data Source

**Dataset provided by:** Masai School — Module 5 Assignment Dataset Pack.

**Google Drive Link:**  https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

**Dataset used in this part:**

| File | Description |
|---|---|
| `ai_usecase_reference_catalog.csv` | Reference catalog of AI use cases across domains — used to select business domain and AI task type |
| `business_kpi_sample.csv` | Sample business KPI data — used to define measurable business impact metrics |
| `data_dictionary.md` | Column descriptions for both reference files |

> Located inside: `part_4_ai_solution_design/` folder in the shared Drive.
