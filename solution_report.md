# AI Solution Design Report
## Automated Chest X-Ray Analysis System
### Domain: Healthcare | Task: Pneumonia Detection via CNN-Based Image Classification

---

> **Assessment:** Part 4 — AI Solution Design
> **Date:** May 2026

---

## Task 1: Choose a Business Domain

**Selected Domain: Healthcare**

The healthcare domain was selected because it presents one of the most impactful and well-defined opportunities for AI-driven improvement. Diagnostic imaging — specifically chest X-ray interpretation — is a high-volume, time-sensitive task where AI assistance can meaningfully reduce errors, accelerate turnaround times, and extend access to expert-level diagnosis in under-resourced settings.

---

## Task 2: Define the Business Problem

### What problem is being solved?

Radiology departments worldwide face a critical workload crisis. A single radiologist may review **80–100 chest X-rays per shift**, and diagnostic fatigue causes error rates to increase by up to 30% in the final hours of a shift. Pneumonia — one of the leading causes of death globally — is primarily diagnosed through chest X-ray interpretation. Delayed or missed diagnoses directly result in preventable patient harm.

The business problem is: **How can hospitals automatically pre-screen chest X-rays to detect pneumonia, prioritise urgent cases, and reduce the diagnostic burden on radiologists?**

### Who are the users or stakeholders?

| Stakeholder | Role | Primary Concern |
|---|---|---|
| **Radiologists** | Primary end-users; final decision-makers | Accuracy, interpretability, workflow integration |
| **Patients** | Recipients of the diagnosis | Safety, timely results, privacy |
| **Hospital Administration** | Deploy and fund the system | Cost, liability, regulatory compliance |
| **IT / Data Engineering Teams** | Infrastructure, EHR integration | Data pipelines, security, uptime |
| **Regulatory Bodies (FDA, HIPAA)** | Oversight and certification | Patient safety, fairness, auditability |
| **Insurance Providers** | Reimburse procedures | Clinical validity, documentation trail |

### What is the current manual or traditional process?

Chest X-rays are captured and routed to a radiologist's reading queue. The radiologist manually reviews each image, dictates a report, and submits it to the referring physician. In busy hospitals, non-urgent X-rays can wait **12–24 hours** for a report. There is no automated triage — urgent and routine cases sit in the same queue.

### What are the limitations of the current process?

- **Diagnostic fatigue:** Accuracy degrades over long shifts as radiologists review hundreds of images
- **No prioritisation:** Critical cases are not automatically surfaced; they wait in turn
- **Geographic inequality:** Rural or low-income hospitals lack access to experienced radiologists
- **No second opinion at scale:** Human second-reads are resource-intensive and rarely applied to all cases
- **Slow turnaround:** Reporting backlogs delay treatment decisions for time-sensitive conditions like pneumonia

---

## Task 3: Identify the AI Task Type

**AI Task Type: Image Classification**

More specifically: **Binary Supervised Image Classification** — given a chest X-ray image, predict whether the patient has pneumonia (`PNEUMONIA`) or not (`NORMAL`).

### Why is this AI task type suitable?

| Reason | Explanation |
|---|---|
| **Labelled historical data exists** | Hospital PACS archives contain thousands of X-rays paired with radiologist reports — ideal for supervised learning |
| **Fixed output classes** | The prediction is one of two mutually exclusive categories: Pneumonia or Normal |
| **Visual input** | The input is an image; CNNs are specifically designed for visual pattern recognition |
| **Clinically validated** | Published research (CheXNet, Rajpurkar et al.) has demonstrated radiologist-level performance on exactly this task using CNNs |

This is not a regression task (no continuous output), not object detection (we need class labels, not bounding boxes), and not segmentation (pixel-level labelling is not required at this stage).

---

## Task 4: Data Requirement Plan

### Type of data needed

Unstructured data — chest X-ray images in **DICOM** (standard medical format) or **JPEG** format, paired with radiologist-confirmed diagnosis labels.

### Structured or unstructured?

**Unstructured** — raw pixel data from imaging scans. Supplementary structured metadata (patient age, sex, acquisition device) may be used as auxiliary features but is not the primary input.

### Input Features

| Feature | Description |
|---|---|
| `chest_xray_image` | Raw X-ray image (224×224 pixels, grayscale or RGB-normalised) |
| `patient_age` | Age in years (auxiliary metadata) |
| `patient_sex` | Male / Female (auxiliary metadata) |
| `acquisition_device` | X-ray machine model/hospital (for bias monitoring) |
| `image_quality_score` | Automated quality gate score (blurriness, rotation) |

### Target Variable / Labels

| Label | Description |
|---|---|
| `PNEUMONIA` | Radiological features of pneumonia confirmed by radiologist |
| `NORMAL` | No significant pathological features detected |

### Data Collection Method

- **Primary source:** Hospital PACS (Picture Archiving and Communication System) — retrospective archive of labelled X-rays
- **Public benchmarks:** NIH ChestX-ray14 (112,120 X-rays), CheXpert — Stanford (224,316 X-rays), Kaggle Chest X-Ray dataset (5,856 images)
- **Labelling:** Radiologist diagnostic reports from CMMS/EHR are joined to image files using study IDs to create ground-truth labels
- **Consent:** Retrospective use under IRB approval; all images de-identified before use

### Data Quality Risks

| Risk | Description | Mitigation |
|---|---|---|
| **Label noise** | Radiologist reports are inconsistently worded across hospitals and systems | Multi-reader consensus labelling; uncertain labels marked ambiguous |
| **Equipment bias** | Different X-ray machines produce images with varying contrast and exposure | CLAHE normalisation; equipment-stratified validation splits |
| **Demographic bias** | Training data may over-represent certain age groups, sex, or ethnicity | Stratified sampling; disaggregated performance reporting by subgroup |
| **Class imbalance** | Normal X-rays typically outnumber pneumonia cases | Class-weighted loss function; oversampling of minority class |
| **Data drift** | New X-ray machines or updated imaging protocols change image distribution post-deployment | Continuous confidence score monitoring; automated retrain triggers |
| **Historical label errors** | Legacy reports may contain outdated diagnostic criteria | Expert re-labelling of a random subset; label audit pipeline |

---

## Task 5: Model Recommendation

**Recommended Model: ResNet-50 with Transfer Learning (CNN)**

### Architecture

```
Input:  224 × 224 × 3  (chest X-ray, RGB normalised)
  │
  ├── ResNet-50 Backbone  (pretrained on ImageNet)
  │     └── First 100 layers: frozen
  │     └── Top layers: fine-tuned on chest X-ray data
  │
  ├── Global Average Pooling 2D
  ├── Dense(256, activation=ReLU)
  ├── Dropout(0.5)
  ├── Dense(2, activation=Softmax)
  │
Output: [P(Normal), P(Pneumonia)]  — calibrated probability scores
        + Grad-CAM heatmap         — visual explanation for radiologist
```

### Why ResNet-50 with Transfer Learning?

| Design Decision | Rationale |
|---|---|
| **CNN architecture** | Images have spatial structure — CNNs preserve 2D relationships through convolution and pooling, unlike flat fully-connected networks |
| **Transfer learning (ImageNet)** | Low-level features (edges, textures, patterns) learned from ImageNet transfer well to X-ray images; reduces required labelled medical data by ~90% |
| **Frozen base layers** | Preserves general visual feature extraction learned during ImageNet pretraining |
| **Fine-tuned top layers** | Adapts the high-level features to chest X-ray specific patterns (lung opacities, infiltrates) |
| **Global Average Pooling** | More parameter-efficient than Flatten; reduces overfitting on small medical datasets |
| **Dropout(0.5)** | Strong regularisation essential for medical imaging where datasets are smaller than general computer vision tasks |
| **Grad-CAM output** | Generates a heatmap highlighting image regions driving the prediction — essential for radiologist trust, interpretability, and clinical accountability |

### Alternative Models Considered

| Model | Verdict |
|---|---|
| **DenseNet-121** | Also excellent for chest X-rays (CheXNet paper); slightly more parameters than ResNet-50 — viable alternative |
| **VGG-16** | Simpler architecture; lower performance ceiling; higher parameter count — not preferred |
| **Custom CNN from scratch** | Requires far more labelled data; lower performance without pretraining — not recommended |
| **Vision Transformer (ViT)** | State-of-the-art for large datasets; requires very large labelled corpus to outperform ResNet-50 with transfer learning — future consideration |

---

## Task 6: Evaluation Plan

### Technical Metrics

| Metric | Formula | Why It Matters |
|---|---|---|
| **Recall (Sensitivity)** | TP / (TP + FN) | Most critical — a missed pneumonia case (False Negative) is clinically far more dangerous than a false alarm |
| **Precision** | TP / (TP + FP) | Controls alert fatigue — too many False Positives waste radiologist time and erode trust |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean balancing recall and precision — primary single-number performance metric |
| **AUC-ROC** | Area under ROC curve | Measures discriminative ability across all decision thresholds, regardless of class balance |
| **Specificity** | TN / (TN + FP) | Proportion of healthy patients correctly cleared — important for managing false alarm rate |

**Target Thresholds:**
- Recall ≥ 92% on Pneumonia class
- F1-Score ≥ 0.90 overall
- AUC-ROC ≥ 0.95

### Business Metrics

| KPI | Baseline (No AI) | Target (With AI) |
|---|---|---|
| X-ray reporting turnaround | 24 hours average | < 4 hours for flagged urgent cases |
| Radiologist cases per shift | 80–100 | 120–140 (AI pre-screens obvious normals) |
| Missed pneumonia rate | ~8% | < 3% |
| Alert-to-action rate | N/A | ≥ 80% (tracks alert quality vs false alarms) |
| Radiologist burnout index | High | Reduced (fewer routine reviews) |

### Possible Failure Cases

- Model trained on one hospital's machines may underperform on a different hospital's equipment
- Paediatric X-rays may not generalise from models trained predominantly on adult data
- Sudden catastrophic lung conditions (e.g., pneumothorax from trauma) are outside the training distribution and will not be flagged
- A new variant of pneumonia with atypical radiological presentation may evade detection

### Human Review / Validation Process

- **Shadow period:** 4-week deployment where predictions are logged but not shown to radiologists — predictions vs actual diagnoses compared to validate real-world performance before go-live
- **Confidence thresholds:** Cases below 85% confidence are mandatorily routed for full radiologist review with no AI pre-annotation
- **Monthly audit:** Random sample of 100 predictions reviewed by senior radiologist each month
- **Override tracking:** Every radiologist override is logged, reviewed weekly, and fed into the retraining pipeline

---

## Task 7: Responsible AI Considerations

### Bias in Data

Training data from a single hospital system may reflect that institution's patient demographics, imaging equipment, and diagnostic practices. A model trained on predominantly adult, male, urban patients may perform poorly on paediatric, female, or rural patient populations. Mitigation requires stratified data collection, demographic subgroup performance reporting, and periodic bias audits by an independent clinical team.

### Incorrect Predictions

A **False Negative** (missed pneumonia) can delay treatment, allowing the condition to progress to sepsis or respiratory failure — potentially fatal. A **False Positive** triggers unnecessary follow-up tests, causes patient anxiety, and wastes radiologist time. The model must be calibrated to prioritise recall (catch as many real cases as possible) while keeping false positives at a clinically acceptable level.

### Privacy Concerns

Chest X-rays are personally identifiable health information protected under HIPAA (US) and GDPR (EU). All patient identifiers (name, date of birth, medical record number) must be stripped from DICOM metadata before the image enters the AI pipeline. Images must be encrypted at rest (AES-256) and in transit (TLS 1.3). The AI module must operate with read-only access to imaging data and write only structured prediction outputs — never raw images — to external systems.

### Over-Reliance on AI

Radiologists under time pressure may accept AI predictions without critically reviewing the underlying image, particularly when the model shows high confidence. This **automation bias** is clinically dangerous. Mitigations include mandatory minimum dwell time on the Grad-CAM heatmap before sign-off, randomised vigilance checks where known-positive cases are periodically routed as if the AI had low confidence, and mandatory AI literacy training for all system users.

### Impact on Users

There is a legitimate concern that automated screening reduces the need for radiologists, potentially threatening jobs. Transparent communication is essential: the system is designed to eliminate routine, repetitive screening of obvious negatives — freeing radiologists to focus on complex, ambiguous, and high-stakes cases where their expertise matters most. It is a productivity multiplier, not a replacement.

### Need for Human Oversight

Per FDA guidance on AI/ML-based Software as a Medical Device (SaMD), the system must function as a **clinical decision support tool**, not an autonomous diagnostic system. A licensed radiologist must have final authority over every diagnosis. The system is deployed in an advisory capacity only — it surfaces predictions, confidence scores, and visual explanations; the radiologist decides.

---

## Task 8: Final Solution Summary

| Section | Detail |
|---|---|
| **Problem** | Radiology departments face unsustainable workloads. Pneumonia — detectable via chest X-ray — is frequently missed or diagnosed late due to radiologist fatigue and reporting backlogs. |
| **Proposed AI Solution** | A CNN-based binary image classification system (ResNet-50 with transfer learning) that analyses chest X-rays in under 2 seconds, classifies them as Pneumonia or Normal, provides a calibrated confidence score, and generates a Grad-CAM heatmap showing the regions driving the prediction. Deployed as a decision-support tool within the hospital radiology workflow. |
| **Required Data** | Chest X-ray images (DICOM/JPEG) from hospital PACS archives and public benchmarks, labelled with radiologist-confirmed diagnoses. Minimum 10,000 labelled images recommended for fine-tuning; supplemented by NIH ChestX-ray14 and CheXpert public datasets. |
| **Model Recommendation** | ResNet-50 pretrained on ImageNet, fine-tuned on chest X-ray data. Base layers frozen; top layers adapted. Global Average Pooling + Dense(256, ReLU) + Dropout(0.5) + Dense(2, Softmax). Grad-CAM for interpretability. |
| **Expected Business Impact** | ≥ 30% reduction in diagnostic turnaround for urgent cases. Radiologist capacity increases from ~100 to ~140 cases per shift. Missed pneumonia rate drops from ~8% to < 3%. Reduced radiologist burnout through elimination of routine normal-screening workload. |
| **Risks and Mitigation Plan** | Demographic bias → stratified data + subgroup audits. Privacy → HIPAA-compliant de-identification + encryption. Over-reliance → mandatory review windows + vigilance checks. Model drift → monthly performance monitoring + automated retrain triggers. Incorrect predictions → human-in-the-loop with final radiologist authority at all times. |

**Architecture Diagram:** See `diagrams/solution_architecture.png`

---

*Part 4 — AI Solution Design | Module 5 Assessment | Healthcare Domain*
