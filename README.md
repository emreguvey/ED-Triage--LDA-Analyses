# ED-Triage--LDA-Analyses
Medical Topic Modeling (LDA) for Emergency Department Texts
This project applies Latent Dirichlet Allocation (LDA) topic modeling to analyze unstructured clinical texts (chief complaints and patient histories) from Emergency Department (ED) visits. The goal is to uncover and interpret the primary clinical themes and patient groupings from free-text data.

The project features an advanced text preprocessing pipeline specifically designed for noisy, short-form Turkish medical text, including artifact removal and term standardization.

## 🚀 Features

* **Specialized Medical NLP:** A custom Turkish NLP pipeline that cleans medical abbreviations, misspellings, and data entry artifacts (e.g., `_xd_`).
* **LDA Modeling:** Optimized LDA modeling using `scikit-learn` to discover latent topics.
* **Medical Topic Interpretation:** A rule-based classifier to automatically label discovered topics with clinical categories (e.g., "Cardiovascular," "Neurological").
* **Comprehensive Visualization:** Exports a full suite of publication-ready charts, including topic prevalence, word clouds, and inter-hospital comparisons.
* **Feature Engineering:** Exports the resulting topic probabilities and features to an Excel file, ready for use in downstream machine learning models (e.g., triage prediction).
