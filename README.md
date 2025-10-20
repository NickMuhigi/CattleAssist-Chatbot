# Cattle Keeping Domain Chatbot

A fine-tuned T5-based conversational AI system specialized in cattle management, health, nutrition, and care.

## Demo Video
**Watch the full demonstration:** https://drive.google.com/file/d/1oTb2KzyjOlByzqENifOqMlQ-3uWN-BqW/view?usp=sharing

---

## Dataset

**Domain:** Agriculture - Cattle Keeping  
**Source:** Custom domain-specific Q&A dataset (`C-Assist_Dataset.csv`)  
**Size:** 340 question-answer pairs  
**Test Set:** 51 samples (15%)  
**Coverage:** Cattle health, feeding, breeding, disease management, vaccination, and general care

### Data Preprocessing
- Text cleaning and normalization
- Removal of extra whitespace and special characters
- Missing value handling
- Train/Validation/Test split: 70%/15%/15%

---

## Model Architecture

**Base Model:** T5-small (Google's Text-to-Text Transfer Transformer)  
**Framework:** PyTorch + Hugging Face Transformers  
**Task:** Generative Question Answering  
**Tokenization:** T5Tokenizer with task prefix "answer question:"

### Model Specifications
- Parameters: ~60M
- Max Input Length: 256 tokens
- Max Output Length: 128 tokens
- Generation Strategy: Beam search with sampling

---

## Hyperparameter Tuning

Four experiments were conducted to optimize performance:

| Experiment | Learning Rate | Batch Size | Final Train Loss | Time (min) |
|------------|---------------|------------|------------------|------------|
| Exp1_LowLR | 1e-5 | 8 | 3.2510 | 113.66 |
| Exp2_HighLR | 1e-4 | 8 | **1.6006** | 116.35 |
| Exp3_Baseline | 5e-5 | 8 | 1.9082 | 117.76 |
| Exp4_LargeBatch | 5e-5 | 16 | 2.6321 | 115.06 |

**Best Configuration:** Exp2_HighLR (High Learning Rate)
- Learning Rate: 1e-4
- Batch Size: 8
- Epochs: 15
- Final Validation Loss: 1.019

---

## Performance Metrics

### Primary Metric
- **Token-Level F1-Score:** 0.2049 ± 0.0949
  - Measures exact token overlap between predictions and references

### ROUGE Scores
- **ROUGE-1 F1:** 0.2364
- **ROUGE-2 F1:** 0.0561
- **ROUGE-L F1:** 0.1894

### Detailed Breakdown
- **ROUGE-1 Precision:** 0.2987 | **Recall:** 0.2016
- **ROUGE-2 Precision:** 0.0728 | **Recall:** 0.0468
- **ROUGE-L Precision:** 0.2376 | **Recall:** 0.1623

### Training Progress
- Loss reduction: ~84% from initial to final epoch
- Consistent validation improvement across all experiments
- No significant overfitting observed
- Evaluation performed on 51 test samples

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (recommended)
```

### Install Dependencies
```bash
pip install transformers datasets torch accelerate sentencepiece
pip install sacrebleu rouge-score gradio pandas numpy
pip install matplotlib seaborn scikit-learn nltk
```

### Clone Repository
```bash
git clone [YOUR_REPO_URL]
cd cattle-chatbot
```

---

## Running the Chatbot

### Option 1: Gradio Web Interface (Recommended)
```python
# Run the notebook or execute:
python chatbot_gradio.py
```
Access at: `http://localhost:7860` or use the public Gradio share link

### Option 2: Command-Line Interface
```python
# In Python or Jupyter:
cli_chatbot()
```
Type your questions and receive instant responses. Type `quit` to exit.

### Option 3: Programmatic Usage
```python
from chatbot import chat_with_bot

question = "What are the symptoms of mastitis?"
answer = chat_with_bot(question, temperature=0.7, num_beams=4)
print(answer)
```

---

## Example Conversations

### Example 1: <img width="1568" height="573" alt="Screenshot 2025-10-20 224623" src="https://github.com/user-attachments/assets/5279e5c8-a2ad-4f23-8d03-290aacb1d9a4" />


### Example 2: <img width="1565" height="576" alt="Screenshot 2025-10-20 224905" src="https://github.com/user-attachments/assets/2d93c1b9-57cd-420f-868d-e43d631e5776" />


### Example 3: <img width="1568" height="580" alt="Screenshot 2025-10-20 225057" src="https://github.com/user-attachments/assets/15ce0413-2f9b-4dfe-a440-6b2c56ed2a5b" />


### Example 4:<img width="1562" height="578" alt="Screenshot 2025-10-20 224510" src="https://github.com/user-attachments/assets/86c7a2e9-7557-44cd-9081-1d7040e9c4ba" />


---

## Repository Structure

```
cattle-chatbot/
├── Cattle-Assist_Bot_NOTEBOOK.ipynb   # Main training notebook
├── Cattle-Assist_REPORT.pdf           # Report
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── models/                            # Saved model checkpoints
│   └── cattle_chatbot_Exp2_HighLR/
└── outputs/                           # Training logs and visualizations
```

---

## Key Features

Domain-specific expertise in cattle keeping  
Hyperparameter-optimized T5 model  
Multiple interface options (Web, CLI, API)  
Configurable generation parameters  
Comprehensive evaluation metrics  
Production-ready deployment  

---

## Customization

### Adjust Response Style
```python
chat_with_bot(
    question="Your question",
    temperature=0.7,    # 0.7-1.0 for creativity
    num_beams=4         # 2-8 for search depth
)
```

### Fine-tune Further
Modify `experiments_config` in the notebook to test additional hyperparameters.

---

**Last Updated:** October 2025
