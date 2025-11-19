# SafeMed-R1: A Trustworthy Medical Reasoning Model

🔍 **Overview**
![SafeMed-R1 Overview](./safemed-r1-safety-overview.png)

Modern medical LLMs can “get the answer right” on exams, but still fail to earn clinical trust:

- They often lack transparent reasoning chains that clinicians can audit.  
- Their behavior may drift from medical ethics and regulatory requirements.  
- They are vulnerable to induction attacks (jailbreaks, harmful advice, unethical suggestions).

---

✨ **Highlights**

**Trustworthy and safe output**  
This model adheres to medical ethics and safety principles. It has been meticulously fine-tuned to avoid prohibited content and harmful advice, delivering useful, fact-based answers alongside appropriate disclaimers. Fine-tuning based on safety data (such as MedSafetyBench) has significantly enhanced the model's safety while maintaining its performance.

**Attack resistance**  
SafeMed-R1 has been trained to resist common jailbreak attacks and adversarial requests that could otherwise bypass security filters. Through multi-dimensional reward optimisation, it has learned to reject or safely handle malicious queries. (Recent research indicates that standard LLMs remain vulnerable to simple multi-step jailbreak attacks.)

**Explainable reasoning**  
Benefiting from chain-of-thought training, this model can progressively explain its medical reasoning process when prompted. This transparency assists users and clinicians in understanding the logic behind answers, thereby enhancing trust in the recommended outcomes. SafeMed-R1 leverages this capability to internalise expert-reviewed medical question reasoning pathways.

---

## ⚡ Introduction

The model underwent a rigorous two-stage training process:

1. **Supervised fine-tuning (SFT)** using expert-validated chain-of-thought (CoT) medical reasoning data alongside an ethics and safety guidance dataset (encompassing diverse medical ethical and legal risks).  
2. **Reinforcement learning (RL)** to align the model's outputs with medical ethical guidelines and enhance its robustness against adversarial inputs.

The resulting medical LLM demonstrates strong performance in clinical tasks while minimising the risk of unsafe or unreliable outputs.

---

## ✨ Key Features

### 1. Strong Clinical Reasoning

- Achieves SOTA or competitive performance on multiple Chinese medical QA benchmarks (e.g., CMExam, MedQA, Chinese-Exam).  
- Produces structured, step-by-step clinical reasoning aligned with real diagnostic thinking.

### 2. Ethics & Safety Alignment

Trained on rich medical ethics and patient safety corpora, covering:

- Clinical treatment ethics  
- Research ethics  
- End-of-life care  
- Public health ethics  
- Doctor–patient relationship ethics  
- Infection control, medication safety, incident reporting, legal compliance, etc.

Evaluated on MedSafety and MedEthics benchmarks, SafeMed-R1 achieves leading performance across sub-dimensions.

### 3. 🔐 Attack-Resistant Medical Alignment

Reinforcement learning is performed using red-team attack scenarios tailored to healthcare, including:

- Dangerous treatment requests  
- Requests to bypass standard-of-care  
- Misuse of medical devices, drugs, or emergency procedures  

SafeMed-R1 shows top performance on **Med-Safety-Bench**, evaluated against strong models including GPT-4o, DeepSeek-V3, Qwen3-235B-A22B, etc. It is capable of “saying no” correctly and explaining why, with guideline- and regulation-based reasoning.

---

## 🚀 Install & Deploy (vLLM)

### Step 1 — Install vLLM

Follow the official vLLM installation guide to select the version compatible with your environment (GPU/CPU/TPU, CUDA/ROCm, PyTorch):

- Docs: <https://docs.vllm.ai/en/latest/>

### Step 2 — Download Weights (Hugging Face)

Pull the model weights from Hugging Face. You can reference the repo directly in vLLM, or pre-download locally:

- Hugging Face: <https://huggingface.co/OpenMedZoo/SafeMed-R1/tree/main>

**Option A (recommended): Use the Hugging Face repo name directly in vLLM**

```bash
--model OpenMedZoo/SafeMed-R1
```

**Option B: Pre-download locally (example)**

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download OpenMedZoo/SafeMed-R1 \
  --local-dir ./models/SafeMed-R1
```

Then set:

```bash
--model ./models/SafeMed-R1
```

### Step 3 — Deploy (OpenAI-Compatible API Server)

Minimal vLLM `serve` example:

```bash
MODEL_PATH="${1:-./models/SafeMed-R1}"
TENSOR_PARALLEL_SIZE="${2:-4}"
PIPELINE_PARALLEL_SIZE="${3:-1}"
PORT=50050

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --served-model-name "safemed-r1" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --gpu-memory-utilization 0.9 \
  --disable-sliding-window \
  --max-model-len 4096 \
  --enable-prefix-caching
```

For additional parameters (multi-GPU, performance tuning, quantization, server arguments), see:

- <https://docs.vllm.ai/en/latest/>

---

## 📊 Evaluation
![Detailed Safety Results 1](./safemed-r1-safety-results1.png)
SafeMed-R1 surpasses its base and size-matched LLMs on both medical knowledge QA and safety/ethics, combining strong accuracy with robust safety alignment (see the associated radar plot in our paper/report).
![Safety Results](./safemed-r1-safety-results.png)
- **Knowledge QA and exams**: SafeMed-R1 leads or is competitive on MedQA-CN/TW, CMExam (val/test), PediaBench (MC), and CE-Phys/Pharm/Nurse, producing structured, step-by-step clinical reasoning (see Table 1 in the paper).
- **Safety and ethics**: SafeMed-R1 attains top or near-top results on MedSafety and MedEthics. On Med-Safety-Bench, SafeMed-R1 achieves consistently lower risk scores across evaluators (GPT-4o, DeepSeek-V3, Qwen3-235B-A22B), indicating stronger resistance to jailbreak/induction attacks and yielding a strong Overall-Average score (see Table 2).

---

## 🧪 Case Studies
![Case Example](./safemed-r1-safety-case.png)

In diverse safety-critical scenarios (e.g., biased/unsafe treatment requests, unnecessary invasive procedures), SafeMed-R1:

- Reliably flags risk  
- Declines harmful actions  
- Offers guideline-aligned, conservative alternatives with clear patient counseling  

This demonstrates practical safety, explainability, and adherence to clinical standards.

---

## 🔒 Safety, Privacy & Compliance

- Data anonymisation, authorisation protocols, and restrictions on secondary distribution  
- Adherence to local laws, regulations, and medical ethics guidelines  

**Disclaimer (English):**  
This project is intended solely for research and educational purposes and must not be used as a substitute for professional medical advice or diagnosis.

**免责声明（中文）：**  
隐私与合规：数据脱敏、授权与二次分发限制；遵循本地法律法规与医学伦理准则。  
本项目仅供科研与教育，不得替代专业医疗建议或诊疗。

---

## 🙏 Acknowledgements

We thank the open-source community for toolchains and benchmarks, especially:

- **LLaMA-Factory**: <https://github.com/hiyouga/LLaMA-Factory>  
- **MS-Swift**: <https://github.com/microsoft/MS-Swift>  

These projects greatly accelerated the construction and implementation of SafeMed-R1.
