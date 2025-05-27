# English-to-Portuguese Neural Machine Translation (NMT) with LSTM-RNN Architecture

| Name            | Role              | LinkedIn                                      |
|-----------------|-------------------|-----------------------------------------------|
| Jason Emmanuel  | NLP Engineer | [linkedin.com/in/jasoneml](https://www.linkedin.com/in/jasoneml/) |

This project implements a **Neural Machine Translation (NMT)** system to translate sentences from **English to Portuguese**. It employs a sequence-to-sequence (Seq2Seq) architecture using **LSTM (Long Short-Term Memory)** networks enhanced with an **attention mechanism**, all developed using **TensorFlow 2.x**. The goal is to provide a deep learning based approach that effectively learns the mapping between English and Portuguese sentences, capturing semantic and syntactic relationships better than traditional statistical or rule-based methods.

![image](https://github.com/user-attachments/assets/2baa89c7-ece8-4d3e-9732-a0cf3db85c86)

---

## 📖 Overview

Neural Machine Translation has revolutionized the field of language translation by using deep learning models that can jointly learn to encode input sentences and decode them into a target language. This project builds such a model specifically for English to Portuguese translation. Compared to phrase-based or rule-based MT systems, neural models are capable of learning more fluent and contextually relevant translations by leveraging large datasets and complex neural network architectures.

This implementation uses an LSTM-based Seq2Seq model with an attention mechanism, which allows the decoder to selectively focus on different parts of the input sentence at each decoding step. This greatly improves translation quality, especially for longer sentences or sentences with complex structures.

---

## 🏗️ Model Architecture

The model consists of three main components:

### 1. Encoder

- **Embedding Layer:** Converts input English tokens into dense vector representations.
- **LSTM Layer:** Processes the embedded input sequence and produces hidden states for each timestep, capturing the context of the input sentence.
- **Outputs:** Final hidden and cell states to initialize the decoder, along with outputs for attention.

### 2. Attention Mechanism

- **Type:** Bahdanau-style additive attention.
- **Function:** Computes a weighted context vector at each decoder step by attending over all encoder outputs. This allows the decoder to focus on relevant parts of the source sentence dynamically.
- **Benefit:** Improves handling of long sentences and helps the model learn alignments between source and target words.

### 3. Decoder

- **Embedding Layer:** Converts Portuguese tokens into vectors.
- **LSTM Layer:** Receives previous output and context vector to generate next hidden state.
- **Dense Output Layer:** Maps LSTM outputs to vocabulary size, producing logits for next-token prediction.

![image](https://github.com/user-attachments/assets/572a74fa-ae51-475d-8633-b761c8352b29)

---

## 📚 Dataset

- **Source:** Parallel English-Portuguese sentence pairs, for example from datasets like [Tatoeba](https://tatoeba.org/) or [ManyThings.org](https://www.manythings.org/anki/).
- **Format:** Each line contains an English sentence and its Portuguese translation separated by a tab.
- **Size:** Typically tens of thousands to hundreds of thousands of sentence pairs, depending on source.
- **Preprocessing:**  
  - Lowercasing all text for normalization.  
  - Removing extra spaces and special characters.  
  - Adding `<start>` and `<end>` tokens to each target sentence to denote sentence boundaries.

---

## 🧹 Preprocessing

- **Tokenization:** Using TensorFlow's `Tokenizer` to convert sentences into sequences of integer token IDs.
- **Vocabulary Construction:** Separate vocabularies for English and Portuguese with size limits (e.g., top 10,000 words).
- **Padding:** Input and output sequences are padded to fixed lengths for batch processing.
- **Batching:** Data is grouped into batches for efficient training.
- **Data Pipeline:** TensorFlow `tf.data.Dataset` API is used to shuffle, batch, and prefetch data.

---

## 🏋️ Training

- **Loss Function:** Sparse categorical cross-entropy with masking to ignore padded tokens.
- **Optimizer:** Adam optimizer with an initial learning rate of 0.001.
- **Teacher Forcing:** During training, the actual target token at the previous step is fed into the decoder instead of the predicted token.
- **Epochs:** Training for multiple epochs until convergence or early stopping criteria met.
- **Metrics:** Accuracy calculated on non-padding tokens; training and validation loss monitored.
- **Checkpointing:** Model weights saved periodically to allow resuming training and to preserve best-performing models.

---

## 🧠 Inference

- **Encoding:** Input English sentence is tokenized and passed through the encoder to generate initial states.
- **Decoding:**  
  - Start token (`<start>`) is fed as initial input.  
  - Decoder predicts next token step-by-step using previous prediction and attention over encoder outputs.  
  - Greedy decoding is applied (choosing highest probability token each step).  
  - Decoding stops when `<end>` token is generated or max length reached.
- **Postprocessing:** Token IDs are converted back to Portuguese words, and special tokens removed.

---

## 📈 Evaluation

- **Qualitative Evaluation:** Manually inspecting sample translations to assess fluency and correctness.
- **Quantitative Metrics:**  
  - BLEU (Bilingual Evaluation Understudy) score to compare model output to reference translations.  
  - Optionally METEOR or ROUGE scores.
- **Visualization:**  
  - Plot training/validation loss and accuracy curves.  
  - Attention heatmaps to visualize where the model focuses during translation.

---

## 🛠️ Tools

| Tool / Library       | Purpose                                        | Notes                                      |
|---------------------|------------------------------------------------|--------------------------------------------|
| **TensorFlow 2.x**   | Deep learning framework for model building and training | Core library for all neural network operations |
| **NumPy**            | Numerical computing and array manipulation     | Used for data handling and preprocessing   |
| **Matplotlib**       | Plotting training metrics and results          | Visualizes loss, accuracy, and attention   |
| **Seaborn**          | Statistical data visualization                  | Enhanced visualization styles for plots    |
| **NLTK**             | Natural Language Processing utilities           | Tokenization, evaluation metrics (optional)|
| **Python 3.7+**      | Programming language                             | Primary language used                       |
| **Git**              | Version control system                           | Source code management                      |
| **Jupyter Notebook** | Experimentation and visualization (optional)   | Useful during development                   |

---

## ⚙️ Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/nmt-en-pt.git
cd nmt-en-pt
pip install -r requirements.txt
