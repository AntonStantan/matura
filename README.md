# Neural Predictive Calculator
Scientific research on the ability of neural networks to solve simple arithmetic expressions.

## Evaluating the Arithmetic Capabilities of Neural Networks

This project investigates why neural networks often struggle with simple arithmetic expressions. It involves building and evaluating various neural network architectures—including Feed-Forward Neural Networks (FNNs), Recurrent Neural Networks (RNNs), attention-based RNNs, and Transformers—to assess their performance on arithmetic tasks.

## Introduction
The goal of this project is to evaluate the arithmetic capabilities of different neural network architectures and understand their limitations with simple arithmetic expressions. This is done by building and testing various models on arithmetic problems.

## Repository Structure
The repository is organized as follows:
- [`FNN/`](./FNN/): Contains notebooks and code related to Feed-Forward Neural Networks.
- [`RNN/`](./RNN/): Contains notebooks and code related to Recurrent Neural Networks.
- [`attentional-RNN/`](./attentional-RNN/): Contains notebooks for RNNs with attention mechanisms.
- [`transformer/`](./transformer/): Contains notebooks for Transformer-based models.
- [`documentation/`](./documentation/): Contains project documentation, such as literature reviews.
- [`pre-trained-tranformers/`](./pre-trained-tranformers/): Contains notebooks for pre-trained transformer models.
- [`zwischenPresentation/`](./zwischenPresentation/): Contains presentation materials.
- [`zwischenProdukt/`](./zwischenProdukt/): Contains intermediate products and results.

## Important Files
Here are some of the most important notebooks and scripts in this repository, showcasing core implementations and utilities:

- **Scripts:**
  - [`GetXY.py`](./GetXY.py): A script for generating the dataset.
  - [`FNN1_1.py`](./FNN1_1.py): A script version of the FNN model.
  - [`get_p-value.py`](./get_p-value.py): A script for obtaining the p-value caluclated for the findings table
  - [`EvaluationMatura.xlsx`](./EvaluationMatura.xlsx): Findings Table
  - [`FNN/NPC-final.py`](./FNN/NPC-final.py): Script for demonstrating the best model `FNN2` on user-requested expressions.
- **Notebooks:**
  - **FNN:**
    - [`FNN/FNN1.ipynb`](./FNN/FNN1.ipynb): Initial Feed-Forward Neural Network implementation. Bugged on github.com, but works on VSC.
    - [`FNN/FNN2.ipynb`](./FNN/FNN2.ipynb): An iteration on the FNN model.
  - **RNN:**
    - [`RNN/RNN0.ipynb`](./RNN/RNN0.ipynb): A basic Recurrent Neural Network implementation for arithmetic tasks.
    - [`RNN/RNN2.ipynb`](./RNN/RNN2.ipynb): An iteration on the RNN model.
  - **Attentional RNN:**
    - [`attentional-RNN/g4gLSTM.ipynb`](./attentional-RNN/g4gLSTM.ipynb): Implements an LSTM-based RNN with an attention mechanism.
  - **Transformer:**
    - [`transformer/transformer0.ipynb`](./transformer/transformer0.ipynb): A basic Transformer architecture for arithmetic tasks.
    - [`transformer/transformer4.ipynb`](./transformer/transformer4.ipynb): Explores the use of a Transformer architecture for arithmetic tasks.
    - [`transformer/transformer5.ipynb`](./transformer/transformer5.ipynb): A further iteration on the Transformer model.
  - **Pre-trained Fine-tuned Transformers:**
    - [`pre-trained-tranformers/gemma_huggingface.ipynb`](./pre-trained-tranformers/gemma_huggingface.ipynb): Fine-tuning the Gemma 270M model from Hugging Face.
    - [`pre-trained-tranformers/big_gemma_huggingface.ipynb`](./pre-trained-tranformers/big_gemma_huggingface.ipynb): Fine-tuning the Gemma 1B model from Hugging Face.
    - [`pre-trained-tranformers/gemini_vertex.ipynb`](./pre-trained-tranformers/gemini_vertex.ipynb): Fine-tuning the Gemini model using Vertex AI.

- **Documentation:**
  - [`documentation/findings/findings.pdf`](./documentation/findings/findings.pdf): Fidnings document
  - [`documentation/methodology/methodology.pdf`](./documentation/methodology/methodology.pdf): Methodology document
  - [`documentation/AI_assistance/ai_assistance.pdf`](./documentation/AI_assistance/ai_assistance.pdf): Document stating the ai assistance used for this project.
  - [`zwischenProdukt/LiteraturstudieAnton.pdf`](./zwischenProdukt/LiteraturstudieAnton.pdf): Literature Review (Zwischenprodukt)
  - [`documentation/Process/`](./documentation/Process/): Arbeitsprozess



## Dependencies
The required Python packages for most of the project are listed in `requirements.txt`. These are the libraries used on the laptop. They will be running on CPU.
These libraries should be enough for running the [`demonstation script`](./FNN/NPC-final.py), try it out!

The packages can be installed using pip:
```bash
pip install -r requirements.txt
```
Incase you have a nvidia Jetson device, try the packages listed in `requirements_GPU.txt`
```bash
pip install -r requirements_GPU.txt
```

**Note:** The notebooks within the [`pre-trained-transformers/`](./pre-trained-transformers/) directory have different dependency requirements. Please refer to the methodology documentation for more details.
