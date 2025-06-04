# Text-to-Website-Generation-Using-CampEdUI
🧠 CampEdUI: Code Generation from Prompts using CodeT5
This project fine-tunes the Salesforce/CodeT5-base model to generate code from natural language prompts. It is aimed at enabling intelligent code generation in educational or structured coding environments where consistent prompt–response behavior is valuable.

📌 Project Overview
CampEdUI is built to support automatic code generation from structured textual prompts. Using a dataset of { "prompt": ..., "code": ... } pairs, the model is trained to learn mappings from descriptive instructions to functional code blocks. The dataset can include any programming language or markup (e.g., HTML, Python, Java).

This is particularly useful in educational contexts, code-assist tools, or prompt-based generation systems where accurate, reliable code output is needed based on specific textual instructions.

🔍 Features
✅ Fine-tunes CodeT5 on prompt–code datasets

✅ Compatible with CPU and GPU (with fp16 support)

✅ Token-level accuracy metric for evaluation

✅ Generates complete code blocks from natural language prompts

✅ Saves the best checkpoint based on validation performance

The Falcon-7B model is fine-tuned specifically for frontend-like code (HTML/CSS) generation tasks from descriptive language prompts. It uses instruction-tuned formatting to guide the model toward precise code completions.

Base model: tiiuae/falcon-7b-instruct

Trained with prompt-response style formatting

Optimized for structured generation (HTML, CSS layouts, UI elements)

Uses PEFT (Parameter-Efficient Fine-Tuning) or LoRA for resource-efficient training

To fine-tune the models effectively, synthetic training data was created using a combination of manual curation and semi-automated prompt generation techniques.

Prompt Templates:
A set of predefined templates was used to generate variations. 
Examples:
"Build a [component] with [style]."
"Write HTML for a [UI element] that is [aligned / colored]."

Code Snippet Bank:
Commonly used HTML/CSS/Python code patterns were stored and reused in different combinations to increase variety and realism.
Randomization:
Element names (e.g., div, button, form) and style properties (color, alignment, font-size) were randomly swapped to create hundreds of diverse pairs.
Manual Review:
After generation, samples were reviewed and cleaned to ensure syntactic correctness and prompt-code alignment.
Dataset Format:
The resulting synthetic datasets were saved as .json files:
train_prompts.json, val_prompts.json for CodeT5


