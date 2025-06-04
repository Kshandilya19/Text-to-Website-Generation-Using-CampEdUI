import torch
from pathlib import Path
from transformers import AutoTokenizer, T5ForConditionalGeneration

def main():
    # 1) Locate the fine‑tuned checkpoint folder
    script_dir     = Path(__file__).parent
    checkpoint_dir = script_dir.parent / "codet5-finetuned"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")

    print("→ Loading tokenizer & model from:", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model     = T5ForConditionalGeneration.from_pretrained(str(checkpoint_dir))

    # 2) Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"→ Using device: {device}")
    model.to(device)

    # 3) Single‑shot prompt (must match exactly the style of your training entries)
    prompt_text = (
        "Generate CampEdUI component: Write a React component named UserProfileCard. "
        "It should import { Card, CardHeader, CardTitle, CardContent, CardFooter } from \"@camped-ui/card\". "
        "Then render:\n\n"
        "<Card>\n"
        "  <CardHeader>\n"
        "    <CardTitle>User Profile</CardTitle>\n"
        "  </CardHeader>\n"
        "  <CardContent>\n"
        "    <div id=\"user-profile\">\n"
        "      <p>Name: John Doe</p>\n"
        "      <p>Email: john.doe@example.com</p>\n"
        "    </div>\n"
        "  </CardContent>\n"
        "  <CardFooter>\n"
        "    <button>Edit Profile</button>\n"
        "  </CardFooter>\n"
        "</Card>\n\n"
        "Output only the raw code for `export function UserProfileCard() { … }`, with no backticks or extra comments."
    )

    # 4) Tokenize (raise max_length to 768 to avoid truncation)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=768,
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 5) Generate with beam search + repetition controls
    print("→ Generating code… (beam search + repetition controls)")
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=768,            # match tokenizer’s max_length
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 6) Decode & print
    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    ).strip()

    print("\n===== GENERATED CODE =====\n")
    print(generated_text)
    print("\n===== END GENERATED CODE =====\n")


if __name__ == "__main__":
    main()
