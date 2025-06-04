import streamlit as st
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="CampEdUI Code Generator",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load the trained model and tokenizer"""
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_code(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generate code from text prompt"""
    if not model or not tokenizer:
        return "Model not loaded"
    
    # Format input
    input_text = f"Generate CampEdUI component: {prompt}"
    
    # Tokenize
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code

def render_component_preview(code):
    """Create a preview of what the component might look like"""
    preview_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .component-preview {{
                padding: 20px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: white;
            }}
        </style>
    </head>
    <body>
        <div class="component-preview">
            {code}
        </div>
    </body>
    </html>
    """
    return preview_html

def main():
    st.title("🎨 CampEdUI Code Generator")
    st.markdown("Generate CampEdUI components from natural language descriptions")
    
    # Sidebar for model configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="./codet5-camped-ui",
        help="Path to your trained CodeT5 model"
    )
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider("Max Length", 128, 1024, 512)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please check the model path.")
        st.info("💡 Make sure you've trained the model first using the training script.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Predefined examples
        st.subheader("Quick Examples")
        examples = [
            "Create a button with text 'Submit' and primary variant",
            "Create an accordion with title 'Settings' and configuration options",
            "Create a card with header 'User Profile' and user information",
            "Create a navigation menu with Home, About, and Contact links",
            "Create a form with email input and submit button",
            "Create a modal dialog with title 'Confirmation' and yes/no buttons"
        ]
        
        selected_example = st.selectbox("Choose an example:", ["Custom"] + examples)
        
        # Text input
        if selected_example == "Custom":
            prompt = st.text_area(
                "Describe the component you want to create:",
                placeholder="e.g., Create a button with text 'Click me' and secondary variant",
                height=100
            )
        else:
            prompt = st.text_area(
                "Describe the component you want to create:",
                value=selected_example,
                height=100
            )
        
        # Generate button
        generate_btn = st.button("🚀 Generate Code", type="primary")
    
    with col2:
        st.header("💻 Output")
        
        if generate_btn and prompt:
            with st.spinner("Generating code..."):
                generated_code = generate_code(model, tokenizer, prompt, max_length, temperature)
            
            st.subheader("Generated Code")
            st.code(generated_code, language="html")
            
            # Download button
            st.download_button(
                label="📥 Download Code",
                data=generated_code,
                file_name="component.jsx",
                mime="text/plain"
            )
            
            # Component preview (basic)
            st.subheader("Preview")
            st.info("💡 This is a basic HTML preview. For full CampEdUI styling, use this code in your React project.")
            
            try:
                # Simple HTML preview
                preview_html = render_component_preview(generated_code)
                st.components.v1.html(preview_html, height=200)
            except Exception as e:
                st.warning(f"Preview not available: {str(e)}")
        
        elif generate_btn and not prompt:
            st.warning("⚠️ Please enter a description first!")
    
    # Additional information
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("📚 Tips for Better Prompts")
        st.markdown("""
        - Be specific about component types (Button, Card, Accordion, etc.)
        - Mention variants when applicable (primary, secondary, outline)
        - Include content details (text, labels, descriptions)
        - Specify interactive behavior when needed
        """)
    
    with col4:
        st.subheader("🛠️ CampEdUI Components")
        st.markdown("""
        Supported components:
        - Button (variants: primary, secondary, outline)
        - Card (with header, content, footer)
        - Accordion (single, multiple)
        - Navigation Menu
        - Form elements
        - Modal/Dialog
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit and CodeT5 | "
        "[CampEdUI Documentation](https://ui.camped.academy/)"
    )

if __name__ == "__main__":
    main()