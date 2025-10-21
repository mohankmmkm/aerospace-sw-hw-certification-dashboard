import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import numpy as np
import zipfile
from pypdf import PdfReader
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="Aerospace Certification Assistant",
    page_icon="✈️",
    layout="wide",
)

# --- Model Loading ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8n model and caches it."""
    return YOLO('yolov8n.pt')

# --- Helper Functions ---

def initialize_session_state():
    """Initializes all the necessary session state variables."""
    # General state
    if "certification_type" not in st.session_state:
        st.session_state.certification_type = None

    # API Keys and Models
    if "text_api_key" not in st.session_state:
        st.session_state.text_api_key = ""
    if "gemini_text_model" not in st.session_state:
        st.session_state.gemini_text_model = None

    # State for each stage (for both SW and HW)
    stages = ["requirements", "design", "code", "test", "equipment_images"]
    for stage in stages:
        if f"{stage}_content" not in st.session_state:
            st.session_state[f"{stage}_content"] = None
        if f"{stage}_source" not in st.session_state:
            st.session_state[f"{stage}_source"] = None # 'AI' or 'Uploaded'
        if f"view_{stage}" not in st.session_state:
            st.session_state[f"view_{stage}"] = False

def configure_text_model():
    """Configures the Gemini text model with the user's key."""
    if not st.session_state.text_api_key:
        st.session_state.gemini_text_model = None
        return
    try:
        genai.configure(api_key=st.session_state.text_api_key)
        st.session_state.gemini_text_model = genai.GenerativeModel('gemini-2.5-flash')
        return True
    except Exception as e:
        st.sidebar.error(f"Text Model Error: {e}")
        st.session_state.gemini_text_model = None
        return

def generate_content_from_gemini(prompt: str, stage_name: str, is_validation=False):
    """Generates content using the Gemini model and handles streaming output."""
    if not st.session_state.gemini_text_model:
        st.error("Gemini model is not configured. Please enter a valid API key.")
        return

    st.info(f"🤖 Generating {stage_name}... Please wait.")
    try:
        # Using stream=True for a better user experience
        response = st.session_state.gemini_text_model.generate_content(prompt, stream=True)
        full_response = ""
        # Create a placeholder to stream the response
        placeholder = st.empty()
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

        if not is_validation:
            # Store the final result in session state if it's a generation task
            st.session_state[f"{stage_name.lower().replace(' ', '_')}_content"] = full_response
            st.session_state[f"{stage_name.lower().replace(' ', '_')}_source"] = "AI"
            st.success(f"✅ {stage_name} generated successfully!")

    except Exception as e:
        st.error(f"An error occurred during generation: {e}")

def analyze_image_with_yolo(image: Image.Image, stage_name: str):
    """Analyzes an image using the YOLO model and stores the result."""
    if not image:
        st.error("No image provided for analysis.")
        return

    st.info(f"🤖 Analyzing {stage_name} with YOLOv8...")
    try:
        yolo_model = load_yolo_model()
        # Convert PIL Image to NumPy array
        image_np = np.array(image)
        # Perform detection
        results = yolo_model(image_np)
        # Plot results on the image
        annotated_image_np = results[0].plot()
        # Convert back to PIL Image
        annotated_image = Image.fromarray(annotated_image_np)

        st.session_state[f"{stage_name.lower().replace(' ', '_')}_content"] = annotated_image
        st.success(f"✅ {stage_name} analyzed successfully with YOLO!")

    except Exception as e:
        st.error(f"An error occurred during YOLO analysis: {e}")

def handle_file_upload(uploaded_file, stage_name: str, is_image=False):
    """Handles file uploads, decodes content, and stores it in session state."""
    if uploaded_file is not None:
        try:
            if is_image:
                # For images, we can also handle zips of images, but for now, we'll stick to single images.
                if uploaded_file.name.endswith('.zip'):
                    st.warning("Zip file uploads for images are not fully supported yet. Please upload a single image.")
                    content = None # Or handle zip extraction for images
                else:
                    content = Image.open(uploaded_file)
            elif uploaded_file.name.endswith('.zip'):
                # Handle zip files
                content = ""
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    for filename in file_list:
                        # Ignore macOS-specific metadata files
                        if not filename.startswith('__MACOSX/'):
                            try:
                                content += f"--- Content from {filename} ---\n"
                                content += zip_ref.read(filename).decode('utf-8') + "\n\n"
                            except (UnicodeDecodeError, KeyError):
                                content += f"--- Could not decode {filename} (likely a binary file) ---\n\n"
            elif uploaded_file.name.endswith('.pdf'):
                # Handle PDF files
                pdf_reader = PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            else:
                # Read as bytes and decode, handling potential errors
                content = uploaded_file.getvalue().decode("utf-8")

            if content:
                st.session_state[f"{stage_name.lower().replace(' ', '_')}_content"] = content
                st.session_state[f"{stage_name.lower().replace(' ', '_')}_source"] = "Uploaded"
                st.success(f"✅ {stage_name} uploaded successfully!")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- UI Rendering Functions ---

def render_sidebar():
    """Renders the sidebar for API key input and certification type selection."""
    with st.sidebar:
        st.title("✈️ Cert-Assist")
        st.markdown("Choose a certification process to begin.")

        col1, col2 = st.columns(2)
        if col1.button("Software", use_container_width=True):
            st.session_state.certification_type = "Software"
        if col2.button("Hardware", use_container_width=True):
            st.session_state.certification_type = "Hardware"

        st.divider()

        if st.session_state.certification_type == "Software":
            st.header("Software Config")
            st.session_state.text_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="API key for text generation (e.g., gemini-2.5-flash).",
                value=st.session_state.text_api_key
            )
            if st.session_state.text_api_key:
                if configure_text_model():
                    st.success("Text model configured!")
            else:
                st.warning("Please enter an API key to enable AI features.")

        elif st.session_state.certification_type == "Hardware":
            st.header("Hardware Config")
            st.session_state.text_api_key = st.text_input(
                "Gemini Text API Key",
                type="password",
                help="API key for text generation (e.g., gemini-2.5-flash).",
                value=st.session_state.text_api_key
            )
            if st.session_state.text_api_key:
                if configure_text_model():
                    st.success("Text model configured!")

def render_stage_ui(stage_title: str, previous_stage_name: str, prompt_placeholder: str, file_types: list, is_image_stage=False, is_yolo_stage=False):
    """
    A generic function to render the UI for a single certification stage.
    This reduces code duplication significantly.
    """
    stage_key = stage_title.lower().replace(' ', '_')

    with st.expander(f"**Stage: {stage_title}**", expanded=st.session_state[f"{previous_stage_name}_content"] is not None):
        # Check if the previous stage is completed
        if not st.session_state[f"{previous_stage_name}_content"]:
            st.warning(f"Please complete the '{previous_stage_name.replace('_', ' ').title()}' stage first.")
            return

        st.markdown(f"Generate the **{stage_title}** based on the previous stage, or upload your own document/image.")

        # For YOLO stage, we only allow upload, not generation.
        if not is_yolo_stage:
            col1, col2 = st.columns(2)
            # --- AI Generation Column ---
            with col1:
                st.subheader(f"🤖 Generate with AI")
                # Disable button if the required model is not configured
                is_button_disabled = (not is_image_stage and not st.session_state.gemini_text_model)

                if st.button(f"Generate {stage_title}", key=f"generate_{stage_key}", disabled=is_button_disabled):
                    # Create a dynamic prompt based on previous stage's content
                    previous_content = st.session_state[f"{previous_stage_name}_content"]

                    if is_image_stage:
                        # This path is now unused for YOLO, but kept for potential future image generation models
                        prompt = f"Generate a photorealistic image of an aerospace hardware component based on the following design description: {previous_content}. {prompt_placeholder}"
                        # A call to a future image generation function would go here.
                    else:
                        # Prompt for text generation
                        prompt = f"You are an expert in aerospace certification (DO-178C for software, DO-254 for hardware). Based on the following {previous_stage_name.replace('_', ' ')}:\n\n---\n{previous_content}\n\n---\n\nGenerate the corresponding detailed **{stage_title}**. {prompt_placeholder}"
                        generate_content_from_gemini(prompt, stage_title)
            upload_col = col2
        else:
            upload_col = st.container()

        # --- File Upload Column ---
        with upload_col:
            st.subheader(f"📄 Upload Existing")
            uploaded_file = st.file_uploader(
                f"Upload {stage_title} file",
                type=file_types,
                key=f"upload_{stage_key}"
            )
            if uploaded_file:
                handle_file_upload(uploaded_file, stage_title, is_image=is_image_stage)

        # --- Display Content ---
        if st.session_state[f"{stage_key}_content"]:
            st.divider()
            source = st.session_state[f"{stage_key}_source"]
            st.subheader(f"Current {stage_title} (Source: {source})")

            content = st.session_state.get(f"{stage_key}_content")

            if source == "AI":
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button(f"View Generated {stage_title}", key=f"view_btn_{stage_key}"):
                    st.session_state[f"view_{stage_key}"] = not st.session_state[f"view_{stage_key}"]

                with btn_col2:
                    if is_image_stage and isinstance(content, Image.Image):
                        buf = io.BytesIO()
                        content.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(f"Download {stage_title}", byte_im, file_name=f"{stage_key}.png", mime="image/png")
                    elif not is_image_stage:
                        st.download_button(f"Download {stage_title}", content, file_name=f"{stage_key}.md")


                if st.session_state[f"view_{stage_key}"]:
                    if is_image_stage and isinstance(content, Image.Image):
                         st.image(content, use_column_width=True)
                    elif not is_image_stage and isinstance(content, str):
                        if stage_key == "code":
                            st.code(content, language='c', line_numbers=True)
                        else:
                            st.markdown(content)

            elif source == "Uploaded":
                # Display what was uploaded
                if is_image_stage and isinstance(content, Image.Image):
                    st.image(content, caption=f"Uploaded {stage_title}", use_column_width=True)
                elif not is_image_stage and isinstance(content, str):
                    with st.expander(f"View Uploaded {stage_title} Content"):
                        st.text_area(
                            label="Uploaded Content",
                            value=content,
                            height=300,
                            disabled=True,
                            key=f"text_area_{stage_key}"
                        )

                if is_yolo_stage and isinstance(content, Image.Image):
                    if st.button(f"Analyze with YOLO", key=f"analyze_{stage_key}"):
                        # The source remains "Uploaded", but we process the content
                        analyze_image_with_yolo(content, "Equipment Images")
                        # After analysis, the content is the annotated image, and we can treat it like an AI result
                        st.session_state[f"{stage_key}_source"] = "AI"
                        st.rerun()

                if st.button(f"Validate Uploaded {stage_title}", key=f"validate_{stage_key}"):
                    st.info(f"Validating {stage_title} against {previous_stage_name.replace('_', ' ')}...")
                    previous_content = st.session_state[f"{previous_stage_name}_content"]
                    current_content = st.session_state[f"{stage_key}_content"]

                    validation_prompt = (f"You are an expert aerospace compliance auditor. Your task is to validate if the `{stage_title}` artifact correctly and completely implements the `{previous_stage_name}` artifact.\n\n"
                                         f"**PREVIOUS STAGE: {previous_stage_name.upper()}**\n---\n{previous_content}\n---\n\n"
                                         f"**CURRENT STAGE: {stage_title.upper()}**\n---\n{current_content}\n---\n\n"
                                         "Please provide a detailed validation report. Identify any gaps, inconsistencies, or requirements that are not met. If all looks good, confirm compliance.")

                    generate_content_from_gemini(validation_prompt, f"{stage_title} Validation", is_validation=True)

def render_software_certification():
    """Renders the UI for the Software Certification process."""
    st.header(" Avionics Software Certification (DO-178C)")

    # --- Stage 1: Requirements ---
    with st.expander("**Stage: Requirements**", expanded=True):
        st.markdown("Start by generating requirements from a prompt or uploading an existing document.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 Generate with AI")
            prompt = st.text_area("Enter a high-level prompt to generate software requirements:",
                                  placeholder="e.g., 'Generate high-level software requirements for a flight control system's pitch control loop.'",
                                  height=150)
            if st.button("Generate Requirements", disabled=not st.session_state.gemini_text_model or not prompt):
                full_prompt = f"You are an expert in aerospace certification (DO-178C). Generate detailed, verifiable high-level software requirements based on this prompt: '{prompt}'"
                generate_content_from_gemini(full_prompt, "Requirements")
        with col2:
            st.subheader("📄 Upload Existing")
            uploaded_file = st.file_uploader("Upload Requirements Document(s)", type=['txt', 'md', 'pdf', 'zip'])
            if uploaded_file:
                handle_file_upload(uploaded_file, "Requirements")

        if st.session_state.requirements_content:
            st.divider()
            source = st.session_state.requirements_source
            st.subheader(f"Current Requirements (Source: {source})")
            if source == "AI":
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button("View Generated Requirements", key="view_sw_reqs"):
                    st.session_state.view_requirements = not st.session_state.view_requirements

                btn_col2.download_button("Download Requirements", st.session_state.requirements_content, file_name="requirements.md")

                if st.session_state.view_requirements:
                    st.markdown(st.session_state.requirements_content)

            elif source == "Uploaded":
                # For the first stage, there's nothing to validate against
                with st.expander("View Uploaded Requirements Content"):
                    st.text_area(
                        label="Uploaded Content",
                        value=st.session_state.requirements_content,
                        height=300,
                        disabled=True,
                        key="text_area_requirements"
                    )
                st.info("Requirements uploaded. Proceed to the next stage to validate the Design against them.")


    # --- Subsequent Stages ---
    file_types = ['txt', 'md', 'pdf', 'zip']
    render_stage_ui("Design", "requirements", "The output should be in a standard design document format.", file_types)
    render_stage_ui("Code", "design", "The code should be in C, well-commented, and adhere to safety standards.", ['c', 'h', 'txt', 'zip'])
    render_stage_ui("Test", "code", "Generate unit test cases and procedures to verify the code against the requirements.", file_types)


def render_hardware_certification():
    """Renders the UI for the Hardware Certification process."""
    st.header("✈️ Airborne Electronic Hardware Certification (DO-254)")

    # --- Stage 1: Requirements ---
    with st.expander("**Stage: Requirements**", expanded=True):
        st.markdown("Start by generating requirements from a prompt or uploading an existing document.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 Generate with AI")
            prompt = st.text_area("Enter a high-level prompt to generate hardware requirements:",
                                  placeholder="e.g., 'Generate high-level hardware requirements for a data acquisition unit for engine sensors.'",
                                  height=150)
            if st.button("Generate Requirements", disabled=not st.session_state.gemini_text_model or not prompt):
                full_prompt = f"You are an expert in aerospace certification (DO-254). Generate detailed, verifiable high-level hardware requirements based on this prompt: '{prompt}'"
                generate_content_from_gemini(full_prompt, "Requirements")
        with col2:
            st.subheader("📄 Upload Existing")
            uploaded_file = st.file_uploader("Upload Requirements Document(s)", type=['txt', 'md', 'pdf', 'zip'])
            if uploaded_file:
                handle_file_upload(uploaded_file, "Requirements")

        if st.session_state.requirements_content:
            st.divider()
            source = st.session_state.requirements_source
            st.subheader(f"Current Requirements (Source: {source})")
            if source == "AI":
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button("View Generated Requirements", key="view_hw_reqs"): # Unique key
                    st.session_state.view_requirements = not st.session_state.view_requirements

                btn_col2.download_button("Download Requirements", st.session_state.requirements_content, file_name="requirements.md")

                if st.session_state.view_requirements:
                    st.markdown(st.session_state.requirements_content)
            elif source == "Uploaded":
                with st.expander("View Uploaded Requirements Content"):
                    st.text_area(
                        label="Uploaded Content",
                        value=st.session_state.requirements_content,
                        height=300,
                        disabled=True,
                        key="text_area_hw_requirements"
                    )
                st.info("Requirements uploaded. Proceed to the next stage to validate the Design against them.")

    # --- Subsequent Stages ---
    file_types = ['txt', 'md', 'pdf', 'zip']
    image_file_types = ['png', 'jpg', 'jpeg', 'zip'] # Zip could contain images
    render_stage_ui("Design", "requirements", "The output should be a conceptual hardware design (e.g., block diagrams, component selection).", file_types, is_image_stage=False)
    render_stage_ui("Equipment Images", "design", "Upload an image of the hardware. You can then analyze it with YOLO.", image_file_types, is_image_stage=True, is_yolo_stage=True)
    render_stage_ui("Test", "equipment_images", "Describe a test setup or a scenario to test the equipment shown in the image.", file_types, is_image_stage=False)


# --- Main Application ---
def main():
    load_yolo_model() # Pre-load the model
    st.title("Aerospace Certification Process Assistant")
    st.markdown("An AI-powered assistant to help streamline the DO-178C and DO-254 certification workflows.")

    initialize_session_state()

    # Reset view state when certification type changes to prevent stale views
    if st.session_state.certification_type != st.session_state.get("_last_cert_type"):
        for key in st.session_state.keys():
            if key.startswith("view_"):
                st.session_state[key] = False
    st.session_state._last_cert_type = st.session_state.certification_type

    render_sidebar()

    if not st.session_state.certification_type:
        st.info("⬅️ Please select a certification process from the sidebar to begin.")
    elif st.session_state.certification_type == "Software":
        render_software_certification()
    else:
        render_hardware_certification()

if __name__ == "__main__":
    main()
