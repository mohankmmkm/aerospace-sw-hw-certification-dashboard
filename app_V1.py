import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os
import zipfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Aerospace Certification Assistant",
    page_icon="✈️",
    layout="wide",
)

# --- Helper Functions ---

def initialize_session_state():
    """Initializes all the necessary session state variables."""
    # General state
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "certification_type" not in st.session_state:
        st.session_state.certification_type = "Software"
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = None

    # State for each stage (for both SW and HW)
    stages = ["requirements", "design", "code", "test", "equipment_images"]
    for stage in stages:
        if f"{stage}_content" not in st.session_state:
            st.session_state[f"{stage}_content"] = None
        if f"{stage}_source" not in st.session_state:
            st.session_state[f"{stage}_source"] = None # 'AI' or 'Uploaded'

def configure_gemini():
    """Configures the Gemini API with the user's key and returns the model."""
    try:
        genai.configure(api_key=st.session_state.api_key)
        # Using gemini-1.5-flash for a balance of speed and capability
        model = genai.GenerativeModel('gemini-2.5-flash')
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.gemini_model = model
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini: {e}")
        st.session_state.gemini_model = None
        return False

def generate_content_from_gemini(prompt: str, stage_name: str):
def generate_content_from_gemini(prompt: str, stage_name: str, is_validation=False):
    """Generates content using the Gemini model and handles streaming output."""
    if not st.session_state.gemini_model:
        st.error("Gemini model is not configured. Please enter a valid API key.")
        return

    st.info(f"🤖 Generating {stage_name}... Please wait.")
    try:
        # Using stream=True for a better user experience
        response = st.session_state.gemini_model.generate_content(prompt, stream=True)
        full_response = ""
        # Create a placeholder to stream the response
        placeholder = st.empty()
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

        # Store the final result in session state
        st.session_state[f"{stage_name.lower().replace(' ', '_')}_content"] = full_response
        st.session_state[f"{stage_name.lower().replace(' ', '_')}_source"] = "AI"
        st.success(f"✅ {stage_name} generated successfully!")
        if not is_validation:
            # Store the final result in session state if it's a generation task
            st.session_state[f"{stage_name.lower().replace(' ', '_')}_content"] = full_response
            st.session_state[f"{stage_name.lower().replace(' ', '_')}_source"] = "AI"
            st.success(f"✅ {stage_name} generated successfully!")

    except Exception as e:
        st.error(f"An error occurred during generation: {e}")

def handle_file_upload(uploaded_file, stage_name: str, is_image=False):
    """Handles file uploads, decodes content, and stores it in session state."""
    if uploaded_file is not None:
        try:
            if is_image:
                content = Image.open(uploaded_file)
                st.image(content, caption=f"Uploaded {stage_name}", use_column_width=True)
            elif uploaded_file.name.endswith('.zip'):
                # Handle zip files
                content = ""
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    st.write(f"Files in zip: `{', '.join(file_list)}`")
                    for filename in file_list:
                        # Ignore macOS-specific metadata files
                        if not filename.startswith('__MACOSX/'):
                            try:
                                content += f"--- Content from {filename} ---\n"
                                content += zip_ref.read(filename).decode('utf-8') + "\n\n"
                            except (UnicodeDecodeError, KeyError):
                                content += f"--- Could not decode {filename} (likely a binary file) ---\n\n"
                st.text_area("Extracted Text Content", content, height=300, disabled=True)
            else:
                # Read as bytes and decode, handling potential errors
                content = uploaded_file.getvalue().decode("utf-8")
                st.text_area("Uploaded Content", content, height=300, disabled=True)



            st.session_state[f"{stage_name.lower().replace(' ', '_')}_content"] = content
            st.session_state[f"{stage_name.lower().replace(' ', '_')}_source"] = "Uploaded"
            st.success(f"✅ {stage_name} uploaded successfully!")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- UI Rendering Functions ---

def render_sidebar():
    """Renders the sidebar for API key input and certification type selection."""
    with st.sidebar:
        st.title("✈️ Configuration")
        st.markdown("Configure the assistant to begin the certification process.")

        # 1. API Key Input
        st.session_state.api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Get your API key from Google AI Studio.",
            value=st.session_state.api_key
        )

        if st.session_state.api_key:
            if configure_gemini():
                st.success("Gemini API Key configured!")
        else:
            st.warning("Please enter your Gemini API Key to enable AI features.")

        st.divider()

        # 2. Certification Type Selection
        st.session_state.certification_type = st.radio(
            "Choose Certification Process",
            ("Software", "Hardware"),
            horizontal=True,
        )

def render_stage_ui(stage_title: str, previous_stage_name: str, prompt_placeholder: str, file_types: list, is_image_stage=False):
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

        col1, col2 = st.columns(2)

        # --- AI Generation Column ---
        with col1:
            st.subheader(f"🤖 Generate with AI")
            if st.button(f"Generate {stage_title}", key=f"generate_{stage_key}", disabled=not st.session_state.api_key):
                # Create a dynamic prompt based on previous stage's content
                previous_content = st.session_state[f"{previous_stage_name}_content"]
                if isinstance(previous_content, Image.Image):
                     # If previous content is an image, we need a different kind of prompt
                    prompt = f"Based on the provided image of aerospace equipment, generate the {stage_title}."
                    # This part would need Gemini Vision Pro, which is handled by the model
                else:
                    prompt = f"You are an expert in aerospace certification (DO-178C for software, DO-254 for hardware). Based on the following {previous_stage_name.replace('_', ' ')}:\n\n---\n{previous_content}\n\n---\n\nGenerate the corresponding detailed **{stage_title}**. {prompt_placeholder}"
                generate_content_from_gemini(prompt, stage_title)

        # --- File Upload Column ---
        with col2:
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

            content = st.session_state[f"{stage_key}_content"]
            if is_image_stage and isinstance(content, Image.Image):
                st.image(content, use_column_width=True)
            elif not is_image_stage and isinstance(content, str):
                if stage_key == "code":
                    st.code(content, language='c', line_numbers=True)
                else:
                    st.markdown(content)

            if source == "AI":
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    with st.expander(f"View Generated {stage_title}"):
                        if is_image_stage and isinstance(content, Image.Image):
                             st.image(content, use_column_width=True)
                        elif not is_image_stage and isinstance(content, str):
                            if stage_key == "code":
                                st.code(content, language='c', line_numbers=True)
                            else:
                                st.markdown(content)
                with btn_col2:
                     if not is_image_stage:
                        st.download_button(f"Download {stage_title}", content, file_name=f"{stage_key}.md")

            elif source == "Uploaded":
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
            if st.button("Generate Requirements", disabled=not st.session_state.api_key or not prompt):
                full_prompt = f"You are an expert in aerospace certification (DO-178C). Generate detailed, verifiable high-level software requirements based on this prompt: '{prompt}'"
                generate_content_from_gemini(full_prompt, "Requirements")
        with col2:
            st.subheader("📄 Upload Existing")
            uploaded_file = st.file_uploader("Upload Requirements Document", type=['txt', 'md', 'pdf'])
            uploaded_file = st.file_uploader("Upload Requirements Document(s)", type=['txt', 'md', 'pdf', 'zip'])
            if uploaded_file:
                handle_file_upload(uploaded_file, "Requirements")

        if st.session_state.requirements_content:
            st.divider()
            source = st.session_state.requirements_source
            st.subheader(f"Current Requirements (Source: {source})")
            st.markdown(st.session_state.requirements_content)
            if source == "AI":
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    with st.expander("View Generated Requirements"):
                        st.markdown(st.session_state.requirements_content)
                with btn_col2:
                    st.download_button("Download Requirements", st.session_state.requirements_content, file_name="requirements.md")
            elif source == "Uploaded":
                # For the first stage, there's nothing to validate against
                st.success("Requirements uploaded. Proceed to the next stage.")


    # --- Subsequent Stages ---
    render_stage_ui("Design", "requirements", "The output should be in a standard design document format.", ['txt', 'md', 'pdf'])
    render_stage_ui("Code", "design", "The code should be in C, well-commented, and adhere to safety standards.", ['c', 'h', 'txt'])
    render_stage_ui("Test", "code", "Generate unit test cases and procedures to verify the code against the requirements.", ['txt', 'md', 'pdf'])
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
            if st.button("Generate Requirements", disabled=not st.session_state.api_key or not prompt):
                full_prompt = f"You are an expert in aerospace certification (DO-254). Generate detailed, verifiable high-level hardware requirements based on this prompt: '{prompt}'"
                generate_content_from_gemini(full_prompt, "Requirements")
        with col2:
            st.subheader("📄 Upload Existing")
            uploaded_file = st.file_uploader("Upload Requirements Document", type=['txt', 'md', 'pdf'])
            uploaded_file = st.file_uploader("Upload Requirements Document(s)", type=['txt', 'md', 'pdf', 'zip'])
            if uploaded_file:
                handle_file_upload(uploaded_file, "Requirements")

        if st.session_state.requirements_content:
            st.divider()
            source = st.session_state.requirements_source
            st.subheader(f"Current Requirements (Source: {source})")
            st.markdown(st.session_state.requirements_content)
            if source == "AI":
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    with st.expander("View Generated Requirements"):
                        st.markdown(st.session_state.requirements_content)
                with btn_col2:
                    st.download_button("Download Requirements", st.session_state.requirements_content, file_name="requirements.md")
            elif source == "Uploaded":
                st.success("Requirements uploaded. Proceed to the next stage.")

    # --- Subsequent Stages ---
    render_stage_ui("Design", "requirements", "The output should be a conceptual hardware design (e.g., block diagrams, component selection).", ['txt', 'md', 'pdf'])
    render_stage_ui("Equipment Images", "design", "The output should be a detailed textual description of what the physical hardware would look like.", ['png', 'jpg', 'jpeg'], is_image_stage=True)
    render_stage_ui("Test", "equipment_images", "Describe a test setup or a scenario to test the equipment shown in the image.", ['png', 'jpg', 'jpeg'], is_image_stage=True)
    file_types = ['txt', 'md', 'pdf', 'zip']
    image_file_types = ['png', 'jpg', 'jpeg', 'zip'] # Zip could contain images
    render_stage_ui("Design", "requirements", "The output should be a conceptual hardware design (e.g., block diagrams, component selection).", file_types)
    render_stage_ui("Equipment Images", "design", "The output should be a detailed textual description of what the physical hardware would look like.", image_file_types, is_image_stage=True)
    render_stage_ui("Test", "equipment_images", "Describe a test setup or a scenario to test the equipment shown in the image.", image_file_types, is_image_stage=True)


# --- Main Application ---
def main():
    st.title("Aerospace Certification Process Assistant")
    st.markdown("An AI-powered assistant to help streamline the DO-178C and DO-254 certification workflows.")

    initialize_session_state()
    render_sidebar()

    if st.session_state.certification_type == "Software":
        render_software_certification()
    else:
        render_hardware_certification()

if __name__ == "__main__":
    main()
