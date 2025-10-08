import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta, date, date as date_class
from typing import Dict, Any, List
from snowflake.snowpark.context import get_active_session
import _snowflake
import io

# Set page configuration
st.set_page_config(
    page_title="Call Center Analytics Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #388e3c;
    }
    .insight-box {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .stButton > button {
        width: 100%;
    }
    .sidebar-section {
        margin: 1rem 0;
    }
    .sidebar .stButton > button {
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .sidebar .stButton > button[data-baseweb="button"][kind="primary"] {
        background-color: #1976d2;
        border-color: #1976d2;
        font-weight: bold;
    }
    .sidebar .stButton > button[data-baseweb="button"][kind="secondary"]:hover {
        background-color: #e3f2fd;
        border-color: #1976d2;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Constants
SEMANTIC_MODEL_STAGE_PATH = "@PUBLIC.SEMANTIC_MODEL_STAGE/call_center_analytics_model.yaml"
API_ENDPOINT = "/api/v2/cortex/analyst/message"
API_TIMEOUT = 60000  # 60 seconds in milliseconds

# @st.cache_data
def get_snowpark_session():
    """Get Snowpark session for SiS"""
    try:
        session = get_active_session()
        return session
    except Exception as e:
        st.error(f"Error getting Snowpark session: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def read_semantic_model_from_stage() -> str:
    """Read semantic model YAML content from Snowflake stage"""
    try:
        session = get_snowpark_session()
        if not session:
            return None
        
        # Read the YAML file content from the stage
        query = f"""
        SELECT $1 as yaml_content 
        FROM {SEMANTIC_MODEL_STAGE_PATH}
        WHERE $1 IS NOT NULL AND LENGTH(TRIM($1)) > 0
        """
        
        result = session.sql(query).collect()
        
        if result and len(result) > 0:
            # Filter out None/empty values and concatenate all rows to get the full YAML content
            yaml_lines = []
            for row in result:
                if row['YAML_CONTENT'] is not None and row['YAML_CONTENT'].strip():
                    yaml_lines.append(row['YAML_CONTENT'])
            
            if yaml_lines:
                yaml_content = '\n'.join(yaml_lines)
                return yaml_content
            else:
                st.error("YAML file appears to be empty or contains only null values")
                return None
        else:
            st.error(f"No valid content found in stage file: {SEMANTIC_MODEL_STAGE_PATH}")
            return None
            
    except Exception as e:
        st.error(f"Error reading semantic model from stage: {str(e)}")
        return None

def query_cortex_analyst_rest_api(question: str, conversation_history: List[Dict] = None) -> Dict[Any, Any]:
    """Query Cortex Analyst using Snowflake internal REST API"""
    try:
        # Read the semantic model YAML content from stage
        semantic_model_yaml = read_semantic_model_from_stage()
        if not semantic_model_yaml:
            return {"error": "Could not read semantic model YAML from stage. Please ensure the YAML file is uploaded to the stage and contains valid content."}
        
        # Prepare messages array
        messages = []
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user question
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                }
            ]
        })
        
        # Prepare request body according to Cortex Analyst API specification
        request_body = {
            "messages": messages,
            "semantic_model": semantic_model_yaml,  # Pass YAML content directly
            "stream": False  # For now, use non-streaming
        }
        
        # Use Snowflake internal API method
        resp = _snowflake.send_snow_api_request(
            "POST",           # method
            API_ENDPOINT,     # path
            {},               # headers
            {},               # params
            request_body,     # body
            None,             # request_guid
            API_TIMEOUT,      # timeout in milliseconds
        )
        
        # Check if the response is successful
        if resp.get("status") == 200:
            response_content = resp.get("content", {})
            
            # Parse JSON if response is a string
            if isinstance(response_content, str):
                try:
                    response_data = json.loads(response_content)
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to parse JSON response: {str(e)}"}
            else:
                response_data = response_content
            
            # Check if response_data is a dictionary
            if not isinstance(response_data, dict):
                return {"error": f"Unexpected response format: expected dict, got {type(response_data).__name__}"}
            
            # Extract the analyst response according to API specification
            if "message" in response_data and "content" in response_data["message"]:
                content = response_data["message"]["content"]
                
                # Parse the content blocks
                result = {
                    "text": "",
                    "sql": "",
                    "suggestions": [],
                    "response_metadata": response_data.get("response_metadata", {}),
                    "warnings": response_data.get("warnings", [])
                }
                
                # Ensure content is iterable
                if not isinstance(content, list):
                    return {"error": f"Unexpected content format: expected list, got {type(content).__name__}"}
                
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            result["text"] = block.get("text", "")
                        elif block.get("type") == "sql":
                            result["sql"] = block.get("statement", "")
                        elif block.get("type") == "suggestions":
                            result["suggestions"] = block.get("suggestions", [])
                
                return result
            else:
                return {"error": "Invalid response format from Cortex Analyst"}
        else:
            status_code = resp.get("status", "unknown")
            error_message = f"API request failed with status {status_code}"
            
            # Try to extract error details from response
            try:
                content = resp.get("content", {})
                if isinstance(content, dict) and "message" in content:
                    error_message += f": {content['message']}"
                elif isinstance(content, str):
                    error_message += f": {content}"
            except:
                pass
            
            return {"error": error_message}
            
    except Exception as e:
        return {"error": f"Unexpected error calling Cortex Analyst API: {str(e)}"}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_onboarding_questions() -> List[str]:
    """Return hardcoded suggested questions for the chat interface - analyst"""
    return [
        "What is the average call duration for each representative in year 2025?",
        "Which representatives have the highest first call resolution rate?",
        "What is the sentiment breakdown by issue type",
        "How many calls were not resolved on the first attempt",
        "What are the most common issues customers call about"
    ]

def execute_sql_query(sql_query: str) -> pd.DataFrame:
    """Execute SQL query and return DataFrame"""
    try:
        session = get_snowpark_session()
        if not session:
            return pd.DataFrame()
        
        # Execute query and convert to pandas DataFrame
        result = session.sql(sql_query).to_pandas()
        return result
        
    except Exception as e:
        st.error(f"SQL execution failed: {str(e)}")
        return pd.DataFrame()



def call_audio_processing_procedure(file_path: str):
    """Call the stored procedure to process uploaded audio file"""
    try:
        session = get_snowpark_session()
        if not session:
            return False, "Could not establish Snowpark session"
        
        # Call the stored procedure
        call_sql = f"CALL process_audio_file('{file_path}')"
        result = session.sql(call_sql).collect()
        
        if result and len(result) > 0:
            return True, result[0][0]  # Return the result message
        else:
            return True, "Procedure executed successfully"
            
    except Exception as e:
        return False, f"Error calling stored procedure: {str(e)}"

def upload_audio_file_to_stage(uploaded_file):
    """Upload audio file to Snowflake stage using session.file.put_stream"""
    try:
        session = get_snowpark_session()
        if not session:
            st.error("Could not establish Snowpark session")
            return False
        
        # Get today's date for folder naming
        today = date.today().strftime("%Y-%m-%d")
        
        # Create a BytesIO object from the uploaded file
        file_stream = io.BytesIO(uploaded_file.getvalue())
        
        # Construct the stage path with date folder
        stage_path = f"@AUDIO_FILES/{today}/{uploaded_file.name}"
        
        # Use session.file.put_stream to upload the file
        session.file.put_stream(
            file_stream,
            stage_path,
            auto_compress=False,
            overwrite=True
        )
        
        # Refresh the stage to ensure the newly uploaded file is visible
        try:
            refresh_command = f"ALTER STAGE AUDIO_FILES REFRESH"

            # refresh_command = f"ALTER STAGE AUDIO_FILES REFRESH SUBPATH='/{today}/'"
            session.sql(refresh_command).collect()
            st.success(f"✅ Successfully uploaded '{uploaded_file.name}' to folder: {today} and refreshed stage")
        except Exception as refresh_error:
            st.warning(f"✅ File uploaded but stage refresh failed: {str(refresh_error)}")
        
        return True, f"{today}/{uploaded_file.name}"
                
    except Exception as e:
        st.error(f"❌ Error uploading file: {str(e)}")
        return False, None

def download_and_play_stage_file(stage_file_path, file_name):
    """Download file from stage and play it using session.file.get_stream"""
    try:
        session = get_snowpark_session()
        if not session:
            st.sidebar.error("Could not establish Snowpark session")
            return
        
        # Use session.file.get_stream to download the file
        with session.file.get_stream(stage_file_path) as file_stream:
            audio_bytes = file_stream.read()
        
        # Simple audio player
        st.sidebar.markdown(f"**🎵 Playing: {file_name}**")
        st.sidebar.audio(audio_bytes)
                
    except Exception as e:
        st.sidebar.error(f"Error playing file: {str(e)}")

def show_file_upload_section():
    """Display simplified file upload section in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎤 Cortex AISQL Powered Audio Transcription")
    
    # Audio transcribe limitations info
    with st.sidebar.expander("📋 Audio Transcribe Limitations", expanded=False):
        st.markdown("""
        **UI Upload (This Interface):**
        - **Formats:** MP3, WAV only
        - **Max Size:** 200 MB for larger than 200 MB upload to stage and process using stored procedures
        - **Max Duration:** 90 minutes
        
        **Cortex AI Transcribe:**
        - Supports up to 700 MB files
        - Max Duration: 90 minutes
        - Only  .mp3 and .wav are officially supported.
    
        **Note:** Audio longer than 90 minutes will be automatically truncated during processing.
    
        """)
    
    
    # Initialize session state for uploaded file path
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'upload_completed' not in st.session_state:
        st.session_state.upload_completed = False
    # Track processed audio status
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False
    if 'processed_file_name' not in st.session_state:
        st.session_state.processed_file_name = None
    
    # File uploader widget
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav'],
        help="Supported formats: MP3, WAV | UI Max: 200MB | Max duration: 90 minutes | Large files: upload to stage"
    )
    
    # Show processed status in sidebar when no file is being uploaded
    if st.session_state.audio_processed and not uploaded_file:
        st.sidebar.success(f"✅ Transcribed and Processed: {st.session_state.processed_file_name}")
    
    if uploaded_file is not None:
        # Reset upload completed flag when a new file is selected
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.upload_completed = False
        
        # Check file size limitation for UI uploads (200 MB = 200 * 1024 * 1024 bytes)
        file_size_bytes = len(uploaded_file.getvalue())
        file_size_mb = file_size_bytes / (1024 * 1024)  # Size in MB
        max_size_mb = 200
        
        st.sidebar.write(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
        
        # File format validation (additional check)
        file_extension = uploaded_file.name.lower().split('.')[-1]
        allowed_formats = ['mp3', 'wav']
        
        if file_extension not in allowed_formats:
            st.sidebar.error(f"❌ **Unsupported format!** Only MP3 and WAV files are allowed. Your file: .{file_extension}")
            st.sidebar.info("💡 Please convert your audio to MP3 or WAV format.")
            return  # Stop processing if format is not supported
        
        # File size validation
        if file_size_mb > max_size_mb:
            st.sidebar.error(f"❌ **File too large for UI upload!** Maximum size is {max_size_mb} MB. Your file is {file_size_mb:.1f} MB.")
            st.sidebar.info("💡 **For large files (200+ MB):** Upload directly to Snowflake stage and process using stored procedures.")
            st.sidebar.info("🔧 **Alternative:** Compress your audio file or select a smaller file for UI upload.")
            return  # Stop processing if file is too large
        
        # Display file size status
        if file_size_mb > max_size_mb * 0.8:  # Warning at 80% of limit (160 MB)
            st.sidebar.warning(f"⚠️ Large file: {file_size_mb:.1f} MB (UI Max: {max_size_mb} MB)")
        else:
            st.sidebar.success(f"✅ File size OK: {file_size_mb:.1f} MB")
        
        # Simple audio preview
        st.sidebar.audio(uploaded_file.getvalue())
        
        # Upload button
        if st.sidebar.button("🚀 Upload to Stage", use_container_width=True, type="primary"):
            with st.spinner("📤 Uploading to stage..."):
                # Upload the file first
                success, file_path = upload_audio_file_to_stage(uploaded_file)
            
            # Show completion status below the button
            if success:
                st.sidebar.success(f"✅ Upload completed successfully!")
                st.sidebar.info(f"📁 Stored as: {file_path}")
                # st.sidebar.balloons()
                # Store the file path and name in session state for processing button
                st.session_state.uploaded_file_path = file_path
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.upload_completed = True
                # st.rerun()
            else:
                st.sidebar.error("❌ Upload failed. Please try again.")
                st.session_state.upload_completed = False
    
    # Show Process Audio File button only after successful upload
    if st.session_state.uploaded_file_path and st.session_state.uploaded_file_name and st.session_state.upload_completed:
        st.sidebar.markdown("---")
        st.sidebar.write(f"📋 **Ready to process:** {st.session_state.uploaded_file_path}")
        
        if st.sidebar.button("🔄 Process Audio File", use_container_width=True, type="secondary"):
            # Create containers for all messages
            info_container = st.sidebar.empty()
            progress_container = st.sidebar.empty()
            status_container = st.sidebar.empty()
            
            # Show initial processing message
            info_container.info("⏱️ Processing may take around 1 minute. Please wait...")
            
            # Step 1: Transcribing text from audio
            progress_container.progress(0.25)
            # status_container.info("🎙️ Step 1/4: Transcribing text from audio...")
            # time.sleep(5)  # Realistic time for audio transcription
            
            # Step 2: Extracting information using Cortex AISQL
            progress_container.progress(0.50)
            # status_container.info("🧠 Step 2/4: Extracting information using Cortex AISQL...")
            # time.sleep(5)  # Time for AI processing
            
            # Step 3: Identifying speaker and customer
            progress_container.progress(0.75)
            # status_container.info("👥 Step 3/4: Identifying speaker and customer...")
            # time.sleep(10)  # Time for speaker identification
            
            # Step 4: Execute the actual processing and populate tables
            progress_container.progress(0.90)
            # status_container.info("📊 Step 4/4: Populating tables for Analytics and for Chatbot...")
            
            proc_success, proc_message = call_audio_processing_procedure(st.session_state.uploaded_file_path)
            
            # Complete progress
            progress_container.progress(1.0)
            
            if proc_success:
                status_container.success(f"🎯 Processing complete: {proc_message}")
                st.sidebar.balloons()
                # Set processed status
                st.session_state.audio_processed = True
                st.session_state.processed_file_name = st.session_state.uploaded_file_name
                # Clear the session state after successful processing
                st.session_state.uploaded_file_path = None
                st.session_state.uploaded_file_name = None
                st.session_state.upload_completed = False
                time.sleep(2)  # Keep success message visible briefly
                info_container.empty()
                progress_container.empty()
                status_container.empty()
                # Show refresh instruction instead of auto-rerun
                st.sidebar.info("💡 Audio Processing Completed. Click '🔄 Refresh List' to see the Transcribed and processed audio file")
                # Removed st.rerun() to prevent clearing the processed status
            else:
                status_container.error(f"❌ Processing failed: {proc_message}")
                info_container.empty()
                progress_container.empty()
    
    # Simple file viewing buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("📂 View Today's Files", use_container_width=True):
        show_stage_files()

def show_stage_files():
    """Display files in today's stage folder with simple play buttons"""
    try:
        session = get_snowpark_session()
        if not session:
            st.sidebar.error("Could not establish Snowpark session")
            return
        
        # Get today's date for folder naming
        today = date.today().strftime("%Y-%m-%d")
        
        # List files in today's folder using modern approach
        stage_path = f"@AUDIO_FILES/{today}/"
        
        try:
            result = session.sql(f"LIST {stage_path}").collect()
            
            if result and len(result) > 0:
                st.sidebar.success(f"📂 **Files in {today} folder:**")
                
                for i, row in enumerate(result[:10]):  # Show max 10 files
                    # Extract file path from the result
                    file_path = row['name'] if 'name' in row else str(row[0])
                    file_name = file_path.split('/')[-1] if file_path else f"file_{i}"
                    
                    # Construct the full stage path for download
                    full_stage_path = f"@AUDIO_FILES/{today}/{file_name}"
                    
                    # Simple file display with play button
                    col1, col2 = st.sidebar.columns([4, 1])
                    with col1:
                        st.write(f"📄 {file_name}")
                    with col2:
                        if st.button("▶️", key=f"play_{i}", help=f"Play {file_name}"):
                            download_and_play_stage_file(full_stage_path, file_name)
                
                if len(result) > 10:
                    st.sidebar.write(f"... and {len(result) - 10} more files")
            else:
                st.sidebar.info(f"📂 No files found in {today} folder")
                
        except Exception as e:
            if "does not exist" in str(e).lower():
                st.sidebar.info(f"📂 Folder {today} doesn't exist yet")
            else:
                st.sidebar.error(f"Error: {str(e)}")
                
    except Exception as e:
        st.sidebar.error(f"Error accessing stage: {str(e)}")

def unified_chat_interface():
    """Simplified unified chat interface - analyst with conversation history"""
    st.subheader("💬 Chat with Your Call Center Data")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []
    
    # Suggested questions section
    show_suggested_questions()
    
    # Display conversation history
    display_chat_history()
    
    # Chat input and controls
    handle_chat_input()

def show_suggested_questions():
    """Display suggested questions in an expandable section"""
    is_first_visit = len(st.session_state.chat_history) == 0
    expander_title = "💡 Suggested Questions" + ("" if is_first_visit else " - Click to expand")
    
    with st.expander(expander_title, expanded=is_first_visit):
        suggested_questions = get_onboarding_questions()
        
        if suggested_questions:
            st.markdown("**Quick questions to get insights:**")
            cols = st.columns(2)
            for i, question in enumerate(suggested_questions):
                col = cols[i % 2]
                if col.button(question, key=f"suggest_{i}"):
                    add_user_message(question)
                    process_user_message(question)
                    # Clear the input after processing suggested question
                    if 'chat_input_value' in st.session_state:
                        st.session_state.chat_input_value = ""
                    st.rerun()
        else:
            st.info("💡 Try asking: 'How many total calls?' or 'What is our FCR rate?'")
    
    st.markdown("---")

def display_chat_history():
    """Display the conversation history"""
    if len(st.session_state.chat_history) == 0:
        st.info("👋 Welcome! Start by asking a question or using suggested questions above.")
        return
    
    st.markdown("### 💬 Conversation History")
    
    for i, message in enumerate(st.session_state.chat_history):
        display_message(message, i)
        
        # Add separator between messages
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")
    
    # Show preservation info
    st.success("✅ All outputs preserved - scroll up to view previous responses")
    st.info("💾 Your entire conversation is saved. Use Export below to download.")

def display_message(message, index):
    """Display a single message (user or assistant)"""
    timestamp = message.get("timestamp", datetime.now().strftime("%H:%M"))
    
    if message["role"] == "user":
        st.markdown(f"""
<div class='chat-message user-message'>
    <strong>🧑‍💼 You</strong> <small style='color: #666;'>({timestamp})</small><br>
    {message["content"]}
</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class='chat-message assistant-message'>
    <strong>🤖 Assistant</strong> <small style='color: #666;'>({timestamp})</small><br>
    {message["content"]}
</div>
        """, unsafe_allow_html=True)
        
        # Display data table
        if "data" in message and message["data"] is not None:
            st.markdown("**📊 Data Results:**")
            st.dataframe(message["data"], use_container_width=True)
        
        # Display chart
        if "chart" in message and message["chart"] is not None:
            st.markdown("**📈 Chart:**")
            st.plotly_chart(message["chart"], use_container_width=True)
        
        # Display SQL query
        if "sql" in message and message["sql"]:
            with st.expander(f"📝 SQL Query #{index+1}"):
                st.code(message["sql"], language='sql')

def handle_chat_input():
    """Handle chat input and controls"""
    st.markdown("---")
    
    # Show stats if conversation exists
    if len(st.session_state.chat_history) > 0:
        show_conversation_stats()
    
    # Initialize input value in session state
    if 'chat_input_value' not in st.session_state:
        st.session_state.chat_input_value = ""
    
    # Input controls
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        user_input = st.text_input(
            "💭 Ask about your call center data:",
            value=st.session_state.chat_input_value,
            placeholder="e.g., What is our FCR rate?",
            key="chat_input_field"
        )
        # Update session state when user types
        if user_input != st.session_state.chat_input_value:
            st.session_state.chat_input_value = user_input
    
    with col2:
        send_button = st.button("💬 Send", type="primary", use_container_width=True)
    
    with col3:
        if st.session_state.chat_history:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            if clear_button:
                st.session_state.chat_history = []
                st.session_state.conversation_context = []
                st.session_state.chat_input_value = ""  # Clear input too
                st.success("Chat cleared!")
                st.rerun()
    
    # Handle send
    if send_button and user_input.strip():
        add_user_message(user_input.strip())
        process_user_message(user_input.strip())
        # Clear the input after sending
        st.session_state.chat_input_value = ""
        st.rerun()
    elif send_button:
        st.warning("Please enter a question.")

def show_conversation_stats():
    """Display conversation statistics and export"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💬 Messages", len(st.session_state.chat_history))
    with col2:
        user_count = sum(1 for msg in st.session_state.chat_history if msg["role"] == "user")
        st.metric("❓ Questions", user_count)
    with col3:
        data_count = sum(1 for msg in st.session_state.chat_history 
                        if msg["role"] == "assistant" and "data" in msg)
        st.metric("📊 Data Responses", data_count)
    with col4:
        # Export conversation
        conversation_text = "\n".join([
            f"[{msg.get('timestamp', 'N/A')}] {'You' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in st.session_state.chat_history
        ])
        st.download_button(
            "📥 Export",
            data=conversation_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    st.markdown("---")

def add_user_message(content):
    """Add user message to chat history"""
    st.session_state.chat_history.append({
        "role": "user",
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M")
    })

def process_user_message(user_input: str):
    """Process user message and add assistant response to chat history"""
    with st.spinner("🤔 Thinking..."):
        # First try to match with verified queries directly
        verified_sql = try_match_verified_query(user_input)
        
        if verified_sql:
            # Use verified query directly
            df = execute_sql_query(verified_sql)
            
            if len(df) > 0:
                content = "📋 Using verified query - here's your data:"
                
                message_data = {
                    "role": "assistant",
                    "content": content,
                    "data": df.copy(),
                    "sql": verified_sql,
                    "timestamp": datetime.now().strftime("%H:%M")
                }
                
                # Add chart if possible
                try:
                    if len(df.columns) >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
                        if "sentiment" in user_input.lower():
                            # For sentiment breakdown, create a grouped bar chart
                            fig = px.bar(df, x=df.columns[0], y=df.columns[-1], 
                                       color=df.columns[1] if len(df.columns) > 2 else None,
                                       title="Sentiment Breakdown by Issue Type")
                        else:
                            fig = px.bar(df, x=df.columns[0], y=df.columns[-1], 
                                       title=f"{df.columns[-1]} by {df.columns[0]}")
                        fig.update_layout(height=400)
                        message_data["chart"] = fig
                except:
                    pass
                
                st.session_state.chat_history.append(message_data)
                return
        
        # Fallback to Cortex Analyst API
        response = query_cortex_analyst_rest_api(user_input, st.session_state.conversation_context)
        
        if 'error' in response:
            add_assistant_message(f"❌ Error: {response['error']}")
            return
        
        # Handle successful response with SQL
        if response.get('sql'):
            df = execute_sql_query(response['sql'])
            
            if len(df) > 0:
                # Determine response text
                if len(df) == 1 and len(df.columns) == 1:
                    content = f"📊 Result: **{df.iloc[0, 0]}**"
                else:
                    content = "📋 Here's your data:"
                
                # Create message with data
                message_data = {
                    "role": "assistant",
                    "content": content,
                    "data": df.copy(),
                    "sql": response['sql'],
                    "timestamp": datetime.now().strftime("%H:%M")
                }
                
                # Add chart if possible
                try:
                    if len(df.columns) == 2 and len(df) > 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                        fig = px.bar(df, x=df.columns[0], y=df.columns[1], 
                                   title=f"{df.columns[1]} by {df.columns[0]}")
                        fig.update_layout(height=400)
                        message_data["chart"] = fig
                except:
                    pass  # Chart creation is optional
                
                st.session_state.chat_history.append(message_data)
                update_conversation_context(user_input, response)
            else:
                add_assistant_message("🔍 No data found. Try rephrasing your question.")
        else:
            # No SQL generated
            text = response.get('text', "🤔 Couldn't generate a query. Try being more specific.")
            add_assistant_message(text)
        
        # Add suggestions if available
        if response.get('suggestions'):
            suggestions = "💡 **Suggestions:**\n" + "\n".join([f"• {s}" for s in response['suggestions']])
            add_assistant_message(suggestions)

def try_match_verified_query(user_input: str):
    """Try to match user input with verified queries and return SQL if found"""
    try:
        import yaml
        semantic_model_yaml = read_semantic_model_from_stage()
        if not semantic_model_yaml:
            return None
            
        yaml_data = yaml.safe_load(semantic_model_yaml)
        verified_queries = yaml_data.get('verified_queries', [])
        
        user_lower = user_input.lower().strip()
        
        for vq in verified_queries:
            # Check main question
            question = vq.get('question', '').lower().strip()
            if question and question == user_lower:
                return vq.get('sql', '').strip()
            
            # Check synonyms
            synonyms = vq.get('synonyms', [])
            for synonym in synonyms:
                if synonym.lower().strip() == user_lower:
                    return vq.get('sql', '').strip()
            
            # Fuzzy matching for key phrases
            if "sentiment" in user_lower and "issue" in user_lower and vq.get('name') == 'sentiment_by_issue_type':
                return vq.get('sql', '').strip()
                
        return None
    except Exception as e:
        return None

def add_assistant_message(content):
    """Add assistant message to chat history"""
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M")
    })

def update_conversation_context(user_input, response):
    """Update conversation context for multi-turn conversations"""
    st.session_state.conversation_context.extend([
        {"role": "user", "content": [{"type": "text", "text": user_input}]},
        {"role": "analyst", "content": [
            {"type": "text", "text": response.get('text', '')},
            {"type": "sql", "statement": response.get('sql', '')}
        ]}
    ])

def show_semantic_model_debug():
    """Show debug information about the current semantic model"""
    st.markdown("### 🔧 Semantic Model Debug Info")
    
    try:
        semantic_model_yaml = read_semantic_model_from_stage()
        if semantic_model_yaml:
            st.success("✅ Semantic model loaded successfully from stage")
            
            # Parse and show basic info
            import yaml
            yaml_data = yaml.safe_load(semantic_model_yaml)
            
            st.metric("📊 Tables", len(yaml_data.get('tables', [])))
            
            if 'verified_queries' in yaml_data:
                st.metric("✅ Verified Queries", len(yaml_data['verified_queries']))
                
                st.markdown("**Verified Queries:**")
                for i, vq in enumerate(yaml_data['verified_queries']):
                    with st.expander(f"Query {i+1}: {vq.get('name', 'Unnamed')}"):
                        st.write(f"**Question:** {vq.get('question', 'N/A')}")
                        st.write(f"**Name:** {vq.get('name', 'N/A')}")
                        st.write(f"**Onboarding:** {vq.get('use_as_onboarding_question', False)}")
                        if vq.get('sql'):
                            st.code(vq['sql'], language='sql')
            else:
                st.warning("❌ No verified queries found in semantic model")
                
            # Show dimensions for call_sentiment and issue_type
            st.markdown("**Key Dimensions:**")
            tables = yaml_data.get('tables', [])
            if tables:
                dimensions = tables[0].get('dimensions', [])
                for dim in dimensions:
                    if dim.get('name') in ['call_sentiment', 'issue_type']:
                        st.write(f"✅ **{dim['name']}** → `{dim.get('expr', 'N/A')}`")
        else:
            st.error("❌ Failed to load semantic model from stage")
            
    except Exception as e:
        st.error(f"❌ Error loading semantic model: {str(e)}")

def test_verified_query_matching():
    """Test if verified queries are matching correctly"""
    st.markdown("### 🧪 Verified Query Matching Test")
    
    # Test the specific question that's not working
    test_question = "What is the sentiment breakdown by issue type?"
    
    st.write(f"**Testing Question:** {test_question}")
    
    # Show what Cortex Analyst actually generates
    with st.spinner("Testing query generation..."):
        response = query_cortex_analyst_rest_api(test_question)
        
        if 'error' in response:
            st.error(f"❌ Error: {response['error']}")
        else:
            if 'sql' in response and response['sql']:
                st.markdown("**Generated SQL:**")
                st.code(response['sql'], language='sql')
                
                # Compare with expected verified query
                st.markdown("**Expected SQL (from verified query - using logical names):**")
                expected_sql = """
                SELECT issue_type, call_sentiment, COUNT(*) as call_count
                FROM call_center_interactions
                WHERE issue_type IS NOT NULL AND call_sentiment IS NOT NULL
                GROUP BY issue_type, call_sentiment
                ORDER BY issue_type, call_sentiment
                """
                st.code(expected_sql, language='sql')
                
                # Check if they match
                if "call_sentiment" in response['sql'] and "COUNT(*)" in response['sql']:
                    st.success("✅ Query appears to match verified query pattern")
                else:
                    st.error("❌ Generated query does NOT match verified query")
                    st.write("**Issues found:**")
                    if "call_sentiment" not in response['sql']:
                        st.write("- Missing `call_sentiment` column")
                    if "COUNT(*)" not in response['sql']:
                        st.write("- Missing `COUNT(*)` aggregation")
            else:
                st.warning("❌ No SQL generated")
                
        if 'text' in response:
            st.write(f"**Response Text:** {response['text']}")

def advanced_analytics():
    """Advanced analytics section"""
    st.subheader("📈 Advanced Analytics")
    
    # Time series analysis
    st.markdown("### 📅 Trend Analysis")
    time_period = st.selectbox("Select time period:", ["Monthly", "Yearly"])
    
    # More specific queries that work with available columns
    if time_period == "Monthly":
        question = "Show call volume by month using the call date, group by month for the last 12 months"
    else:  # Yearly
        question = "Show call volume by year using the call date, group by year"
    
    if st.button("Generate Trend Analysis"):
        with st.spinner("Generating trend analysis..."):
            try:
                response = query_cortex_analyst_rest_api(question)
                
                # Show the generated SQL for debugging
                if 'sql' in response and response['sql']:
                    with st.expander("📝 Generated SQL Query"):
                        st.code(response['sql'], language='sql')
                
                if 'error' in response:
                    st.error(f"Query generation failed: {response['error']}")
                elif 'sql' in response and response['sql']:
                    df = execute_sql_query(response['sql'])
                    if len(df) > 0:
                        st.success(f"Found {len(df)} data points for trend analysis")
                        
                        # Create time series chart
                        try:
                            # More flexible chart creation
                            if len(df.columns) >= 2:
                                x_col = df.columns[0]
                                y_col = df.columns[1]
                                
                                fig = px.line(df, x=x_col, y=y_col, 
                                            title=f"{time_period} Call Volume Trends")
                                fig.update_layout(
                                    height=500, 
                                    showlegend=True,
                                    xaxis_title=x_col,
                                    yaxis_title=y_col
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create trend chart: {str(e)}")
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No trend data available for the selected period")
                else:
                    st.warning("Could not generate SQL query for trend analysis")
            except Exception as e:
                st.error(f"Trend analysis failed: {str(e)}")
    
    # Comparative analysis
    st.markdown("### 🔍 Comparative Analysis")
    comparison_type = st.selectbox(
        "What would you like to compare?",
        ["Representatives Performance", "Issue Types Resolution", "Sentiment Analysis", "Call Duration Analysis"]
    )
    
    # More specific and achievable comparison queries
    comparison_queries = {
        "Representatives Performance": "Show each representative with their total calls, average call duration, and first call resolution rate",
        "Issue Types Resolution": "Show each issue type with total calls, average duration, and resolution statistics",
        "Sentiment Analysis": "Show call sentiment breakdown by issue type with counts and percentages",
        "Call Duration Analysis": "Show average call duration by representative and by issue type"
    }
    
    if st.button("Run Comparison Analysis"):
        with st.spinner("Running comparison analysis..."):
            try:
                question = comparison_queries[comparison_type]
                response = query_cortex_analyst_rest_api(question)
                
                # Show the generated SQL for debugging
                if 'sql' in response and response['sql']:
                    with st.expander("📝 Generated SQL Query"):
                        st.code(response['sql'], language='sql')
                
                if 'error' in response:
                    st.error(f"Query generation failed: {response['error']}")
                elif 'sql' in response and response['sql']:
                    df = execute_sql_query(response['sql'])
                    if len(df) > 0:
                        st.success(f"Found {len(df)} records for comparison analysis")
                        st.dataframe(df, use_container_width=True)
                        
                        # Create appropriate visualization if we have multiple numeric columns
                        try:
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            if len(numeric_cols) >= 1 and len(df.columns) >= 2:
                                # Use first column as x-axis and first numeric column as y-axis
                                x_col = df.columns[0]
                                
                                if len(numeric_cols) == 1:
                                    # Single metric - simple bar chart
                                    fig = px.bar(df, x=x_col, y=numeric_cols[0], 
                                               title=f"{comparison_type} - {numeric_cols[0]}")
                                else:
                                    # Multiple metrics - grouped bar chart (max 3 for readability)
                                    metrics_to_plot = numeric_cols[:3]
                                    fig = px.bar(df, x=x_col, y=metrics_to_plot, 
                                               title=f"{comparison_type}")
                                
                                fig.update_layout(height=500, showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Data doesn't contain numeric columns suitable for visualization")
                        except Exception as e:
                            st.info(f"Visualization could not be generated: {str(e)}")
                    else:
                        st.warning("No comparative data available")
                else:
                    st.warning("Could not generate SQL query for comparison analysis")
            except Exception as e:
                st.error(f"Comparison analysis failed: {str(e)}")

# simple_agent_chat function removed - contained troubleshooting sections

def overview_dashboard():
    """Overview dashboard section with audio analytics"""
    
    session = get_snowpark_session()
    if not session:
        st.error("Could not establish Snowpark session")
        return

    # Helper function to get date range
    @st.cache_data
    def get_date_range():
        try:
            query = """
                SELECT MIN(DATE) as min_date, MAX(DATE) as max_date 
                FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP 
                WHERE DATE IS NOT NULL
            """
            result = session.sql(query).collect()
            if result:
                return result[0]['MIN_DATE'], result[0]['MAX_DATE']
            return date_class(2024, 1, 1), date_class.today()
        except:
            return date_class(2024, 1, 1), date_class.today()

    @st.cache_data
    def load_data(query_of_interest):
        try:
            return session.sql(query_of_interest).to_pandas()
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()

    # Date range selector
    st.markdown('-----')
    min_date, max_date = get_date_range()
    
    d_col1, _, d_col2 = st.columns([4, 1, 4])
    with d_col1:
        s_date = st.date_input("Start date", min_date, key='overview_start_date')
    with d_col2:
        e_date = st.date_input("End date", max_date, key='overview_end_date')

    st.markdown('-----')

    # Key metrics
    col1, col2 = st.columns(2)
    
    # Get all audio call details
    base_query = f"""
        SELECT 
            DATE,
            TRANSCRIPTION_DURATION_SECONDS as CALL_DURATION_SECONDS,
            REPRESENTATIVE,
            CUSTOMER,
            CONVERSATION_SENTIMENT,
            CALL_INTENT,
            CONVERSATION_SUMMARY,
            CLAIM_NUMBER,
            POLICY_NUMBER,
            CALL_TO_ACTION,
            PURPOSE_OF_CALL,
            ISSUE,
            RESOLUTION,
            FIRST_CALL_RESOLUTION,
            AUDIO_FILE,
            CONVERSATION_STRUCTURED,
            YEAR(DATE) || '-' || MONTHNAME(DATE) as YEAR_MONTH
        FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
        WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
    """
    
    all_audio_calls = load_data(base_query)
    
    if all_audio_calls.empty:
        st.warning("No data found for the selected date range")
        return

    # Display key metrics
    with st.container(border=True):
        with col1:
            st.metric(label="Total Calls", value=len(all_audio_calls))
        with col2:
            total_duration = all_audio_calls["CALL_DURATION_SECONDS"].sum() / 60  # Convert to minutes
            st.metric(label="Total Call Duration (mins)", value=f"{total_duration:.1f}")

    st.markdown('----')

    # Audio file selection and details (moved above charts)
    if not all_audio_calls.empty and 'AUDIO_FILE' in all_audio_calls.columns:
        audio_files_list = all_audio_calls['AUDIO_FILE'].dropna().unique().tolist()
        audio_files_list.sort()

        if audio_files_list:
            # Header with refresh button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("🎧 Audio File Details")
            with col2:
                if st.button("🔄 Refresh List", key="refresh_audio_files", use_container_width=True):
                    # Clear cache to force refresh of data
                    st.cache_data.clear()
                    # Clear processed status when refreshing
                    st.session_state.audio_processed = False
                    st.session_state.processed_file_name = None
                    st.rerun()
            
            select_audio_file = st.selectbox("Select Audio File", audio_files_list, key='overview_audiofile')
            
            # Get details for selected audio file
            selected_call = all_audio_calls[all_audio_calls['AUDIO_FILE'] == select_audio_file].iloc[0]
            
            # Action buttons
            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
            
            # Initialize session state for button tracking
            if 'selected_action' not in st.session_state:
                st.session_state.selected_action = None
            
            with opt_col1:
                if st.button("🎵 Play Audio", key="overview_play", use_container_width=True):
                    st.session_state.selected_action = 'play'
            
            with opt_col2:
                if st.button("📄 Show Call Summary", key="overview_summary", use_container_width=True):
                    st.session_state.selected_action = 'summary'
            
            with opt_col3:
                if st.button("📊 Show Call Details", key="overview_details", use_container_width=True):
                    st.session_state.selected_action = 'details'
            
            with opt_col4:
                if st.button("ℹ️ Show Related Info", key="overview_related", use_container_width=True):
                    st.session_state.selected_action = 'related'
            
            # Display results in full width below the buttons
            if st.session_state.selected_action is None and st.session_state.get('audio_processed', False):
                st.markdown("---")
                st.markdown("### ✅ Processing Status")
                st.success(f"**Transcribed and Processed:** {st.session_state.get('processed_file_name', 'Audio file')}")
                st.info("💡 Select an option above to view audio details, summary, or play the file.")
            elif st.session_state.selected_action == 'play':
                st.markdown("---")
                st.markdown("### 🎵 Audio Player")
                
                # Try to play audio from stage using the audio file name
                audio_file_name = selected_call.get('AUDIO_FILE', '')
                if audio_file_name:
                    try:
                        # Get the date from the call to construct the stage path
                        call_date = selected_call.get('DATE')
                        if call_date:
                            # Convert date to date string (YYYY-MM-DD format)
                            if hasattr(call_date, 'strftime'):
                                # If it's already a date object
                                date_str = call_date.strftime("%Y-%m-%d")
                            else:
                                # Handle string date format
                                from datetime import datetime
                                if isinstance(call_date, str):
                                    # Try to parse the date string
                                    date_obj = datetime.strptime(call_date, "%Y-%m-%d")
                                    date_str = date_obj.strftime("%Y-%m-%d")
                                else:
                                    date_str = str(call_date)
                            
                            # Construct stage path: @AUDIO_FILES/YYYY-MM-DD/filename.mp3
                            stage_path = f"@AUDIO_FILES/{date_str}/{audio_file_name}"
                            
                            st.info(f"🎧 Playing: **{audio_file_name}**")
                            st.info(f"📁 Stage Path: `{stage_path}`")
                            
                            # Use the existing download_and_play_stage_file function
                            try:
                                session = get_snowpark_session()
                                if session:
                                    # Use session.file.get_stream to download the file
                                    with session.file.get_stream(stage_path) as file_stream:
                                        audio_bytes = file_stream.read()
                                    
                                    # Determine audio format from file extension
                                    file_extension = audio_file_name.lower().split('.')[-1]
                                    if file_extension == 'mp3':
                                        audio_format = 'audio/mpeg'
                                    elif file_extension == 'wav':
                                        audio_format = 'audio/wav'
                                    else:
                                        audio_format = None  # Let Streamlit auto-detect
                                    
                                    # Display audio player with correct format
                                    if audio_format:
                                        st.audio(audio_bytes, format=audio_format)
                                    else:
                                        st.audio(audio_bytes)  # Auto-detect format
                                    
                                    st.success("✅ Audio loaded successfully!")
                                else:
                                    st.error("❌ Could not establish Snowpark session")
                            except Exception as audio_error:
                                st.error(f"❌ Could not load audio: {str(audio_error)}")
                                st.info("💡 Audio file might not be available in the stage or path might be incorrect.")
                        else:
                            st.error("❌ Could not determine call date for audio file location")
                    except Exception as e:
                        st.error(f"❌ Error processing audio playback: {str(e)}")
                else:
                    st.warning("⚠️ No audio file name found for this call")
                    
            elif st.session_state.selected_action == 'summary':
                st.markdown("---")
                st.markdown("### 📄 Call Summary")
                summary_text = selected_call.get('CONVERSATION_SUMMARY', 'No summary available')
                if summary_text and summary_text != 'No summary available':
                    st.markdown(summary_text)
                else:
                    st.info("No call summary available for this audio file")
                    
            elif st.session_state.selected_action == 'details':
                st.markdown("---")
                st.markdown("### 📊 Complete Call Details")
                # Create a better formatted details display
                details_df = pd.DataFrame([selected_call]).T
                details_df.columns = ['Value']
                details_df.index.name = 'Field'
                st.dataframe(details_df, use_container_width=True)
                
            elif st.session_state.selected_action == 'related':
                st.markdown("---")
                st.markdown("### ℹ️ Key Call Information")
                
                # Create a more organized display using columns
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown("**👤 People Involved:**")
                    st.write(f"• **Representative:** {selected_call.get('REPRESENTATIVE', 'N/A')}")
                    st.write(f"• **Customer:** {selected_call.get('CUSTOMER', 'N/A')}")
                    
                    st.markdown("**📞 Call Context:**")
                    st.write(f"• **Intent:** {selected_call.get('CALL_INTENT', 'N/A')}")
                    st.write(f"• **Duration:** {selected_call.get('CALL_DURATION_SECONDS', 'N/A')} seconds")
                    
                    st.markdown("**📝 Structured Conversation:**")
                    structured_conv = selected_call.get('CONVERSATION_STRUCTURED', 'N/A')
                    if structured_conv and structured_conv != 'N/A' and len(str(structured_conv).strip()) > 0:
                        # Display structured conversation in a scrollable text area
                        st.text_area("Structured Data", structured_conv, height=150, disabled=True, key="structured_display")
                    else:
                        st.write("• **No structured conversation data available**")
                
                with info_col2:
                    st.markdown("**🔍 Issue & Resolution:**")
                    st.write(f"• **Issue:** {selected_call.get('ISSUE', 'N/A')}")
                    st.write(f"• **Resolution:** {selected_call.get('RESOLUTION', 'N/A')}")
                    
                    st.markdown("**📈 Performance:**")
                    st.write(f"• **First Call Resolution:** {selected_call.get('FIRST_CALL_RESOLUTION', 'N/A')}")
                    st.write(f"• **Sentiment:** {selected_call.get('CONVERSATION_SENTIMENT', 'N/A')}")
            
            # Add a clear button
            if st.session_state.selected_action:
                st.markdown("---")
                if st.button("🗑️ Clear Display", key="clear_display"):
                    st.session_state.selected_action = None
                    st.rerun()
        else:
            st.info("No audio files found in selected date range")
    else:
        st.info("No audio file data available")

    st.markdown('----')

    # Enhanced Charts section with Business Intelligence
    st.subheader("📊 Call Analytics & Business Intelligence")
    
    # Primary Distribution Charts - Enhanced with better visuals
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 **Top 5 Purpose of Call**")
        with st.container(border=True):
            # Purpose of call distribution with business insights
            purpose_query = f"""
                SELECT 
                    PURPOSE_OF_CALL as name, 
                    COUNT(*) as value,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage,
                    AVG(TRANSCRIPTION_DURATION_SECONDS) as avg_duration,
                    AVG(CASE WHEN FIRST_CALL_RESOLUTION = 'Yes' THEN 1 ELSE 0 END) * 100 as fcr_rate
                FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
                WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
                  AND PURPOSE_OF_CALL IS NOT NULL
                GROUP BY PURPOSE_OF_CALL
                ORDER BY value DESC
                LIMIT 5
            """
            purpose_df = load_data(purpose_query)
            
            if not purpose_df.empty:
                # Create enhanced donut chart with business colors
                fig = px.pie(
                    values=purpose_df["VALUE"], 
                    names=purpose_df["NAME"], 
                    hole=0.4,  # Donut chart
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
                
                # Enhanced styling
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=12,
                    marker=dict(line=dict(color='#FFFFFF', width=2))
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="v", 
                        yanchor="top", 
                        y=1, 
                        xanchor="left", 
                        x=1.05,
                        font=dict(size=10)
                    ),
                    margin=dict(t=50, b=50, l=50, r=50),
                    font=dict(color="white")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show purpose details with FCR and duration
                with st.expander("📋 Call Purpose Analytics", expanded=False):
                    display_df = purpose_df.copy()
                    display_df['AVG_DURATION'] = (display_df['AVG_DURATION']/60).round(1)
                    display_df['FCR_RATE'] = display_df['FCR_RATE'].round(1)
                    display_df = display_df[['NAME', 'VALUE', 'PERCENTAGE', 'AVG_DURATION', 'FCR_RATE']]
                    display_df.columns = ['Purpose', 'Total Calls', 'Percentage %', 'Avg Duration (min)', 'FCR Rate %']
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.info("📋 No purpose data available for the selected date range")

    with col2:
        st.markdown("#### 😊 **Customer Sentiment Analysis**")
        with st.container(border=True):
            # Customer sentiment distribution with satisfaction insights
            sentiment_query = f"""
                SELECT 
                    CONVERSATION_SENTIMENT as name, 
                    COUNT(*) as value,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage,
                    AVG(TRANSCRIPTION_DURATION_SECONDS) as avg_duration,
                    AVG(CASE WHEN FIRST_CALL_RESOLUTION = 'Yes' THEN 1 ELSE 0 END) * 100 as fcr_rate
                FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
                WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
                  AND CONVERSATION_SENTIMENT IS NOT NULL
                GROUP BY CONVERSATION_SENTIMENT
                ORDER BY value DESC
                -- LIMIT 8
            """
            sentiment_df = load_data(sentiment_query)
            
            if not sentiment_df.empty:
                # Create enhanced donut chart with sentiment-based colors
                fig = px.pie(
                    values=sentiment_df["VALUE"], 
                    names=sentiment_df["NAME"], 
                    hole=0.4,  # Donut chart
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                
                # Enhanced styling
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=12,
                    marker=dict(line=dict(color='#FFFFFF', width=2))
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="v", 
                        yanchor="top", 
                        y=1, 
                        xanchor="left", 
                        x=1.05,
                        font=dict(size=10)
                    ),
                    margin=dict(t=50, b=50, l=50, r=50),
                    font=dict(color="white")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sentiment details with performance metrics
                with st.expander("😊 Sentiment Performance Insights", expanded=False):
                    display_df = sentiment_df.copy()
                    display_df['AVG_DURATION'] = (display_df['AVG_DURATION']/60).round(1)
                    display_df['FCR_RATE'] = display_df['FCR_RATE'].round(1)
                    display_df = display_df[['NAME', 'VALUE', 'PERCENTAGE', 'AVG_DURATION', 'FCR_RATE']]
                    display_df.columns = ['Sentiment', 'Total Calls', 'Percentage %', 'Avg Duration (min)', 'FCR Rate %']
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.info("😊 No sentiment data available for the selected date range")
    
    st.markdown("---")
    
    # Business Intelligence Metrics Row
    st.markdown("#### 🏢 **Business Performance Metrics**")
    
    # Key Business Metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        with st.container(border=True):
            st.markdown("**📞 Call Volume Trend (MoM)**")
            volume_query = f"""
                WITH monthly_data AS (
                    SELECT 
                        YEAR(DATE) as call_year,
                        MONTH(DATE) as call_month,
                        COUNT(*) as total_calls,
                        COUNT(*) / COUNT(DISTINCT DATE) as avg_daily_calls
                    FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
                    WHERE DATE >= DATEADD(month, -8, CURRENT_DATE())
                    GROUP BY YEAR(DATE), MONTH(DATE)
                    ORDER BY call_year DESC, call_month DESC
                    LIMIT 8
                )
                SELECT 
                    call_year,
                    call_month,
                    total_calls,
                    avg_daily_calls,
                    LAG(avg_daily_calls) OVER (ORDER BY call_year, call_month) as prev_month_avg
                FROM monthly_data
                ORDER BY call_year DESC, call_month DESC
            """
            volume_df = load_data(volume_query)
            
            if not volume_df.empty and len(volume_df) >= 1:
                current_avg = volume_df['AVG_DAILY_CALLS'].iloc[0]
                
                # Calculate month-over-month trend
                if len(volume_df) >= 2 and volume_df['PREV_MONTH_AVG'].iloc[0] is not None:
                    prev_month_avg = volume_df['AVG_DAILY_CALLS'].iloc[1]
                    mom_trend = ((current_avg - prev_month_avg) / prev_month_avg * 100) if prev_month_avg > 0 else 0
                    
                    st.metric(
                        label="Avg Daily Calls",
                        value=f"{current_avg:.0f}",
                        delta=f"{mom_trend:+.1f}% MoM"
                    )
                else:
                    st.metric(
                        label="Avg Daily Calls",
                        value=f"{current_avg:.0f}",
                        delta="No previous data"
                    )
            else:
                st.metric("Avg Daily Calls", "N/A")
    
    with metrics_col2:
        with st.container(border=True):
            st.markdown("**⏱️ Response Efficiency**")
            efficiency_query = f"""
                SELECT 
                    AVG(TRANSCRIPTION_DURATION_SECONDS) as avg_duration,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY TRANSCRIPTION_DURATION_SECONDS) as median_duration
                FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
                WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
                  AND TRANSCRIPTION_DURATION_SECONDS > 0
            """
            eff_df = load_data(efficiency_query)
            
            if not eff_df.empty:
                avg_duration = eff_df['AVG_DURATION'].iloc[0]
                st.metric(
                    label="Avg Call Duration",
                    value=f"{avg_duration/60:.1f} min",
                    delta=f"Median: {eff_df['MEDIAN_DURATION'].iloc[0]/60:.1f}m"
                )
            else:
                st.metric("Avg Call Duration", "N/A")
    
    with metrics_col3:
        with st.container(border=True):
            st.markdown("**😊 Customer Sentiment**")
            sentiment_query = f"""
                SELECT 
                    CONVERSATION_SENTIMENT,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
                FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
                WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
                  AND CONVERSATION_SENTIMENT IS NOT NULL
                GROUP BY CONVERSATION_SENTIMENT
                ORDER BY count DESC
                -- LIMIT 1
            """
            sent_df = load_data(sentiment_query)
            
            if not sent_df.empty:
                top_sentiment = sent_df['CONVERSATION_SENTIMENT'].iloc[0]
                sentiment_pct = sent_df['PERCENTAGE'].iloc[0]
                
                # Color code based on sentiment
                delta_color = "normal"
                if top_sentiment.lower() in ['positive', 'satisfied', 'happy']:
                    delta_color = "normal"
                elif top_sentiment.lower() in ['negative', 'angry', 'frustrated']:
                    delta_color = "inverse"
                
                st.metric(
                    label="Dominant Sentiment",
                    value=top_sentiment.title(),
                    delta=f"{sentiment_pct}% of calls"
                )
            else:
                st.metric("Dominant Sentiment", "N/A")
    
    with metrics_col4:
        with st.container(border=True):
            st.markdown("**🎯 Resolution Quality**")
            resolution_query = f"""
                SELECT 
                    AVG(CASE WHEN FIRST_CALL_RESOLUTION = 'Yes' THEN 1 ELSE 0 END) * 100 as fcr_rate,
                    COUNT(CASE WHEN FIRST_CALL_RESOLUTION = 'Yes' THEN 1 END) as resolved_count,
                    COUNT(*) as total_count
                FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
                WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
                  AND FIRST_CALL_RESOLUTION IS NOT NULL
            """
            res_df = load_data(resolution_query)
            
            if not res_df.empty:
                fcr_rate = res_df['FCR_RATE'].iloc[0]
                resolved = res_df['RESOLVED_COUNT'].iloc[0]
                total = res_df['TOTAL_COUNT'].iloc[0]
                
                st.metric(
                    label="First Call Resolution",
                    value=f"{fcr_rate:.1f}%",
                    delta=f"{resolved}/{total} resolved"
                )
            else:
                st.metric("First Call Resolution", "N/A")
    
    st.markdown("---")
    
    # Additional Business Intelligence Charts
    st.markdown("#### 📈 **Advanced Business Analytics**")
    
    # Representative Workload Analysis (full-width)
    st.markdown("**👥 Representative Workload**")
    with st.container(border=True):
        rep_workload_query = f"""
            SELECT 
                REPRESENTATIVE as rep_name,
                COUNT(*) as total_calls,
                AVG(TRANSCRIPTION_DURATION_SECONDS) as avg_duration,
                AVG(CASE WHEN FIRST_CALL_RESOLUTION = 'Yes' THEN 1 ELSE 0 END) * 100 as fcr_rate,
                SUM(TRANSCRIPTION_DURATION_SECONDS) as total_time
            FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
            WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
              AND REPRESENTATIVE IS NOT NULL
            GROUP BY REPRESENTATIVE
            ORDER BY total_calls DESC
            LIMIT 8
        """
        rep_df = load_data(rep_workload_query)
        
        if not rep_df.empty:
            fig = px.scatter(
                rep_df, 
                x='TOTAL_CALLS', 
                y='FCR_RATE',
                size='TOTAL_TIME',
                hover_data=['REP_NAME', 'AVG_DURATION'],
                title="Rep Performance: Calls vs FCR Rate",
                labels={
                    'TOTAL_CALLS': 'Total Calls Handled', 
                    'FCR_RATE': 'First Call Resolution %',
                    'TOTAL_TIME': 'Total Call Time'
                },
                color='FCR_RATE',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                height=400,  # Increased height for full-width display
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performer insight
            best_rep = rep_df.loc[rep_df['FCR_RATE'].idxmax(), 'REP_NAME']
            best_fcr = rep_df['FCR_RATE'].max()
            st.success(f"🏆 **Top Performer:** {best_rep} with {best_fcr:.1f}% FCR rate")
        else:
            st.info("No representative data available")

    # First Call Resolution Stats - Centered
    st.markdown("### 📊 Performance Over Time")
    
    # Create centered column layout for the FCR chart
    fcr_col1, fcr_col2, fcr_col3 = st.columns([1, 3, 1])
    
    with fcr_col2:
        # First call resolution over time
        fcr_query = f"""
            SELECT 
                FIRST_CALL_RESOLUTION,
                YEAR(DATE) || '-' || MONTHNAME(DATE) as YEAR_MONTH,
                COUNT(*) as CALL_COUNT,
                YEAR(DATE) as call_year,
                MONTH(DATE) as call_month
            FROM PUBLIC.ANALYZED_TRANSCRIPTIONS_APP
            WHERE DATE >= '{s_date}' AND DATE <= '{e_date}'
              AND FIRST_CALL_RESOLUTION IS NOT NULL
            GROUP BY FIRST_CALL_RESOLUTION, YEAR_MONTH, YEAR(DATE), MONTH(DATE)
            ORDER BY call_year ASC, call_month ASC, FIRST_CALL_RESOLUTION
        """
        fcr_df = load_data(fcr_query)
        
        if not fcr_df.empty:
            fig = px.bar(
                fcr_df, x='YEAR_MONTH', y='CALL_COUNT',
                color='FIRST_CALL_RESOLUTION', barmode='group',
                labels={'CALL_COUNT': 'Total Count'}
            )
            fig.update_layout(
                yaxis=dict(title='Total Calls', side='left'),
                height=500,
                annotations=[
                    dict(
                        x=0.5, y=1.15,
                        xref="paper", yref="paper",
                        text="<b>First Call Resolution Performance</b>",
                        showarrow=False,
                        font=dict(size=20, color="white", family="sans-serif"),
                        xanchor="center"
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No FCR data available")



def main():
    """Main application"""
    st.markdown("<h1 class='main-header'>📞 Cortex AISQL Powered Call Center Analytics Dashboard</h1>", 
                unsafe_allow_html=True)
    
    # Check Snowpark session
    session = get_snowpark_session()
    if not session:
        st.error("❌ Could not establish Snowpark session. Please ensure this app is running in Streamlit in Snowflake.")
        st.stop()
    
    # Sidebar
    # st.sidebar.success("✅ Connected to Snowflake")
    
    # Initialize page state if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Overview Dashboard"
    
    # Navigation sections
    st.sidebar.markdown("### 📊 Analytics & Reporting")
    if st.sidebar.button("📈 Overview Dashboard", use_container_width=True, 
                        type="primary" if st.session_state.current_page == "📊 Overview Dashboard" else "secondary"):
        st.session_state.current_page = "📊 Overview Dashboard"
        # Clear processed status when navigating
        st.session_state.audio_processed = False
        st.session_state.processed_file_name = None
        st.rerun()
    if st.sidebar.button("📊 Advanced Analytics", use_container_width=True,
                        type="primary" if st.session_state.current_page == "📈 Advanced Analytics" else "secondary"):
        st.session_state.current_page = "📈 Advanced Analytics"
        # Clear processed status when navigating
        st.session_state.audio_processed = False
        st.session_state.processed_file_name = None
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI-Powered Analysis")
    if st.sidebar.button("💬 Conversational Chat Interface - Analyst", use_container_width=True,
                        type="primary" if st.session_state.current_page == "💬 Conversational Chat Interface - Analyst" else "secondary"):
        st.session_state.current_page = "💬 Conversational Chat Interface - Analyst"
        # Clear processed status when navigating
        st.session_state.audio_processed = False
        st.session_state.processed_file_name = None
        st.rerun()

    # Audio file upload section
    show_file_upload_section()
    
    page = st.session_state.current_page
    
    # Cache management and debugging
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Semantic Model", use_container_width=True):
        st.cache_data.clear()
        st.sidebar.success("✅ Cache cleared! Semantic model will be reloaded.")
        st.rerun()
    
    # Debug section
    with st.sidebar.expander("🔧 Debug Info"):
        if st.button("📋 Show Semantic Model Info", use_container_width=True):
            show_semantic_model_debug()
        if st.button("🧪 Test Verified Query", use_container_width=True):
            test_verified_query_matching()
    
    # Add some info about the current session
    with st.sidebar.expander("ℹ️ Session Info"):
        st.write(f"**Database**: {session.get_current_database()}")
        st.write(f"**Schema**: {session.get_current_schema()}")
        st.write(f"**Warehouse**: {session.get_current_warehouse()}")
        st.write(f"**API Endpoint**: {API_ENDPOINT}")
        st.write(f"**Semantic Model**: {SEMANTIC_MODEL_STAGE_PATH}")
    
    # Main content based on selection
    try:
        if page == "📊 Overview Dashboard":
            overview_dashboard()
        elif page == "💬 Conversational Chat Interface - Analyst":
            unified_chat_interface()

        elif page == "📈 Advanced Analytics":
            advanced_analytics()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or selecting a different section.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; margin-top: 2rem;'>"
        "🔮 Powered by Snowflake Cortex Analyst REST API | Running in Streamlit in Snowflake"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 