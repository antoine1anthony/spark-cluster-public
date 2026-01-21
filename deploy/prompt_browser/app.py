#!/usr/bin/env python3
"""
Prompt Browser - Streamlit UI for Agent System Prompts

Browse, preview, and copy agent prompts from the Prompt Registry.
One-click copy + Open in OpenWebUI workflow.
"""

import os
import streamlit as st
import requests

# Configuration - via environment variables
REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://localhost:8010")
OPENWEBUI_URL = os.environ.get("OPENWEBUI_URL", "http://localhost:8080")

# Page config
st.set_page_config(
    page_title="Prompt Browser",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .prompt-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .prompt-category {
        color: #6b7280;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    .copy-success {
        color: #10b981;
        font-weight: 500;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def copy_to_clipboard_js(text: str, button_id: str) -> str:
    """Generate JavaScript to copy text to clipboard (works on HTTP)."""
    # Escape for JS string
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    return f"""
    <script>
    function copyToClipboard_{button_id}() {{
        const text = `{escaped}`;
        
        // Fallback method for HTTP (execCommand)
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        textarea.style.top = '-9999px';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        
        try {{
            document.execCommand('copy');
            document.getElementById('copy-status-{button_id}').innerHTML = '✓ Copied!';
            setTimeout(function() {{
                document.getElementById('copy-status-{button_id}').innerHTML = '';
            }}, 2000);
        }} catch (err) {{
            document.getElementById('copy-status-{button_id}').innerHTML = '✗ Failed';
        }}
        
        document.body.removeChild(textarea);
    }}
    </script>
    <button onclick="copyToClipboard_{button_id}()" style="
        background-color: #3b82f6;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        font-size: 0.875rem;
        margin-right: 0.5rem;
    ">📋 Copy to Clipboard</button>
    <span id="copy-status-{button_id}" style="color: #10b981; font-weight: 500;"></span>
    """


def open_webui_js() -> str:
    """Generate JavaScript to open OpenWebUI in a new tab."""
    return f"""
    <a href="{OPENWEBUI_URL}" target="_blank" style="
        display: inline-block;
        background-color: #8b5cf6;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.375rem;
        cursor: pointer;
        font-size: 0.875rem;
        text-decoration: none;
    ">🚀 Open WebUI</a>
    """


@st.cache_data(ttl=60)
def fetch_prompts() -> dict:
    """Fetch all prompts from the registry."""
    try:
        response = requests.get(f"{REGISTRY_URL}/prompts", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch prompts: {e}")
        return {"prompts": {}, "total": 0}


@st.cache_data(ttl=60)
def fetch_prompt_content(category: str, name: str) -> dict:
    """Fetch a specific prompt's content."""
    try:
        response = requests.get(
            f"{REGISTRY_URL}/prompts/system/{category}/{name}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch prompt: {e}")
        return {"role": "system", "content": "Error loading prompt"}


def format_category_name(category: str) -> str:
    """Format category name for display."""
    return category.replace("_", " ").title()


def main():
    # Header
    st.title("📋 Prompt Browser")
    st.markdown("Browse and copy agent system prompts for use in OpenWebUI")
    
    # Fetch all prompts
    data = fetch_prompts()
    prompts_by_category = data.get("prompts", {})
    total = data.get("total", 0)
    
    if not prompts_by_category:
        st.warning("No prompts available. Is the Prompt Registry running?")
        st.code(f"Registry URL: {REGISTRY_URL}")
        return
    
    st.sidebar.markdown(f"**{total} prompts available**")
    st.sidebar.divider()
    
    # Category selector in sidebar
    categories = sorted(prompts_by_category.keys())
    selected_category = st.sidebar.selectbox(
        "Category",
        categories,
        format_func=format_category_name
    )
    
    # Prompt selector in sidebar
    if selected_category:
        prompts_in_category = prompts_by_category[selected_category]
        prompt_names = [p["name"] for p in prompts_in_category]
        prompt_titles = {p["name"]: p.get("title", p["name"]) for p in prompts_in_category}
        
        selected_prompt = st.sidebar.selectbox(
            "Prompt",
            prompt_names,
            format_func=lambda x: prompt_titles.get(x, x)
        )
        
        if selected_prompt:
            # Fetch and display the prompt
            prompt_data = fetch_prompt_content(selected_category, selected_prompt)
            content = prompt_data.get("content", "")
            
            # Display header
            st.markdown(f'<div class="prompt-title">{prompt_titles.get(selected_prompt, selected_prompt)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prompt-category">{format_category_name(selected_category)} / {selected_prompt}</div>', unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns([2, 2, 6])
            with col1:
                st.components.v1.html(
                    copy_to_clipboard_js(content, f"{selected_category}_{selected_prompt}"),
                    height=50
                )
            with col2:
                st.components.v1.html(open_webui_js(), height=50)
            
            st.divider()
            
            # Content preview
            st.subheader("Preview")
            
            # Show as markdown or code
            view_mode = st.radio(
                "View as",
                ["Rendered", "Raw"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            if view_mode == "Rendered":
                st.markdown(content)
            else:
                st.code(content, language="markdown")
            
            # Metadata
            st.sidebar.divider()
            st.sidebar.markdown("**Quick Info**")
            st.sidebar.markdown(f"- Characters: {len(content):,}")
            st.sidebar.markdown(f"- Lines: {content.count(chr(10)) + 1}")
            
            # Instructions
            st.sidebar.divider()
            st.sidebar.markdown("**How to use**")
            st.sidebar.markdown("""
            1. Click **Copy to Clipboard**
            2. Click **Open WebUI**
            3. Start a new chat
            4. Paste into **System Prompt**
            5. Chat with the agent!
            """)


if __name__ == "__main__":
    main()
