"""Web chat interface for the Text-to-SQL system."""

import logging
from typing import Dict, Optional

import gradio as gr

from .text_to_sql_optimized import TextToSQLConfig, TextToSQLSystem


logger = logging.getLogger(__name__)


_SYSTEM: Optional[TextToSQLSystem] = None


def get_system() -> TextToSQLSystem:
    """Lazily initialize and cache the Text-to-SQL system."""
    global _SYSTEM
    if _SYSTEM is None:
        logger.info("Initializing TextToSQLSystem for web chat")
        config = TextToSQLConfig(
            llm_model="defog/sqlcoder-7b-2",
            use_gguf=True,
            database_path="./sample_database.db",
            max_correction_attempts=3,
            n_ctx=2048,
            n_threads=0,
            n_gpu_layers=-1,
        )
        _SYSTEM = TextToSQLSystem(config)
    return _SYSTEM


def format_response(result: Dict) -> str:
    """Create a formatted text response for the chat UI."""
    lines = []

    schema_info = result.get("schema_info")
    if schema_info:
        if schema_info.get("source") == "query":
            lines.append("**Schema (extracted from your question):**")
            table = schema_info.get("table") or "unknown"
            columns = schema_info.get("columns") or []
            lines.append(f"- Table: `{table}`")
            if columns:
                col_list = ", ".join(f"`{col}`" for col in columns)
                lines.append(f"- Columns: {col_list}")
        else:
            tables = schema_info.get("tables") or []
            if tables:
                table_list = ", ".join(f"`{tbl}`" for tbl in tables)
                lines.append("**Schema (retrieved from database metadata):**")
                lines.append(f"- Tables: {table_list}")

    sql = result.get("sql") or "-- No SQL generated --"
    lines.append("\n**SQL Query:**")
    lines.append(f"```sql\n{sql}\n```")

    status_text = result.get("result") or "SQL generated."
    lines.append(f"\n_Status:_ {status_text}")

    return "\n".join(lines)


def _process_message(message: str) -> str:
    system = get_system()
    logger.info("Received chat message")
    result = system.query(message, skip_execution=True)
    return format_response(result)


def _add_user_message(message: str, history):
    message = message.strip()
    if not message:
        return gr.update(value=""), history
    history = history + [(message, None)]
    return gr.update(value=""), history


def _generate_bot_reply(history):
    if not history:
        return history
    user_message, _ = history[-1]
    reply = _process_message(user_message)
    history[-1] = (user_message, reply)
    return history


def _clear_history():
    return []


def main():
    description = (
        "Type your analytics request as if you were texting a teammate. "
        "Include the table and columns so the assistant can build precise SQL."
    )

    css = """
    .gradio-container {background: #ece5dd; font-family: 'Helvetica Neue', Arial, sans-serif;}
    #chat-header {background: #075e54; color: #fff; padding: 16px 24px; border-radius: 12px;}
    #chat-header h1 {margin: 0; font-size: 1.35rem;}
    #chat-header p {margin: 4px 0 0 0; font-size: 0.9rem; opacity: 0.9;}
    .chatbot-container {border-radius: 16px; box-shadow: 0 12px 24px rgba(7, 94, 84, 0.18); overflow: hidden;}
    .chatbot-container [data-testid="chatbot"] {background: #ece5dd;}
    .chatbot-container .message.user {background: #dcf8c6; color: #2a2a2a; border-radius: 16px 16px 0 16px;}
    .chatbot-container .message.bot {background: #ffffff; color: #1f2326; border-radius: 16px 16px 16px 0;}
    .message {padding: 0.65rem 0.9rem !important;}
    .message > p {margin: 0;}
    .input-row {background: #f0f0f0; padding: 12px; border-radius: 16px;}
    .input-row textarea {border: none !important; box-shadow: none !important; background: #fff !important; border-radius: 12px !important;}
    .send-btn {background: #25d366 !important; color: #fff !important; border: none !important; border-radius: 12px !important;}
    .clear-btn {background: transparent !important; color: #075e54 !important; border: 1px solid #075e54 !important; border-radius: 12px !important;}
    """

    with gr.Blocks(title="Text-to-SQL Chat", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
            <div id="chat-header">
              <h1>Text-to-SQL Assistant</h1>
              <p>🔍 Ask questions about your tables. Mention column names to keep the SQL precise.</p>
            </div>
            """,
            elem_id="chat-header"
        )

        chatbot = gr.Chatbot(
            label="",
            height=520,
            bubble_full_width=False,
            show_copy_button=True,
            render_markdown=True,
            elem_classes=["chatbot-container"],
        )

        with gr.Row(elem_classes=["input-row"]):
            user_input = gr.Textbox(
                placeholder="Type your request... (e.g., 'Total disbursement per zone for 2023. Columns: ...')",
                autofocus=True,
                lines=2,
                show_label=False,
            )
            send_button = gr.Button("Send", elem_classes=["send-btn"], scale=0)
        clear_button = gr.Button("Reset Conversation", elem_classes=["clear-btn"])

        user_input.submit(_add_user_message, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            _generate_bot_reply, chatbot, chatbot
        )
        send_button.click(_add_user_message, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            _generate_bot_reply, chatbot, chatbot
        )
        clear_button.click(_clear_history, outputs=chatbot, queue=False)

        demo.load(lambda: [], outputs=chatbot, queue=False)

    demo.launch(share=True)


if __name__ == "__main__":
    main()
