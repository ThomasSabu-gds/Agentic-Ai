import os, json, re
from typing import Dict, Optional
from utils import utility
from copy import deepcopy
from autogen.agentchat import AssistantAgent
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# --------------------------------------------------
# MODEL REGISTRY (LLM)
# --------------------------------------------------

AVAILABLE_MODELS = {
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "api_type": "azure",
        "base_url": os.environ.get(
            "AZURE_OPENAI_BASE_URL",
            "https://model-azure.openai.azure.com/"
        ),
        "api_version": os.environ.get(
            "DEFAULT_API_VERSION",
            "2025-01-01-preview"
        ),
        "temperature": 0.4,
        "max_tokens": 1000,
    }
}

# --------------------------------------------------
# LLM CONFIG BUILDER
# --------------------------------------------------

def build_llm_config(model_key: str) -> dict:
    info = AVAILABLE_MODELS[model_key]
    return {
        "config_list": [
            {
                "model": info["model"],
                "api_key": info["api_key"],
                "api_type": info["api_type"],
                "base_url": info["base_url"],
                "api_version": info["api_version"],
            }
        ],
        "temperature": info["temperature"],
        "max_tokens": info["max_tokens"],
    }

# --------------------------------------------------
# LOAD AGENTS FROM AZURE TABLE
# --------------------------------------------------

def load_agents_from_db(table_client) -> Dict[str, dict]:
    agents = {}

    for ent in table_client.list_entities():
        if ent.get("PartitionKey") != "agents":
            continue

        name = ent["RowKey"].strip()
        if not name.isidentifier():
            continue

        agents[name] = {
            "name": name,
            "prompt": ent.get("prompt", ""),
            "model": ent.get("model", "gpt-4.1-mini"),
            "agent_type": ent.get("agent_type", "llm"),
        }

    return agents

# --------------------------------------------------
# FORM RECOGNIZER HANDLER (UNCHANGED)
# --------------------------------------------------

def run_form_recognizer(file_bytes: bytes, model_id: str):
    endpoint = os.environ.get("AZURE_DI_ENDPOINT")
    key = os.environ.get("AZURE_DI_KEY")

    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    poller = client.begin_analyze_document(
        model_id=model_id,
        document=file_bytes
    )

    result = poller.result()

    out = utility.fetch_results(result, model_id)

    return out

# --------------------------------------------------
# NEW: DOCUMENT INTENT DETECTOR
# --------------------------------------------------

def detect_document_type_from_prompt(prompt: str) -> str:
    p = prompt.lower()

    invoice_words = ["invoice", "vendor", "invoice number", "invoice id", "billing", "amount due", "tax"]
    receipt_words = ["receipt", "merchant", "tip", "subtotal", "total tax"]

    if any(w in p for w in invoice_words):
        return "invoice"
    if any(w in p for w in receipt_words):
        return "receipt"

    return "general"

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_pipeline(
    task: str,
    table_client,
    file_bytes: Optional[bytes] = None,
    doc_type: Optional[str] = None,
    filename: Optional[str] = None
) -> dict:

    if not task or not task.strip():
        return {"status": "error", "message": "Please provide a task."}

    # --------------------------------------------------
    # NO FILE CASE
    # --------------------------------------------------

    if not file_bytes:
        if any(word in task.lower() for word in ["invoice", "receipt", "document", "extract", "summarize", "tax"]):
            return {
                "status": "success",
                "agent": "System",
                "output": "No document found."
            }

    # --------------------------------------------------
    # DOCUMENT UPLOADED FLOW
    # --------------------------------------------------

    if file_bytes:

        doc_intent = detect_document_type_from_prompt(task)

        if doc_intent == "invoice":
            model_id = "prebuilt-invoice"
            extracted = run_form_recognizer(file_bytes, model_id)

        elif doc_intent == "receipt":
            model_id = "prebuilt-receipt"
            extracted = run_form_recognizer(file_bytes, model_id)

        else:
            # GENERAL DOCUMENT → ask user confirmation
            if task.strip().lower() in ["yes", "y"]:
                from utils.utility import extract_text_from_file

                text = extract_text_from_file(file_bytes, filename)

                summary_agent = AssistantAgent(
                    name="SummaryAgent",
                    system_message="""
You are a document summarization assistant.
Provide a clear, point-by-point summary of the document in a vertical list format.
Each point should be concise, informative, and written on a new line.
Do not write paragraphs.

""",
                    llm_config=build_llm_config("gpt-4.1-mini"),
                )

                summary = summary_agent.generate_reply(
                    messages=[{
                        "role": "user",
                        "content": f"Summarize:\n{text}"
                    }]
                )

                return {
                    "status": "success",
                    "agent": "SummaryAgent",
                    "output": summary
                }

            return {
                "status": "success",
                "agent": "System",
                "output": "This looks like a general document. Do you want me to extract data from it? ",
                "needs_confirmation": True
            }

        # --------------------------------------------------
        # DOC QA (Invoice/Receipt)
        # --------------------------------------------------

        extracted_text = "\n".join(f"{k}: {v}" for k, v in extracted.items())

        qa_agent = AssistantAgent(
            name="DocQA",
            system_message="""
You are a document question answering assistant.
Answer ONLY from the document data.
""",
            llm_config=build_llm_config("gpt-4.1-mini"),
        )

        final_answer = qa_agent.generate_reply(
            messages=[{
                "role": "user",
                "content": f"""
Document Data:
{extracted_text}

User Question:
{task}
"""
            }]
        )

        return {
            "status": "success",
            "agent": "DocQA",
            "output": final_answer
        }

    # --------------------------------------------------
    # NORMAL LLM AGENT FLOW (NO FILE)
    # --------------------------------------------------

    agents_meta = load_agents_from_db(table_client)

    supervisor_meta = agents_meta["Supervisor"]

    supervisor = AssistantAgent(
        name="Supervisor",
        system_message=supervisor_meta["prompt"],
        llm_config=build_llm_config(supervisor_meta["model"]),
    )

    agent_catalog = "\n".join(
        f"- {n}: {m['prompt'].replace(chr(10), ' ')}"
        for n, m in agents_meta.items() if n != "Supervisor"
    )

    supervisor_input = f"""
USER TASK:
{task}

AVAILABLE AGENTS:
{agent_catalog}
"""

    selected_agent = supervisor.generate_reply(
        messages=[{"role": "user", "content": supervisor_input}]
    ).strip()

    agent_meta = agents_meta[selected_agent]

    agent = AssistantAgent(
        name=agent_meta["name"],
        system_message=agent_meta["prompt"],
        llm_config=build_llm_config(agent_meta["model"]),
    )

    output = agent.generate_reply(
        messages=[{"role": "user", "content": task}]
    )

    return {
        "status": "success",
        "agent": selected_agent,
        "output": utility.format_output(output)
    }
