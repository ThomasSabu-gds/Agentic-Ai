import os
from typing import Dict, Optional

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
            "agent_type": ent.get("agent_type", "llm"),  # llm | service
        }

    return agents


# --------------------------------------------------
# BUILD AGENT CATALOG
# --------------------------------------------------

def build_agent_catalog(
    agents_meta: Dict[str, dict],
    file_bytes: Optional[bytes]
) -> str:
    lines = []

    for name, meta in agents_meta.items():
        if name == "Supervisor":
            continue

        if meta["agent_type"] == "service" and not file_bytes:
            continue

        role = meta["prompt"].replace("\n", " ").strip()
        lines.append(f"- {name}: {role}")

    return "\n".join(lines)


# --------------------------------------------------
# CLEAN VALUE EXTRACTION (SDK SAFE)
# --------------------------------------------------

def extract_clean_value(field):
    """
    SAFE for azure-ai-formrecognizer 3.3.3
    Returns only user-meaningful data
    """
    if field is None:
        return None

    if field.value is not None:
        return field.value

    if field.content:
        return field.content

    return None


# --------------------------------------------------
# FORM RECOGNIZER HANDLER (FILTERED OUTPUT)
# --------------------------------------------------

def run_form_recognizer(
    file_bytes: bytes,
    model_id: str
) -> str:
    endpoint = os.environ.get("AZURE_DI_ENDPOINT")
    key = os.environ.get("AZURE_DI_KEY")

    if not endpoint or not key:
        return "Document Intelligence credentials not configured."

    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    poller = client.begin_analyze_document(
        model_id=model_id,
        document=file_bytes
    )

    result = poller.result()
    output_lines = []

    for document in result.documents:
        output_lines.append("=== EXTRACTED FIELDS ===")

        for field_name, field in document.fields.items():
            value = extract_clean_value(field)
            if value is None:
                continue

            output_lines.append(f"{field_name}: {value}")

    if len(output_lines) == 1:
        output_lines.append("No meaningful fields detected.")

    return "\n".join(output_lines)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_pipeline(
    task: str,
    table_client,
    file_bytes: Optional[bytes] = None,
    doc_type: Optional[str] = None
) -> dict:

    if not task or not task.strip():
        return {
            "status": "error",
            "message": "Please provide a task."
        }

    agents_meta = load_agents_from_db(table_client)

    if "Supervisor" not in agents_meta:
        return {
            "status": "error",
            "message": "Supervisor agent missing in database."
        }

    supervisor_meta = agents_meta["Supervisor"]

    supervisor = AssistantAgent(
        name="Supervisor",
        system_message=supervisor_meta["prompt"],
        llm_config=build_llm_config(supervisor_meta["model"]),
    )

    agent_catalog = build_agent_catalog(agents_meta, file_bytes)

    supervisor_input = f"""
USER TASK:
{task}

FILE_UPLOADED:
{"YES" if file_bytes else "NO"}

AVAILABLE AGENTS:
{agent_catalog}
""".strip()

    selected_agent = supervisor.generate_reply(
        messages=[{"role": "user", "content": supervisor_input}]
    ).strip()

    if selected_agent == "NONE":
        return {
            "status": "no_suitable_agent",
            "message": "No suitable agent found for this task."
        }

    if selected_agent not in agents_meta:
        return {
            "status": "error",
            "message": "Supervisor selected an unknown agent.",
            "raw": selected_agent
        }

    agent_meta = agents_meta[selected_agent]

    # --------------------------------------------------
    # SERVICE AGENT EXECUTION
    # --------------------------------------------------

    if agent_meta["agent_type"] == "service":
        if not file_bytes:
            return {
                "status": "error",
                "message": "No document provided."
            }

        if doc_type == "invoice":
            model_id = "prebuilt-invoice"
        elif doc_type == "receipt":
            model_id = "prebuilt-receipt"
        elif doc_type == "identity":
            model_id = "prebuilt-idDocument"
        else:
            return {
                "status": "error",
                "message": "Unknown document type."
            }

        output = run_form_recognizer(file_bytes, model_id)

        return {
            "status": "success",
            "agent": selected_agent,
            "output": output,
        }

    # --------------------------------------------------
    # LLM AGENT EXECUTION
    # --------------------------------------------------

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
        "output": output,
    }
