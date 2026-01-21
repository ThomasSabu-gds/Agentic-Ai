import os
from typing import Dict, Optional

from autogen.agentchat import AssistantAgent
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential


# -------------------------------
# MODEL REGISTRY
# -------------------------------

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


# -------------------------------
# LLM CONFIG BUILDER
# -------------------------------

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


# -------------------------------
# LOAD AGENTS FROM AZURE TABLE
# -------------------------------

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


# -------------------------------
# BUILD AGENT CATALOG FOR SUPERVISOR
# -------------------------------

def build_agent_catalog(
    agents_meta: Dict[str, dict],
    file_bytes: Optional[bytes]
) -> str:
    lines = []

    for name, meta in agents_meta.items():
        if name == "Supervisor":
            continue

        # Hide service agents if no file uploaded
        if meta.get("agent_type") == "service" and not file_bytes:
            continue

        role = meta["prompt"].replace("\n", " ").strip()
        lines.append(f"- {name}: {role}")

    return "\n".join(lines)


# -------------------------------
# DOCUMENT INTELLIGENCE (INVOICE)
# -------------------------------

def run_document_intelligence(file_bytes: bytes) -> str:
    endpoint = os.environ["AZURE_DI_ENDPOINT"]
    key = os.environ["AZURE_DI_KEY"]

    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    poller = client.begin_analyze_document(
        model_id="prebuilt-invoice",
        document=file_bytes
    )

    result = poller.result()
    output = []

    for idx, invoice in enumerate(result.documents):
        output.append(f"=== INVOICE #{idx + 1} FIELDS ===")

        for field_name, field in invoice.fields.items():
            value = field.value if field.value is not None else "N/A"
            confidence = (
                f"{field.confidence:.2%}"
                if field.confidence is not None
                else "N/A"
            )
            output.append(
                f"{field_name}: {value} (confidence: {confidence})"
            )

    if not output:
        output.append("No invoice fields detected.")

    return "\n".join(output)


# -------------------------------
# MAIN PIPELINE (SINGLE AGENT)
# -------------------------------

def run_pipeline(
    task: str,
    table_client,
    file_bytes: Optional[bytes] = None
) -> dict:

    if not task or not task.strip():
        return {
            "status": "error",
            "message": "Please provide a task."
        }

    task = task.strip()
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

    selected = supervisor.generate_reply(
        messages=[{"role": "user", "content": supervisor_input}]
    ).strip()

    if selected == "NONE":
        return {
            "status": "no_suitable_agent",
            "message": "No suitable agent found for this task."
        }

    if selected not in agents_meta:
        return {
            "status": "error",
            "message": "Supervisor selected an unknown agent.",
            "raw": selected
        }

    agent_meta = agents_meta[selected]

    # -------------------------------
    # EXECUTION
    # -------------------------------

    if agent_meta["agent_type"] == "service":
        if not file_bytes:
            return {
                "status": "error",
                "message": "No document provided for Invoice processing."
            }

        output = run_document_intelligence(file_bytes)

        return {
            "status": "success",
            "agent": selected,
            "output": output,
        }

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
        "agent": selected,
        "output": output,
    }
