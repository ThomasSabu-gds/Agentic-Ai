import os
import re
from typing import Dict, List
 
from autogen.agentchat import AssistantAgent
 
 
# MODEL REGISTRY
 
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
 
# LLM CONFIG BUILDER
 
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
 
 
# LOAD AGENTS FROM AZURE TABLE
 
def load_agents_from_db(table_client) -> Dict[str, dict]:
    agents = {}
 
    for ent in table_client.list_entities():
        if ent.get("PartitionKey") != "agents":
            continue
 
        name = ent["RowKey"].strip()
 
        if not name.isidentifier():
            continue
 
        model = ent.get("model", "gpt-4.1-mini")
        if model not in AVAILABLE_MODELS:
            continue
 
        agents[name] = {
            "name": name,
            "prompt": ent.get("prompt", ""),
            "model": model,
        }
 
    return agents
 
 
# BUILD AGENT CATALOG FOR SUPERVISOR
 
def build_agent_catalog(agents_meta: Dict[str, dict]) -> str:
    lines = []
    for name, meta in agents_meta.items():
        if name == "Supervisor":
            continue
        role = meta["prompt"].replace("\n", " ").strip()
        lines.append(f"- {name}: {role}")
    return "\n".join(lines)
 
 
# EXTRACT PLAN
 
def extract_plan(text: str) -> List[str]:
    plan = []
    text = text.replace("PLAN:", "PLAN:\n")
 
    for line in text.splitlines():
        match = re.match(r"\s*\d+\.\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if match:
            plan.append(match.group(1))
 
    return plan
 
 
# CONFIDENCE (BACKEND, DETERMINISTIC)
 
def compute_confidence(task: str, agent_prompt: str) -> str:
    task_words = set(task.lower().split())
    prompt_words = set(agent_prompt.lower().split())
 
    overlap = len(task_words & prompt_words)
 
    if overlap >= 3:
        return "high"
    if overlap >= 1:
        return "medium"
    return "low"
 
 
# MAIN PIPELINE (NO CHAINING)
 
def run_pipeline(task: str, table_client) -> dict:
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
 
 
    # SUPERVISOR — PLAN ONLY
 
    supervisor_meta = agents_meta["Supervisor"]
 
    supervisor = AssistantAgent(
        name="Supervisor",
        system_message=supervisor_meta["prompt"],
        llm_config=build_llm_config(supervisor_meta["model"]),
    )
 
    agent_catalog = build_agent_catalog(agents_meta)
 
    supervisor_input = f"""
USER TASK:
{task}
 
AVAILABLE AGENTS:
{agent_catalog}
""".strip()
 
    supervisor_output = supervisor.generate_reply(
        messages=[{"role": "user", "content": supervisor_input}]
    )
 
    plan = extract_plan(supervisor_output)
 
    if not plan:
        return {
            "status": "error",
            "message": "Supervisor did not produce a valid plan.",
            "raw": supervisor_output,
        }
 
 
    # EXECUTE EACH AGENT INDEPENDENTLY (NO CHAINING)
 
    results = []
    blocked_agents = []
 
    for agent_name in plan:
        if agent_name not in agents_meta:
            blocked_agents.append({
                "agent": agent_name,
                "reason": "Agent not found"
            })
            continue
 
        meta = agents_meta[agent_name]
 
        confidence = compute_confidence(task, meta["prompt"])
 
        if confidence == "low":
            blocked_agents.append({
                "agent": agent_name,
                "confidence": confidence
            })
            continue
 
        agent = AssistantAgent(
            name=agent_name,
            system_message=meta["prompt"],
            llm_config=build_llm_config(meta["model"]),
        )
        output = agent.generate_reply(
            messages=[{"role": "user", "content": task}]
        )
 
        results.append({
            "agent": agent_name,
            "confidence": confidence,
            "output": output,
        })
 
 
    # FINAL RESPONSE
 
    if not results:
        return {
            "status": "no_suitable_agent",
            "message": "No suitable agent found for this task.",
            "blocked_agents": blocked_agents,
        }
 
    return {
        "status": "success",
        "results": results,
        "blocked_agents": blocked_agents,
    }
 
 