"""Prompt composition helper for agent calls."""

from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import AgentConfig


def build_agent_prompt(agent_cfg: AgentConfig, context: dict[str, Any]) -> str:
    payload = {
        "agent_id": agent_cfg.agent_id,
        "required_data": agent_cfg.required_data,
        "indicators": agent_cfg.indicators,
        "context": context,
        "output_schema": agent_cfg.output_schema,
    }
    return f"{agent_cfg.system_prompt}\n\nINPUT:\n{json.dumps(payload, separators=(',', ':'), default=str)}"
