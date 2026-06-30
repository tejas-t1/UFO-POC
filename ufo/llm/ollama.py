# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import copy
import io
import json
import logging
import time
from typing import Any, Optional, Dict, List

import requests
from openai import OpenAI
from PIL import Image

from .openai import BaseOpenAIService

logger = logging.getLogger(__name__)


class OllamaService(BaseOpenAIService):
    """
    A service class for Ollama models.
    """

    def __init__(self, config, agent_type: str):
        """
        Initialize the Ollama service.
        :param config: The configuration.
        :param agent_type: The agent type.
        """
        agent_cfg = config[agent_type]
        base_url = agent_cfg["API_BASE"]
        agent_cfg["API_KEY"] = "ollama"
        super().__init__(config, agent_type, "openai", f"{base_url}/v1")

        # Optional extras to support remote Ollama deployments fronted by a
        # reverse proxy (HTTP Basic Auth and/or self-signed TLS certs).
        username = agent_cfg.get("API_USERNAME")
        password = agent_cfg.get("API_PASSWORD")
        verify_ssl = agent_cfg.get("VERIFY_SSL", True)

        if username is not None or verify_ssl is False:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - httpx ships with openai
                raise RuntimeError(
                    "httpx is required for Ollama basic-auth / custom TLS support"
                ) from exc

            auth = (
                httpx.BasicAuth(username, password or "")
                if username is not None
                else None
            )
            http_client = httpx.Client(
                verify=bool(verify_ssl),
                auth=auth,
                timeout=self.config["TIMEOUT"],
            )
            self.client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key=agent_cfg["API_KEY"],
                max_retries=self.max_retry,
                timeout=self.config["TIMEOUT"],
                http_client=http_client,
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generates completions for a given conversation using the Ollama API.
        :param messages: The list of messages in the conversation.
        :param n: The number of completions to generate.
        :param stream: Whether to stream the API response.
        :param temperature: The temperature parameter for randomness in the output.
        :param max_tokens: The maximum number of tokens in the generated completion.
        :param top_p: The top-p parameter for nucleus sampling.
        :param kwargs: Additional keyword arguments to pass to the OpenAI API.
        :return: A tuple containing a list of generated completions and the estimated cost.
        """
        return super()._chat_completion(
            messages,
            False,
            temperature,
            max_tokens,
            top_p,
            response_format={"type": "json_object"},
            **kwargs,
        )
