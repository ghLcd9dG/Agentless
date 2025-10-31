import time
from typing import Dict, Union

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None

import tiktoken

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

from call_gpt import AzureChatCompletionResult, call_azure_openai


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(
    config,
    logger,
    backend: str = "openai",
    base_url=None,
    max_retries=5,
    timeout=100,
):
    if backend == "azure":
        request_config = dict(config)
        messages = request_config.pop("messages")
        mode = request_config.pop("model", "gpt-5-mini")
        max_tokens = request_config.pop("max_tokens", None)
        temperature = request_config.pop("temperature", None)
        n = request_config.pop("n", None)

        azure_kwargs = {
            "messages": messages,
            "mode": mode,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
        }

        if temperature not in (None, 1, 1.0):
            logger.warning(
                "Azure backend ignores unsupported temperature=%s; using service default",
                temperature,
            )
        # Azure GPT-5 mini currently only supports its default temperature, so we omit it.

        if n not in (None, 1):
            logger.warning(
                "Azure backend does not support n=%s samples; defaulting to 1", n
            )

        try:
            ret = call_azure_openai(**azure_kwargs)
        except Exception as e:
            logger.error("Azure OpenAI request failed", exc_info=True)
            raise e

        if not isinstance(ret, AzureChatCompletionResult):
            raise TypeError("Unexpected response type from Azure OpenAI request")

        return ret

    if backend == "openai" and openai is None:
        raise ImportError(
            "openai package not available; install it or use the azure backend"
        )

    ret = None
    retries = 0

    client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    if anthropic is None:
        raise ImportError(
            "anthropic package not available; install it or avoid using the anthropic backend"
        )

    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret
