import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

MODEL = os.environ.get("LLM_MODEL", "global.anthropic.claude-sonnet-5-5")
client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "ap-northeast-2"))
log = logging.getLogger("llm")


MAX_RETRIES = 3        # ThrottlingException일 때만 최대 3번 재시도
BASE_DELAY = 1.0       # 지수 백오프: 1s, 2s, 4s


def _converse_with_retry(**kwargs) -> dict:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.converse(**kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            # AccessDenied, ValidationException 등은 재시도하지 않고 바로 올린다
            if code != "ThrottlingException" or attempt == MAX_RETRIES:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            log.warning("ThrottlingException, %.1fs 후 재시도 (%d/%d)", delay, attempt + 1, MAX_RETRIES)
            time.sleep(delay)


def ask(system: str, user: str, max_tokens: int = 1024) -> dict:
    t0 = time.time()
    resp = _converse_with_retry(
        modelId=MODEL,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    # output.message.content: 응답 블록 목록 (text 외 블록 포함 가능)
    text = "".join(b["text"] for b in resp["output"]["message"]["content"] if "text" in b)
    return {"text": text, "usage": resp["usage"], "stop": resp["stopReason"],
            "ms": int((time.time() - t0) * 1000)}


def ask_logged(system, user, **kw):
    r = ask(system, user, **kw)
    log.info(json.dumps({
        "prompt_version": os.environ.get("PROMPT_VERSION", "v1"),
        "in": r["usage"]["inputTokens"], "out": r["usage"]["outputTokens"],
        "stop": r["stop"], "ms": r["ms"],
    }))
    # max_tokens면 답이 잘린 것
    if r["stop"] == "max_tokens":
        log.warning("응답이 maxTokens에서 잘렸습니다")
    return r
