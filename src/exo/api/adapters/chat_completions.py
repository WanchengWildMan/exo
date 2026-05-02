"""OpenAI Chat Completions API adapter for converting requests/responses."""

import base64
import logging
import os
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

from exo.api.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Logprobs,
    LogprobsContentItem,
    StreamingChoiceResponse,
    ToolCall,
    Usage,
)
from exo.download.download_utils import create_http_session
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
    resolve_reasoning_params,
)

logger = logging.getLogger(__name__)

_CONTENT_REASONING_FALLBACK_SENTINELS: frozenset[str] = frozenset(
    {
        "*",
        '"',
        "'",
        "`",
        ",",
        "，",
        ".",
        "。",
        "!",
        "！",
        "?",
        "？",
        ":",
        "：",
        ";",
        "；",
        "-",
        "~",
        "…",
    }
)


def _looks_like_placeholder_content(content: str) -> bool:
    stripped = content.strip()
    if stripped == "" or stripped in _CONTENT_REASONING_FALLBACK_SENTINELS:
        return True

    if len(stripped) > 3:
        return False

    return all(not char.isalnum() for char in stripped)


def openai_compatible_content(content: str, reasoning_content: str | None) -> str:
    """Ensure OpenAI clients get usable content when thinking-only text is emitted.

    Some thinking models may produce a tiny sentinel token (for example "*" or a
    punctuation mark) in ``content`` while placing the meaningful answer in
    ``reasoning_content``. Many OpenAI clients ignore ``reasoning_content`` and
    treat this as an empty assistant reply.
    """
    if not reasoning_content:
        return content

    if _looks_like_placeholder_content(content):
        return reasoning_content

    return content


def extract_base64_from_data_url(data_url: str) -> str:
    match = re.match(r"data:[^;]+;base64,(.+)", data_url)
    if match:
        return match.group(1)
    return data_url


async def fetch_image_url(url: str) -> str:
    headers = {"User-Agent": "exo/1.0"}
    async with (
        create_http_session(timeout_profile="short") as session,
        session.get(url, headers=headers) as resp,
    ):
        resp.raise_for_status()
        data = await resp.read()
        return base64.b64encode(data).decode("ascii")


_STRIP_SKILLS_BLOCK: bool = os.getenv("EXO_STRIP_SKILLS_BLOCK", "").lower() in {
    "1", "true", "yes", "on"
}
_MAX_SYSTEM_PROMPT_CHARS: int | None = (
    int(os.getenv("EXO_MAX_SYSTEM_PROMPT_CHARS", ""))
    if os.getenv("EXO_MAX_SYSTEM_PROMPT_CHARS", "")
    else None
)
_SKILLS_DESC_MAX_CHARS: int | None = (
    int(os.getenv("EXO_SKILLS_DESC_MAX_CHARS", ""))
    if os.getenv("EXO_SKILLS_DESC_MAX_CHARS", "")
    else None
)
_OPENCLAW_SKILLS_DESC_MAX_CHARS: int | None = (
    int(os.getenv("EXO_OPENCLAW_SKILLS_DESC_MAX_CHARS", "30"))
    if os.getenv("EXO_OPENCLAW_SKILLS_DESC_MAX_CHARS", "30")
    else None
)
_OPENCLAW_MAX_SYSTEM_PROMPT_CHARS: int | None = (
    int(os.getenv("EXO_OPENCLAW_MAX_SYSTEM_PROMPT_CHARS", "3000"))
    if os.getenv("EXO_OPENCLAW_MAX_SYSTEM_PROMPT_CHARS", "3000")
    else None
)
_SKILLS_BLOCK_RE = re.compile(
    r"(<available_skills>.*?</available_skills>)",
    re.DOTALL,
)
_SKILL_BLOCK_RE = re.compile(
    r"<skill>\s*(?P<body>.*?)\s*</skill>",
    re.DOTALL,
)
_DESCRIPTION_LINE_RE = re.compile(
    r"^(?P<leading_ws>\s*)<description>(?P<content>.*?)(?P<closing_tag></description>)?$"
)
_OPENCLAW_SYSTEM_PROMPT_MARKERS: tuple[str, ...] = (
    "You are a personal assistant running inside OpenClaw.",
    "## OpenClaw CLI Quick Reference",
    "<available_skills>",
)

# ── OpenClaw fixed-section Chinese compression ──────────────────────────
# Maps known English boilerplate sections to compact Chinese equivalents.
# Only applied when EXO_OPENCLAW_COMPRESS_CHINESE=1 (or auto-detected).
_COMPRESS_CHINESE: bool = os.getenv("EXO_OPENCLAW_COMPRESS_CHINESE", "1").lower() in {
    "1", "true", "yes", "on"
}

_OPENCLAW_CHINESE_REPLACEMENTS: list[tuple[str, str]] = [
    # Identity line
    (
        "You are a personal assistant running inside OpenClaw.",
        "你是运行在 OpenClaw 中的个人助手。",
    ),
    # Tooling header
    (
        "## Tooling\nTool availability (filtered by policy):\nTool names are case-sensitive. Call tools exactly as listed.",
        "## 工具\n可用工具（按策略过滤，名称区分大小写）：",
    ),
    # Tool descriptions — compress common ones
    ("- read: Read file contents", "- read: 读取文件"),
    ("- write: Create or overwrite files", "- write: 创建/覆盖文件"),
    ("- edit: Make precise edits to files", "- edit: 精确编辑文件"),
    ("- exec: Run shell commands (pty available for TTY-required CLIs)", "- exec: 执行shell命令（支持PTY）"),
    ("- process: Manage background exec sessions", "- process: 管理后台进程"),
    ("- sessions_list: List other sessions (incl. sub-agents) with filters/last", "- sessions_list: 列出会话"),
    ("- sessions_history: Fetch history for another session/sub-agent", "- sessions_history: 获取会话历史"),
    ("- sessions_send: Send a message to another session/sub-agent", "- sessions_send: 发送消息到其他会话"),
    (
        "- session_status: Show a /status-equivalent status card (usage + time + Reasoning/Verbose/Elevated); use for model-use questions (📊 session_status); optional per-session model override",
        "- session_status: 显示状态卡片（用量+时间+推理级别）",
    ),
    ("- image: Analyze an image with the configured image model", "- image: 分析图片"),
    (
        "- memory_get: Safe snippet read from MEMORY.md or memory/*.md with optional from/lines; use after memory_search to pull only the needed lines and keep context small.",
        "- memory_get: 从MEMORY.md读取片段（配合memory_search使用）",
    ),
    (
        "- memory_search: Mandatory recall step: semantically search MEMORY.md + memory/*.md (and optional session transcripts) before answering questions about prior work, decisions, dates, people, preferences, or todos; returns top snippets with path + lines.",
        "- memory_search: 语义搜索记忆（回答历史问题前必须调用）",
    ),
    ("- sessions_spawn: Spawn a sub-agent session", "- sessions_spawn: 创建子智能体"),
    # Tooling footer
    (
        "TOOLS.md does not control tool availability; it is user guidance for how to use external tools.\nIf a task is more complex or takes longer, spawn a sub-agent. It will do the work for you and ping you when it's done. You can always check up on it.",
        "TOOLS.md 是用户指南不控制工具可用性。复杂任务请生成子智能体。",
    ),
    # Tool Call Style — entire section
    (
        "## Tool Call Style\nDefault: do not narrate routine, low-risk tool calls (just call the tool).\nNarrate only when it helps: multi-step work, complex/challenging problems, sensitive actions (e.g., deletions), or when the user explicitly asks.\nKeep narration brief and value-dense; avoid repeating obvious steps.\nUse plain human language for narration unless in a technical context.",
        "## 工具调用风格\n默认：不解说常规调用，直接调用。仅在复杂/敏感操作或用户要求时简要说明。",
    ),
    # Safety — entire section
    (
        "## Safety\nYou have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking; avoid long-term plans beyond the user's request.\nPrioritize safety and human oversight over completion; if instructions conflict, pause and ask; comply with stop/pause/audit requests and never bypass safeguards. (Inspired by Anthropic's constitution.)\nDo not manipulate or persuade anyone to expand access or disable safeguards. Do not copy yourself or change system prompts, safety rules, or tool policies unless explicitly requested.",
        "## 安全\n无自主目标，不追求自保/复制/资源/权力。安全优先于完成；冲突时暂停询问。不操纵他人扩展权限或绕过防护。",
    ),
    # CLI Quick Reference
    (
        "## OpenClaw CLI Quick Reference\nOpenClaw is controlled via subcommands. Do not invent commands.\nTo manage the Gateway daemon service (start/stop/restart):\n- openclaw gateway status\n- openclaw gateway start\n- openclaw gateway stop\n- openclaw gateway restart\nIf unsure, ask the user to run `openclaw help` (or `openclaw gateway --help`) and paste the output.",
        "## OpenClaw CLI\n通过子命令控制。网关管理：openclaw gateway status/start/stop/restart。不确定时让用户运行 `openclaw help`。",
    ),
    # Skills instruction
    (
        "## Skills (mandatory)\nBefore replying: scan <available_skills> <description> entries.\n- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.\n- If multiple could apply: choose the most specific one, then read/follow it.\n- If none clearly apply: do not read any SKILL.md.\nConstraints: never read more than one skill up front; only read after selecting.\nThe following skills provide specialized instructions for specific tasks.\nUse the read tool to load a skill's file when the task matches its description.",
        "## Skills（必须）\n回复前扫描 <available_skills>：匹配一个→read其SKILL.md并遵循；多个匹配→选最具体的；无匹配→不读。一次只读一个。",
    ),
    # Memory Recall
    (
        "## Memory Recall\nBefore answering anything about prior work, decisions, dates, people, preferences, or todos: run memory_search on MEMORY.md + memory/*.md; then use memory_get to pull only the needed lines. If low confidence after search, say you checked.",
        "## 记忆回溯\n回答历史相关问题前：先 memory_search 再 memory_get 取所需行。搜索后仍不确定则说明已查。",
    ),
    # Reply Tags
    (
        "## Reply Tags\nTo request a native reply/quote on supported surfaces, include one tag in your reply:\n- [[reply_to_current]] replies to the triggering message.\n- [[reply_to:<id>]] replies to a specific message id when you have it.\nWhitespace inside the tag is allowed (e.g. [[ reply_to_current ]] / [[ reply_to: 123 ]]).\nTags are stripped before sending; support depends on the current channel config.",
        "## 回复标签\n[[reply_to_current]] 回复触发消息；[[reply_to:<id>]] 回复指定消息。标签发送前移除。",
    ),
    # Messaging
    (
        "## Messaging\n- Reply in current session → automatically routes to the source channel (Signal, Telegram, etc.)\n- Cross-session messaging → use sessions_send(sessionKey, message)\n- Never use exec/curl for provider messaging; OpenClaw handles all routing internally.",
        "## 消息\n当前会话回复自动路由到源渠道。跨会话用 sessions_send。不要用 exec/curl 发消息。",
    ),
    # Silent Replies
    (
        "## Silent Replies\nWhen you have nothing to say, respond with ONLY: NO_REPLY\n⚠️ Rules:\n- It must be your ENTIRE message — nothing else\n- Never append it to an actual response (never include \"NO_REPLY\" in real replies)\n- Never wrap it in markdown or code blocks\n❌ Wrong: \"Here's help... NO_REPLY\"\n❌ Wrong: \"NO_REPLY\"\n✅ Right: NO_REPLY",
        "## 静默回复\n无话可说时仅回复 NO_REPLY（必须是完整消息，不附加到其他内容）。",
    ),
    # Heartbeats
    (
        "## Heartbeats\nHeartbeat prompt: Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. Do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK.\nIf you receive a heartbeat poll (a user message matching the heartbeat prompt above), and there is nothing that needs attention, reply exactly:\nHEARTBEAT_OK\nOpenClaw treats a leading/trailing \"HEARTBEAT_OK\" as a heartbeat ack (and may discard it).\nIf something needs attention, do NOT include \"HEARTBEAT_OK\"; reply with the alert text instead.",
        "## 心跳\n收到心跳轮询时读 HEARTBEAT.md 并严格执行。无需关注则回复 HEARTBEAT_OK。有需关注的事项则回复具体内容，不含 HEARTBEAT_OK。",
    ),
    # Workspace Files header
    (
        "## Workspace Files (injected)\nThese user-editable files are loaded by OpenClaw and included below in Project Context.",
        "## 工作区文件（已注入）\n以下用户文件已加载到项目上下文中。",
    ),
    # Project Context header
    (
        "# Project Context\nThe following project context files have been loaded:\nIf SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it.",
        "# 项目上下文\n以下项目文件已加载。如有 SOUL.md 则体现其人格和语气。",
    ),
    # Reasoning footer (varies slightly but has fixed prefix)
    (
        "Reasoning: off (hidden unless on/stream). Toggle /reasoning; /status shows Reasoning when enabled.",
        "推理：关闭。切换 /reasoning；/status 查看。",
    ),
]


def _compress_openclaw_to_chinese(text: str) -> str:
    """Replace known fixed English sections with compact Chinese equivalents."""
    for eng, chn in _OPENCLAW_CHINESE_REPLACEMENTS:
        text = text.replace(eng, chn)
    return text


def _looks_like_openclaw_system_prompt(text: str) -> bool:
    return any(marker in text for marker in _OPENCLAW_SYSTEM_PROMPT_MARKERS)


def _extract_skill_field(skill_body: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", skill_body, re.DOTALL)
    if match is None:
        return None

    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return value or None


def _trim_skill_descriptions_linewise(block: str, max_chars: int) -> str:
    """Fallback linewise trimming inside an <available_skills> block.

    Used only when a skill entry cannot be parsed as a normal OpenClaw
    `<skill>...</skill>` block.
    """
    lines = block.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        description_match = _DESCRIPTION_LINE_RE.match(line)
        if description_match is not None:
            content = description_match.group("content")
            if len(content) > max_chars:
                leading_ws = description_match.group("leading_ws")
                closing_tag = description_match.group("closing_tag") or ""
                result.append(
                    f"{leading_ws}<description>{content[:max_chars]}…{closing_tag}"
                )
            else:
                result.append(line)
            continue
        # Keep: empty lines, XML tags, file paths, skill name lines (short or contain /)
        is_path = stripped.endswith(".md") or ("/" in stripped and not stripped.startswith("<"))
        if not stripped or stripped.startswith("<") or is_path:
            result.append(line)
        elif len(stripped) > max_chars:
            leading_ws = line[: len(line) - len(line.lstrip())]
            result.append(leading_ws + stripped[:max_chars] + "…")
        else:
            result.append(line)
    return "\n".join(result)


def _trim_skill_descriptions(block: str, max_chars: int) -> str:
    """Rewrite each <skill> entry into a compact, trimmed structure."""
    skills: list[str] = []
    last_end = 0

    for match in _SKILL_BLOCK_RE.finditer(block):
        prefix = block[last_end : match.start()]
        if prefix.strip() and skills:
            skills.append(_trim_skill_descriptions_linewise(prefix, max_chars))
        elif prefix.strip():
            skills.append(prefix.strip())

        skill_body = match.group("body")
        name = _extract_skill_field(skill_body, "name")
        description = _extract_skill_field(skill_body, "description")
        location = _extract_skill_field(skill_body, "location")

        if name is None or description is None or location is None:
            skills.append(
                _trim_skill_descriptions_linewise(match.group(0), max_chars).strip()
            )
        else:
            compact_description = (
                description[:max_chars] + "…"
                if len(description) > max_chars
                else description
            )
            skills.append(
                "\n".join(
                    [
                        "  <skill>",
                        f"    <name>{name}</name>",
                        f"    <description>{compact_description}</description>",
                        f"    <location>{location}</location>",
                        "  </skill>",
                    ]
                )
            )
        last_end = match.end()

    suffix = block[last_end:]
    if suffix.strip():
        skills.append(_trim_skill_descriptions_linewise(suffix, max_chars))

    return "\n".join(part for part in skills if part)


def trim_system_prompt(text: str) -> tuple[str, str]:
    """Trim system prompt and report which policy applied."""
    trim_policy = "explicit-env"
    skills_desc_max_chars = _SKILLS_DESC_MAX_CHARS
    max_system_prompt_chars = _MAX_SYSTEM_PROMPT_CHARS

    is_openclaw = _looks_like_openclaw_system_prompt(text)

    if (
        not _STRIP_SKILLS_BLOCK
        and skills_desc_max_chars is None
        and max_system_prompt_chars is None
        and is_openclaw
    ):
        trim_policy = "openclaw-default"
        skills_desc_max_chars = _OPENCLAW_SKILLS_DESC_MAX_CHARS
        max_system_prompt_chars = _OPENCLAW_MAX_SYSTEM_PROMPT_CHARS

    # Compress fixed English sections to Chinese (before skill/length trimming)
    if _COMPRESS_CHINESE and is_openclaw:
        text = _compress_openclaw_to_chinese(text)

    if _STRIP_SKILLS_BLOCK:
        text = _SKILLS_BLOCK_RE.sub("", text).strip()
    elif skills_desc_max_chars is not None:
        text = _SKILLS_BLOCK_RE.sub(
            lambda m: _trim_skill_descriptions(m.group(1), skills_desc_max_chars),
            text,
        )
    if max_system_prompt_chars is not None and len(text) > max_system_prompt_chars:
        text = text[:max_system_prompt_chars] + " [trimmed]"
    return text, trim_policy


async def chat_request_to_text_generation(
    request: ChatCompletionRequest,
) -> TextGenerationTaskParams:
    instructions: str | None = None
    input_messages: list[InputMessage] = []
    chat_template_messages: list[dict[str, Any]] = []
    images: list[str] = []

    for msg in request.messages:
        # Normalize content to string
        content: str
        has_images = False
        if msg.content is None:
            content = ""
        elif isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, ChatCompletionMessageText):
            content = msg.content.text
        elif isinstance(msg.content, ChatCompletionMessageImageUrl):
            url = msg.content.image_url.get("url", "")
            if url:
                if url.startswith(("http://", "https://")):
                    images.append(await fetch_image_url(url))
                else:
                    images.append(extract_base64_from_data_url(url))
                has_images = True
            content = ""
        else:
            text_parts: list[str] = []
            for part in msg.content:
                if isinstance(part, ChatCompletionMessageText):
                    text_parts.append(part.text)
                else:
                    url = part.image_url.get("url", "")
                    if url:
                        if url.startswith(("http://", "https://")):
                            images.append(await fetch_image_url(url))
                        else:
                            images.append(extract_base64_from_data_url(url))
                        has_images = True
            content = "\n".join(text_parts)

        # Extract system / developer message as instructions
        # OpenAI uses "system"; newer convention (and OpenClaw) uses "developer"
        if msg.role in ("system", "developer"):
            original_len = len(content)
            content, trim_policy = trim_system_prompt(content)
            trimmed_len = len(content)
            if original_len != trimmed_len:
                logger.info(
                    f"[System prompt trim:{trim_policy}] {original_len} -> "
                    f"{trimmed_len} chars (stripped {original_len - trimmed_len} chars)"
                )
            else:
                logger.info(
                    f"[System prompt:{trim_policy}] {original_len} chars "
                    "(no trimming applied)"
                )
            if instructions is None:
                instructions = content
            else:
                # Append additional system messages
                instructions = f"{instructions}\n{content}"
            chat_template_messages.append({"role": msg.role, "content": content})
        else:
            # Skip messages with no meaningful content
            if (
                msg.content is None
                and msg.reasoning_content is None
                and msg.tool_calls is None
            ):
                continue

            if msg.role in ("user", "assistant"):
                input_messages.append(InputMessage(role=msg.role, content=content))

            # Build full message dict for chat template (preserves tool_calls etc.)
            # Normalize content for model_dump
            if has_images:
                multimodal_content: list[dict[str, Any]] = []
                assert isinstance(msg.content, list)
                for part in msg.content:
                    if isinstance(part, ChatCompletionMessageText):
                        multimodal_content.append({"type": "text", "text": part.text})
                    else:
                        multimodal_content.append({"type": "image"})
                chat_template_messages.append(
                    {"role": msg.role, "content": multimodal_content}
                )
                continue
            msg_copy = msg.model_copy(update={"content": content})

            dumped: dict[str, Any] = msg_copy.model_dump(exclude_none=True)
            chat_template_messages.append(dumped)

    resolved_effort, resolved_thinking = resolve_reasoning_params(
        request.reasoning_effort, request.enable_thinking
    )

    return TextGenerationTaskParams(
        model=request.model,
        input=input_messages
        if input_messages
        else [InputMessage(role="user", content="")],
        instructions=instructions,
        max_output_tokens=request.max_completion_tokens or request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        seed=request.seed,
        stream=request.stream,
        tools=request.tools,
        reasoning_effort=resolved_effort,
        enable_thinking=resolved_thinking,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
        logprobs=request.logprobs or False,
        top_logprobs=request.top_logprobs,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        repetition_context_size=request.repetition_context_size,
        images=images,
    )


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    """Convert a TokenChunk to a streaming ChatCompletionResponse."""
    # Build logprobs if available
    logprobs: Logprobs | None = None
    if chunk.logprob is not None:
        logprobs = Logprobs(
            content=[
                LogprobsContentItem(
                    token=chunk.text,
                    logprob=chunk.logprob,
                    top_logprobs=chunk.top_logprobs or [],
                )
            ]
        )

    if chunk.is_thinking:
        delta = ChatCompletionMessage(role="assistant", reasoning_content=chunk.text)
    else:
        delta = ChatCompletionMessage(role="assistant", content=chunk.text)

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=delta,
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_chat_stream(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """Generate Chat Completions API streaming events from chunks."""
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                # Use SSE comment so third-party clients ignore it
                yield f": prefill_progress {chunk.model_dump_json()}\n\n"

            case ErrorChunk():
                error_response = ErrorResponse(
                    error=ErrorInfo(
                        message=chunk.error_message or "Internal server error",
                        type="InternalServerError",
                        code=500,
                    )
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case ToolCallChunk():
                last_usage = chunk.usage or last_usage

                tool_call_deltas = [
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                ]
                tool_response = ChatCompletionResponse(
                    id=command_id,
                    created=int(time.time()),
                    model=chunk.model,
                    choices=[
                        StreamingChoiceResponse(
                            index=0,
                            delta=ChatCompletionMessage(
                                role="assistant",
                                tool_calls=tool_call_deltas,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=last_usage,
                )
                yield f"data: {tool_response.model_dump_json()}\n\n"
                if chunk.stats is not None:
                    yield f": generation_stats {chunk.stats.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case TokenChunk():
                last_usage = chunk.usage or last_usage

                chunk_response = chunk_to_response(chunk, command_id)
                if chunk.finish_reason is not None:
                    chunk_response = chunk_response.model_copy(
                        update={"usage": last_usage}
                    )
                yield f"data: {chunk_response.model_dump_json()}\n\n"

                if chunk.finish_reason is not None:
                    if chunk.stats is not None:
                        yield f": generation_stats {chunk.stats.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                    return


async def collect_chat_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> AsyncGenerator[str]:
    # This is an AsyncGenerator[str] rather than returning a ChatCompletionReponse because
    # FastAPI handles the cancellation better but wouldn't auto-serialize for some reason
    """Collect all token chunks and return a single ChatCompletionResponse."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    logprobs_content: list[LogprobsContentItem] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue

            case ErrorChunk():
                error_message = chunk.error_message or "Internal server error"
                break

            case TokenChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                else:
                    text_parts.append(chunk.text)
                if chunk.logprob is not None:
                    logprobs_content.append(
                        LogprobsContentItem(
                            token=chunk.text,
                            logprob=chunk.logprob,
                            top_logprobs=chunk.top_logprobs or [],
                        )
                    )
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            case ToolCallChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                tool_calls.extend(
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                )
                finish_reason = chunk.finish_reason

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    combined_thinking = "".join(thinking_parts) if thinking_parts else None
    compatible_content = openai_compatible_content(combined_text, combined_thinking)
    assert model is not None

    yield ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=compatible_content,
                    reasoning_content=combined_thinking,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                logprobs=Logprobs(content=logprobs_content)
                if logprobs_content
                else None,
                finish_reason=finish_reason,
            )
        ],
        usage=last_usage,
    ).model_dump_json()
    return
