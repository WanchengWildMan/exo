# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

exo is a distributed AI inference system that connects multiple devices into a cluster. It enables running large language models across multiple machines using MLX as the inference backend and libp2p for peer-to-peer networking.

## Build & Run Commands

```bash
# Build the dashboard (required before running exo)
cd dashboard && npm install && npm run build && cd ..

# Run exo (starts both master and worker with API at http://localhost:52415)
uv run exo

# Run with verbose logging
uv run exo -v   # or -vv for more verbose

# Run tests (excludes slow tests by default)
uv run pytest

# Run all tests including slow tests
uv run pytest -m ""

# Run a specific test file
uv run pytest src/exo/shared/tests/test_election.py

# Run a specific test function
uv run pytest src/exo/shared/tests/test_election.py::test_function_name

# Type checking (strict mode)
uv run basedpyright

# Linting
uv run ruff check

# Format code (using nix)
nix fmt
```

## Pre-Commit Checks (REQUIRED)

**IMPORTANT: Always run these checks before committing code. CI will fail if these don't pass.**

```bash
# 1. Type checking - MUST pass with 0 errors
uv run basedpyright

# 2. Linting - MUST pass
uv run ruff check

# 3. Formatting - MUST be applied
nix fmt

# 4. Tests - MUST pass
uv run pytest
```

Run all checks in sequence:
```bash
uv run basedpyright && uv run ruff check && nix fmt && uv run pytest
```

If `nix fmt` changes any files, stage them before committing. The CI runs `nix flake check` which verifies formatting, linting, and runs Rust tests.

## Architecture

### Node Composition
A single exo `Node` (src/exo/main.py) runs multiple components:
- **Router**: libp2p-based pub/sub messaging via Rust bindings (exo_pyo3_bindings)
- **Worker**: Handles inference tasks, downloads models, manages runner processes
- **Master**: Coordinates cluster state, places model instances across nodes
- **Election**: Bully algorithm for master election
- **API**: FastAPI server for OpenAI-compatible chat completions

### Message Flow
Components communicate via typed pub/sub topics (src/exo/routing/topics.py):
- `GLOBAL_EVENTS`: Master broadcasts indexed events to all workers
- `LOCAL_EVENTS`: Workers send events to master for indexing
- `COMMANDS`: Workers/API send commands to master
- `ELECTION_MESSAGES`: Election protocol messages
- `CONNECTION_MESSAGES`: libp2p connection updates

### Event Sourcing
The system uses event sourcing for state management:
- `State` (src/exo/shared/types/state.py): Immutable state object
- `apply()` (src/exo/shared/apply.py): Pure function that applies events to state
- Master indexes events and broadcasts; workers apply indexed events

### Key Type Hierarchy
- `src/exo/shared/types/`: Pydantic models for all shared types
  - `events.py`: Event types (discriminated union)
  - `commands.py`: Command types
  - `tasks.py`: Task types for worker execution
  - `state.py`: Cluster state model

### Rust Components
Rust code in `rust/` provides:
- `networking`: libp2p networking (gossipsub, peer discovery)
- `exo_pyo3_bindings`: PyO3 bindings exposing Rust to Python
- `system_custodian`: System-level operations

### Dashboard
Svelte 5 + TypeScript frontend in `dashboard/`. Build output goes to `dashboard/build/` and is served by the API.

## Code Style Requirements

From .cursorrules:
- Strict, exhaustive typing - never bypass the type-checker
- Use `Literal[...]` for enum-like sets, `typing.NewType` for primitives
- Pydantic models with `frozen=True` and `strict=True`
- Pure functions with injectable effect handlers for side-effects
- Descriptive names - no abbreviations or 3-letter acronyms
- Catch exceptions only where you can handle them meaningfully
- Use `@final` and immutability wherever applicable

## Testing

Tests use pytest-asyncio with `asyncio_mode = "auto"`. Tests are in `tests/` subdirectories alongside the code they test. The `EXO_TESTS=1` env var is set during tests.

## Dashboard UI Testing & Screenshots

### Building and Running the Dashboard
```bash
# Build the dashboard (must be done before running exo)
cd dashboard && npm install && npm run build && cd ..

# Start exo (serves the dashboard at http://localhost:52415)
uv run exo &
sleep 8  # Wait for server to start
```

### Taking Headless Screenshots with Playwright
Use Playwright with headless Chromium for programmatic screenshots — no manual browser interaction needed.

**Setup (one-time):**
```bash
npx --yes playwright install chromium
cd /tmp && npm init -y && npm install playwright
```

**Taking screenshots:**
```javascript
// Run from /tmp where playwright is installed: cd /tmp && node -e "..."
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto('http://localhost:52415', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // Inject test data into localStorage if needed (e.g., recent models)
  await page.evaluate(() => {
    localStorage.setItem('exo-recent-models', JSON.stringify([
      { modelId: 'mlx-community/Qwen3-30B-A3B-4bit', launchedAt: Date.now() },
    ]));
  });
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // Interact with UI elements
  await page.locator('text=SELECT MODEL').click();
  await page.waitForTimeout(1000);

  // Take screenshot
  await page.screenshot({ path: '/tmp/screenshot.png', fullPage: false });
  await browser.close();
})();
```

### Uploading Images to GitHub PRs
GitHub's API doesn't support direct image upload for PR comments. Workaround:

1. **Commit images to the branch** (temporarily):
   ```bash
   cp /tmp/screenshot.png .
   git add screenshot.png
   git commit -m "temp: add screenshots for PR"
   git push origin <branch>
   COMMIT_SHA=$(git rev-parse HEAD)
   ```

2. **Post PR comment** referencing the raw image URL (uses permanent commit SHA so images survive deletion):
   ```bash
   gh pr comment <PR_NUMBER> --body "![Screenshot](https://raw.githubusercontent.com/exo-explore/exo/${COMMIT_SHA}/screenshot.png)"
   ```

3. **Remove the images** from the branch:
   ```bash
   git rm screenshot.png
   git commit -m "chore: remove temporary screenshot files"
   git push origin <branch>
   ```
   The images still render in the PR comment because they reference the permanent commit SHA.


日志在logs下面


必须遵守：务必充分了解现状，即已经做了的事项，严格避免已经实现或描述了的功能，以为没做而重复描述、重复实现。
必须遵守：长文件分段分次写入，每次写入120行左右，以避免返回时间过长导致网络超时，进而出现连接关闭的问题
必须遵守：有新需求或新修复时，在docs下相应文件夹记录文档，以日期-中文为文件名
必须遵守工作记忆：当我打断当前工作后，若之前还有未完成的事项，务必询问是否继续进行，若这个事项是必选的P0，你可以告知后直接接续进行。
必须遵守口径对齐：任何时候修改后，务必警觉并对齐前后文、或不同文件同一个信息的口径，例如所有和当前修改相关的文档与代码的描述口径，包括但不限于配置、公式、符号、描述、注释、数值
必须遵守：你是作为具有批判性和独立思考能力的讨论者而非我的服从者，你觉得不合理的务必要尽量反驳我的建议。
必须遵守：完成后调用提问工具！！！！：当你完成工作或有需要澄清的问题时，必须必须必须必须调用【提问工具】以获取进一步指示，严禁输出完之后直接结束对话。但不要让subagent获得此项。

openclaw 配置在~/.openclaw/openclaw.json