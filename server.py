"""OpenReward environment for SWE-rebench-V2."""
import base64
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import Environment, Server, tool
from openreward.environments.types import Blocks, JSONObject, TextBlock, ToolOutput
from pydantic import BaseModel, Field

from log_parsers import TestStatus

# ---------------------------------------------------------------------------
# Dataset loading from parquet
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "/orwd_data"))
_table = None  # pyarrow.Table, lazily loaded


def _load_table():
    """Load parquet files from DATA_DIR into a global Arrow table (lazy, cached)."""
    global _table
    if _table is None:
        parquet_files = sorted(DATA_DIR.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {DATA_DIR}")
        _table = pq.read_table(DATA_DIR)
    return _table


def _row_to_dict(table, index: int) -> dict:
    """Extract a single row from the Arrow table as a Python dict."""
    row = table.slice(index, 1).to_pydict()
    return {k: v[0] for k, v in row.items()}


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------

class InstallConfig(BaseModel):
    test_cmd: str
    log_parser: str
    install: str = ""
    base_image_name: str = ""
    docker_specs: dict = {}


class TaskSpec(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    test_patch: str
    problem_statement: str
    image_name: str
    language: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    install_config: InstallConfig


# ---------------------------------------------------------------------------
# Tool input models
# ---------------------------------------------------------------------------

ENVIRONMENT_NAME = "GeneralReasoning/swe-rebench-v2"


class BashInput(BaseModel):
    """Input for bash command execution."""
    command: str = Field(..., description="Bash command to run in container")
    description: str = Field(..., description="Why I'm running this command")


class StrReplaceInput(BaseModel):
    """Input for string replacement in files."""
    path: str = Field(..., description="Path to the file to edit")
    old_str: str = Field(..., description="String to replace (must be unique in file)")
    new_str: str = Field(default="", description="String to replace with (empty to delete)")
    description: str = Field(..., description="Why I'm making this edit")


class ViewInput(BaseModel):
    """Input for viewing files and directories."""
    path: str = Field(..., description="Absolute path to file or directory")
    view_range: tuple[int, int] | None = Field(
        default=None,
        description="Optional line range for text files. Format: [start_line, end_line] where lines are indexed starting at 1. Use [start_line, -1] to view from start_line to end."
    )
    description: str = Field(..., description="Why I need to view this")


class CreateFileInput(BaseModel):
    """Input for creating new files."""
    description: str = Field(..., description="Why I'm creating this file")
    path: str = Field(..., description="Path to the file to create")
    file_text: str = Field(..., description="Content to write to the file")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_output(text: str, finished: bool = False) -> ToolOutput:
    return ToolOutput(blocks=[TextBlock(text=text)], finished=finished)


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _get_workdir(repo: str) -> str:
    """Derive the working directory inside the sandbox from the repo name."""
    # Repos in the dataset are like "owner/name"; the sandbox image typically
    # clones into /testbed or /<repo_name>.  SWE-bench convention is /testbed.
    return "/testbed"


def _get_log_parser(parser_name: str):
    """Import and return the log parser function by name."""
    import log_parsers
    fn = getattr(log_parsers, parser_name, None)
    if fn is None:
        raise ValueError(f"Unknown log parser: {parser_name}")
    return fn


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SWERebenchV2(Environment):
    """OpenReward environment for SWE-rebench-V2 tasks."""

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.parsed = TaskSpec.model_validate(task_spec)

        self.or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        self.workdir = _get_workdir(self.parsed.repo)
        self.sandbox_settings = SandboxSettings(
            environment=ENVIRONMENT_NAME,
            image=self.parsed.image_name,
            machine_size="1:2"
        )
        self.sandbox = self.or_client.sandbox(self.sandbox_settings)

    # ----- splits / tasks (class methods) -----

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        raise NotImplementedError(
            "Dataset has 32K+ tasks — use num_tasks/get_task instead"
        )

    @classmethod
    async def num_tasks(cls, split: str) -> int:
        if split != "train":
            raise ValueError(f"Unknown split: {split!r}")
        table = _load_table()
        return len(table)

    @classmethod
    async def get_task(cls, split: str, index: int) -> JSONObject:
        if split != "train":
            raise ValueError(f"Unknown split: {split!r}")
        table = _load_table()
        if index < 0 or index >= len(table):
            raise IndexError(f"index {index} out of range (0..{len(table) - 1})")
        row = _row_to_dict(table, index)
        # Build a TaskSpec-shaped dict, excluding the gold patch
        install_config = row.get("install_config", {})
        if isinstance(install_config, str):
            install_config = json.loads(install_config)
        fail_to_pass = row.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass)
        pass_to_pass = row.get("PASS_TO_PASS", [])
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass)
        return {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "test_patch": row["test_patch"],
            "problem_statement": row["problem_statement"],
            "image_name": row["image_name"],
            "language": row["language"],
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
            "install_config": install_config,
        }

    # ----- lifecycle -----

    async def setup(self):
        await self.sandbox.start()
        # Configure git
        await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && "
            "git config --global --add safe.directory '*' && "
            "git config user.email 'agent@openreward.dev' && "
            "git config user.name 'Agent'"
        )
        # Checkout the base commit
        await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && "
            f"git checkout {_shell_quote(self.parsed.base_commit)}"
        )
        # Remove git history beyond base_commit so the agent can't peek at the fix
        await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && "
            "git reflog expire --expire=now --all && "
            "git gc --prune=now --quiet"
        )

    async def teardown(self):
        await self.sandbox.stop()

    def get_prompt(self) -> Blocks:
        text = (
            f"You are a software engineer working on the repository **{self.parsed.repo}** "
            f"(language: {self.parsed.language}).\n\n"
            f"## Problem Statement\n\n{self.parsed.problem_statement}\n\n"
            f"## Instructions\n\n"
            f"The repository is cloned at `{self.workdir}` and checked out to the commit "
            f"before the fix. Your task is to modify the code so that the failing tests pass.\n\n"
            f"Use the available tools to explore the codebase, understand the problem, "
            f"make edits, and then call `submit_answer` when you are done.\n\n"
            f"Do NOT modify or create tests — only fix the source code."
        )
        return [TextBlock(text=text)]

    # ----- tools -----

    @tool
    async def bash(self, input: BashInput) -> ToolOutput:
        """Run a bash command in the container."""
        cmd = f"cd {_shell_quote(self.workdir)} && {input.command}"
        output, exit_code = await self.sandbox.run(cmd)
        s = output if output else "(no output)"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def str_replace(self, input: StrReplaceInput) -> ToolOutput:
        """Replace a unique string in a file with another string."""
        res = await self.sandbox.run(f"cat -- {_shell_quote(input.path)}")
        content = res.output
        exit_code = res.return_code
        if exit_code != 0:
            s = content if content else "(no output)"
            return _text_output(f"{s}\nExit code: {exit_code}")

        count = content.count(input.old_str)
        if count == 0:
            return _text_output(f"Error: The string to replace was not found in {input.path}\nExit code: 1")
        if count > 1:
            return _text_output(f"Error: The string to replace appears {count} times in {input.path}. It must be unique.\nExit code: 1")

        new_content = content.replace(input.old_str, input.new_str, 1)
        encoded = base64.b64encode(new_content.encode('utf-8')).decode('ascii')
        write_cmd = f"echo '{encoded}' | base64 -d > {_shell_quote(input.path)}"
        output, exit_code = await self.sandbox.run(write_cmd)

        s = output if output else f"Successfully replaced string in {input.path}"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def view(self, input: ViewInput) -> ToolOutput:
        """View file contents or directory listings."""
        res = await self.sandbox.run(f"test -d {_shell_quote(input.path)} && echo 'dir' || echo 'file'")
        output = res.output
        is_dir = output.strip() == "dir"

        if is_dir:
            cmd = f"find {_shell_quote(input.path)} -maxdepth 2 -not -path '*/\\.*' -not -path '*/node_modules/*' | head -100"
        else:
            if input.view_range:
                start, end = input.view_range
                if end == -1:
                    cmd = f"cat -n {_shell_quote(input.path)} | tail -n +{start}"
                else:
                    cmd = f"cat -n {_shell_quote(input.path)} | sed -n '{start},{end}p'"
            else:
                cmd = f"cat -n {_shell_quote(input.path)}"

        res = await self.sandbox.run(cmd)
        output = res.output
        exit_code = res.return_code

        if len(output) > 16000:
            lines = output.split('\n')
            mid = len(lines) // 2
            keep_start = mid // 2
            keep_end = mid // 2
            output = '\n'.join(lines[:keep_start]) + \
                    f"\n\n... [truncated {len(lines) - keep_start - keep_end} lines] ...\n\n" + \
                    '\n'.join(lines[-keep_end:])

        s = output if output else "(no output)"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def create_file(self, input: CreateFileInput) -> ToolOutput:
        """Create a new file with the specified content."""
        parent_dir = "/".join(input.path.rsplit("/", 1)[:-1])
        if parent_dir:
            await self.sandbox.run(f"mkdir -p {_shell_quote(parent_dir)}")

        encoded = base64.b64encode(input.file_text.encode('utf-8')).decode('ascii')
        write_cmd = f"echo '{encoded}' | base64 -d > {_shell_quote(input.path)}"
        output, exit_code = await self.sandbox.run(write_cmd)

        s = output if output else f"Successfully created {input.path}"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def submit_answer(self) -> ToolOutput:
        """Submit your solution. Applies the test patch, runs the test suite, and scores."""
        # 1. Write test_patch to a file and apply it
        test_patch_encoded = base64.b64encode(
            self.parsed.test_patch.encode('utf-8')
        ).decode('ascii')
        await self.sandbox.run(
            f"echo '{test_patch_encoded}' | base64 -d > /tmp/test_patch.diff"
        )
        apply_output, apply_code = await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && git apply /tmp/test_patch.diff"
        )
        if apply_code != 0:
            # Try with --3way as fallback
            apply_output, apply_code = await self.sandbox.run(
                f"cd {_shell_quote(self.workdir)} && git apply --3way /tmp/test_patch.diff"
            )
            if apply_code != 0:
                return ToolOutput(
                    blocks=[TextBlock(text=f"Failed to apply test patch:\n{apply_output}")],
                    reward=0.0,
                    finished=True,
                )

        # 2. Run test command
        test_cmd = self.parsed.install_config.test_cmd
        res = await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && {test_cmd}",
            timeout=600,
        )
        test_output, test_code = res.output, res.return_code

        # 3. Parse test output
        parser_name = self.parsed.install_config.log_parser
        try:
            parser_fn = _get_log_parser(parser_name)
            test_results = parser_fn(test_output)
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Log parser error ({parser_name}): {e}\n\nRaw output:\n{test_output[:4000]}")],
                reward=0.0,
                finished=True,
            )

        # 4. Check FAIL_TO_PASS and PASS_TO_PASS
        fail_to_pass_ok = all(
            test_results.get(t) == TestStatus.PASSED.value
            for t in self.parsed.FAIL_TO_PASS
        )
        pass_to_pass_ok = all(
            test_results.get(t) == TestStatus.PASSED.value
            for t in self.parsed.PASS_TO_PASS
        )

        reward = 1.0 if (fail_to_pass_ok and pass_to_pass_ok) else 0.0

        # Build summary
        f2p_detail = []
        for t in self.parsed.FAIL_TO_PASS:
            status = test_results.get(t, "NOT_FOUND")
            f2p_detail.append(f"  {t}: {status}")
        p2p_total = len(self.parsed.PASS_TO_PASS)
        p2p_passed = sum(
            1 for t in self.parsed.PASS_TO_PASS
            if test_results.get(t) == TestStatus.PASSED.value
        )

        summary = (
            f"Test command exit code: {test_code}\n"
            f"FAIL_TO_PASS ({len(self.parsed.FAIL_TO_PASS)}):\n" +
            "\n".join(f2p_detail) + "\n"
            f"PASS_TO_PASS: {p2p_passed}/{p2p_total} passed\n"
            f"Reward: {reward}"
        )

        return ToolOutput(
            blocks=[TextBlock(text=summary)],
            reward=reward,
            finished=True,
        )


if __name__ == "__main__":
    Server(environments=[SWERebenchV2]).run()
