# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adaptive prompt refinement for code-evolution agents.

Two-phase feedback loop per generation:

1. **Analyze** — After each agent finishes, read its ``journal.log`` and
   produce a behavioural critique (iteration efficiency, time allocation,
   metric progression).
2. **Refine** — After all agents in a generation finish, feed every
   critique plus the current prompt template to the LLM and produce a
   refined template for the next generation.
"""

import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI

import printer
from config import LLMConfig
from utils import invoke_llm_with_retry, retry_on_parse_failure
from workspace_fs import file_exists_under_root, read_text_under_root

from .json_utils import strip_code_fences, extract_json_fragment
from .models import JournalAnalysis

REQUIRED_PLACEHOLDERS = [
    "{original_query}",
    "{assigned_approach_section}",
    "{metric_direction}",
]


class PromptEngineer:
    """Analyzes code-evolution agent journals and refines the agent prompt template."""

    JOURNAL_ANALYSIS_PROMPT = """You are reviewing the execution log of an autonomous code-evolution agent. Identify **behavioural patterns only** — how the agent used its tools and time budget. Do **not** evaluate the algorithm, the code, or the assigned approach: that is handled by a different component.

## Context
- The agent was given a task and an assigned approach. Its goal is to iteratively write, evaluate, and refine code to optimise a metric.
- It has tools: file read/write/edit/copy, code execution, and an evaluation tool that scores the solution and returns a metric.
- The agent's workflow is: implement → evaluate → analyse result → refine code → re-evaluate. More iterations generally yield better metrics.

## Scope of this analysis (strict)
You MUST restrict your observations to **process** signals — tool-call counts, sequencing, timing, repetition, error-handling. You MUST NOT comment on:
- Whether the algorithm is good, simple, complex, fast, slow, novel, or appropriate.
- Whether the agent should have chosen a different algorithm, family, hyperparameters, or complexity tier.
- Whether the assigned approach was a good idea, or whether the agent's deviation from it was justified.
- Anything that requires reading the *content* of the code being written.

The fact of a deviation from the assigned approach is captured elsewhere; do not re-state or judge it here.

## Structured statistics from this run
{stats_text}

## Condensed execution log (may be truncated)
{condensed_journal}

## Your task
In 3–5 bullet points, describe purely behavioural observations such as:
1. Iteration discipline — number of evaluate→refine cycles, gaps between evaluations, whether evaluations were preceded by a code change.
2. Time / tool-call allocation — long stretches without tool calls, excessive file reads relative to writes, excessive hyperparameter tuning without intervening code changes, premature exit while time remained.
3. Process anti-patterns — re-evaluations without intervening code change, repeated identical file writes, persistence on a failing tool call without a different fix, debugging the same error for many turns without trying a different angle, oscillating between two saved versions.

Reply with **only** a JSON object with one key:
{{"behavioral_summary": "- bullet 1\\n- bullet 2\\n..."}}"""

    REFINE_PROMPT_PROMPT = """You are a prompt engineer optimizing the instruction prompt for autonomous code-evolution agents. These agents iteratively write, evaluate, and refine code to optimize a metric. The agent prompt is **task-agnostic with respect to algorithmic content**: the algorithmic approach for any given run is delivered separately, via the {assigned_approach_section} placeholder, by a different component (the orchestrator). Your role is to improve **how the agent works**, not **what it works on**.

## Current prompt template
The template below is what each agent receives. It has three placeholders that MUST be preserved exactly as shown (including the curly braces): {original_query}, {assigned_approach_section}, {metric_direction}.

```
{current_template}
```

## Behavioral analyses from generation {generation}
Each entry summarizes one agent's execution — purely behavioural observations on tool usage and time allocation. Focus on **process** patterns that appear across multiple agents (iteration discipline, tool-call mix, time spent stuck, etc.).

{analyses_text}

## Your task
Produce a refined version of the prompt template that:
1. Keeps the same overall structure (Context, Task, Assigned approach, Workflow steps, Rules).
2. Preserves ALL three placeholders exactly as they appear: {original_query}, {assigned_approach_section}, {metric_direction}.
3. Addresses common **process** failure modes observed — add targeted tips, warnings, or modified instructions to steer agents toward more productive iteration patterns (e.g., more evaluate-refine cycles, better time allocation, fewer redundant file reads, recovering from a failing tool call by changing approach instead of repeating it).
4. **Strict deny-list — must not appear in the refined template under any form, including reworded or implicit:**
   - No guidance favouring or disfavouring any algorithmic family, technique, or library.
   - No complexity targets, hyperparameter values, kernel sizes, iteration counts, learning rates, thresholds, or other task-specific numerics.
   - No guidance about whether to follow, deviate from, or "stay the course" on the assigned approach. Whether the agent should follow the assigned approach is a decision owned by the orchestrator's idea-generation prompt, not by this template.
   - No task-specific calibration ("metric values around X mean Y", "scores below N are usually due to Z"). The template is reused across tasks; numbers from one task must not leak into it.
   - No anti-oscillation or anti-rewrite rules that effectively forbid the agent from switching strategy mid-run; those interact badly with assignments that genuinely require a fresh approach.
5. Does NOT remove any existing rules that agents were already following well — *unless* those rules violate item 4, in which case remove them.
6. Stays concise — do not bloat the prompt beyond ~50% of the current length.

If the current template already contains content that violates item 4 (e.g. accumulated algorithmic guidance from previous refinement rounds), strip it out as part of this revision; do not preserve it just because it was already there.

Reply with **only** the refined template wrapped in a fenced block. Only the last fenced block is read. If the template itself contains ``` blocks, use a longer outer fence (e.g. ````)."""

    def __init__(
        self,
        llm_config: LLMConfig,
        log_path: Path,
        higher_is_better: bool,
        log_lock: Optional[threading.Lock] = None,
    ):
        self._llm_config = llm_config
        self._llm: Optional[ChatOpenAI] = None
        self._llm_init_lock = threading.Lock()
        self._log_path = log_path
        self._higher_is_better = higher_is_better
        self._log_lock = log_lock or threading.Lock()

    def _get_llm(self) -> ChatOpenAI:
        """Lazily create the shared LLM instance (thread-safe)."""
        if self._llm is None:
            with self._llm_init_lock:
                if self._llm is None:
                    self._llm = ChatOpenAI(**asdict(self._llm_config))
        return self._llm

    def _log(self, role: str, content: str) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n[{role}]\n{content}\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_journal(
        self, workspace_root: Path, workspace_id: str
    ) -> JournalAnalysis:
        """Read and analyze a single agent's ``journal.log``.

        Extracts structured statistics programmatically, then asks the
        LLM for a behavioural critique.
        """
        ws_path = workspace_root / workspace_id
        if not file_exists_under_root(ws_path, "journal.log"):
            printer.log(f"No journal found for {workspace_id}, skipping analysis.")
            return JournalAnalysis(
                workspace_id=workspace_id,
                behavioral_summary="No journal found.",
            )
        journal_text = read_text_under_root(
            ws_path, "journal.log", errors="replace",
        )

        printer.log(f"Analyzing journal for {workspace_id}...")
        stats = _extract_journal_stats(journal_text)
        condensed = _condense_journal(journal_text)
        behavioral_summary = self._llm_analyze(stats, condensed)

        printer.log(
            f"Journal analysis for {workspace_id}: "
            f"{stats['num_eval_attempts']} evals, "
            f"{stats['num_eval_successes']} successes."
        )
        return JournalAnalysis(
            workspace_id=workspace_id,
            behavioral_summary=behavioral_summary,
            num_tool_calls=stats["num_tool_calls"],
            num_eval_attempts=stats["num_eval_attempts"],
            num_eval_successes=stats["num_eval_successes"],
            metric_trajectory=stats["metric_trajectory"],
            timed_out=stats["timed_out"],
        )

    def refine_prompt(
        self,
        current_template: str,
        analyses: list[JournalAnalysis],
        generation: int,
    ) -> str:
        """Produce a refined prompt template from behavioural analyses.

        Returns the current template unchanged if the LLM output fails
        validation (missing placeholders).
        """
        printer.log(
            f"Refining prompt template using {len(analyses)} analyses "
            f"from generation {generation}..."
        )
        analyses_lines: list[str] = []
        for a in analyses:
            analyses_lines.append(f"### {a.workspace_id}")
            analyses_lines.append(
                f"  Evals: {a.num_eval_attempts} attempts, "
                f"{a.num_eval_successes} successes | "
                f"Tool calls: {a.num_tool_calls} | "
                f"Timed out: {a.timed_out}"
            )
            # Note: raw metric_trajectory deliberately omitted from the
            # refiner's input — task-specific numeric scores would invite
            # task-specific calibration in the refined template, which the
            # deny-list in REFINE_PROMPT_PROMPT explicitly forbids.
            analyses_lines.append(f"  {a.behavioral_summary}")
            analyses_lines.append("")

        # The prompt template uses single-brace placeholders like {original_query}.
        # We need to escape them for .format() by doubling the braces in the
        # template we embed, but the REFINE_PROMPT_PROMPT itself also uses
        # .format(). To avoid collisions we do manual replacement.
        prompt = self.REFINE_PROMPT_PROMPT
        prompt = prompt.replace("{n}", str(len(analyses)))
        prompt = prompt.replace("{generation}", str(generation))
        prompt = prompt.replace("{current_template}", current_template)
        prompt = prompt.replace("{analyses_text}", "\n".join(analyses_lines))

        self._log("PROMPT (refine prompt)", prompt)
        llm = self._get_llm()

        def _attempt() -> str:
            response = invoke_llm_with_retry(llm, prompt, context="refine prompt")
            refined = response.content.strip()
            self._log("RESPONSE (refine prompt)", refined)

            cleaned = strip_code_fences(refined)

            if not _validate_template(cleaned):
                self._log(
                    "VALIDATION FAILED",
                    f"Missing placeholders in:\n{cleaned[:500]}...",
                )
                raise ValueError(
                    "Refined prompt template is missing required placeholders."
                )
            return cleaned

        try:
            refined = retry_on_parse_failure(_attempt, context="refine prompt")
        except ValueError:
            printer.log(
                "WARNING: Refined prompt template failed validation after "
                "retries — keeping previous template."
            )
            return current_template

        printer.log("Prompt template refined successfully.")
        return refined

    # ------------------------------------------------------------------
    # LLM journal analysis
    # ------------------------------------------------------------------

    def _llm_analyze(self, stats: dict, condensed_journal: str) -> str:
        """Ask the LLM for a behavioural critique of a single journal."""
        metric_direction = (
            "Higher metric values are better."
            if self._higher_is_better
            else "Lower metric values are better."
        )
        stats_lines = [
            f"- Total tool calls: {stats['num_tool_calls']}",
            f"- Evaluation attempts: {stats['num_eval_attempts']}",
            f"- Evaluation successes: {stats['num_eval_successes']}",
            f"- Evaluation failures: {stats['num_eval_failures']}",
            f"- File writes/edits: {stats['num_file_writes']}",
            f"- File reads: {stats['num_file_reads']}",
            f"- Timed out: {stats['timed_out']}",
        ]
        if stats["metric_trajectory"]:
            traj = " → ".join(f"{m:.4f}" for m in stats["metric_trajectory"])
            stats_lines.append(f"- Metric trajectory: {traj}")

        prompt = self.JOURNAL_ANALYSIS_PROMPT.format(
            metric_direction=metric_direction,
            stats_text="\n".join(stats_lines),
            condensed_journal=condensed_journal,
        )
        self._log("PROMPT (journal analysis)", prompt)
        llm = self._get_llm()

        def _attempt() -> str:
            response = invoke_llm_with_retry(
                llm, prompt, context="journal analysis"
            )
            content = response.content.strip()
            self._log("RESPONSE (journal analysis)", content)
            parsed = self._parse_behavioral_summary(content)
            if parsed is None:
                raise ValueError(
                    f"Failed to parse journal analysis from LLM response:\n{content}"
                )
            return parsed

        try:
            return retry_on_parse_failure(_attempt, context="journal analysis")
        except ValueError:
            return "Behavioral analysis unavailable (parse failure)."

    @staticmethod
    def _parse_behavioral_summary(response_text: str) -> Optional[str]:
        text = strip_code_fences(response_text)
        fragment = extract_json_fragment(text, "{")
        if fragment is None:
            return None
        try:
            import json
            data = json.loads(fragment)
            if isinstance(data, dict):
                s = data.get("behavioral_summary")
                if s is not None:
                    return str(s).strip()
        except (ValueError, TypeError):
            pass
        return None


# ------------------------------------------------------------------
# Journal processing helpers (stateless)
# ------------------------------------------------------------------

_EVAL_CALL_RE = re.compile(r"\[TOOL_CALL\].*_evaluate", re.IGNORECASE)
_EVAL_RESP_RE = re.compile(r"\[TOOL_RESPONSE\].*_evaluate", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"^SUCCESS\s*,\s*([+-]?[\d.]+)", re.IGNORECASE)
_FAILURE_RE = re.compile(r"^FAILURE\s*,", re.IGNORECASE)
_WRITE_EDIT_RE = re.compile(
    r"\[TOOL_CALL\].*(_write_file|_edit_file)", re.IGNORECASE
)
_READ_RE = re.compile(r"\[TOOL_CALL\].*_read_file", re.IGNORECASE)
def _extract_journal_stats(journal_text: str) -> dict:
    """Extract key signals from a journal without an LLM."""
    lines = journal_text.split("\n")
    stats: dict = {
        "num_tool_calls": 0,
        "num_eval_attempts": 0,
        "num_eval_successes": 0,
        "num_eval_failures": 0,
        "metric_trajectory": [],
        "num_file_writes": 0,
        "num_file_reads": 0,
        "timed_out": False,
        "total_lines": len(lines),
    }
    for line in lines:
        if "[TOOL_CALL]" in line:
            stats["num_tool_calls"] += 1
            if _EVAL_CALL_RE.search(line):
                stats["num_eval_attempts"] += 1
            if _WRITE_EDIT_RE.search(line):
                stats["num_file_writes"] += 1
            if _READ_RE.search(line):
                stats["num_file_reads"] += 1
        m = _SUCCESS_RE.match(line)
        if m:
            stats["num_eval_successes"] += 1
            try:
                stats["metric_trajectory"].append(float(m.group(1)))
            except ValueError:
                pass
        if _FAILURE_RE.match(line):
            stats["num_eval_failures"] += 1
        if "[TIMEOUT]" in line:
            stats["timed_out"] = True
    return stats


def _condense_journal(
    journal_text: str,
    max_chars: int = 15000,
) -> str:
    """Produce a condensed journal suitable for LLM analysis.

    Strips the ``[QUERY]`` block (the LLM already knows the prompt
    template) and truncates long ``[TOOL_RESPONSE]`` blocks.  If the
    result still exceeds *max_chars*, keeps the first and last portions.
    """
    lines = journal_text.split("\n")

    # Strip the [QUERY] block (everything up to the next timestamped line)
    condensed: list[str] = []
    in_query_block = False
    for line in lines:
        if "[QUERY]" in line:
            in_query_block = True
            condensed.append("[QUERY] (omitted — the LLM already knows the prompt)")
            continue
        if in_query_block:
            if re.match(r"^\[\d{4}-\d{2}-\d{2}", line):
                in_query_block = False
            else:
                continue
        condensed.append(line)

    # Truncate long TOOL_RESPONSE blocks (keep first 10 lines of each)
    final: list[str] = []
    in_response = False
    response_line_count = 0
    max_response_lines = 10
    for line in condensed:
        if "[TOOL_RESPONSE]" in line:
            in_response = True
            response_line_count = 0
            final.append(line)
            continue
        if in_response:
            response_line_count += 1
            if response_line_count <= max_response_lines:
                final.append(line)
            elif response_line_count == max_response_lines + 1:
                final.append("    ... (truncated) ...")
            if re.match(r"^\[\d{4}-\d{2}-\d{2}", line):
                in_response = False
                final.append(line)
            continue
        final.append(line)

    result = "\n".join(final)
    if len(result) <= max_chars:
        return result

    # Still too long — keep first and last portions
    head_budget = max_chars // 4
    tail_budget = max_chars - head_budget - 40
    return (
        result[:head_budget]
        + "\n\n... (middle truncated) ...\n\n"
        + result[-tail_budget:]
    )


def _validate_template(template: str) -> bool:
    """Check that all required placeholders are present as single-brace tokens.

    Rejects templates where a placeholder is double-braced (e.g.
    ``{{original_query}}``) because ``.replace()`` would then leave
    stray braces around the substituted value.
    """
    for p in REQUIRED_PLACEHOLDERS:
        if p not in template:
            return False
        doubled = "{" + p
        if doubled in template:
            return False
    return True
