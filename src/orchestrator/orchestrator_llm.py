# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""LLM-powered idea generation and solution summarization for the orchestrator."""

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI

import printer
from config import LLMConfig
from utils import invoke_llm_with_retry, retry_on_parse_failure

from .json_utils import strip_code_fences, extract_json_fragment
from .models import GenerationSummary


class OrchestratorLLM:
    """Handles all LLM interactions: idea generation and solution summarization.

    Owns the prompt templates, JSON parsing, and the orchestrator log file.
    """

    IDEAS_INITIAL_PROMPT = """There is no prior conversation: use only the information below. You are a research lead. Decompose the task below into exactly {n} distinct algorithmic approaches. Each will be assigned to a separate implementing agent. Each idea must be actionable, self-contained, and different in kind (not minor variants of the same approach).

The goal is to build a Pareto front of solutions trading off task metric ({metric_direction}) against complexity (lower is better). Where two ideas are otherwise comparable, prefer the simpler one. Propose approaches that span different points on this trade-off.

Each idea must be actionable, self-contained, and different in kind (not minor variants of the same approach).

## Task
{query}

## Output format
Reply with **only** a JSON array of exactly {n} objects. No other text.

Each object has one key:
- "description": string — the approach in enough detail for an implementer to follow.

Example for n=2:
[{{"description": "Enumerate candidate solutions and evaluate each; keep the best."}}, {{"description": "Build the solution step by step with a local greedy rule at each step."}}]

Your response (JSON array only):"""

    IDEAS_FROM_RESULTS_PROMPT = """There is no prior conversation: use only the information below. You are a research lead in an iterative optimization loop. Previous generations of agents have run; their results are below.

## Task
{query}

## Objective
Build a 2-D Pareto front over:
- **Metric** ({metric_direction})
- **Complexity** (lower is better)

Where two ideas are otherwise comparable, prefer the simpler one. Propose approaches that span different points on this trade-off.

Propose exactly {n} ideas for the next generation: refinements, simplifications, combinations, or new approaches.

## Previous results
Two groups of candidates from previous generations are shown below. Each entry includes ``workspace_id:hp_index``, the assigned-idea description, the verdict-prefixed summary, the metric, the complexity, and the actual code that ran. A cluster appears in **at most one** group: PARETO FRONT or OFF FRONT, never both.

### PARETO FRONT
Candidates currently on the metric/complexity Pareto front. These are the strongest known trade-offs and are the natural targets for refinement proposals. Verdicts here are mixed (``yes`` / ``partial`` / ``no``): a Pareto-front entry with ``partial`` or ``no`` verdict means the agent substituted a different algorithm than its cluster description claims — read the verdict and code to see what the entry actually is, and treat the cluster description as untested in that case.

{front_text}

### OFF FRONT
Best ``yes``-verdict attempt per cluster, for clusters not on the front. ``partial``/``no`` entries are filtered out here, so for OFF FRONT the cluster description is a faithful label of what each entry does. These surface algorithmic families that the front view alone would hide — e.g. a promising idea that scored decently on its first attempt and was never revisited. Refining an OFF FRONT entry is encouraged when you can articulate why a specific change would close its current metric gap, especially when the front is dominated by a single algorithmic family.

{off_front_text}

## Diversification
- Ensure the {n} ideas are diverse at the level of the **underlying algorithmic family**, not merely different hyperparameters or complexity tiers of the same approach.
- When assessing which family is over-represented, count what the code actually is (per the verdict and summary), not the cluster label — a ``partial``/``no`` Pareto-front entry does not establish that its assigned family has been tried.
- When the families *actually present* across PARETO FRONT and OFF FRONT are dominated by one approach, prefer a refinement of an OFF FRONT entry from a different family, or a fresh-start idea, over yet another refinement of the front leader. A fresh-start idea is acceptable even if its expected metric is worse than the current leader on day one.

## Output format
Reply with **only** a JSON array of exactly {n} objects. No other text.

Each object has:
- "description": string — the approach for the next generation.
- "reference_workspaces": comma-separated references (e.g. "gen00-0001:0"). Use **only** for direct refinements of an entry shown above (PARETO FRONT or OFF FRONT), and only when the description matches what that entry's code actually does (per the verdict and summary), not the cluster label it was assigned to. **Omit this field entirely** for fresh-start ideas; supplying a reference for a fresh-start idea anchors the agent on unrelated code.

Example for n=2:
[{{"description": "Take gen00-0001 and replace its boxcar smoothing with a Gaussian kernel; everything else identical.", "reference_workspaces": "gen00-0001:0"}}, {{"description": "Estimate the channel by clustering received samples around constellation points using a Gaussian-mixture model fitted with EM, then demap with per-cluster posteriors. Do not reuse any prior code."}}]

Your response (JSON array only):"""

    SUMMARIZE_PROMPT = """You are summarizing a single implementation for a research coordinator. There is no prior conversation: use only the information below.

## Context
- An agent was assigned one specific approach (idea). It produced the solution code below.
- Your summary will be used to decide the next set of ideas for future agents. Be factual and concise: what was actually implemented, and how does it relate to the assigned idea?

## Assigned idea (the approach this agent was supposed to follow)
{idea_description}

## Solution code produced by the agent
```python
{code}
```

## Your task
Write the summary in this exact form:

"Followed assigned approach: <yes|partial|no>. <2–4 sentences describing (1) what approach/techniques were actually used in the code, and (2) how they align with or deviate from the assigned idea.>"

Rules for the verdict:
- "yes" — the code implements the core mechanism named in the assigned idea.
- "partial" — the code implements some of the assigned idea but the central mechanism is missing or replaced. Use "partial" whenever the headline algorithm in the idea is not the headline algorithm in the code.
- "no" — the code is from a clearly different algorithmic family than the assigned idea.

The verdict reflects only what the code does relative to the assigned idea — it must not be influenced by whether the code is good, fast, or scores well. No preamble, no meta-commentary, no markdown.

Reply with **only** a JSON object with one key: "summary" (the summary text). Example: {{"summary": "Followed assigned approach: partial. The code implements ..."}}"""

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
        """Lazily create the shared LLM instance with thread-safe initialization."""
        if self._llm is None:
            with self._llm_init_lock:
                if self._llm is None:
                    self._llm = ChatOpenAI(**asdict(self._llm_config))
        return self._llm

    def _log(self, role: str, content: str) -> None:
        """Append a turn to the orchestrator log file."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n[{role}]\n{content}\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_initial_ideas(self, query: str, n: int) -> list[str]:
        """Ask the LLM for *n* initial ideas from the task query.

        Returns a list of description strings.
        Raises on LLM or parsing failure after retries.
        """
        printer.log(f"Generating {n} initial ideas...")
        llm = self._get_llm()
        metric_direction = "HIGHER is better" if self._higher_is_better else "LOWER is better"
        prompt = self.IDEAS_INITIAL_PROMPT.format(n=n, query=query, metric_direction=metric_direction)
        self._log("PROMPT (initial ideas)", prompt)

        def _attempt() -> list[str]:
            response = invoke_llm_with_retry(llm, prompt, context="initial ideas")
            content = response.content.strip()
            self._log("RESPONSE (initial ideas)", content)
            parsed = self._parse_initial_ideas_json(content, n)
            if parsed is None:
                raise ValueError(
                    f"Failed to parse initial ideas from LLM response:\n{content}"
                )
            return parsed

        parsed = retry_on_parse_failure(_attempt, context="initial ideas")
        printer.log(f"Generated {len(parsed)} initial ideas.")
        return parsed

    def generate_ideas_from_results(
        self,
        front_summaries: list[GenerationSummary],
        off_front_summaries: list[GenerationSummary],
        n: int,
        query: str,
        front_codes: Optional[list[str]] = None,
        off_front_codes: Optional[list[str]] = None,
    ) -> tuple[list[str], list[list[str]]]:
        """Generate *n* ideas informed by previous generation summaries.

        Two parallel inputs are accepted:
          * ``front_summaries`` / ``front_codes``: entries currently on the
            metric/complexity Pareto front.
          * ``off_front_summaries`` / ``off_front_codes``: a sample of
            yes-verdict candidates from clusters *not* on the front,
            shown to surface under-explored families.

        Code lists are parallel to the corresponding summaries; pass
        ``None`` (or an empty string per entry) to omit code rendering.

        Returns ``(descriptions, refs_per_idea)`` where *refs_per_idea[i]*
        is a list of workspace IDs to use as reference code for idea *i*.
        Raises on LLM or parsing failure.
        """
        total = len(front_summaries) + len(off_front_summaries)
        printer.log(
            f"Generating {n} ideas from {len(front_summaries)} front + "
            f"{len(off_front_summaries)} off-front previous results "
            f"(total {total})..."
        )
        front_text = self._build_results_text(
            front_summaries, codes=front_codes
        ) or "(none yet)"
        off_front_text = self._build_results_text(
            off_front_summaries, codes=off_front_codes
        ) or "(no candidates available yet)"
        metric_direction = (
            "Higher metric values are better (e.g. reward, throughput)."
            if self._higher_is_better
            else "Lower metric values are better (e.g. loss, error rate)."
        )
        llm = self._get_llm()
        prompt = self.IDEAS_FROM_RESULTS_PROMPT.format(
            n=n,
            query=query,
            metric_direction=metric_direction,
            front_text=front_text,
            off_front_text=off_front_text,
        )
        self._log("PROMPT (ideas from results)", prompt)

        def _attempt() -> tuple[list[str], list[list[str]]]:
            response = invoke_llm_with_retry(llm, prompt, context="ideas from results")
            content = response.content.strip()
            self._log("RESPONSE (ideas from results)", content)
            descriptions, refs = self._parse_ideas_from_results_json(content, n)
            if not descriptions:
                raise ValueError(
                    f"Failed to parse ideas from LLM response:\n{content}"
                )
            return descriptions, refs

        descriptions, refs = retry_on_parse_failure(
            _attempt, context="ideas from results"
        )
        printer.log(f"Generated {len(descriptions)} ideas from results.")
        return descriptions, refs

    def summarize_solution(
        self, code: str, idea_description: str, workspace_id: str = ""
    ) -> str:
        """Ask the LLM for a concise summary of a solution.

        Falls back to the raw LLM response text if JSON parsing fails.
        """
        if not code.strip():
            return "No code provided."
        label = f" for {workspace_id}" if workspace_id else ""
        printer.log(f"Summarizing solution{label}...")
        llm = self._get_llm()
        prompt = self.SUMMARIZE_PROMPT.format(
            idea_description=idea_description,
            code=code,
        )
        self._log("PROMPT (summarize)", prompt)

        def _attempt() -> str:
            response = invoke_llm_with_retry(llm, prompt, context="summarize")
            content = response.content.strip() or "No summary."
            self._log("RESPONSE (summarize)", content)
            summary = self._parse_summary_json(content)
            if summary is None:
                raise ValueError(
                    f"Failed to parse summary from LLM response:\n{content}"
                )
            return summary

        try:
            return retry_on_parse_failure(_attempt, context="summarize")
        except ValueError:
            return "Summary unavailable (parse failure)."

    # ------------------------------------------------------------------
    # Results text formatting
    # ------------------------------------------------------------------

    # Cap per-entry code rendering to keep prompt size bounded. Most
    # solutions in our tasks fit comfortably under this; longer ones get
    # truncated with a tail marker.
    _MAX_CODE_LINES = 200

    @classmethod
    def _truncate_code(cls, code: str) -> str:
        if not code:
            return ""
        lines = code.splitlines()
        if len(lines) <= cls._MAX_CODE_LINES:
            return code
        head = lines[: cls._MAX_CODE_LINES]
        tail = f"# ... ({len(lines) - cls._MAX_CODE_LINES} more lines truncated) ..."
        return "\n".join(head + [tail])

    @classmethod
    def _build_results_text(
        cls,
        summaries: list[GenerationSummary],
        codes: Optional[list[str]] = None,
    ) -> str:
        """Format generation summaries into the text block for the prompt.

        If ``codes`` is provided, it must be parallel to ``summaries`` and
        each entry's code is rendered (truncated) below its summary line.
        """
        if not summaries:
            return ""
        if codes is None:
            codes = [""] * len(summaries)
        if len(codes) != len(summaries):
            raise ValueError(
                "codes and summaries must have the same length"
            )

        by_gen: dict[int, dict[int, list[tuple[GenerationSummary, str]]]] = {}
        idea_descriptions: dict[int, str] = {}
        for s, code in zip(summaries, codes):
            idea_descriptions[s.cluster_id] = s.idea_description
            by_gen.setdefault(s.generation, {}).setdefault(
                s.cluster_id, []
            ).append((s, code))

        lines: list[str] = []
        for generation in sorted(by_gen):
            lines.append(f"=== Generation {generation} ===")
            for cluster_id, entries in by_gen[generation].items():
                desc = idea_descriptions.get(cluster_id, "")
                lines.append(f"  --- Cluster {cluster_id} ---")
                lines.append(f"  Description: {desc}")
                for entry, code in entries:
                    ref = f"{entry.workspace_id}:{entry.hp_index}"
                    lines.append(
                        f"    {ref}: "
                        f"Summary: {entry.summary}; "
                        f"Metric: {entry.metric}; "
                        f"Complexity: {entry.complexity}"
                    )
                    if entry.info:
                        lines.append(
                            f"      Additional evaluation info: {entry.info}"
                        )
                    if code:
                        lines.append("      Code:")
                        lines.append("```python")
                        lines.append(cls._truncate_code(code))
                        lines.append("```")
                lines.append("")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON parsing helpers (all stateless)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_initial_ideas_json(
        response_text: str, n: int
    ) -> Optional[list[str]]:
        """Parse a JSON array of initial ideas. Returns ``None`` on failure."""
        text = strip_code_fences(response_text)
        text = extract_json_fragment(text, "[")
        if text is None:
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, list) or len(data) < n:
                return None
            descriptions: list[str] = []
            for item in data[:n]:
                if not isinstance(item, dict):
                    return None
                desc = item.get("description")
                if desc is not None:
                    desc = str(desc).strip()
                descriptions.append(desc or "No description")
            return descriptions[:n]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def _parse_summary_json(response_text: str) -> Optional[str]:
        """Parse a JSON object with a ``summary`` key. Returns ``None`` on failure."""
        text = strip_code_fences(response_text)
        text = extract_json_fragment(text, "{")
        if text is None:
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            s = data.get("summary")
            return str(s).strip() if s is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def _parse_ideas_from_results_json(
        response_text: str, n: int
    ) -> tuple[list[str], list[list[str]]]:
        """Parse a JSON array of ideas with optional reference workspaces.

        Returns ``(descriptions, refs_per_idea)``.  On failure both lists
        are empty.
        """
        empty: tuple[list[str], list[list[str]]] = ([], [[] for _ in range(n)])
        text = strip_code_fences(response_text)
        text = extract_json_fragment(text, "[")
        if text is None:
            return empty
        try:
            data = json.loads(text)
            if not isinstance(data, list) or len(data) < n:
                return empty
            descriptions: list[str] = []
            refs_per_idea: list[list[str]] = []
            for item in data[:n]:
                if not isinstance(item, dict):
                    return empty
                desc = item.get("description")
                if desc is not None:
                    desc = str(desc).strip()
                descriptions.append(desc or "No description")
                refs_raw = item.get("reference_workspaces")
                refs: list[str] = []
                if refs_raw is not None:
                    if isinstance(refs_raw, list):
                        refs = [str(x).strip() for x in refs_raw if str(x).strip()]
                    else:
                        refs = [
                            w.strip()
                            for w in str(refs_raw).replace(",", " ").split()
                            if w.strip()
                        ]
                refs_per_idea.append(refs)
            return descriptions[:n], refs_per_idea[:n]
        except (json.JSONDecodeError, TypeError, ValueError):
            return empty
