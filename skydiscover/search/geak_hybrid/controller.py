"""
GEAK Hybrid Controller: AdaEvolve outer loop + GEAK-style inner optimization.

Each outer iteration runs a bounded inner loop of:
  Router → Reviewer → Editor → Evaluate → Interpreter

The inner loop refines one candidate through multiple rounds before committing
the best result to the population database.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import apply_diff, extract_diffs, parse_full_rewrite
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ephemeral dataclasses (controller-local, not persisted)
# ---------------------------------------------------------------------------

@dataclass
class RoutePlan:
    strategy: str = "exploit"
    focus_metrics: List[str] = field(default_factory=lambda: ["combined_score"])
    edit_goals: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ReviewPlan:
    acceptance_checks: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    review_summary: str = ""


@dataclass
class Interpretation:
    outcome: str = "no_change"
    lessons: List[str] = field(default_factory=list)
    next_focus: List[str] = field(default_factory=list)
    continue_loop: bool = True
    summary: str = ""


@dataclass
class GEAKRoundTrace:
    round_index: int = 0
    source_score: Optional[float] = None
    candidate_score: Optional[float] = None
    route_strategy: str = "exploit"
    outcome: str = "pending"
    error: Optional[str] = None
    accepted: bool = False


class GEAKHybridController(AdaEvolveController):
    """AdaEvolve outer loop with GEAK-style bounded inner optimization per iteration.

    Inherits all AdaEvolve behavior (UCB islands, adaptive intensity, paradigm
    breakthroughs, migration) and overrides only _run_normal_step to add a
    multi-round Router → Reviewer → Editor → Evaluate → Interpreter cycle.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # GEAK-specific config from database config
        db_config = self.config.search.database
        self.inner_loop_budget = getattr(db_config, "inner_loop_budget", 3)
        self.max_invalid_rounds = getattr(db_config, "max_invalid_rounds", 2)
        self.accept_eps = getattr(db_config, "accept_improvement_epsilon", 1e-6)

        if self.inner_loop_budget < 1:
            raise ValueError(f"inner_loop_budget must be >= 1, got {self.inner_loop_budget}")

        # Use guide models for Router/Reviewer/Interpreter if available,
        # otherwise fall back to main LLM pool
        self._stage_llms = self.guide_llms if self.guide_llms else self.llms

        logger.info(
            f"GEAKHybridController: inner_loop_budget={self.inner_loop_budget}, "
            f"max_invalid_rounds={self.max_invalid_rounds}"
        )

    # ------------------------------------------------------------------
    # Override: inner GEAK loop replaces single-shot generation
    # ------------------------------------------------------------------

    async def _run_normal_step(self, iteration: int) -> SerializableResult:
        """Run GEAK bounded inner loop for one outer iteration."""
        try:
            # === SAMPLE ONCE PER OUTER ITERATION ===
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            self._ensure_all_islands_seeded()

            parent_dict, context_programs_dict = self.database.sample(
                self.num_context_programs
            )
            if not parent_dict:
                return SerializableResult(error="Empty parent from sample()", iteration=iteration)

            parent_label = list(parent_dict.keys())[0]
            parent = list(parent_dict.values())[0]
            base_score = get_score(parent.metrics) if parent.metrics else 0.0

            # Paradigm handling (same as AdaEvolve)
            paradigm = None
            if self.database.use_paradigm_breakthrough:
                paradigm = self.database.get_current_paradigm()
            if paradigm:
                best = self.database.get_best_program()
                if best:
                    parent = best
                    base_score = get_score(best.metrics) if best.metrics else 0.0

            # Record paradigm usage (mirrors AdaEvolve._generate_child)
            if paradigm:
                self.database.use_paradigm()

            # Build tracking info
            parent_info = (parent_label, parent.id)
            context_info = [
                (lbl, p.id) for lbl, progs in context_programs_dict.items() for p in progs
            ]
            context_program_ids = [
                p.id for progs in context_programs_dict.values() for p in progs
            ]

            # === GEAK INNER LOOP ===
            # Track a full working candidate state across rounds
            working_solution = parent.solution
            working_metrics = dict(parent.metrics) if parent.metrics else {}
            working_score = base_score
            best_result: Optional[SerializableResult] = None
            best_score = -float("inf")
            invalid_rounds = 0
            traces: List[GEAKRoundTrace] = []
            last_interpretation: Optional[Interpretation] = None
            # The original DB parent — always used for lineage tracking
            persisted_parent = parent

            for round_idx in range(self.inner_loop_budget):
                if self.shutdown_event.is_set():
                    break

                trace = GEAKRoundTrace(round_index=round_idx, source_score=working_score)

                # --- 1. ROUTER: identify bottleneck ---
                route = await self._geak_router(
                    working_solution, working_metrics, paradigm, last_interpretation,
                )
                trace.route_strategy = route.strategy

                # --- 2. REVIEWER: form hypothesis + checks ---
                review = await self._geak_reviewer(
                    working_solution, route, working_metrics,
                )

                # --- 3. EDITOR: generate code via normal LLM + evaluate ---
                # Build a round-local Program representing the current working state.
                # This is critical: _execute_generation applies diffs against
                # round_parent.solution, so it must reflect the latest accepted code.
                round_parent = Program(
                    id=persisted_parent.id,
                    solution=working_solution,
                    metrics=working_metrics,
                    generation=persisted_parent.generation,
                    language=getattr(persisted_parent, 'language', None),
                )

                context = {
                    "program_metrics": working_metrics,
                    "other_context_programs": context_programs_dict,
                    "paradigm": paradigm,
                    "siblings": [],
                    "error_context": None,
                }
                for k, v in self._prompt_context.items():
                    if k not in context:
                        context[k] = v

                prompt = self.context_builder.build_prompt(
                    {parent_label: round_parent},
                    context,
                )

                # Inject GEAK guidance into the user message
                geak_guidance = self._format_geak_guidance(route, review, last_interpretation)
                if geak_guidance:
                    prompt["user"] = prompt["user"] + "\n\n" + geak_guidance

                # Generate + evaluate using round_parent so diffs apply to
                # the working solution, but track lineage to persisted_parent
                result = await self._execute_generation(
                    round_parent, prompt, iteration,
                    parent_info=parent_info,
                    context_info=context_info,
                    context_program_ids=context_program_ids,
                    other_context_programs=context_programs_dict,
                )

                # --- Score the result ---
                if result.error:
                    trace.outcome = "invalid"
                    trace.error = result.error[:200]
                    invalid_rounds += 1
                else:
                    child_metrics = result.child_program_dict.get("metrics", {})
                    candidate_score = get_score(child_metrics)
                    trace.candidate_score = candidate_score

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_result = result

                    improved_over_base = candidate_score > base_score + self.accept_eps
                    improved_over_working = candidate_score >= working_score - self.accept_eps

                    if improved_over_working and result.child_program_dict.get("solution"):
                        working_solution = result.child_program_dict["solution"]
                        working_metrics = result.child_program_dict.get("metrics", working_metrics)
                        working_score = candidate_score
                        trace.accepted = True

                    if improved_over_base:
                        trace.outcome = "improved"
                    elif candidate_score >= working_score - self.accept_eps:
                        trace.outcome = "no_change"
                    else:
                        trace.outcome = "regressed"

                traces.append(trace)

                # --- 4. INTERPRETER: classify + extract learnings ---
                last_interpretation = await self._geak_interpreter(
                    route, trace, parent.metrics or {},
                    best_score if best_score > -float("inf") else None,
                )

                # --- Early termination checks ---
                if trace.outcome == "improved":
                    logger.info(
                        f"GEAK round {round_idx}: improved {base_score:.4f} → "
                        f"{candidate_score:.4f}, accepting early"
                    )
                    break

                if invalid_rounds >= self.max_invalid_rounds:
                    logger.info(f"GEAK: {invalid_rounds} invalid rounds, stopping inner loop")
                    break

                if not last_interpretation.continue_loop:
                    logger.info(f"GEAK: interpreter says stop after round {round_idx}")
                    break

            # === FINALIZE: return best result with GEAK metadata ===
            if best_result is None:
                return SerializableResult(
                    error=f"GEAK inner loop produced no valid candidate after "
                          f"{len(traces)} rounds",
                    iteration=iteration,
                )

            # Attach GEAK metadata to the child program
            child_dict = best_result.child_program_dict
            if "metadata" not in child_dict or child_dict["metadata"] is None:
                child_dict["metadata"] = {}
            child_dict["metadata"]["geak_hybrid"] = {
                "rounds_used": len(traces),
                "accepted_round": next(
                    (t.round_index for t in traces if t.accepted and t.candidate_score == best_score),
                    None,
                ),
                "base_parent_id": parent.id,
                "base_parent_score": base_score,
                "best_inner_score": best_score,
                "final_outcome": traces[-1].outcome if traces else "none",
                "rounds": [
                    {
                        "round": t.round_index,
                        "route_strategy": t.route_strategy,
                        "source_score": t.source_score,
                        "candidate_score": t.candidate_score,
                        "outcome": t.outcome,
                        "error": t.error,
                        "accepted": t.accepted,
                    }
                    for t in traces
                ],
                "final_interpreter_summary": (
                    last_interpretation.summary if last_interpretation else ""
                ),
            }

            return best_result

        except Exception as e:
            logger.exception(f"GEAK generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)

    # ------------------------------------------------------------------
    # GEAK stage implementations
    # ------------------------------------------------------------------

    async def _geak_router(
        self,
        solution: str,
        metrics: Dict[str, Any],
        paradigm: Optional[Dict] = None,
        last_interp: Optional[Interpretation] = None,
    ) -> RoutePlan:
        """Router: identify the primary bottleneck and optimization strategy."""
        system = (
            "You are a GPU kernel performance analyst. Identify the primary "
            "bottleneck and recommend an optimization strategy. Respond in JSON only:\n"
            '{"strategy": "exploit|explore|repair", "focus_metrics": [...], '
            '"edit_goals": [...], "reasoning": "..."}'
        )
        user_parts = [f"## Current Code (excerpt)\n```\n{solution[:4000]}\n```"]
        if metrics:
            user_parts.append(f"## Metrics\n{json.dumps(metrics, default=str)}")
        if paradigm:
            user_parts.append(f"## Paradigm Shift\n{paradigm.get('description', '')}")
        if last_interp:
            user_parts.append(f"## Previous Round\n{last_interp.summary}")
            if last_interp.next_focus:
                user_parts.append(f"Focus: {', '.join(last_interp.next_focus)}")

        try:
            response = await self._stage_llms.generate(
                system_message=system,
                messages=[{"role": "user", "content": "\n\n".join(user_parts)}],
            )
            data = _parse_json(response.text)
            return RoutePlan(
                strategy=data.get("strategy", "exploit"),
                focus_metrics=data.get("focus_metrics", ["combined_score"]),
                edit_goals=data.get("edit_goals", []),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.debug(f"Router parse failed: {e}")
            return RoutePlan(reasoning=f"Fallback: {e}")

    async def _geak_reviewer(
        self,
        solution: str,
        route: RoutePlan,
        metrics: Dict[str, Any],
    ) -> ReviewPlan:
        """Reviewer: form acceptance checks and forbidden patterns."""
        system = (
            "You are a code review assistant for GPU kernel optimization. "
            "Given the router's strategy, define acceptance checks. JSON only:\n"
            '{"acceptance_checks": [...], "forbidden_patterns": [...], '
            '"review_summary": "..."}'
        )
        user = (
            f"Strategy: {route.strategy}\n"
            f"Goals: {', '.join(route.edit_goals)}\n"
            f"Metrics: {json.dumps(metrics, default=str)}\n"
            f"Code excerpt:\n```\n{solution[:2000]}\n```"
        )
        try:
            response = await self._stage_llms.generate(
                system_message=system,
                messages=[{"role": "user", "content": user}],
            )
            data = _parse_json(response.text)
            return ReviewPlan(
                acceptance_checks=data.get("acceptance_checks", []),
                forbidden_patterns=data.get("forbidden_patterns", []),
                review_summary=data.get("review_summary", ""),
            )
        except Exception as e:
            logger.debug(f"Reviewer parse failed: {e}")
            return ReviewPlan(review_summary=f"Fallback: {e}")

    async def _geak_interpreter(
        self,
        route: RoutePlan,
        trace: GEAKRoundTrace,
        parent_metrics: Dict[str, Any],
        best_score_so_far: Optional[float],
    ) -> Interpretation:
        """Interpreter: classify result and extract learnings."""
        system = (
            "You are an experiment interpreter. Classify the result and extract "
            "learnings for the next round. JSON only:\n"
            '{"outcome": "improved|no_change|regressed|invalid", '
            '"lessons": [...], "next_focus": [...], '
            '"continue_loop": true/false, "summary": "..."}'
        )
        user = (
            f"Strategy: {route.strategy}\n"
            f"Source score: {trace.source_score}\n"
            f"Candidate score: {trace.candidate_score}\n"
            f"Outcome: {trace.outcome}\n"
            f"Error: {trace.error or 'none'}\n"
            f"Best so far: {best_score_so_far}\n"
        )
        try:
            response = await self._stage_llms.generate(
                system_message=system,
                messages=[{"role": "user", "content": user}],
            )
            data = _parse_json(response.text)
            return Interpretation(
                outcome=data.get("outcome", trace.outcome),
                lessons=data.get("lessons", []),
                next_focus=data.get("next_focus", []),
                continue_loop=data.get("continue_loop", True),
                summary=data.get("summary", ""),
            )
        except Exception as e:
            logger.debug(f"Interpreter parse failed: {e}")
            return Interpretation(
                outcome=trace.outcome,
                continue_loop=trace.round_index < self.inner_loop_budget - 1,
                summary=f"Fallback: {e}",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_geak_guidance(
        self,
        route: RoutePlan,
        review: ReviewPlan,
        last_interp: Optional[Interpretation],
    ) -> str:
        """Format GEAK stage outputs as guidance for the editor prompt."""
        sections = []

        if route.edit_goals:
            sections.append(
                "## GEAK ROUTER PLAN\n"
                f"Strategy: {route.strategy}\n"
                f"Goals:\n" + "\n".join(f"- {g}" for g in route.edit_goals)
            )
        if route.reasoning:
            sections.append(f"Reasoning: {route.reasoning}")

        if review.acceptance_checks:
            sections.append(
                "## GEAK REVIEW CHECKS\n"
                + "\n".join(f"- {c}" for c in review.acceptance_checks)
            )
        if review.forbidden_patterns:
            sections.append(
                "## FORBIDDEN\n"
                + "\n".join(f"- {p}" for p in review.forbidden_patterns)
            )

        if last_interp and last_interp.lessons:
            sections.append(
                "## PREVIOUS ROUND LESSONS\n"
                + "\n".join(f"- {l}" for l in last_interp.lessons)
            )
        if last_interp and last_interp.next_focus:
            sections.append(
                "## FOCUS FOR THIS ROUND\n"
                + "\n".join(f"- {f}" for f in last_interp.next_focus)
            )

        return "\n\n".join(sections) if sections else ""


def _parse_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response text."""
    # Try direct parse
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try markdown code block
    if "```json" in text:
        parts = text.split("```json", 1)
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0].strip()
            try:
                return json.loads(code)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}")
