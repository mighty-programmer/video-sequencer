import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hybrid_agentic import (
    HybridAgentConfig,
    HybridResult,
    HybridAgentToolBridge,
    HybridAgentTools,
    HybridRunConfig,
    HybridVideoPrismConfig,
    VideoTextMatcher,
    _build_codex_exec_command,
    _codex_failure_message,
    _codex_agent_prompt,
    _find_codex_cli,
    _hard_constraint_evidence,
    _is_transient_codex_failure,
    _known_candidates_for_decisions,
    _payload_validation_score,
    analyze_full_audit_cycles,
    apply_agent_decisions,
    codex_cycle_critic,
    find_ambiguous_segments,
    generate_grid_configs,
    gate_full_audit_decisions,
    jax_runtime_status,
    load_segments,
    make_contact_sheet,
    run_grid,
    validate_run_config,
)
from hybrid_options import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    HYBRID_OPTIONS,
    HYBRID_PROMPT_MODES,
    load_codex_model_catalog,
)
from web_backend import EditorSessionManager, _apply_best_grid_search_config, build_job_command


class FakeIndexer:
    def __init__(self, metadata):
        self.metadata = metadata

    def search_by_embedding(self, _embedding, k):
        return [(self.metadata[0].video_id, 0.9, self.metadata[0])][:k]


class FakeBridgeTools:
    initial_assignment = [0]
    matrix = np.array([[0.9]])
    metadata = [SimpleNamespace()]

    def get_global_script_context(self):
        return {"segments": []}

    def search_video_index(self, query, top_k, exclude_assigned):
        return [{"candidate_index": 0, "query": query}]

    def get_candidate_catalog(self):
        return {"candidate_count": 1, "candidates": [{"candidate_index": 0, "anonymous_clip_id": "clip_000"}]}

    def inspect_clip(self, candidate_index):
        return {"candidate_index": candidate_index, "anonymous_clip_id": f"clip_{candidate_index:03d}", "contact_sheet": None}


class FakeEditorMatcher:
    def __init__(self, metadata):
        self.metadata = metadata

    def compute_similarity_matrix(self, _segments, **_kwargs):
        return np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64), self.metadata


def grid_args(output):
    return argparse.Namespace(
        benchmark="6",
        video_dir=None,
        segments=None,
        ground_truth=None,
        output=str(output),
        cache_dir="./cache",
        device="cpu",
        models=None,
        frames=None,
        resolutions=None,
        dual_softmax=None,
        prompt_modes=None,
        shortlist_sizes=None,
        ambiguity_margins=None,
        max_agent_segments_list=None,
        agent_search_budgets=None,
        agent_decision_modes=None,
        agent_review_scopes=None,
        model="videoprism_lvt_public_v1_large",
        num_frames=8,
        resolution=288,
        use_dual_softmax=False,
        prompt_mode="none",
        shortlist_size=5,
        ambiguity_margin_threshold=0.05,
        max_agent_segments=5,
        agent_search_budget_per_segment=2,
        agent_decision_mode="advisory",
        agent_review_scope="ambiguous",
        fast_grid=True,
        query_mode="original",
        context_window_size=1,
        query_llm_model=None,
        use_query_cache=True,
        force_refresh_expansions=False,
        disable_llm_expansion=False,
        score_normalization="none",
        csls_k=5,
        no_windowing=True,
        window_size=5.0,
        window_overlap=0.5,
        disable_agent=False,
        use_codex=False,
        no_agent_exclusions=False,
        no_agent_candidate_expansion=False,
        no_contact_sheets=False,
        advisory_boost=0.03,
        codex_model=DEFAULT_CODEX_MODEL,
        codex_reasoning_effort=DEFAULT_CODEX_REASONING_EFFORT,
        codex_timeout_seconds=600,
        hard_lock_min_confidence=0.9,
        allow_uninspected_hard_constraints=False,
        no_cycle_critic=False,
        cycle_critic_min_confidence=0.75,
    )


class HybridSanityTests(unittest.TestCase):
    def test_empty_narration_is_preserved_without_ordinal_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.json"
            path.write_text(json.dumps({"segments": [
                {"text": "", "start_time": 0.0, "end_time": 1.0},
                {"start_time": 1.0, "end_time": 2.0},
            ]}), encoding="utf-8")
            segments = load_segments(str(path))
        self.assertEqual([segment["text"] for segment in segments], ["", ""])
        self.assertNotIn("segment_", " ".join(segment["text"] for segment in segments))

    def test_live_codex_catalog_shape_and_preferred_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "models_cache.json"
            cache_path.write_text(json.dumps({"models": [{
                "slug": "gpt-5.6-sol",
                "display_name": "GPT-5.6-Sol",
                "visibility": "list",
                "default_reasoning_level": "low",
                "supported_reasoning_levels": [{"effort": "low"}, {"effort": "xhigh"}],
            }]}), encoding="utf-8")
            catalog = load_codex_model_catalog(cache_path)
        self.assertEqual(catalog[0]["value"], DEFAULT_CODEX_MODEL)
        self.assertIn(DEFAULT_CODEX_REASONING_EFFORT, catalog[0]["supported_reasoning_efforts"])

    def test_codex_discovery_finds_user_local_install_without_path(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / ".local" / "bin" / "codex"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            with mock.patch("hybrid_agentic.shutil.which", return_value=None), mock.patch(
                "hybrid_agentic.Path.home",
                return_value=Path(directory),
            ):
                self.assertEqual(_find_codex_cli(), str(executable))

    def test_codex_command_explicitly_sets_model_and_reasoning(self):
        config = HybridAgentConfig(
            use_codex=True,
            codex_model="gpt-5.6-sol",
            codex_reasoning_effort="xhigh",
        )
        command = _build_codex_exec_command(
            "/usr/bin/codex",
            Path("/tmp/workspace"),
            Path("/tmp/schema.json"),
            Path("/tmp/response.json"),
            Path("/tmp/proxy.py"),
            ["search_video_index"],
            config,
        )
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-sol")
        self.assertIn('model_reasoning_effort="xhigh"', command)

    def test_web_job_command_forwards_codex_model_and_reasoning(self):
        _, command = build_job_command("hybrid-agentic-grid-search", {
            "benchmark": "6",
            "use_codex": True,
            "agent_review_scopes": ["all"],
            "codex_model": "gpt-5.6-sol",
            "codex_reasoning_effort": "xhigh",
            "no_cycle_critic": True,
        }, {})
        self.assertEqual(command[command.index("--codex-model") + 1], "gpt-5.6-sol")
        self.assertEqual(command[command.index("--codex-reasoning-effort") + 1], "xhigh")
        self.assertEqual(command[command.index("--agent-review-scopes") + 1], "all")
        self.assertIn("--no-cycle-critic", command)

    def test_legacy_hybrid_winner_does_not_silently_enable_cycle_critic(self):
        config = {"agent_verify_assignment_cycles": True}
        best = {
            "retrieval_mode": "hybrid_agentic",
            "config": {
                "model_name": "videoprism_lvt_public_v1_base",
                "agent_enabled": True,
            },
        }
        with mock.patch("web_backend.hybrid_codex_status", return_value={"ready": True}):
            applied = _apply_best_grid_search_config(config, best, "hybrid_agentic")
        self.assertFalse(applied["agent_verify_assignment_cycles"])

    def test_recorded_cycle_critic_setting_is_preserved(self):
        best = {
            "retrieval_mode": "hybrid_agentic",
            "config": {
                "model_name": "videoprism_lvt_public_v1_base",
                "agent_verify_assignment_cycles": True,
            },
        }
        with mock.patch("web_backend.hybrid_codex_status", return_value={"ready": True}):
            applied = _apply_best_grid_search_config({}, best, "hybrid_agentic")
        self.assertTrue(applied["agent_verify_assignment_cycles"])

    def test_shared_options_include_every_parser_prompt_and_real_budget_four(self):
        prompt_values = tuple(option["value"] for option in HYBRID_OPTIONS["prompt_modes"])
        budget_values = {option["value"] for option in HYBRID_OPTIONS["search_budgets"]}
        self.assertEqual(prompt_values, HYBRID_PROMPT_MODES)
        self.assertIn("4", budget_values)

    def test_config_validation_rejects_invalid_values(self):
        run = HybridRunConfig()
        run.agent.search_budget_per_segment = -1
        with self.assertRaisesRegex(ValueError, "search_budget"):
            validate_run_config(run)
        run.agent.search_budget_per_segment = 2
        run.video.prompt_mode = "template:not-real"
        with self.assertRaisesRegex(ValueError, "prompt mode"):
            validate_run_config(run)
        run.video.prompt_mode = "none"
        run.video.no_windowing = False
        with self.assertRaisesRegex(ValueError, "temporal windowing"):
            validate_run_config(run)
        run.video.no_windowing = True
        run.agent.review_scope = "oracle"
        with self.assertRaisesRegex(ValueError, "review scope"):
            validate_run_config(run)

    def test_cycle_critic_requires_visual_evidence_when_enabled(self):
        run = HybridRunConfig()
        run.agent.use_codex = True
        run.agent.review_scope = "all"
        run.agent.verify_assignment_cycles = True
        run.agent.allow_contact_sheet = False
        with mock.patch("hybrid_agentic.load_codex_model_catalog", return_value=[{
            "value": run.agent.codex_model,
            "supported_reasoning_efforts": [run.agent.codex_reasoning_effort],
        }]):
            with self.assertRaisesRegex(ValueError, "cycle critic requires contact-sheet"):
                validate_run_config(run)

    def test_cuda_request_cannot_silently_fall_back_to_cpu(self):
        fake_jax = SimpleNamespace(
            default_backend=lambda: "cpu",
            devices=lambda: [SimpleNamespace(id=0, platform="cpu", device_kind="CPU")],
        )
        with mock.patch.dict(sys.modules, {"jax": fake_jax}):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                jax_runtime_status("cuda:0")

    def test_config_validation_rejects_unsupported_codex_reasoning_pair(self):
        run = HybridRunConfig()
        run.agent.use_codex = True
        run.agent.codex_model = "test-model"
        run.agent.codex_reasoning_effort = "xhigh"
        with mock.patch("hybrid_agentic.load_codex_model_catalog", return_value=[{
            "value": "test-model",
            "supported_reasoning_efforts": ["low", "medium"],
        }]):
            with self.assertRaisesRegex(ValueError, "does not support reasoning effort"):
                validate_run_config(run)

    def test_hard_constraints_require_codex_and_contact_sheet_inspection(self):
        run = HybridRunConfig()
        run.agent.decision_mode = "hard_lock"
        with self.assertRaisesRegex(ValueError, "requires the authenticated Codex"):
            validate_run_config(run)
        run.agent.use_codex = True
        run.agent.allow_contact_sheet = False
        with self.assertRaisesRegex(ValueError, "requires contact-sheet inspection"):
            with mock.patch("hybrid_agentic.load_codex_model_catalog", return_value=[{
                "value": run.agent.codex_model,
                "supported_reasoning_efforts": [run.agent.codex_reasoning_effort],
            }]):
                validate_run_config(run)

    def test_agent_search_bypasses_baseline_prompt_wrapper_and_hides_duration(self):
        meta = SimpleNamespace(video_id="video", file_path="/hidden/video.mp4", window_start=0.0, is_windowed=False)
        matcher = SimpleNamespace()
        with tempfile.TemporaryDirectory() as directory:
            tools = HybridAgentTools(
                matcher,
                FakeIndexer([meta]),
                [{"segment_id": 0, "text": "script"}],
                np.array([[0.8]]),
                [meta],
                [0],
                Path(directory),
            )
            with mock.patch.object(
                VideoTextMatcher,
                "get_text_embeddings_batch",
                return_value=np.array([[1.0, 0.0]], dtype=np.float32),
            ) as raw_encoder:
                results = tools.search_video_index("agent authored query", top_k=1)
            raw_encoder.assert_called_once_with(matcher, ["agent authored query"])
            self.assertNotIn("duration", results[0])
            self.assertEqual(results[0]["score_space"], "raw_videoprism_cosine_similarity")
            self.assertNotIn("duration", tools.get_candidate_scores(0, top_k=1)[0])

    def test_search_budget_four_is_enforced_as_four(self):
        config = HybridAgentConfig(search_budget_per_segment=4)
        bridge = HybridAgentToolBridge(FakeBridgeTools(), [{"segment_index": 0}], config)
        for index in range(4):
            result = bridge._execute("search_video_index", {"segment_index": 0, "query": f"query {index}", "top_k": 1})
            self.assertEqual(result["search_number"], index + 1)
            self.assertEqual(result["search_budget"], 4)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            bridge._execute("search_video_index", {"segment_index": 0, "query": "query 5", "top_k": 1})

    def test_full_audit_can_catalog_and_inspect_unranked_candidates(self):
        bridge = HybridAgentToolBridge(
            FakeBridgeTools(),
            [{"segment_index": 0}],
            HybridAgentConfig(review_scope="all", allow_contact_sheet=True),
        )
        catalog = bridge._execute("get_candidate_catalog", {})
        self.assertEqual(catalog["candidate_count"], 1)
        inspected = bridge._execute("inspect_candidate", {"candidate_index": 0})
        self.assertEqual(inspected["candidate_index"], 0)
        self.assertFalse(inspected["contact_sheet_available"])
        known = _known_candidates_for_decisions(bridge, [0], bridge.config)
        self.assertEqual(known[0], {0})
        with self.assertRaisesRegex(ValueError, "full-sequence audit"):
            HybridAgentToolBridge(
                FakeBridgeTools(),
                [{"segment_index": 0}],
                HybridAgentConfig(review_scope="ambiguous"),
            )._execute("get_candidate_catalog", {})

    def test_contact_sheet_uses_three_high_resolution_samples(self):
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 100
        capture.read.return_value = (True, np.zeros((90, 160, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "hybrid_agentic.cv2.VideoCapture",
            return_value=capture,
        ), mock.patch("hybrid_agentic.cv2.imwrite", return_value=True) as imwrite:
            self.assertTrue(make_contact_sheet(Path("hidden.mp4"), Path(directory) / "sheet.jpg"))
        sheet = imwrite.call_args.args[1]
        self.assertEqual(sheet.shape, (288, 1536, 3))
        self.assertEqual(capture.read.call_count, 3)

    def test_advisory_mode_uses_soft_adjustments_only(self):
        matrix = np.array([[0.5, 0.4]], dtype=np.float64)
        decisions = {
            "locks": {"0": 1},
            "exclusions": {"0": [0]},
            "advisory_scores": {"0": {"1": 1.0}},
            "advisory_penalties": {"0": {"0": 0.5}},
        }
        refined, locks, exclusions = apply_agent_decisions(matrix, decisions, HybridAgentConfig(decision_mode="advisory", advisory_boost=0.1))
        self.assertEqual(locks, {})
        self.assertEqual(exclusions, {})
        self.assertAlmostEqual(refined[0, 1], 0.5)
        self.assertAlmostEqual(refined[0, 0], 0.45)

    def test_full_audit_cycle_gate_accepts_one_component_and_rejects_disconnected_rewrites(self):
        one_cycle = {
            "advisory_scores": {"0": {"1": 0.95}, "1": {"0": 0.95}},
            "advisory_penalties": {}, "locks": {}, "exclusions": {},
        }
        gated = gate_full_audit_decisions(one_cycle, [0, 1, 2, 3])
        self.assertTrue(gated["full_audit_cycle_gate"]["accepted"])
        self.assertEqual(gated["full_audit_cycle_gate"]["component_count"], 1)
        self.assertTrue(gated["advisory_scores"])

        disconnected = {
            "advisory_scores": {
                "0": {"1": 0.95}, "1": {"0": 0.95},
                "2": {"3": 0.95}, "3": {"2": 0.95},
            },
            "advisory_penalties": {}, "locks": {}, "exclusions": {},
        }
        analysis = analyze_full_audit_cycles(disconnected, [0, 1, 2, 3])
        self.assertTrue(analysis["structurally_valid"])
        self.assertEqual(
            [component["component_id"] for component in analysis["components"]],
            ["component_0_1", "component_2_3"],
        )
        gated = gate_full_audit_decisions(disconnected, [0, 1, 2, 3])
        self.assertFalse(gated["full_audit_cycle_gate"]["accepted"])
        self.assertEqual(gated["full_audit_cycle_gate"]["component_count"], 2)
        self.assertEqual(gated["advisory_scores"], {})
        self.assertIn("gated_out_adjustments", gated)

    def test_independent_critic_can_accept_one_complete_cycle_and_reject_another(self):
        decisions = {
            "advisory_scores": {
                "0": {"1": 0.95}, "1": {"0": 0.95},
                "2": {"3": 0.95}, "3": {"2": 0.95},
            },
            "advisory_penalties": {}, "locks": {}, "exclusions": {},
        }
        response = {
            "summary": "The first cycle is visually supported; the second is ambiguous.",
            "components": [
                {
                    "component_id": "component_0_1",
                    "verdict": "accept",
                    "confidence": 0.92,
                    "reason": "Both proposed placements are visibly stronger.",
                    "row_checks": [
                        {
                            "segment_index": 0,
                            "baseline_candidate_index": 0,
                            "proposed_candidate_index": 1,
                            "verdict": "improves",
                            "reason": "Supported.",
                        },
                        {
                            "segment_index": 1,
                            "baseline_candidate_index": 1,
                            "proposed_candidate_index": 0,
                            "verdict": "improves",
                            "reason": "Supported.",
                        },
                    ],
                },
                {
                    "component_id": "component_2_3",
                    "verdict": "reject",
                    "confidence": 0.88,
                    "reason": "One placement is visually ambiguous.",
                    "row_checks": [
                        {
                            "segment_index": 2,
                            "baseline_candidate_index": 2,
                            "proposed_candidate_index": 3,
                            "verdict": "contradicts",
                            "reason": "Ambiguous.",
                        },
                        {
                            "segment_index": 3,
                            "baseline_candidate_index": 3,
                            "proposed_candidate_index": 2,
                            "verdict": "neutral",
                            "reason": "Supported.",
                        },
                    ],
                },
            ],
        }

        class FakeCriticBridge:
            def __init__(self, _tools, _items, _config):
                self.token = "token"
                self.audit_log = [
                    {
                        "tool": "inspect_candidate",
                        "arguments": {"candidate_index": candidate},
                        "result": {"contact_sheet_available": True},
                    }
                    for candidate in range(4)
                ]

            def start(self):
                return "http://127.0.0.1:1/tool"

            def close(self):
                return None

        def fake_run(command, **_kwargs):
            response_path = Path(command[command.index("--output-last-message") + 1])
            response_path.write_text(json.dumps(response), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        config = HybridAgentConfig(
            use_codex=True,
            review_scope="all",
            verify_assignment_cycles=True,
            cycle_critic_min_confidence=0.75,
        )
        tools = SimpleNamespace(metadata=[SimpleNamespace() for _ in range(4)])
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "hybrid_agentic.HybridAgentToolBridge", FakeCriticBridge
        ), mock.patch("hybrid_agentic.subprocess.run", side_effect=fake_run):
            result = codex_cycle_critic(
                tools,
                decisions,
                [0, 1, 2, 3],
                config,
                Path(directory),
                {"ready": True, "path": "/usr/bin/codex"},
            )
        self.assertEqual(set(result["advisory_scores"]), {"0", "1"})
        self.assertEqual(
            result["full_audit_cycle_gate"]["accepted_component_ids"],
            ["component_0_1"],
        )
        self.assertEqual(
            result["full_audit_cycle_gate"]["rejected_component_ids"],
            ["component_2_3"],
        )

    def test_hard_mode_keeps_constraints_and_still_applies_recommendations(self):
        matrix = np.array([[0.5, 0.4]], dtype=np.float64)
        decisions = {
            "locks": {"0": 1},
            "exclusions": {"0": [0]},
            "advisory_scores": {"0": {"1": 1.0}},
            "advisory_penalties": {},
        }
        refined, locks, exclusions = apply_agent_decisions(matrix, decisions, HybridAgentConfig(decision_mode="hard_lock", advisory_boost=0.1))
        self.assertEqual(locks, {0: 1})
        self.assertEqual(exclusions, {0: [0]})
        self.assertAlmostEqual(refined[0, 1], 0.5)

    def test_ambiguity_ranking_prioritizes_local_uncertainty_and_records_global_stability(self):
        matrix = np.array([
            [0.90, 0.89, 0.00, 0.00],
            [0.00, 0.90, 0.00, 0.00],
            [0.00, 0.00, 0.90, 0.70],
            [0.00, 0.00, 0.75, 0.90],
        ], dtype=np.float64)
        segments = [{"segment_id": index, "text": str(index)} for index in range(4)]
        ambiguous = find_ambiguous_segments(
            matrix,
            [0, 1, 2, 3],
            segments,
            threshold=1.0,
            max_segments=1,
        )
        self.assertEqual(ambiguous[0]["segment_index"], 0)
        self.assertAlmostEqual(ambiguous[0]["margin"], 0.01)
        self.assertAlmostEqual(ambiguous[0]["global_assignment_margin"], 0.91)

    def test_ambiguity_review_includes_rows_linked_by_alternative_assignment(self):
        matrix = np.array([
            [0.90, 0.89],
            [0.10, 0.95],
        ], dtype=np.float64)
        segments = [{"segment_id": index, "text": str(index)} for index in range(2)]
        ambiguous = find_ambiguous_segments(
            matrix,
            [0, 1],
            segments,
            threshold=0.02,
            max_segments=2,
        )
        self.assertEqual([item["segment_index"] for item in ambiguous], [0, 1])
        self.assertEqual(ambiguous[0]["selection_role"], "ambiguity_seed")
        self.assertEqual(ambiguous[1]["selection_role"], "assignment_link")
        self.assertEqual(ambiguous[1]["review_group_seed"], 0)

    def test_full_review_scope_includes_every_segment_and_ignores_cap(self):
        matrix = np.array([
            [0.95, 0.10],
            [0.10, 0.95],
        ], dtype=np.float64)
        segments = [{"segment_id": index, "text": str(index)} for index in range(2)]
        reviewed = find_ambiguous_segments(
            matrix,
            [0, 1],
            segments,
            threshold=0.0,
            max_segments=1,
            review_scope="all",
        )
        self.assertEqual({item["segment_index"] for item in reviewed}, {0, 1})
        self.assertTrue(all(item["selection_role"] == "global_audit" for item in reviewed))
        prompt = _codex_agent_prompt(reviewed, HybridAgentConfig(review_scope="all", max_agent_segments=1))
        self.assertIn("segment indices [0, 1]", prompt)
        self.assertIn("untrusted proposal", prompt)
        self.assertIn("complete one-to-one storyboard independently", prompt)
        self.assertIn("inspect every", prompt)
        self.assertIn("inspect_candidate", prompt)
        self.assertLess(prompt.index("get_candidate_catalog"), prompt.index("get_initial_assignment"))
        self.assertIn("draft an independent complete storyboard before calling get_initial_assignment", prompt)

    def test_codex_json_error_extraction_and_transient_classification(self):
        stdout = "\n".join([
            json.dumps({"type": "error", "message": "Selected model is at capacity. Please try a different model."}),
            json.dumps({"type": "turn.failed", "error": {"message": "Selected model is at capacity. Please try a different model."}}),
        ])
        message = _codex_failure_message(stdout, "Reading additional input from stdin")
        self.assertEqual(message, "Selected model is at capacity. Please try a different model.")
        self.assertTrue(_is_transient_codex_failure(message))
        self.assertFalse(_is_transient_codex_failure("MCP response failed schema validation"))

    def test_hard_constraints_require_confidence_and_same_pair_inspection(self):
        config = HybridAgentConfig(decision_mode="hard_lock")
        audit = [{
            "tool": "inspect_clip",
            "arguments": {"segment_index": 2, "candidate_index": 4},
            "result": {"contact_sheet_available": True},
        }]
        self.assertEqual(_hard_constraint_evidence(2, 4, 0.95, audit, config), (True, None))
        supported, reason = _hard_constraint_evidence(2, 5, 0.95, audit, config)
        self.assertFalse(supported)
        self.assertIn("contact-sheet inspection", reason)
        failed_audit = [{
            "tool": "inspect_clip",
            "arguments": {"segment_index": 2, "candidate_index": 4},
            "result": {"contact_sheet_available": False},
        }]
        supported, reason = _hard_constraint_evidence(2, 4, 0.95, failed_audit, config)
        self.assertFalse(supported)
        self.assertIn("successful", reason)
        supported, reason = _hard_constraint_evidence(2, 4, 0.89, audit, config)
        self.assertFalse(supported)
        self.assertIn("below", reason)

    def test_grid_results_preserve_execution_order_and_rank_only_post_hoc(self):
        def result(score):
            return HybridResult(
                config={"score_marker": score},
                exact_match_accuracy=score,
                top_3_accuracy=score,
                top_5_accuracy=score,
                mrr=score / 100.0,
                avg_similarity=0.0,
                baseline_exact_match_accuracy=10.0,
                baseline_top_3_accuracy=10.0,
                baseline_top_5_accuracy=10.0,
                baseline_mrr=0.1,
                exact_match_delta=score - 10.0,
                indexing_time=0.0,
                matching_time=0.0,
                total_time=0.0,
                reviewed_segments=0,
                codex_available=False,
                agent_mode="deterministic_heuristic",
                log_file="",
            )

        with tempfile.TemporaryDirectory() as directory:
            configs = [SimpleNamespace(output_dir="", video_dir="v", segments_file="s", ground_truth_file="g") for _ in range(2)]
            args = SimpleNamespace(output=directory, estimate_only=False, fast_grid=False, video_dir="v", segments="s", ground_truth="g")
            with mock.patch("hybrid_agentic.generate_grid_configs", return_value=configs), mock.patch(
                "hybrid_agentic.run_hybrid_benchmark",
                side_effect=[(result(10.0), {}), (result(90.0), {})],
            ):
                returned = run_grid(args)
            self.assertEqual([item.exact_match_accuracy for item in returned], [10.0, 90.0])
            payload = json.loads((Path(directory) / "hybrid_agentic_grid_search_results.json").read_text(encoding="utf-8"))
            self.assertEqual([item["exact_match_accuracy"] for item in payload["results"]], [10.0, 90.0])
            self.assertEqual(payload["posthoc_tuning_ranking"][0]["result_index"], 1)
            self.assertFalse(payload["evaluation_protocol"]["oracle_output_selection"])

    def test_fast_grid_uses_best_saved_backbone_and_only_eight_agent_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            hybrid_output = Path(directory) / "output" / "benchmark_6" / "hybrid_agentic"
            videoprism_output = hybrid_output.parent / "videoprism"
            videoprism_output.mkdir(parents=True)
            (videoprism_output / "videoprism_grid_search_results.json").write_text(json.dumps({
                "results": [{
                    "exact_match_accuracy": 50.0,
                    "config": {
                        "model_name": "videoprism_lvt_public_v1_base",
                        "num_frames": 8,
                        "resolution": 396,
                        "use_dual_softmax": False,
                        "prompt_mode": "template:cooking",
                        "query_mode": "context_window",
                    },
                }],
            }), encoding="utf-8")
            configs = generate_grid_configs(grid_args(hybrid_output))
            self.assertEqual(len(configs), 8)
            self.assertEqual({config.video.model_name for config in configs}, {"videoprism_lvt_public_v1_base"})
            self.assertEqual({config.video.resolution for config in configs}, {396})
            self.assertEqual({config.video.prompt_mode for config in configs}, {"template:cooking"})
            self.assertEqual({config.video.query_mode for config in configs}, {"original"})
            self.assertEqual({config.agent.max_agent_segments for config in configs}, {5, 10})
            self.assertEqual({config.agent.review_scope for config in configs}, {"all"})

    def test_weaker_grid_does_not_replace_best_known_canonical_result(self):
        def result(score):
            return HybridResult(
                config={"score_marker": score},
                exact_match_accuracy=score,
                top_3_accuracy=score,
                top_5_accuracy=score,
                mrr=score / 100.0,
                avg_similarity=0.0,
                baseline_exact_match_accuracy=50.0,
                baseline_top_3_accuracy=50.0,
                baseline_top_5_accuracy=50.0,
                baseline_mrr=0.5,
                exact_match_delta=score - 50.0,
                indexing_time=0.0,
                matching_time=0.0,
                total_time=0.0,
                reviewed_segments=10,
                codex_available=True,
                agent_mode="codex_mcp",
                log_file="",
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = {"results": [{"exact_match_accuracy": 62.5, "mrr": 0.71, "config": {"state": "previous"}}]}
            (output / "hybrid_agentic_grid_search_results_previous.json").write_text(json.dumps(previous), encoding="utf-8")
            configs = [SimpleNamespace(output_dir="", video_dir="v", segments_file="s", ground_truth_file="g")]
            args = SimpleNamespace(output=directory, estimate_only=False, fast_grid=False, video_dir="v", segments="s", ground_truth="g")
            with mock.patch("hybrid_agentic.generate_grid_configs", return_value=configs), mock.patch(
                "hybrid_agentic.run_hybrid_benchmark",
                return_value=(result(56.25), {}),
            ):
                run_grid(args)
            canonical = json.loads((output / "hybrid_agentic_grid_search_results.json").read_text(encoding="utf-8"))
            self.assertEqual(_payload_validation_score(canonical), (62.5, 0.71))

    def test_editor_hybrid_path_runs_real_refinement_contract(self):
        metadata = [
            SimpleNamespace(video_id="a", file_path="/tmp/a.mp4", window_start=0.0, duration=1.0),
            SimpleNamespace(video_id="b", file_path="/tmp/b.mp4", window_start=0.0, duration=1.0),
        ]
        segments = [
            SimpleNamespace(segment_id=0, text="first", duration=1.0, duration_multiplier=1.0, start_time=0.0, end_time=1.0),
            SimpleNamespace(segment_id=1, text="second", duration=1.0, duration_multiplier=1.0, start_time=1.0, end_time=2.0),
        ]
        config = {
            "agent_enabled": True,
            "agent_use_codex": False,
            "agent_decision_mode": "advisory",
            "agent_shortlist_size": 5,
            "agent_ambiguity_margin_threshold": 0.05,
            "agent_max_segments": 5,
            "agent_search_budget_per_segment": 4,
            "agent_allow_contact_sheet": False,
            "videoprism_model": "videoprism_lvt_public_v1_base",
            "num_frames": 8,
            "videoprism_resolution": 396,
            "use_dual_softmax": False,
            "prompt_mode": "template:cooking",
            "query_mode": "original",
            "score_normalization": "none",
            "csls_k": 5,
            "windowing": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimpleNamespace(
                session=SimpleNamespace(config=config, segments=segments),
                session_dir=Path(directory),
                matcher=FakeEditorMatcher(metadata),
                indexer=SimpleNamespace(),
            )
            sequence, diagnostics = EditorSessionManager()._run_hybrid_agentic_editor(
                runtime,
                [{"segment_id": 0, "text": "baseline first", "duration": 1.0}, {"segment_id": 1, "text": "baseline second", "duration": 1.0}],
            )
            self.assertEqual(len(sequence), 2)
            self.assertEqual(config["agent_runtime"], "no_ambiguous_segments")
            self.assertEqual(diagnostics["hybrid_agentic"]["agent_search_prompt_mode"], "none")
            self.assertTrue(Path(config["agent_log_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
