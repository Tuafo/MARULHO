from __future__ import annotations

from typing import Any

import torch


class ServiceReportingMixin:
    """Architecture and grounding-probe reporting helpers."""

    def architecture_summary(self) -> dict[str, Any]:
        """Return a current runtime-driven description of the active Terminus architecture."""
        with self._lock:
            model = self._trainer.model
            config = self._trainer.config
            sensory = self._brain_config.get("sensory") or {}
            autonomy = self._brain_config.get("autonomy") or {}
            predictive_enabled = bool(getattr(model, "predictive", None) is not None)
            cortex_snapshot = self.cortex_snapshot() if self._thought_loop is not None else {"enabled": False}
            layers: list[dict[str, Any]] = []
            layers.append({
                "id": "input_encoding",
                "name": "Input + Stream Ingestion",
                "enabled": True,
                "type": "input",
                "params": {
                    "input_dim": int(config.input_dim),
                    "representation": config.input_representation,
                    "background_sources": int(len(self._brain_source_runtimes)),
                    "background_routing": "focus_aware_allocation",
                    "sensory_sources": int(len(self._sensory_source_runtimes)),
                    "learned_chunking": bool(config.enable_learned_chunking),
                },
            })
            layers.append({
                "id": "competitive_routing",
                "name": "GPCSN Column Field",
                "enabled": True,
                "type": "core",
                "params": {
                    "n_columns": int(config.n_columns),
                    "k_routing": int(config.k_routing),
                    "plasticity_mode": config.plasticity_mode,
                    "plasticity_rule": config.plasticity_rule,
                },
            })
            layers.append({
                "id": "predictive_columns",
                "name": "Predictive Columns",
                "enabled": predictive_enabled,
                "type": "prediction",
                "params": {
                    "enabled": predictive_enabled,
                    "prediction_error_driven": predictive_enabled,
                } if predictive_enabled else {},
            })
            layers.append({
                "id": "context_prediction",
                "name": f"Context Attractor ({config.context_mode})",
                "enabled": model.context_layer is not None,
                "type": "context",
                "params": {
                    "context_mode": config.context_mode,
                },
            })
            layers.append({
                "id": "binding",
                "name": "Hypercube Binding + Hubs",
                "enabled": model.binding_layer is not None,
                "type": "binding",
                "params": {
                    "n_bindings": int(config.binding_n_bindings),
                    "fan_in": int(config.binding_fan_in),
                    "topology": type(model.binding_layer).__name__ if model.binding_layer is not None else "disabled",
                } if model.binding_layer is not None else {},
            })
            layers.append({
                "id": "abstraction",
                "name": "Abstraction Layer",
                "enabled": model.abstraction_layer is not None,
                "type": "abstraction",
                "params": {
                    "n_concepts": int(config.abstraction_n_concepts),
                } if model.abstraction_layer is not None else {},
            })
            layers.append({
                "id": "cross_modal_grounding",
                "name": "Real Cross-Modal Grounding",
                "enabled": model.cross_modal is not None,
                "type": "grounding",
                "params": {
                    "dim_visual": int(config.cross_modal_dim_visual),
                    "dim_audio": int(config.cross_modal_dim_audio),
                    "visual_confidence": float(model.cross_modal.visual_confidence.mean().item()) if model.cross_modal else 0.0,
                    "audio_confidence": float(model.cross_modal.audio_confidence.mean().item()) if model.cross_modal else 0.0,
                    "sensory_active": bool(sensory.get("enabled", False)),
                },
            })
            layers.append({
                "id": "memory_consolidation",
                "name": "Dual Memory + Consolidation",
                "enabled": True,
                "type": "memory",
                "params": {
                    "memory_capacity": int(config.memory_capacity),
                    "stc_tag_duration_strong": float(config.stc_tag_duration_strong),
                },
            })
            layers.append({
                "id": "nim_cortex",
                "name": "NIM Mind Layer",
                "enabled": bool(cortex_snapshot.get("enabled", False)),
                "type": "cortex",
                "params": {
                    "thoughts_generated": int(cortex_snapshot.get("thoughts_generated", 0) or 0),
                    "working_memory": bool(cortex_snapshot.get("working_memory") is not None),
                    "narrative_self": bool(cortex_snapshot.get("narrative_self") is not None),
                } if bool(cortex_snapshot.get("enabled", False)) else {},
            })
            layers.append({
                "id": "autonomy_guidance",
                "name": "Active Exploration + Grounded-Family-Summary Lineage-Reconvergent Divergence-Split Trajectory-Sensitive Compacted Age-Sensitive Consequence-Calibrated Real-Source Guidance",
                "enabled": bool(autonomy.get("enabled", False)) or bool(sensory.get("enabled", False)),
                "type": "autonomy",
                "params": {
                    "autonomy_enabled": bool(autonomy.get("enabled", False)),
                    "candidate_count": int(len(autonomy.get("candidate_bank", []))) if autonomy else 0,
                    "adaptive_focus_budgeting": bool(autonomy.get("enabled", False)),
                    "grounded_outcome_calibration": bool(autonomy.get("enabled", False)),
                    "evidence_provenance_credit": True,
                    "delayed_multi_turn_consequence_tracking": True,
                    "contradiction_decay_penalties": True,
                    "mixed_evidence_forgiveness_scheduling": True,
                    "age_sensitive_consequence_cooling": True,
                    "consequence_state_retirement": True,
                    "consequence_record_compaction": True,
                    "trajectory_sensitive_consequence_families": True,
                    "divergence_sensitive_consequence_splitting": True,
                    "lineage_aware_consequence_remerge": True,
                    "grounded_family_summary_calibration": True,
                    "sensory_enabled": bool(sensory.get("enabled", False)),
                    "items_per_episode": int(sensory.get("items_per_episode", 0)) if sensory else 0,
                },
            })
            return {
                "model_name": "Terminus",
                "core_name": "GPCSN",
                "version": "current",
                "family": "hybrid_snn_llm",
                "layers": layers,
                "config": {
                    "context_mode": config.context_mode,
                    "plasticity_rule": config.plasticity_rule,
                    "n_columns": int(config.n_columns),
                    "cross_modal": bool(model.cross_modal is not None),
                },
            }

    def run_grounding_probe(self) -> dict[str, Any]:
        """Run the 50-triple grounding probe and return results.

        When cross-modal grounding is enabled, the probe vector blends
        the routing key with the visual prediction from W_tv, so that
        concrete concepts with strong visual grounding produce distinct
        representations from abstract concepts (§8.7).
        """
        from hecsn.evaluation.grounding_probe import evaluate_grounding_probe
        with self._lock:
            trainer = self._trainer
            encoder = self._encoder
            cross_modal = trainer.model.cross_modal

            def _vector_fn(text: str) -> torch.Tensor:
                patterns = list(encoder.iter_char_patterns(text, window_size=8, learn=False))
                if not patterns:
                    return torch.zeros(trainer.config.n_columns, device=trainer.model.device)
                vecs = [trainer.model.routing_key_from_pattern(p[1]) for p in patterns]
                routing_key = torch.stack(vecs).mean(dim=0)

                if cross_modal is not None and routing_key.shape[0] == cross_modal.W_tv.shape[0]:
                    # Predict visual representation and blend with routing key
                    pred_visual = torch.mv(cross_modal.W_tv.T, routing_key)
                    visual_conf = float(cross_modal.visual_confidence.mean().item())
                    if pred_visual.norm() > 1e-6 and visual_conf > 0.01:
                        # Project visual prediction back to text space
                        visual_feedback = torch.mv(cross_modal.W_vt.T, pred_visual)
                        if visual_feedback.shape == routing_key.shape:
                            blend = min(0.3, visual_conf)
                            routing_key = (1.0 - blend) * routing_key + blend * visual_feedback
                return routing_key

            result = evaluate_grounding_probe(_vector_fn)
            return {
                "total_accuracy": float(result.total_accuracy),
                "concrete_accuracy": float(result.concrete_accuracy),
                "abstract_accuracy": float(result.abstract_accuracy),
                "concreteness_gap": float(result.concreteness_gap),
                "visual_text_accuracy": float(result.visual_text_accuracy),
                "audio_text_accuracy": float(result.audio_text_accuracy),
                "visual_text_count": result.visual_text_count,
                "audio_text_count": result.audio_text_count,
                "sample_count": result.total_count,
            }

