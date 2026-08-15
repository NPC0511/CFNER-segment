# Semantic Risk Graph Memory Plan

## 1. Goal

The current project already has a semantic-agent extension, but it is still close to:

```text
RDP + rule-based semantic retrieval + LLM verifier
```

The stronger target is:

```text
Semantic Risk Graph-Guided Continual NER
```

The core claim:

```text
Forgetting in class-incremental NER is not uniform.
New entity types mainly interfere with old types that are semantically similar,
contextually co-occurring, annotation-confusable, or historically confused.

Therefore, continual NER should explicitly build and update a semantic risk graph,
then use the graph to control distillation, prototype anchoring, contrastive separation,
pseudo-label thresholds, and optional LLM verification.
```

The LLM-Agent is not a token-level NER model. It is a graph builder and policy controller.

## 2. What Is the Semantic Risk Graph?

The graph is a task-aware directed graph:

```text
G_t = (V_t, E_t)
```

Nodes:

```text
entity types observed up to task t
```

Edges:

```text
new_type -> old_type
```

Edge meaning:

```text
How strongly learning the new type may interfere with, confuse, or forget the old type.
```

This is not only semantic similarity. It is continual-learning interference risk.

Example:

```json
{
  "task_id": 3,
  "new_types": ["organisation"],
  "old_types": ["person", "location"],
  "edges": [
    {
      "source": "organisation",
      "target": "location",
      "risk": 0.92,
      "risk_type": ["boundary_confusion", "context_overlap"],
      "reason": "Organisation names can contain place names or refer to sports teams named after cities.",
      "evidence_ids": ["rule_conll_org_loc_001", "example_conll_org_loc_002"]
    },
    {
      "source": "organisation",
      "target": "person",
      "risk": 0.71,
      "risk_type": ["co_occurrence"],
      "reason": "Person names frequently co-occur with organisations in news and sports contexts.",
      "evidence_ids": ["rule_conll_org_person_001"]
    }
  ]
}
```

## 3. Memory Types

The method should maintain four memories.

### 3.1 Semantic Memory

Current files:

```text
semantic_memory/conll2003/entity_types.json
semantic_memory/conll2003/annotation_rules.json
semantic_memory/conll2003/examples.jsonl
```

Contents:

```text
type definitions
annotation guidelines
positive examples
negative examples
confusable relations
```

This memory exists already, but it should be expanded beyond CoNLL2003 later.

### 3.2 Risk Graph Memory

New directory:

```text
semantic_cache/risk_graph/
```

Suggested files:

```text
semantic_cache/risk_graph/conll2003_task_1.json
semantic_cache/risk_graph/conll2003_task_2.json
semantic_cache/risk_graph/conll2003_task_3.json
semantic_cache/risk_graph/conll2003_latest.json
```

Each file stores:

```text
task id
new types
old types
risk edges
LLM/raw rule source
evidence
generated training policy
```

### 3.3 Prototype Memory

New directory:

```text
semantic_cache/prototype_memory/
```

Store compact old-class representation memory, not old training sentences.

Suggested content:

```text
label name
entity type
prototype vector
feature count
task id
optional variance
```

This preserves the strict incremental setting better than saving old raw samples.

### 3.4 Reflection Memory

New directory:

```text
semantic_cache/reflection/
```

After each task, record:

```text
old-class F1 before/after
observed forgetting
confusion pairs
risk edges that were confirmed
risk edges that were missed
LLM verifier statistics
```

Example:

```json
{
  "task_id": 3,
  "new_types": ["organisation"],
  "observed_forgetting": {
    "location": 4.8,
    "person": 2.1
  },
  "observed_confusion": [
    {
      "source": "organisation",
      "target": "location",
      "count": 37
    }
  ],
  "confirmed_edges": [
    ["organisation", "location"]
  ],
  "missed_edges": []
}
```

This is what makes the system agent-like: it does not only generate a plan, it remembers whether the plan matched actual training behavior.

## 4. How to Build the Risk Graph

Risk can be estimated from several signals:

```text
risk(new, old)
  = alpha * semantic_similarity
  + beta  * annotation_conflict
  + gamma * context_overlap
  + delta * historical_confusion
  + eta   * llm_judgment
```

### 4.1 Rule-Based First Version

Use existing `annotation_rules.json` confusion rules.

Mapping:

```text
confusion rule weight -> risk score
```

For example:

```text
weight 1.7 -> risk 0.85
weight 1.5 -> risk 0.75
weight 1.1 -> risk 0.55
```

This can be implemented without any extra model.

### 4.2 LLM-Agent Version

Use local Qwen once per task.

Input:

```text
new entity types
old entity types
retrieved definitions
annotation rules
examples
previous reflection memory
```

Output strict JSON:

```json
{
  "risk_edges": [
    {
      "source": "organisation",
      "target": "location",
      "risk": 0.92,
      "risk_type": ["boundary_confusion", "context_overlap"],
      "reason": "..."
    }
  ],
  "training_policy": {
    "distill_weights": {
      "location": 1.8
    },
    "prototype_anchor_weights": {
      "location": 2.0
    },
    "contrastive_pairs": [
      ["organisation", "location"]
    ],
    "pseudo_label_thresholds": {
      "location": 0.9
    },
    "verifier_budget": {
      "location": 2
    }
  }
}
```

LLM calls should be task-level and cached.

Do not call the LLM for every batch by default.

## 5. How the Graph Controls Training

### 5.1 Risk-Weighted Distillation

Already partially implemented.

Current code:

```text
semantic_plan["old_label_weights"] -> KL distillation class weights
```

Required upgrade:

```text
risk_edges -> old_label_weights
```

Higher risk means stronger old-class KL protection.

### 5.2 Risk-Weighted Prototype Anchoring

Use old prototypes as virtual old-class features.

For each old label prototype:

```text
old_proto -> current classifier -> old_label
```

Loss:

```text
L_proto_anchor = CE(classifier(old_proto), old_label)
```

Weight:

```text
prototype_anchor_weight(old_label) from risk graph
```

Why this matters:

```text
No old raw samples are stored.
Old class representation knowledge is replayed through prototypes.
High-risk old classes receive stronger anchoring.
```

### 5.3 Risk-Guided Contrastive Separation

For high-risk new-old pairs:

```text
new_type -> old_type
```

Encourage current new-class token/span representations to be:

```text
closer to their own new-class prototype
farther from high-risk old-class prototype
```

Possible loss:

```text
L_risk_contrast =
  risk(new, old) * max(0, margin + sim(h_new, proto_old) - sim(h_new, proto_new))
```

First implementation can use token-level features.

Later version can move to span-level features.

### 5.4 Risk-Aware Pseudo-Label Threshold

Current RDP accepts teacher/prototype pseudo labels relatively directly.

Upgrade:

```text
If old label belongs to a high-risk edge, require higher confidence.
If confidence is below threshold, keep O or send to LLM verifier.
```

This specifically targets the continual NER O-class problem:

```text
old entities in new-task data may be hidden as O.
wrong old pseudo labels can also pollute training.
```

### 5.5 Optional LLM Verifier

The existing verifier should become auxiliary.

Use it only when:

```text
edge risk is high
pseudo-label confidence is low
budget allows
```

The main method should not rely on frequent LLM calls.

## 6. Visualization Plan

Risk graph visualization is important for both debugging and paper presentation.

### 6.1 Task-Level Graph

Output:

```text
experiments/<exp_name>/<exp_id>/risk_graph_task_<t>.png
```

Visual encoding:

```text
old nodes: gray/blue
new nodes: orange/red
edge width: risk score
edge color: risk score
edge label: risk value
```

### 6.2 Risk vs Forgetting Plot

For each edge:

```text
x = predicted risk
y = observed F1 drop of target old type
```

If these correlate, the graph has predictive value.

Output:

```text
risk_vs_forgetting_task_<t>.png
```

### 6.3 Case Study Table

Example:

```text
New type      Old type      Risk    Reason                         F1 drop
organisation location      0.92    city/team ambiguity             4.8
organisation person        0.71    news co-occurrence              2.1
```

This should be saved as JSON/CSV and can be converted into a paper table.

## 7. Current Code Status

### 7.1 Completed

```text
[done] Task-level RiskGraph data structure and JSON cache.
[done] Rule risk graph from annotation confusion rules, including low background-risk edges for uncovered pairs.
[done] Local semantic retrieval of definitions, annotation rules, and examples.
[done] Per-task, cumulative, and final risk graph PNG outputs.
[done] Risk-weighted KL distillation.
[done] Risk-weighted prototype-anchor loss and prototype-memory serialization.
[done] Pairwise local-LLM risk generation for every new_type -> old_type pair.
[done] LLM output validation, raw-output cache, FP32 local inference path, and safe rule fallback.
[done] Rule + LLM hybrid initial risk.
[done] Task-end reflection memory: per-class F1, observed forgetting, token-level new-to-old confusion, and edge validation.
[done] Reflection update of later-task risks from historical forgetting and confusion.
[done] Online teacher new-to-old confusion from the current development set, used to raise pairwise risk before training.
[done] Two-tier risk-aware pseudo-label thresholds: candidates below the strict risk threshold are sent to verifier review; only the subset below a lower fallback confidence is reverted to O, preserving usable old-class supervision.
[done] Cumulative-development macro-F1 checkpoint selection, preventing new-class-only model selection from discarding old-class performance.
[done] Symmetric fallback for one-direction annotation-confusion rules when no explicit new-to-old rule exists.
[done] Risk-guided token-level contrastive loss between current new-type features and high-risk old prototypes.
[done] Per-stage experiment summary.csv.
```

Current implementation files:

```text
src/risk_graph.py
src/risk_graph_visualizer.py
src/prototype_memory.py
src/reflection_memory.py
src/result_summary.py
```

### 7.2 Current Risk Definition

The current system does not treat an LLM decimal as an unverified ground truth. It uses three explicit layers.

1. LLM component risk. The local LLM predicts three integer components for every pair:

```text
semantic_overlap    in {0, 1, 2}
annotation_conflict in {0, 1, 2}
context_overlap     in {0, 1, 2}
```

The application computes, rather than trusts an arbitrary LLM decimal:

```text
llm_risk = 0.05 + 0.95 *
  (0.30 * semantic_overlap
 + 0.45 * annotation_conflict
 + 0.25 * context_overlap) / 2
```

2. Rule + LLM initial risk. When annotation confusion rules exist:

```text
initial_risk =
  (rule_risk_weight * rule_risk + llm_risk_weight * llm_risk)
  / (rule_risk_weight + llm_risk_weight)
```

The default weights are `0.6` for rules and `0.4` for LLM. For a dataset without handcrafted confusion rules, set `rule_risk_weight: 0.0` and `llm_risk_weight: 1.0`.

3. Reflection-updated final risk. If a previous-task reflection contains evidence for the old target type:

```text
empirical_risk =
  (forgetting_weight * normalized_forgetting
 + confusion_weight * new_to_old_confusion_rate)
  / (forgetting_weight + confusion_weight)

blended_risk =
  (1 - reflection_update_weight) * initial_risk
  + reflection_update_weight * empirical_risk

reflection_risk = max(initial_risk, blended_risk, empirical_risk)
```

The default reflection update weight is `0.3`. The CoNLL2003 semantic configurations map a 5-point old-class F1 drop to maximum normalized forgetting risk. Each cached edge stores `rule_risk`, `llm_risk`, `initial_risk`, final `risk`, and, when used, `reflection_update` metadata.

4. Online teacher confusion. Before training task `t`, the old teacher is evaluated on the current development split. If it maps a gold new type to an old type, the matching edge risk is raised:

```text
model_risk = min(model_confusion_risk_scale * new_to_old_confusion_rate, 1.0)
final_risk = max(reflection_risk, model_risk)
```

This turns the graph from a semantic prior into a model-aware training controller.

### 7.3 LLM Accuracy Is Not Yet Proven

The LLM risk is a semantic prior, not a calibrated probability. Correctness must be evaluated against training outcomes, not against whether the explanation sounds plausible.

For each edge at task `t`:

```text
observed_forgetting(old, t)
  = F1_old_before_task_t - F1_old_after_task_t
```

The required validation table is:

```text
task_id, source_new, target_old,
llm_risk, rule_risk, initial_risk, final_risk,
observed_forgetting, confusion_rate
```

Required metrics:

```text
Spearman correlation between predicted risk and observed forgetting.
Top-K hit rate / precision@K for high-risk old types.
Mean forgetting for high-risk versus low-risk edge groups.
Old-class F1 and average-forgetting gains in ablation experiments.
```

`risk_analysis.py` writes `risk_vs_forgetting.csv`, `risk_prediction_metrics.json`, and task/global risk-versus-forgetting plots. A single CoNLL2003 order still has too few edges for reliable correlation; aggregate multiple class orders, random seeds, and datasets.

### 7.4 Still Missing

```text
[done] risk_vs_forgetting.csv, risk-vs-forgetting plots, Spearman correlation, and Top-K metrics.
[done] Risk-guided token-level contrastive loss.
[done] Two-tier risk-aware pseudo-label thresholds with a retained medium-confidence band.
[done] Risk-controlled LLM verifier scheduling: candidates are ranked by risk descending and confidence ascending; each task has a total budget and each batch has a safety cap. Under the two-tier pseudo-label policy, medium-confidence candidates remain as teacher labels unless the verifier explicitly rejects them; hard-dropped candidates require an explicit accept to be restored.
[partial] Deterministic policy generation now outputs contrastive_pairs and pseudo_label_thresholds.
[done] Risk graph now exports `verifier_label_risks` and the trainer executes the corresponding budget. Old cached plans are upgraded automatically when loaded.
[done] Offline LLM semantic-memory builder: a separate command validates an LLM draft, freezes a reviewed human-memory backbone for training, stores LLM additions separately as candidates, records hashes, and marks the version `pending_human_review`. Training remains read-only.
[done] Old-class vulnerability risk: task-end cumulative per-class F1 below a configured floor, and observed forgetting, produce a persisted vulnerability score. On the next task, it raises the relevant old-target edge within a bounded remaining-risk range before the new forgetting event occurs.
[todo] OntoNotes semantic memory with type definitions/examples but no handcrafted confusion rules.
[todo] i2b2 semantic memory with type definitions/examples and clinical ambiguity rules.
[todo] Full ablation runs across multiple orders, seeds, and datasets.
```

### 7.5 Development Status

The current implementation has a complete risk-graph training loop:

```text
semantic memory / rules / task-level LLM / teacher confusion / reflection
-> task-aware new_type -> old_type risk graph
-> deterministic training policy
-> risk-weighted distillation, prototype anchoring, pseudo-label filtering,
   and token-level contrastive separation
-> task-end forgetting and confusion reflection
-> next-task risk update
```

Implemented training-policy fields:

```text
old_label_weights
prototype_anchor_weights
pseudo_label_thresholds
contrastive_pairs
verifier_label_risks
old_type_vulnerability
```

The local LLM currently estimates risk components only. Python code translates
the validated risk graph into the policy above and schedules verifier calls
under deterministic risk and task-budget constraints; it does not allow the LLM
to choose arbitrary loss coefficients.

Formal evaluation is intentionally deferred until the model features above are
frozen. During development, one fixed-seed run is used only to confirm that a
new module is wired into training and produces its expected log fields.

### 7.6 Current Code Map

```text
semantic_memory/*.json(l)             reviewed definitions, rules, and examples
src/semantic_retriever.py             retrieves task-relevant semantic evidence
src/local_llm_agent.py                task-level LLM risk-graph generation
src/semantic_agent.py                 deterministic graph fusion and policy generation
src/risk_graph.py                     serializable task risk graph
src/prototype_memory.py               persistent old feature prototypes
src/trainer.py                        RDP losses plus graph-controlled protection/replay
src/llm_client.py                     cached local LLM pseudo-label verifier
src/reflection_memory.py              task-end F1, forgetting, vulnerability feedback
src/risk_analysis.py                  risk-versus-forgetting artifacts and metrics
main_CL.py                            incremental task orchestration and logging
build_semantic_memory.py              offline LLM memory draft/freeze command
```

### 7.7 Verification Status Before Formal Experiments

```text
[verified] v2 frozen memory is a byte-identical human-memory backbone.
[verified] task-level LLM risk graphs, teacher confusion, reflection, and vulnerability
           are consumed by the next task's training policy.
[verified] two-tier pseudo-label routing and verifier scheduling are active.
[verified] verifier backend now produces parseable decisions after dtype correction.
[pending] verifier quality is not established: the latest full run accepted every scheduled candidate.
[pending] LLM candidate memory has not passed human review or been incorporated into a v3 training memory.
[pending] risk prediction quality is not established; one CoNLL order has too few edges and weak correlation.
[pending] OntoNotes and i2b2 semantic memory/source specs are absent.
[pending] reproducible result management should use each experiment directory's summary.csv;
          the workspace-root summary.csv can be stale and must not be used across runs.
```

## 8. Updated Implementation Order

```text
1. [done] Rule-based risk graph.
2. [done] Risk graph JSON cache.
3. [done] Risk graph visualization.
4. [done] Risk-weighted KL distillation.
5. [done] Risk-weighted prototype anchoring.
6. [done] Task-end reflection memory.
7. [done] Reflection update of later-task risk graphs.
8. [done] Aggregate risk-versus-forgetting analysis and validate LLM risk prediction.
9. [done] Two-tier risk-aware pseudo-label thresholds with a retained medium-confidence band.
10. [done] Risk-guided token-level contrastive separation.
11. [done] Risk-controlled LLM verifier budget and low-confidence candidate routing.
12. [done] Offline LLM semantic-memory builder, human review, and frozen memory versioning.
13. [done] Old-class vulnerability risk from historical per-class F1 and forgetting.
14. [todo] Freeze the method, then run the full ablation study.
```

## 9. Minimal Viable Version

The first publishable prototype can be:

```text
Rule/LLM-induced Semantic Risk Graph
Risk-weighted KL distillation
Risk-weighted prototype anchoring
Risk graph visualization
Risk-vs-forgetting analysis
```

This is already stronger than:

```text
LLM verifier only
```

because the graph becomes:

```text
a structured memory object
a training controller
an explainable artifact
an analyzable predictor of forgetting
```

## 10. Ablation Plan

```text
RDP
RDP + weighted KL from semantic agent
RDP + risk graph weighted KL
RDP + risk graph weighted KL + prototype anchoring
RDP + risk graph weighted KL + prototype anchoring + reflection update
RDP + full method + optional LLM verifier
```

Main metrics:

```text
overall F1
macro F1
old-class F1
new-class F1
average forgetting
last-task average F1
risk-vs-forgetting correlation
confusing-pair F1
```

The most important analysis:

```text
Does a high-risk edge predict a larger old-class F1 drop?
Does prototype anchoring reduce that drop?
Does reflection improve later risk prediction?
```

## 11. Final Method Framing

Recommended name:

```text
SRG-CLNER: Semantic Risk Graph for Class-Incremental Named Entity Recognition
```

Alternative name:

```text
AgentSRG: LLM-Agent Induced Semantic Risk Graph for Continual NER
```

One-sentence description:

```text
We build a task-aware semantic risk graph over entity types, use an LLM-agent and semantic memory to estimate old-new interference risk, and use the graph to control distillation, prototype anchoring, contrastive separation, and pseudo-label validation under a strict no-old-sample continual NER setting.
```

## 12. Current Implementation Audit (2026-08-15)

This section is the operational source of truth for the current code.  It
supersedes earlier aspirational statements in this document when they disagree.

### 12.1 Repository and Local Git

```text
GitHub repository: https://github.com/NPC0511/CFNER-segment.git
Remote name:       origin
Default branch:    main
Local Git command: E:\Git\cmd\git.exe
Git command path:  E:\Git\cmd
```

`E:\Git\cmd` must be present in the Windows user or system `Path` for new
PowerShell sessions to resolve `git` directly.  The repository can always be
operated with the explicit command path above.

### 12.2 Actual Training Dataflow

```text
new task labels + previous model + semantic memory
    -> retrieve definitions, annotation rules, examples, and rule edges
    -> local LLM pairwise ranking of old targets (semantic interference prior)
    -> rule/LLM hybrid risk edges
    -> optional reflection and teacher-confusion updates
    -> training_policy
    -> RDP pseudo labels, weighted KL, prototype losses, contrastive loss,
       optional feature alignment, task-end evaluation and reflection
```

The graph edge `new_type -> old_type` is an *interference prior*, not a direct
measurement of catastrophic forgetting.  Semantic overlap, contextual
ambiguity, and annotation-boundary conflict can identify a candidate old type,
but cannot by themselves predict encoder parameter drift or the final old-class
F1 drop.  Real forgetting must be measured after training and fed back through
reflection.

### 12.3 Implemented and Active Components

```text
[active]   Rule risk from reviewed confusion rules.
[active]   Local-LLM pairwise ranking of old targets, then rule/LLM risk fusion.
[active]   Teacher new-to-old confusion evidence before each incremental task.
[active]   Risk threshold -> old-label KL weights, prototype-classifier weights,
           pseudo-label thresholds, contrastive pairs, and verifier candidates.
[active]   RDP pseudo-label, soft-label, and logits distillation losses.
[active]   Persistent class prototypes and prototype-classifier anchor loss.
[active]   Task-end per-class F1, forgetting, confusion, and risk-analysis output.
[active]   CosineLinear normalizes token features along hidden_dim (`dim=-1`).
[active]   Prototype distances use cosine distance, consistent with the classifier.
[configured] Risk feature alignment, reflection update, and old-class
             vulnerability are enabled in the current CoNLL semantic config.
```

### 12.4 Implemented but Not Connected to the Main Path

The following functions or parameters currently do not affect a normal
`main_CL.py` training run.  Their configuration flags must not be interpreted
as evidence that the feature is active.

```text
is_collect_prototype_similarity
    get_new_to_old_prototype_similarity_pairs() is not called and the returned
    pairs are never attached to graph edges.

is_use_instance_context_risk
    get_entity_context_examples() is not called and no extracted contexts are
    supplied to SemanticAgent or LocalLLMAgent.

_build_risk_evidence_cards()
    Exists, but is not invoked while constructing the LLM prompt.

_apply_risk_calibration()
    Exists, but build_plan() does not call it; the ridge calibrator therefore
    cannot alter an edge even when configured.

semantic_prior_max_delta
    Is accepted by SemanticAgent but not used in risk fusion.  It currently
    cannot limit an LLM-induced increase of rule risk.
```

### 12.5 Reflection Status

Reflection records are saved after a task with observed old-class F1 drop and
old-to-new confusion.  They affect a later plan only when all conditions hold:

```text
reflection_update_weight > 0
the configured reflection_cache_dir contains an earlier task reflection
the earlier reflection task_id is lower than the current task_id
```

The historical run represented by workspace-root `train.log` used
`reflection_update_weight: 0.0`; reflection was recorded but had no training
effect, and `final_risk` therefore matched `initial_risk`.  The current CoNLL
config sets `reflection_update_weight: 0.3`, which can affect only a new run
with compatible earlier reflection files.

### 12.6 Confirmed Risks and Defects

Priority order for the next code changes:

```text
P0  Prototype feature anchor currently selects high-confidence old-teacher
    labels without excluding current-task new-class tokens.  The old teacher
    cannot recognize a new class, so a high-confidence teacher error can pull a
    supervised new token toward an old prototype.  Restrict this loss to
    original O tokens (the hidden-old-token region), or otherwise explicitly
    exclude all current new-class labels.

P1  With is_train_by_steps=True, the scheduler is stepped both per batch and at
    the epoch boundary.  Scheduler ownership must be exclusive: step-based or
    epoch-based, never both.

P1  Prototype feature anchoring, risk feature alignment, and contrastive loss
    must log selected-token/pair counts and their weighted contribution.  A
    configured loss can otherwise silently be zero for an entire task.

P1  Reflection updates apply empirical old-class risk to every later edge that
    targets that old class.  This is a class vulnerability signal, not a
    source-target causal estimate; do not interpret it as pairwise ground truth.

P2  `eval()` is used for schedules and entity-list parsing.  Replace it with
    `ast.literal_eval()` or native YAML lists before using untrusted configs.

P2  Device use is hard-coded through `.cuda()`.  Replace it with device-aware
    `.to(device)` and tensor factories derived from an existing tensor.

P2  `torch.load()` uses pickle semantics.  Only load trusted checkpoints or use
    a restricted loading path where supported.
```

### 12.7 Recommended Next Implementation Order

```text
1. Correct the feature-anchor mask so it never anchors known new-class tokens
   to old prototypes; add selected-token diagnostics.
2. Make scheduler stepping mutually exclusive for epoch and step modes.
3. Wire prototype-similarity extraction and instance-context extraction into
   build_plan(), but treat them as evidence features rather than direct
   forgetting labels.
4. Call risk calibration only after enough prior reflections exist and log its
   mode, sample count, coefficients, and final edge deltas.
5. Enforce semantic_prior_max_delta in the fusion function, or remove the
   parameter to avoid misleading experiment configuration.
6. Run one fixed-seed smoke training job and verify every configured loss has a
   nonzero activation count before evaluating F1 claims.
7. After the mechanism is stable, run ablations and compare old-class F1 drop,
   feature drift, and risk-ranking quality against RDP.
```
