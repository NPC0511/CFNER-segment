# BAN-SSM Ideas Transfer Plan for CFNER/RDP

## 1. Objective

Improve the FG12 first-task NER performance without changing the official BIO evaluation protocol or replacing the current RDP incremental-learning framework.

The current FG12 base task has strong performance on CARDINAL, DATE, MONEY, and ORDINAL, but weak performance on LANGUAGE, NORP, EVENT, GPE, LAW, and LOC. The transfer should therefore target two error families:

1. Boundary errors: the model identifies an entity incompletely or includes neighboring tokens.
2. Type errors: the boundary is plausible, but the entity is assigned a confusable type.

## 2. What To Transfer

The source paper proposes a span-based dual-path model. Its complete architecture is not directly compatible with this project because this project uses BIO token classification. We transfer the underlying ideas instead:

- Boundary-aware supervision: explicitly teach the model where entity spans begin.
- Content-adaptive local neighborhoods: allow a token to selectively use nearby token representations.
- Type consistency and separation: bring an entity representation toward its own type and away from semantically confusable types.
- Gated residual fusion: add auxiliary information without overwriting the existing BERT representation.

We will not initially transfer the following components:

- Two-dimensional span matrices.
- SS2D or multi-directional span-matrix scans.
- Span-level BCE decoding.
- A new span-based evaluation protocol.

## 3. Target Architecture

```text
BERT encoder
    |
    +--> local neighborhood enhancement
    |        |
    |        +--> gated residual fusion
    |
    +--> BIO classifier (existing classifier interface)
    |
    +--> boundary auxiliary head
    |
    +--> type prototype / hard-negative loss
```

The main BIO classifier remains the official prediction head. Auxiliary modules only affect the training representation and losses.

## 4. Stage A: Error Attribution

Before adding a module, run FG12 with the current baseline and record:

- Per-type precision, recall, and F1.
- Gold entity count and predicted entity count per type.
- BIO-tag confusion matrix.
- Entity-type confusion matrix.
- O absorption rate for each weak type.
- Boundary-only errors and type-only errors.

Priority types:

```text
LANGUAGE, NORP, EVENT, GPE, FAC, LAW, LOC
```

Decision rules:

- Low recall with many predictions as O: prioritize sampling, O control, and boundary supervision.
- Similar predicted and gold counts but low F1: prioritize hard-negative type separation.
- Correct type but wrong span extent: prioritize the boundary auxiliary task and local neighborhood module.

This stage is diagnostic only and must not change the official metric.

## 5. Stage B: Boundary Auxiliary Task

Add a small boundary head over the fused token features.

Initial boundary labels:

```text
B-X -> 1
I-X -> 0
O   -> 0
```

Loss:

```text
L_boundary = BCEWithLogitsLoss(boundary_logits, boundary_labels)
L_total = L_main + lambda_boundary * L_boundary
```

Initial values:

```text
lambda_boundary = 0.1, 0.2, 0.3
```

The boundary task is intended to improve entity start detection and reduce O absorption. It should be evaluated first without the local module or prototype loss.

## 6. Stage C: Hard-Negative Type Separation

Reuse the semantic memory's confusion relations as training pairs during the first task. In the first task these relations are not old-class protection signals; they are hard-negative type relations.

Initial hard-negative pairs:

```text
LANGUAGE - NORP
LANGUAGE - GPE
NORP - GPE
GPE - LOC
GPE - FAC
GPE - ORG
EVENT - ORG
CARDINAL - MONEY
CARDINAL - ORDINAL
DATE - TIME
```

For each entity token representation h, compute a positive type prototype p_pos and a confusable negative prototype p_neg:

```text
L_type = max(0, margin + sim(h, p_neg) - sim(h, p_pos))
```

Initial values:

```text
margin = 0.1, 0.2, 0.3
lambda_type = 0.05, 0.1, 0.2
```

Only entity tokens should contribute to prototype estimation. O tokens must not be included in entity prototypes.

The prototype loss must be safe when a type has too few valid tokens in a batch. Such pairs should be skipped, not assigned an artificial prototype.

## 7. Stage D: Gated Local Neighborhood Enhancement

Add a lightweight one-dimensional local attention block over BERT token features. Start with a window radius of 2.

For each token i:

```text
n_i = LocalAttention(h_(i-2), ..., h_i, ..., h_(i+2))
gate_i = sigmoid(W_g [h_i ; n_i] + b_g)
h_fused_i = h_i + gate_i * n_i
```

Use padding and sentence masks so tokens from different sentences cannot attend to each other. Initialize the gate conservatively, for example with a negative bias, so the new module starts as a small residual correction.

The local block is intended to help EVENT, FAC, GPE, LOC, LAW, and WORK_OF_ART when their errors depend on nearby context or boundary cues.

## 8. Stage E: Combined Model

After the isolated ablations, combine only components that show independent benefit:

```text
L = L_BIO
  + lambda_boundary * L_boundary
  + lambda_type * L_type
```

If the local module is enabled:

```text
h_fused = h + gate * h_local
```

For incremental tasks, retain the current RDP and semantic losses:

```text
L_incremental = L_RDP
                + lambda_boundary * L_boundary
                + lambda_type * L_type
                + existing semantic/prototype/risk losses
```

The first implementation should not change `dim=1`, the official seqeval protocol, the task split, or the existing risk graph semantics.

## 9. Experiment Matrix

Run each condition on FG12-PG1 first:

1. Current CosineLinear baseline.
2. Current code with the ordinary Linear classifier control.
3. Baseline plus boundary loss.
4. Baseline plus hard-negative type loss.
5. Baseline plus local neighborhood block.
6. Boundary loss plus hard-negative type loss.
7. Full selected combination.

After FG12 is validated, run the selected configuration on:

- FG8-PG1.
- FG8-PG2.
- CoNLL2003.

Each experiment must use independent checkpoint and semantic-cache directories. Keep seed and all non-target hyperparameters fixed within each comparison.

## 10. Evaluation Criteria

The main first-task criteria are:

- FG12 stage-1 macro-F1.
- LANGUAGE F1.
- NORP F1.
- EVENT, GPE, LAW, LOC, and FAC F1.
- Stage-1 micro-F1, to detect precision/recall trade-offs.

The method is useful only if it improves weak types without materially damaging strong types:

```text
LANGUAGE and NORP must improve substantially.
EVENT/GPE/LAW/LOC should improve or remain stable.
CARDINAL/DATE/MONEY/ORDINAL should not drop materially.
```

For incremental evaluation, also check:

- old-class average F1;
- new-class average F1;
- cumulative macro-F1;
- forgetting per task;
- risk-filtered pseudo-label counts.

## 11. Recommended Implementation Order

1. Add and validate error attribution outputs.
2. Run the ordinary Linear control already implemented in the codebase.
3. Add the boundary auxiliary head.
4. Add semantic-memory hard-negative prototype loss.
5. Add the gated local neighborhood block.
6. Combine only validated components.
7. Extend the selected base-task configuration to incremental tasks.

Do not implement span matrices or SS2D until the lightweight BIO-compatible experiments demonstrate that boundary/context modeling is the actual bottleneck.

## 12. Success Target

The first milestone is not immediately reaching 70-80 macro-F1. The first milestone is to raise the FG12 stage-1 macro-F1 from approximately 53 while specifically improving LANGUAGE and NORP, without sacrificing CARDINAL, DATE, MONEY, and ORDINAL.

Once the base model is reliable, the same improvements can be evaluated in the incremental setting. This separates base-task learning gains from continual-learning gains and prevents semantic-memory changes from masking a weak first-stage model.
