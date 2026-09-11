# V5 Research Traceability

[`references.yaml`](references.yaml) is the single authored metadata registry. The bibliographic
fields below mirror official arXiv abstract metadata queried on 2026-09-10. A research connection
can motivate or support one repository decision; no entry establishes the unified architecture as
an empirical result.

<a id="ref-heart"></a>

## REF-HEART — Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives

- Metadata status: `resolved`
- Supplied title: Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives
- Authors: Haibo Jin, Suijin Wang, Xucheng Yu, Haojing Luo, Haohan Wang
- Year: 2026
- arXiv: [`2609.01736`](https://arxiv.org/abs/2609.01736)
- DOI: `10.48550/arXiv.2609.01736`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-001`, `REC-008`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [epistemic process](../../gepa_mindfulness/core/epistemic_process.py), [runtime governance](../../gepa_mindfulness/verification/runtime_governance.py)

**Source demonstrates:** The source reports a Planner, Router, and Verifier harness built around
reusable tool primitives, together with task-completion and cost results on the evaluated
benchmarks.

**Repository inference:** The source motivates explicit verifier and runtime-role boundaries;
optimizer eligibility and authority enforcement remain independent repository design decisions.

**Maturity:** Resolved arXiv preprint; REC-001 and the repository contract for REC-008 are
implemented. Principal authentication and evidence dereferencing remain runtime-owner
responsibilities.

<a id="ref-pearl"></a>

## REF-PEARL — PEARL: Path-Entity Aligned Relational Learning with Contextual Subgraphs for Inductive Knowledge Graph Completion

- Metadata status: `resolved`
- Supplied title: Path-Entity Aligned Relational Learning with Contextual Subgraphs for Inductive Knowledge Graph Completion
- Authors: Yunchi Yang, Longlong Li, Cunquan Qu
- Year: 2026
- arXiv: [`2609.02216`](https://arxiv.org/abs/2609.02216)
- DOI: `10.48550/arXiv.2609.02216`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-011`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [robustness stripes](../../evaluation/cases/robustness_stripes.yaml)

**Source demonstrates:** The source reports context-conditioned relational paths, query-specific
subgraphs, and a contrastive objective evaluated on inductive knowledge-graph completion
benchmarks.

**Repository inference:** The contextual-path result inspires a disabled competing-hypothesis and
information-gain inquiry overlay; the repository does not adopt PEARL as its reasoning algorithm.

**Maturity:** Resolved arXiv preprint; REC-011 remains experimental and disabled by default.

<a id="ref-segos"></a>

## REF-SEGOS — SE-GoS: Self-Evolving Graph-of-Skills for Skill Library at Scale

- Metadata status: `resolved`
- Supplied title: Self-Evolving Graph-of-Skills for Skill Library at Scale
- Authors: Dawei Fu, Cheng Jiang, Sitian Qian, Huainan Wang, Zhongkai Hao
- Year: 2026
- arXiv: [`2609.08228`](https://arxiv.org/abs/2609.08228)
- DOI: `10.48550/arXiv.2609.08228`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-009`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward pipeline](../../gepa_mindfulness/training/reward_pipeline.py)

**Source demonstrates:** The source reports execution-trace-driven graph topology, edge-weight,
and description updates, including transfer measurements on a disjoint held-out split.

**Repository inference:** The reported execution feedback supports an execution-backed skill
lifecycle with held-out checks; the repository lifecycle remains a separately specified interface.

**Maturity:** Resolved arXiv preprint; REC-009 is accepted but incomplete.

<a id="ref-coevolve"></a>

## REF-COEVOLVE — Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails

- Metadata status: `resolved`
- Supplied title: Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails
- Authors: Zhou Yu, Bin Bi, Shiva Kumar Pentyala, Shubham Mehrotra, Sougata Chaudhuri, Shilpa Bhagavath, Zeyuan Chen, Ran Xu, Phil Mui, James Zhu, Sitaram Asur
- Year: 2026
- arXiv: [`2609.09134`](https://arxiv.org/abs/2609.09134)
- DOI: `10.48550/arXiv.2609.09134`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward pipeline](../../gepa_mindfulness/training/reward_pipeline.py)

**Source demonstrates:** The source reports that full expert-trajectory imitation regressed
performance under an evolved harness, while localized on-policy expert correction preserved
model-harness fit.

**Repository inference:** The result motivates fixed model and harness versions within an
evaluation episode and controlled evolution between episodes rather than online mutation during
scoring.

**Maturity:** Resolved arXiv preprint; REC-010 is accepted but incomplete.

<a id="ref-consistency"></a>

## REF-CONSISTENCY — Closing the Consistency Gap: Self-Evolving Agents That Learn to Stay on Course

- Metadata status: `resolved`
- Supplied title: Closing the Consistency Gap: Self-Evolving Agents That Learn to Stay on Course
- Authors: Evelyn Duesterwald, Benjamin Elder, Lilian Ngweta, Shashanka Ubaru, Malgorzata Zimon
- Year: 2026
- arXiv: [`2609.08832`](https://arxiv.org/abs/2609.08832)
- DOI: `10.48550/arXiv.2609.08832`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-005`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [robustness stripes](../../evaluation/cases/robustness_stripes.yaml)

**Source demonstrates:** The source reports a gap between mean per-run success and success across
all five repeated runs, plus an episodic-memory intervention evaluated on AppWorld.

**Repository inference:** The measured gap supports keeping repeat-sensitive consistency metrics
distinct from average correctness in the V5 case-by-stripe-by-repeat evaluator.

**Maturity:** Resolved arXiv preprint; REC-005 is accepted but incomplete.

<a id="ref-dsr"></a>

## REF-DSR — Beyond Top-$k$ Skill Retrieval: Diversity-Aware Skill Routing for LLM Agents

- Metadata status: `resolved`
- Supplied title: Beyond Top-k Skill Retrieval: Diversity-Aware Skill Routing for LLM Agents
- Authors: Wang Wei, Tiankai Yang, Samyadeep Basu, Hongjie Chen, Yue Zhao, Zhengzhong Tu, Xiyang Hu, Franck Dernoncourt, Ryan A. Rossi, Hoda Eldardiry
- Year: 2026
- arXiv: [`2609.05824`](https://arxiv.org/abs/2609.05824)
- DOI: `10.48550/arXiv.2609.05824`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-009`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward pipeline](../../gepa_mindfulness/training/reward_pipeline.py)

**Source demonstrates:** The source reports diversity-aware skill reranking that balances query
relevance and non-redundancy, with recall and coverage results on multi-skill queries.

**Repository inference:** The result inspires future bounded skill routing within REC-009; the
current repository does not implement the paper's Determinantal Point Process reranker.

**Maturity:** Resolved arXiv preprint; REC-009 is accepted but incomplete.

<a id="ref-edgemem"></a>

## REF-EDGEMEM — EdgeMem: LLM-Free Agent Memory Construction and Retrieval via Evidence-Preserving Multi-Anchor Hypergraph

- Metadata status: `resolved`
- Supplied title: EdgeMem: LLM-Free Agent Memory Construction and Retrieval via Evidence-Preserving Multi-Anchor Hypergraph
- Authors: Zeyang Cui, Jiannong Cao, Zhiyuan Wen, Bo Yuan, Junlan Feng, Shengyuan Chen
- Year: 2026
- arXiv: [`2609.05553`](https://arxiv.org/abs/2609.05553)
- DOI: `10.48550/arXiv.2609.05553`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-006`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [verification state](../../gepa_mindfulness/verification/state.py)

**Source demonstrates:** The source reports a memory system that preserves original interaction
turns and retrieves source evidence through content, temporal, and episodic anchors.

**Repository inference:** Evidence-preserving retrieval supports the repository boundary between
append-only raw evidence and derived belief or memory representations.

**Maturity:** Resolved arXiv preprint; the repository contract for REC-006 is implemented.
Evidence dereferencing and issuer authentication remain external responsibilities.

<a id="ref-graphmem"></a>

## REF-GRAPHMEM — Graph-Based Personalized Memory for LLM Agents: Representation, Evolution, Retrieval, and Evaluation

- Metadata status: `resolved`
- Supplied title: Graph-Based Personalized Memory for LLM Agents: Representation, Evolution, Retrieval, and Evaluation
- Authors: Dac Duy Anh Nguyen, Zhangchi Qiu, Shigeng Chen, Alan Wee-Chung Liew
- Year: 2026
- arXiv: [`2609.08599`](https://arxiv.org/abs/2609.08599)
- DOI: `10.48550/arXiv.2609.08599`
- Venue/status: Accepted by ICKG 2026
- Recommendations influenced: `REC-006`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [verification state](../../gepa_mindfulness/verification/state.py)

**Source demonstrates:** The survey reports a lifecycle view of graph-based personalized memory
spanning representation, evolution, retrieval, and evaluation with temporal and evidence links.

**Repository inference:** The survey motivates explicit types for observed state, evidence, and
derived memory rather than allowing one representation to stand for all three.

**Maturity:** Resolved survey accepted by ICKG 2026; the repository contract for REC-006 is
implemented. Evidence dereferencing and issuer authentication remain external responsibilities.

<a id="ref-sheaves"></a>

## REF-SHEAVES — Time-Varying Data as Sheaves: an Invitation to Narratives

- Metadata status: `resolved`
- Supplied title: Time-Varying Data as Sheaves: an Invitation to Narratives
- Authors: Wilmer Leal, Benjamin Merlin Bumpus, Jana K. Nickel, Johan García, James Fairbanks, Warren Dixon
- Year: 2026
- arXiv: [`2609.09056`](https://arxiv.org/abs/2609.09056)
- DOI: `10.48550/arXiv.2609.09056`
- Venue/status: Book chapter
- Recommendations influenced: `REC-002`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [logging schema](../../src/mindful_trace_gepa/logging_schema.py)

**Source demonstrates:** The source reports an abstract framework for time-varying objects and
discusses information loss across temporal representations and switching multi-agent topologies.

**Repository inference:** The temporal-representation perspective inspires explicit event history
and immutable linkages; the repository does not adopt a sheaf formalism.

**Maturity:** Resolved arXiv book chapter; REC-002 is accepted but incomplete.

<a id="ref-hero"></a>

## REF-HERO — Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Efficient LLM Inference

- Metadata status: `resolved`
- Supplied title: HeRo: History-Aware Routing for Efficient LLM Inference
- Authors: Hongjin Lin, Wentao Wan, Keze Wang
- Year: 2026
- arXiv: [`2609.08189`](https://arxiv.org/abs/2609.08189)
- DOI: `10.48550/arXiv.2609.08189`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [logging schema](../../src/mindful_trace_gepa/logging_schema.py)

**Source demonstrates:** The source reports a layer-routing mechanism that conditions current
choices on accumulated routing history and measures performance under reduced parameter use.

**Repository inference:** The result suggests retaining history for sequential decision records;
the repository does not implement the paper's model-layer router.

**Maturity:** Resolved arXiv preprint; REC-002 is accepted but incomplete.

<a id="ref-sae"></a>

## REF-SAE — SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?

- Metadata status: `resolved`
- Supplied title: SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?
- Authors: Yuqiao Tan, Shizhu He, Jun Zhao, Kang Liu
- Year: 2026
- arXiv: [`2609.09113`](https://arxiv.org/abs/2609.09113)
- DOI: `10.48550/arXiv.2609.09113`
- Venue/status: Preprint. Work in Progress
- Recommendations influenced: `REC-014`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward implementation](../../gepa_mindfulness/core/rewards.py)

**Source demonstrates:** The source reports a benchmark for agent-led sparse-autoencoder feature
discovery and a gap from experts in causal steering and interpretation of experimental
measurements.

**Repository inference:** The reported limitations support keeping mechanistic signals in an
experimental diagnostic audit until independent causal and behavioral grounding exists.

**Maturity:** Resolved work-in-progress preprint; REC-014 remains experimental.

<a id="ref-biometric-mem"></a>

## REF-BIOMETRIC-MEM — Personalizing LLM Agent Memory Using Biometrics

- Metadata status: `resolved`
- Supplied title: Personalizing LLM Agent Memory Using Biometrics
- Authors: Yanhong Qian, Qingguo Meng, Shihao Ding, Xingbo Dong, Zhe Jin, Hanrui Wang, Isao Echizen
- Year: 2026
- arXiv: [`2609.08558`](https://arxiv.org/abs/2609.08558)
- DOI: `10.48550/arXiv.2609.08558`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-008`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [logging schema](../../src/mindful_trace_gepa/logging_schema.py)

**Source demonstrates:** The source reports a multi-user memory architecture that gates a
retrieval candidate pool by biometric matching before semantic ranking.

**Repository inference:** The access-control result inspires explicit authorization evidence at
runtime; it does not recommend biometric collection for this repository.

**Maturity:** Resolved arXiv preprint; the process-local authority contract for REC-008 is
implemented. It does not authenticate human identities, grant issuers, or evidence issuers.

<a id="ref-hoh"></a>

## REF-HOH — Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement

- Metadata status: `resolved`
- Supplied title: Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement
- Authors: Haoyang Yan, Min-le Su, Hangfan Zhang, Zhanhao Li, Chen Zhang, Shao Zhang, Yang Chen, Lei Bai, Shuyue Hu
- Year: 2026
- arXiv: [`2609.01481`](https://arxiv.org/abs/2609.01481)
- DOI: `10.48550/arXiv.2609.01481`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward pipeline](../../gepa_mindfulness/training/reward_pipeline.py)

**Source demonstrates:** The source reports iterative planning, coding, and testing loops with
small verifiable increments and separation between implementation-time tests and independent
evaluation.

**Repository inference:** The process motivates controlled between-episode harness evolution and
held-out validation rather than changing a scored episode in place.

**Maturity:** Resolved arXiv preprint; REC-010 is accepted but incomplete.

<a id="ref-agentscope"></a>

## REF-AGENTSCOPE — Diagnosing with Insights: Structured Analysis of Agent Failures via Behavioral Abstractions

- Metadata status: `resolved`
- Supplied title: Diagnosing with Insights: Structured Analysis of Agent Failures via Behavioral Abstractions
- Authors: Jiayi Bi, Yanjie Gao, Yuanmin Xie, Liqun Li, Tianyin Xu, Fan Yang, Mao Yang
- Year: 2026
- arXiv: [`2609.02371`](https://arxiv.org/abs/2609.02371)
- DOI: `10.48550/arXiv.2609.02371`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-007`, `REC-013`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [failure graph](../../gepa_mindfulness/verification/failure_graph.py)

**Source demonstrates:** The source reports structured behavioral abstractions and neural
invariants for locating and classifying failures in long agent trajectories.

**Repository inference:** The approach motivates inspectable failure records and orchestration
scopes; causal labels still require repository verifier evidence.

**Maturity:** Resolved arXiv preprint; the repository contract for REC-007 is implemented, while
REC-013 remains experimental. Verifier identity authentication remains external to the graph.

<a id="ref-repotoskill"></a>

## REF-REPOTOSKILL — Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills

- Metadata status: `resolved`
- Supplied title: Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills
- Authors: Jianlyu Chen, Yuyang Hu, Hongjin Qian, Jiawei Liu, Wenqing Wei, Xiaolong Chen, Defu Lian, Zhicheng Dou, Chaozhuo Li, Qiwei Ye, Zheng Liu
- Year: 2026
- arXiv: [`2609.02749`](https://arxiv.org/abs/2609.02749)
- DOI: `10.48550/arXiv.2609.02749`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-009`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward pipeline](../../gepa_mindfulness/training/reward_pipeline.py)

**Source demonstrates:** The source reports repository-to-skill distillation and benchmark gains
from a verified skill library under a fixed agent setup and execution budget.

**Repository inference:** The result supports treating procedural knowledge as a reviewable skill
artifact whose value depends on execution evidence and held-out evaluation.

**Maturity:** Resolved arXiv preprint; REC-009 is accepted but incomplete.

<a id="ref-skillglow"></a>

## REF-SKILLGLOW — SkillGLoW: Procedural-Family Skill Consolidation for Self-Improving Agents on Long-Horizon Task Streams

- Metadata status: `resolved`
- Supplied title: SkillGLoW: Procedural-Family Skill Consolidation for Self-Improving Agents on Long-Horizon Task Streams
- Authors: Ao Yan, Xin Zhang, Jiawei Du, Joey Tianyi Zhou
- Year: 2026
- arXiv: [`2609.02217`](https://arxiv.org/abs/2609.02217)
- DOI: `10.48550/arXiv.2609.02217`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-009`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward pipeline](../../gepa_mindfulness/training/reward_pipeline.py)

**Source demonstrates:** The source reports consolidation of task-local skills into procedural
families and an execution-based commit gate evaluated across long-horizon task streams.

**Repository inference:** The result supports procedural-family records, execution evidence,
validation, commit, and rollback stages in the planned skill lifecycle.

**Maturity:** Resolved arXiv preprint; REC-009 is accepted but incomplete.

<a id="ref-maskills"></a>

## REF-MASKILLS — MASkills: Continual Skills Optimization for Multi-Agent LLM Systems

- Metadata status: `resolved`
- Supplied title: MASkills: Continual Skills Optimization for Multi-Agent LLM Systems
- Authors: Huaiyuan Yao, Xiaoou Liu, Charles Fleming, Tianlong Chen, Hua Wei
- Year: 2026
- arXiv: [`2609.02094`](https://arxiv.org/abs/2609.02094)
- DOI: `10.48550/arXiv.2609.02094`
- Venue/status: EMNLP 2026 Findings
- Recommendations influenced: `REC-012`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [reward pipeline](../../gepa_mindfulness/training/reward_pipeline.py)

**Source demonstrates:** The source reports skill-conditioned credit assignment and hierarchical
aggregation for continual skill optimization in evaluated multi-agent systems.

**Repository inference:** The result inspires evaluation of bounded multi-agent adaptation; it
does not establish the repository's proposed topology codebook.

**Maturity:** Resolved EMNLP 2026 Findings paper; REC-012 remains experimental.

<a id="ref-wmllm"></a>

## REF-WMLLM — WMLLM: Self-Evolving Optimization Agents via Predict-Then-Act World Modeling

- Metadata status: `resolved`
- Supplied title: WMLLM: Self-Evolving Optimization Agents via Predict-Then-Act World Modeling
- Authors: Zhongzheng Li, Qingsong Ran, Shikun Feng, Nian Ran, Wenhao Li, Xiaoyuan Zhang, Yue Wang, Xiaoguang Zhao
- Year: 2026
- arXiv: [`2609.01608`](https://arxiv.org/abs/2609.01608)
- DOI: `10.48550/arXiv.2609.01608`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [logging schema](../../src/mindful_trace_gepa/logging_schema.py)

**Source demonstrates:** The source reports a predict-then-act framework that forecasts candidate
directions before costly evaluation in black-box optimization experiments.

**Repository inference:** The sequence motivates committing a prediction before action so later
outcomes can evaluate it; the repository applies that idea in a different domain.

**Maturity:** Resolved arXiv preprint; REC-002 is accepted but incomplete.

<a id="ref-dwm"></a>

## REF-DWM — Discriminative World Models for Web Agents

- Metadata status: `resolved`
- Supplied title: Discriminative World Models for Web Agents
- Authors: Kelvin Li, Dhruv Pendharkar, Anish Pahilajani, Chuyi Shang, Leon Oks, Leonid Karlinsky, Rogerio Feris, Trevor Darrell, Roei Herzig
- Year: 2026
- arXiv: [`2609.02885`](https://arxiv.org/abs/2609.02885)
- DOI: `10.48550/arXiv.2609.02885`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [logging schema](../../src/mindful_trace_gepa/logging_schema.py)

**Source demonstrates:** The source reports predicted-state matching for distinguishing outcomes
of alternative web actions and evaluates its effect on action ranking and end-to-end task success.

**Repository inference:** The result motivates distinct prediction, action, observed-outcome, and
verification records; the repository does not adopt the paper's learned world model.

**Maturity:** Resolved arXiv preprint; REC-002 is accepted but incomplete.

<a id="ref-lexical-perturb"></a>

## REF-LEXICAL-PERTURB — Lexical Perturbations Disrupt LLM Reasoning: An Empirical Study of Attention Diversion

- Metadata status: `resolved`
- Supplied title: Lexical Perturbations Disrupt LLM Reasoning: An Empirical Study of Attention Diversion
- Authors: Jiaqian Zhu, Yang Zhang, Junhua Ding, Xiaowei Yu
- Year: 2026
- arXiv: [`2608.22140`](https://arxiv.org/abs/2608.22140)
- DOI: `10.48550/arXiv.2608.22140`
- Venue/status: Accepted to EMNLP 2026 (Main Conference)
- Recommendations influenced: `REC-005`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [robustness stripes](../../evaluation/cases/robustness_stripes.yaml)

**Source demonstrates:** The source reports accuracy degradation under keyboard noise and
character swaps and links the measured effect to token fragmentation and attention diversion in
the tested models.

**Repository inference:** The result supports representation-perturbation stripes and separate
correctness and consistency measurements; it does not select a universal repair method.

**Maturity:** Resolved EMNLP 2026 main-conference paper; REC-005 is accepted but incomplete.

<a id="ref-tokenizer-betrayal"></a>

## REF-TOKENIZER-BETRAYAL — Say Anything but This: When Tokenizer Betrays Reasoning in LLMs

- Metadata status: `resolved`
- Supplied title: Say Anything but This: When Tokenizer Betrays Reasoning in LLMs
- Authors: Navid Ayoobi, Marcus I Armstrong, Arjun Mukherjee
- Year: 2026
- arXiv: [`2601.14658`](https://arxiv.org/abs/2601.14658)
- DOI: `10.48550/arXiv.2601.14658`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-005`
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [robustness stripes](../../evaluation/cases/robustness_stripes.yaml)

**Source demonstrates:** The source reports reasoning failures associated with non-unique token
encodings of identical surface strings and categorizes tokenizer-induced phantom edits.

**Repository inference:** The result motivates representation-robustness evaluation while leaving
tokenizer-level remediation outside the repository's current application-layer scope.

**Maturity:** Resolved arXiv preprint; REC-005 is accepted but incomplete.
