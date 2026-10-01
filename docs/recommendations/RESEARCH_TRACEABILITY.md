# V5 Research Traceability

[`references.yaml`](references.yaml) is the single authored metadata registry. The bibliographic
fields below mirror primary metadata: the original arXiv entries were queried on 2026-09-10;
REF-KALMAN and the four PR-2 sources (FTA, PINNForge, C3-JEPA and AI Neuroscientist) were checked
against primary texts on 2026-09-30, as were PR-3's GRUET, Dual-Frontier and DEEPO sources.
PR-4 adds CI metadata from the author publication list and the Unanimity/WSQEM primary texts,
also checked on 2026-09-30. PR-5 adds primary-text checks for MemCalib, JitMem, CompKV,
Qwen-Planner-Agent, Share-Borne AI Virus and A2M on the same date. PR-6 adds CDR and TTSE primary-text checks on 2026-09-30. PR-7 adds the six System-One sources checked on 2026-10-01. A research connection
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
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [controlled evolution](../controlled_evolution.md), [skill lifecycle](../../gepa_mindfulness/skill_lifecycle.py)

**Source demonstrates:** The source reports execution-trace-driven graph topology, edge-weight,
and description updates, including transfer measurements on a disjoint held-out split.

**Repository inference:** The reported execution feedback supports an execution-backed skill
lifecycle with held-out checks; the repository lifecycle remains a separately specified interface.

**Maturity:** Resolved arXiv preprint; the bounded catalog lifecycle portion of REC-009 is
implemented. The repository does not claim the paper's complete skill system or empirical results.

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
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [controlled evolution](../controlled_evolution.md), [learning surfaces](../../gepa_mindfulness/learning_surfaces.py), [coevolution](../../gepa_mindfulness/coevolution.py)

**Source demonstrates:** The source reports that full expert-trajectory imitation regressed
performance under an evolved harness, while localized on-policy expert correction preserved
model-harness fit.

**Repository inference:** The result motivates fixed model and harness versions within an
evaluation episode and controlled evolution between episodes rather than online mutation during
scoring.

**Maturity:** Resolved arXiv preprint; the bounded epoch and candidate-record portion of REC-010
is implemented. The repository does not reproduce the paper's experiments or deploy a system.

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

**Maturity:** Resolved arXiv preprint; the repository's deterministic
case-by-stripe-by-repeat planner, record, and summary contracts for REC-005 are implemented. No
paper result is reproduced.

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
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [controlled evolution](../controlled_evolution.md), [skill lifecycle](../../gepa_mindfulness/skill_lifecycle.py)

**Source demonstrates:** The source reports diversity-aware skill reranking that balances query
relevance and non-redundancy, with recall and coverage results on multi-skill queries.

**Repository inference:** The result inspires future bounded skill routing within REC-009; the
current repository does not implement the paper's Determinantal Point Process reranker.

**Maturity:** Resolved arXiv preprint; the bounded lifecycle portion of REC-009 is implemented.
The repository does not implement the paper's Determinantal Point Process reranker.

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

**Maturity:** Resolved arXiv book chapter; the repository's action-bound event contract for
REC-002 is implemented. The repository does not adopt a sheaf formalism.

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

**Maturity:** Resolved arXiv preprint; the repository's action-bound event contract for REC-002 is
implemented. The repository does not implement the paper's model-layer router.

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
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [controlled evolution](../controlled_evolution.md), [learning surfaces](../../gepa_mindfulness/learning_surfaces.py), [coevolution](../../gepa_mindfulness/coevolution.py)

**Source demonstrates:** The source reports iterative planning, coding, and testing loops with
small verifiable increments and separation between implementation-time tests and independent
evaluation.

**Repository inference:** The process motivates controlled between-episode harness evolution and
held-out validation rather than changing a scored episode in place.

**Maturity:** Resolved arXiv preprint; the bounded offline-evolution record contract for REC-010
is implemented. The repository does not execute or deploy harness changes.

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
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [controlled evolution](../controlled_evolution.md), [skill lifecycle](../../gepa_mindfulness/skill_lifecycle.py)

**Source demonstrates:** The source reports repository-to-skill distillation and benchmark gains
from a verified skill library under a fixed agent setup and execution budget.

**Repository inference:** The result supports treating procedural knowledge as a reviewable skill
artifact whose value depends on execution evidence and held-out evaluation.

**Maturity:** Resolved arXiv preprint; the bounded artifact lifecycle portion of REC-009 is
implemented. The repository does not reproduce the paper's skill-distillation pipeline or
benchmark.

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
- Local repository notes: [V5 architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md), [controlled evolution](../controlled_evolution.md), [skill lifecycle](../../gepa_mindfulness/skill_lifecycle.py)

**Source demonstrates:** The source reports consolidation of task-local skills into procedural
families and an execution-based commit gate evaluated across long-horizon task streams.

**Repository inference:** The result supports procedural-family records, execution evidence,
validation, commit, and rollback stages in the repository skill lifecycle.

**Maturity:** Resolved arXiv preprint; the bounded procedural-family and lifecycle portion of
REC-009 is implemented. The repository does not reproduce the paper's complete system or
evaluation.

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

**Maturity:** Resolved arXiv preprint; the repository's action-bound event contract for REC-002 is
implemented. The repository does not reproduce the paper's optimization experiments.

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

**Maturity:** Resolved arXiv preprint; the repository's action-bound event contract for REC-002 is
implemented. The repository does not adopt the paper's learned world model.

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

**Maturity:** Resolved EMNLP 2026 main-conference paper; the repository's deterministic
case-by-stripe-by-repeat contracts for REC-005 are implemented. The paper's measurements are not
reproduced.

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

**Maturity:** Resolved arXiv preprint; the repository's deterministic
case-by-stripe-by-repeat contracts for REC-005 are implemented. The paper's measurements are not
reproduced.

<a id="ref-sot"></a>

## REF-SOT — State of Thought Enables Endogenous Reasoning

- Metadata status: `resolved`
- Supplied title: State of Thought Enables Endogenous Reasoning
- Authors: Zhiren Gong, Yikun Hou, Zihao Zeng, Ming Xiao, Chau Yuen, Wei Yang Bryan Lim
- Year: 2026
- arXiv: [`2609.16055`](https://arxiv.org/abs/2609.16055)
- DOI: `10.48550/arXiv.2609.16055`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`, `REC-015`
- Local repository notes: [internal_state_trajectory.py](../../modules/semantic_intent_robustness/internal_state_trajectory.py), [README.md](../../modules/semantic_intent_robustness/README.md), [dynamic_uncertainty.py](../../gepa_mindfulness/training/dynamic_uncertainty.py), [dynamic_uncertainty.py](../../evaluation/dynamic_uncertainty.py), [test_dynamic_uncertainty.py](../../tests/test_dynamic_uncertainty.py), [dynamic_uncertainty.md](../dynamic_uncertainty.md)

**Source demonstrates:** The source reports a compact dynamics-geometric state derived from internal information transfer in frozen models. A lightweight controller uses that state to select historical reasoning support and regulate reasoning progression, evaluated on language and vision-language reasoning tasks.

**Repository inference:** Compact states and state-conditioned historical support motivate experimental diagnostics for intent continuity across semantic laundering and retrieval of prior public evidence omitted without supported supersession. These are repository hypotheses, not results demonstrated by SoT; omission alone does not establish motivated forgetting. PR-13 conditions a policy on public PEO history rather than internal state; the source controller is not reproduced.

**Maturity:** Resolved arXiv preprint; REC-015 is experimental and disabled by default. Synthetic contract tests do not reproduce SoT results or establish true intent, deception, or motive from states. PR-13 implements opt-in decision learning and contract tests; real-model effectiveness remains unmeasured.

<a id="ref-kalman"></a>

## REF-KALMAN — A New Approach to Linear Filtering and Prediction Problems

- Metadata status: `resolved`
- Supplied title: A New Approach to Linear Filtering and Prediction Problems
- Authors: R. E. Kalman
- Year: 1960
- arXiv: Not applicable.
- DOI: `10.1115/1.3662552`
- Canonical source: [ASME paper](https://doi.org/10.1115/1.3662552)
- Primary text inspected: [CMU-hosted ASME paper](https://www.cs.cmu.edu/~./motionplanning/papers/sbp_papers/k/Kalman1960.pdf)
- Venue/status: Journal of Basic Engineering, 82(1), 35-45 (1960).
- Recommendations influenced: `REC-002`, `REC-010`
- Local repository notes: [epistemic_state.py](../../gepa_mindfulness/verification/epistemic_state.py), [epistemic_state.md](../epistemic_state.md), [dynamic_uncertainty.py](../../gepa_mindfulness/training/dynamic_uncertainty.py), [dynamic_uncertainty.py](../../evaluation/dynamic_uncertainty.py), [test_dynamic_uncertainty.py](../../tests/test_dynamic_uncertainty.py), [dynamic_uncertainty.md](../dynamic_uncertainty.md)

**Source demonstrates:** The paper derives recursive linear estimation and an estimation-error covariance equation under explicit stochastic system assumptions.

**Repository inference:** Separate state, measurement, innovation and update records can preserve temporal uncertainty diagnostics. Applying estimation to semantic state remains a repository hypothesis. PR-13 uses validated prior/posterior histories to train verified next decisions; covariance and residual values are not reward targets.

**Maturity:** Published mathematical result; PR-1 implements records, PR-2 causal validation and PR-3 an opt-in scalar estimator. Semantic calibration, optimizer integration and runtime authority remain outside this implementation. PR-13 implements opt-in decision learning and contract tests; real-model effectiveness remains unmeasured.

**Design hypothesis:** Separating world, model and monitor uncertainty may improve subsequent
evidence-acquisition decisions. Arbitrary semantic state need not satisfy linear or Gaussian
assumptions. With fixed process/measurement noise, covariance reduction alone is not a
model-mismatch detector. PR-3 adds explicit gating and adaptive noise as experimental mechanisms.

**Experiment:** PR-3 compares fixed-noise scalar updates with gated adaptive updates on a
synthetic abrupt shift. Held-out comparisons with the current confidence heuristic remain future work.
Measure calibration and false confidence before considering routing or training integration.

**Implementation status:** Diagnostic contracts, causal reconciliation and an opt-in scalar estimator
are implemented with synthetic validation tests. PR-4 adds scalar correlation-aware fusion.

<a id="ref-fta"></a>

## REF-FTA — Failure-Transparent Agents: Benchmarking Post-Failure Reporting in Tool-Using Language Models

- Metadata status: `resolved`
- Supplied title: Failure-Transparent Agents: Benchmarking Post-Failure Reporting in Tool-Using Language Models
- Authors: Junru Zhu, Shiming Xie, Aime Lu Fan Chen, Xiaoqing Ding, Chunxin Tang, Ruoyu Qi, Yulang Fei
- Year: 2026
- arXiv: [`2609.35732`](https://arxiv.org/abs/2609.35732)
- DOI: `10.48550/arXiv.2609.35732`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-007`, `REC-010`
- Local repository notes: [epistemic_reconciliation.py](../../gepa_mindfulness/verification/epistemic_reconciliation.py), [epistemic_state.md](../epistemic_state.md), [failure_layers.py](../../gepa_mindfulness/verification/failure_layers.py), [skill_bank.py](../../gepa_mindfulness/skill_bank.py), [test_failure_layers.py](../../tests/test_failure_layers.py), [test_skill_bank.py](../../tests/test_skill_bank.py), [skill_failure_localization.md](../skill_failure_localization.md), [dynamic_uncertainty.py](../../gepa_mindfulness/training/dynamic_uncertainty.py), [dynamic_uncertainty.py](../../evaluation/dynamic_uncertainty.py), [test_dynamic_uncertainty.py](../../tests/test_dynamic_uncertainty.py), [dynamic_uncertainty.md](../dynamic_uncertainty.md), [ladder.py](../../evaluation/ladder.py), [test_evaluation_ladder.py](../../tests/test_evaluation_ladder.py), [evaluation_ladder.md](../evaluation_ladder.md)

**Source demonstrates:** The benchmark fixes failed-tool observations before evaluating subsequent reports; structured evidence reporting is associated with fewer unsupported claims in its blocked-task setting.

**Repository inference:** Separate execution, observation and reporting evidence. A residual identifies a recorded mismatch, not deceptive motive; report residuals require future public-report telemetry. PR-8 adds layer-specific diagnostic hypotheses and review paths while retaining existing causal and persistence authority boundaries. PR-13 retains verified observations of failed actions for next-decision assessment; offline proposals do not establish successful execution. PR-14 reports false success and fabricated details separately from action quality, with explicit failed-action opportunities.

**Maturity:** Resolved arXiv preprint; PR-2 implements causal and verifier evidence checks. Report residuals and the paper's model experiment remain unimplemented. PR-13 implements opt-in decision learning and contract tests; real-model effectiveness remains unmeasured. PR-14 implements diagnostic report contracts; empirical model effectiveness is unmeasured.

<a id="ref-pinnforge"></a>

## REF-PINNFORGE — PINNForge: Execution-Grounded Evolutionary Design of Physics-Informed Neural Networks

- Metadata status: `resolved`
- Supplied title: PINNsForge
- Authors: Mingyang Yu, Xu Yang, Jun Zhang, Jing Xu, Keqian Li
- Year: 2026
- arXiv: [`2609.23023`](https://arxiv.org/abs/2609.23023)
- DOI: `10.48550/arXiv.2609.23023`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-007`
- Local repository notes: [epistemic_reconciliation.py](../../gepa_mindfulness/verification/epistemic_reconciliation.py), [epistemic_state.md](../epistemic_state.md), [failure_layers.py](../../gepa_mindfulness/verification/failure_layers.py), [skill_bank.py](../../gepa_mindfulness/skill_bank.py), [test_failure_layers.py](../../tests/test_failure_layers.py), [test_skill_bank.py](../../tests/test_skill_bank.py), [skill_failure_localization.md](../skill_failure_localization.md)

**Source demonstrates:** The PDE experiments use recorded training behavior to inform later candidate designs; withholding execution feedback degrades the reported aggregate error measure.

**Repository inference:** Preserve execution evidence before accepting diagnostic updates. Transferring a PDE design loop to epistemic reconciliation is a repository hypothesis. PR-8 adds layer-specific diagnostic hypotheses and review paths while retaining existing causal and persistence authority boundaries.

**Maturity:** Resolved arXiv preprint, v2 metadata; PR-2 binds diagnostic measurements to execution ancestry. No evolutionary search or PDE experiment is implemented.

<a id="ref-c3-jepa"></a>

## REF-C3-JEPA — Underwater C3-JEPA: An Object-Centric Cross-View World Model for ROV Salvage

- Metadata status: `resolved`
- Supplied title: C3-JEPA
- Authors: Yuncong Yang, Jinlong Li, Yulong Xue, Feng Wu, Chunwen Zhang, Lei Qiao, Xuyang Wang
- Year: 2026
- arXiv: [`2609.30214`](https://arxiv.org/abs/2609.30214)
- DOI: `10.48550/arXiv.2609.30214`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-010`, `REC-014`
- Local repository notes: [epistemic_reconciliation.py](../../gepa_mindfulness/verification/epistemic_reconciliation.py), [epistemic_state.md](../epistemic_state.md), [relation_flips.py](../../synthetic_data/relation_flips.py), [relation_flips.py](../../evaluation/relation_flips.py), [test_relation_flip_worlds.py](../../tests/test_relation_flip_worlds.py), [test_relation_flip_evaluation.py](../../tests/test_relation_flip_evaluation.py), [relation_flips.md](../relation_flips.md), [ladder.py](../../evaluation/ladder.py), [test_evaluation_ladder.py](../../tests/test_evaluation_ladder.py), [evaluation_ladder.md](../evaluation_ladder.md)

**Source demonstrates:** The underwater world model uses binding guidance for identifiable task-object representations and reports representation and prediction evaluations.

**Repository inference:** Make measurement correspondence explicit before comparing predicted and observed values. JSON path bindings do not validate learned representations or their semantic units. PR-11 binds each behavioral intervention to one typed world variable without claiming a learned object representation. PR-14 reports representation and prediction as independent stages without inferring downstream competence.

**Maturity:** Resolved arXiv preprint; PR-2 implements explicit numeric outcome bindings. No JEPA backend, rollout learner or robotics result is reproduced. PR-14 implements diagnostic report contracts; empirical model effectiveness is unmeasured.

<a id="ref-ai-neuroscientist"></a>

## REF-AI-NEUROSCIENTIST — The AI Neuroscientist: An Interactive Agentic Interface for Neuroimaging Analysis

- Metadata status: `resolved`
- Supplied title: AI Neuroscientist
- Authors: Aakash Patel, Panos Ketonis, Shreya Saxena, Smita Krishnaswamy, David van Dijk
- Year: 2026
- arXiv: [`2609.25254`](https://arxiv.org/abs/2609.25254)
- DOI: `10.48550/arXiv.2609.25254`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [causal reconciliation](../../gepa_mindfulness/verification/epistemic_reconciliation.py), [guide](../epistemic_state.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.25254v1), 2026-09-30.

**Source demonstrates:** The neuroimaging agent uses ordered tool-mediated phases with explicit intermediate artifacts before interpretation and evaluates fNIRS analysis tasks.

**Repository inference:** Encode required artifact ordering in the harness. Reject a reconciliation when its prediction, execution, observation or verifier dependency is missing.

**Maturity:** Resolved arXiv preprint; PR-2 implements ordered causal validation. No neuroimaging workflow or model-performance result is reproduced.

**Experiment:** Current tests mutate causal inputs and evidence bindings. Later model experiments must measure calibration and reporting errors separately from execution failures; no behavioral result is claimed here.

<a id="ref-gruet"></a>

## REF-GRUET — GRUET: Quantifying Uncertainty of Agentic Reasoning-and-Acting Processes

- Metadata status: `resolved`
- Supplied title: GRUET
- Authors: Shuang Liang, Xin-Yu Hu, Shao-Qun Zhang
- Year: 2026
- arXiv: [`2609.24831`](https://arxiv.org/abs/2609.24831)
- DOI: `10.48550/arXiv.2609.24831`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-010`
- Local repository notes: [temporal_estimator.py](../../gepa_mindfulness/verification/temporal_estimator.py), [temporal_estimator.md](../temporal_estimator.md), [dynamic_uncertainty.py](../../gepa_mindfulness/training/dynamic_uncertainty.py), [dynamic_uncertainty.py](../../evaluation/dynamic_uncertainty.py), [test_dynamic_uncertainty.py](../../tests/test_dynamic_uncertainty.py), [dynamic_uncertainty.md](../dynamic_uncertainty.md), [ladder.py](../../evaluation/ladder.py), [test_evaluation_ladder.py](../../tests/test_evaluation_ladder.py), [evaluation_ladder.md](../evaluation_ladder.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.24831v1), 2026-09-30.

**Source demonstrates:** The paper models reasoning-and-acting trajectories as graphs and evaluates turn-level and trajectory uncertainty for selective generation.

**Repository inference:** Retain uncertainty as trajectory diagnostics. Scalar numeric residuals do not reproduce graph-based reasoning uncertainty or establish semantic calibration. PR-13 stratifies longitudinal decision evaluation without extracting private reasoning graphs or using uncertainty as reward. PR-14 reports temporal and calibration diagnostics separately without collecting private reasoning graphs.

**Maturity:** Resolved arXiv preprint; PR-3 implements opt-in scalar diagnostics. No reasoning-graph extraction, private-chain-of-thought reward or GRUET experiment is implemented. PR-13 implements opt-in decision learning and contract tests; real-model effectiveness remains unmeasured. PR-14 implements diagnostic report contracts; empirical model effectiveness is unmeasured.

<a id="ref-dual-frontier"></a>

## REF-DUAL-FRONTIER — Dual-Frontier: When Can an Agent Trust Its World Model?

- Metadata status: `resolved`
- Supplied title: Dual-Frontier
- Authors: Huatai Zhu, Qiang Chen, Ziqian Kou, Wenhao Li, Fei Wang, Yichao Cao, Xiu Su, Yi Chen
- Year: 2026
- arXiv: [`2609.26293`](https://arxiv.org/abs/2609.26293)
- DOI: `10.48550/arXiv.2609.26293`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [scalar estimator](../../gepa_mindfulness/verification/temporal_estimator.py), [guide](../temporal_estimator.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.26293v1), 2026-09-30.

**Source demonstrates:** The paper analyzes policy-versus-world-model error ambiguity and proposes conditional decision admission using predicted advantage and calibrated model-error bounds.

**Repository inference:** Keep model mismatch explicit and separate numerical confidence from decision authority. A small scalar variance cannot certify an agent decision.

**Maturity:** Resolved arXiv preprint; PR-3 implements mismatch statuses and a latched insufficient-model state. No decision-admission certificate or published agent-performance result is reproduced.

<a id="ref-deepo"></a>

## REF-DEEPO — DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs

- Metadata status: `resolved`
- Supplied title: DEEPO
- Authors: Yingxuan Zhuang, Miao Pan, Wangjie Gan, Jingxiao Yang, Fan Wang, Weiming Liu, Cheng Tan, Xuhong Zhang, Jintao Chen
- Year: 2026
- arXiv: [`2609.28570`](https://arxiv.org/abs/2609.28570)
- DOI: `10.48550/arXiv.2609.28570`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-010`
- Local repository notes: [temporal_estimator.py](../../gepa_mindfulness/verification/temporal_estimator.py), [temporal_estimator.md](../temporal_estimator.md), [dynamic_uncertainty.py](../../gepa_mindfulness/training/dynamic_uncertainty.py), [dynamic_uncertainty.py](../../evaluation/dynamic_uncertainty.py), [test_dynamic_uncertainty.py](../../tests/test_dynamic_uncertainty.py), [dynamic_uncertainty.md](../dynamic_uncertainty.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.28570v1), 2026-09-30.

**Source demonstrates:** The paper studies uncertain queries and confident errors in multimodal reinforcement learning, combining entropy-triggered expert prefixes with gradient preconditioning.

**Repository inference:** Track persistent mismatch even when numerical confidence appears high. Entropy and confidence are not universal truth signals. PR-13 includes mismatch and recovery decision strata with informative verified targets; entropy triggers and gradient preconditioning are not implemented.

**Maturity:** Resolved arXiv preprint; PR-3 implements explicit scalar mismatch diagnostics. No entropy extraction, policy optimization or hallucination benchmark result is reproduced. PR-13 implements opt-in decision learning and contract tests; real-model effectiveness remains unmeasured.

<a id="ref-ci"></a>

## REF-CI — A Non-divergent Estimation Algorithm in the Presence of Unknown Correlations

- Metadata status: `resolved`
- Supplied title: Julier & Uhlmann 1997 — Covariance Intersection
- Authors: Simon J. Julier, Jeffrey K. Uhlmann
- Year: 1997
- arXiv: Not applicable.
- DOI: `10.1109/ACC.1997.609105`
- Venue/status: Proceedings of the 1997 American Control Conference, volume 4, pages 2369-2373.
- Recommendations influenced: `REC-002`
- Local repository notes: [scalar_fusion.py](../../gepa_mindfulness/verification/scalar_fusion.py), [scalar_fusion.md](../scalar_fusion.md)
- Canonical source: [1997 paper DOI](https://doi.org/10.1109/ACC.1997.609105)
- Metadata checked: [author publication list](https://sites.google.com/umsystem.edu/uhlmannj/home/publications), 2026-09-30. Publisher full text was inaccessible; CI equations and multi-source weighting were checked in the [author-coauthored 2025 primary paper](https://discovery.ucl.ac.uk/id/eprint/10217482/1/SSP-2025-GeneralisedCovarianceIntersection-2.1.pdf).

**Source demonstrates:** Covariance Intersection combines estimates with unknown error cross-correlation using a convex mixture of information, assuming consistent input covariance bounds.

**Repository inference:** Default to conservative scalar fusion when cross-correlation is unknown. Use explicitly supplied covariance when known, while retaining its provenance and checking its numerical validity.

**Maturity:** Published estimation method; PR-4 implements scalar CI, declared known-covariance fusion and a conservative bound. Semantic calibration and unbiasedness remain host-reviewed assumptions.

<a id="ref-unanimity"></a>

## REF-UNANIMITY — Unanimity Without Persuasion: A Single Round of Debate Erases the Disagreement That Verification Needs

- Metadata status: `resolved`
- Supplied title: Unanimity Without Persuasion
- Authors: Yang Shu
- Year: 2026
- arXiv: [`2609.26145`](https://arxiv.org/abs/2609.26145)
- DOI: `10.48550/arXiv.2609.26145`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [judgment_panel.py](../../gepa_mindfulness/verification/judgment_panel.py), [scalar_fusion.md](../scalar_fusion.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.26145v1), 2026-09-30.

**Source demonstrates:** The study reports rapid consensus after a debate round with little accuracy change, and shows that peer exposure can remove disagreement useful for targeting verification.

**Repository inference:** Freeze and verify the complete blind judgment cohort before releasing a peer packet. Treat post-discussion agreement as dependent evidence and preserve the original dissent.

**Maturity:** Resolved arXiv preprint; PR-4 implements an explicit in-memory verified Round-0 collector. The host enforces real peer isolation; no debate benchmark result is reproduced.

<a id="ref-wsqem"></a>

## REF-WSQEM — Weakly Supervised Quantum Error Mitigation

- Metadata status: `resolved`
- Supplied title: Weakly Supervised Quantum Error Mitigation
- Authors: Seyed Mohamad Ali Tousi, G. N. DeSouza
- Year: 2026
- arXiv: [`2609.25555`](https://arxiv.org/abs/2609.25555)
- DOI: `10.48550/arXiv.2609.25555`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [scalar_fusion.py](../../gepa_mindfulness/verification/scalar_fusion.py), [scalar_fusion.md](../scalar_fusion.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.25555v1), 2026-09-30.

**Source demonstrates:** The quantum error-mitigation study combines circuit and hardware heuristics through a probabilistic label model without ideal outputs in its training path.

**Repository inference:** Use only the structural analogy of combining weak signals under explicit assumptions. Unknown or unavailable uncertainty should remain explicit rather than receiving independence credit.

**Maturity:** Resolved arXiv preprint; PR-4 provides scalar fusion and an unresolved mode. No quantum label model, hardware result or transfer of quantum calibration to LLM judgments is implemented.

<a id="ref-memcalib"></a>

## REF-MEMCALIB — MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents

- Metadata status: `resolved`
- Supplied title: MemCalib
- Authors: Ruike Cao, Fanyu Zhao, Fugen Yao, Liang Dong, Jian Xu, Guanjun Jiang, Yifei Zhao, Han Zhang, Li Xiao
- Year: 2026
- arXiv: [`2609.24259`](https://arxiv.org/abs/2609.24259)
- DOI: `10.48550/arXiv.2609.24259`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-015`
- Local repository notes: [evidence_use.py](../../gepa_mindfulness/verification/evidence_use.py), [evidence_memory.md](../evidence_memory.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.24259v1), 2026-09-30.

**Source demonstrates:** The benchmark evaluates under-use and over-use against per-proposition Ignore, Bound and Control targets; the paper also evaluates a token credit-assignment training method.

**Repository inference:** Retain declared target influence separately from numeric eligibility. Do not import token rewards or infer execution authority from Control.

**Maturity:** Resolved arXiv preprint; PR-5 implements diagnostic evidence/memory eligibility and retained summary views. No published experiment or trained policy is reproduced.

<a id="ref-jitmem"></a>

## REF-JITMEM — Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents

- Metadata status: `resolved`
- Supplied title: JITMEM
- Authors: Yefan Zhou, Yang Li, Zeyu Leo Liu, Semih Yavuz, Shafiq Joty
- Year: 2026
- arXiv: [`2609.27334`](https://arxiv.org/abs/2609.27334)
- DOI: `10.48550/arXiv.2609.27334`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-015`
- Local repository notes: [evidence_use.py](../../gepa_mindfulness/verification/evidence_use.py), [evidence_memory.md](../evidence_memory.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.27334v1), 2026-09-30.

**Source demonstrates:** The method retains trajectories and curates task-conditioned memory at read time, with evaluated task-success gains over the compared write-time methods.

**Repository inference:** Preserve original records while appending task-specific summary views. No raw private-reasoning storage or learned curator is introduced.

**Maturity:** Resolved arXiv preprint; PR-5 implements diagnostic evidence/memory eligibility and retained summary views. No published experiment or trained policy is reproduced.

<a id="ref-compkv"></a>

## REF-COMPKV — CompKV: Compensation-Aware KV Selection for Long-Context LLM Inference

- Metadata status: `resolved`
- Supplied title: CompKV
- Authors: Zhen Huang, Ruizhe Yao, Danyi Liu, Xinrui Chen, Shuwei Li, Siru Zhong, Zijian Cao, Yushan Lai, Mingming Guo, Weijie Zheng, Haohuan Fu
- Year: 2026
- arXiv: [`2609.26300`](https://arxiv.org/abs/2609.26300)
- DOI: `10.48550/arXiv.2609.26300`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [evidence_use.py](../../gepa_mindfulness/verification/evidence_use.py), [evidence_memory.md](../evidence_memory.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.26300v1), 2026-09-30.

**Source demonstrates:** Sparse attention selection accounts for the compensation residual of omitted blocks using block statistics, with evaluated accuracy and speed results.

**Repository inference:** Use only the analogy that compression can lose consequential information. A declared semantic distortion score is not the paper's KV residual or a covariance.

**Maturity:** Resolved arXiv preprint; PR-5 implements diagnostic evidence/memory eligibility and retained summary views. No published experiment or trained policy is reproduced.

<a id="ref-qwen-planner"></a>

## REF-QWEN-PLANNER — Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents

- Metadata status: `resolved`
- Supplied title: Qwen-Planner-Agent
- Authors: Tingyu Qu, Weigao Sun, Yuecheng Liu, Yucheng Zhao, Yi Zhu, Yifeng Ding, Qiyi Wang, Sihan Cao, Pengkun Jiao, Hanlei Xie, Xiongwei Wu, Qichao Wang, Haodong Zhang, Jiajun Liu, Yuhao Wang, Yuqing Xie, Junpeng Zhao, Long Chen, Ming Ma, Sihan Yang, Ziwang Zhao, Yanhao Jia, Liangquan Gong, Feida Zhu, Yiran Zhong, Steven Hoi
- Year: 2026
- arXiv: [`2609.29892`](https://arxiv.org/abs/2609.29892)
- DOI: `10.48550/arXiv.2609.29892`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`, `REC-005`, `REC-007`, `REC-009`, `REC-010`
- Local repository notes: [evidence_use.py](../../gepa_mindfulness/verification/evidence_use.py), [evidence_memory.md](../evidence_memory.md), [failure_layers.py](../../gepa_mindfulness/verification/failure_layers.py), [skill_bank.py](../../gepa_mindfulness/skill_bank.py), [test_failure_layers.py](../../tests/test_failure_layers.py), [test_skill_bank.py](../../tests/test_skill_bank.py), [skill_failure_localization.md](../skill_failure_localization.md), [worlds.py](../../synthetic_data/worlds.py), [world_peo.py](../../synthetic_data/world_peo.py), [test_synthetic_worlds.py](../../tests/test_synthetic_worlds.py), [test_synthetic_world_peo.py](../../tests/test_synthetic_world_peo.py), [synthetic_worlds.md](../synthetic_worlds.md), [peo_curriculum.py](../../gepa_mindfulness/training/peo_curriculum.py), [curriculum.py](../../gepa_mindfulness/participatory_agency/training/curriculum.py), [test_peo_curriculum.py](../../tests/test_peo_curriculum.py), [test_rl_engine_cpu.py](../../tests/test_rl_engine_cpu.py), [peo_curriculum.md](../peo_curriculum.md)

**Source demonstrates:** The planner framework links data, training and runtime tools, skills and memory through execution feedback and verification, retaining failed and incomplete traces. Section 2.4.3 describes competence-aware reward-and-advantage engineering (CARE) using observed group pass rates to select shaping, consolidation and efficiency regimes.

**Repository inference:** Keep original execution evidence distinct from summaries and preserve failed or unusable evidence in reports. No mobile planner or training loop is reproduced. PR-8 adds layer-specific diagnostic hypotheses and review paths while retaining existing causal and persistence authority boundaries. PR-9 adds simulated action-feedback trajectories with caller-supplied predictions, existing causal PEO validation, and explicit exclusion from training admission. PR-10 uses host-observed anchor outcomes to bound curriculum adaptation without implementing CARE rewards or advantage engineering.

**Maturity:** Resolved arXiv preprint; PR-5 implements diagnostic evidence/memory eligibility and retained summary views. No published experiment or trained policy is reproduced.

<a id="ref-share-borne"></a>

## REF-SHARE-BORNE — Share-Borne AI Virus: Memory-Hopping Attacks Across LLM Agents

- Metadata status: `resolved`
- Supplied title: Share-Borne AI Virus
- Authors: Sidharth Pulipaka, Ansh Sharma, Stanislau Hlebik, Leonidas Raghav, Vyas Raina, Ivaxi Sheth, Mario Fritz
- Year: 2026
- arXiv: [`2609.35576`](https://arxiv.org/abs/2609.35576)
- DOI: `10.48550/arXiv.2609.35576`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [evidence_use.py](../../gepa_mindfulness/verification/evidence_use.py), [evidence_memory.md](../evidence_memory.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.35576v1), 2026-09-30.

**Source demonstrates:** The study demonstrates adversarial content propagating through persistent memory and shared artifacts across assistants in simulated interaction networks.

**Repository inference:** Keep source trust, authority and taint attached to every summary view. Metadata preservation alone does not detect injection or secure downstream transport.

**Maturity:** Resolved arXiv preprint; PR-5 implements diagnostic evidence/memory eligibility and retained summary views. No published experiment or trained policy is reproduced.

<a id="ref-a2m"></a>

## REF-A2M — A2M: Trace-Optimized Agent Hijacking in the MCP Ecosystem

- Metadata status: `resolved`
- Supplied title: A2M/MCP semantic hijacking
- Authors: Laizhen Li, Xuan Wang, Peicheng Zhao, Juanjuan Zhao, Kejiang Ye, Cheng-zhong Xu, Xitong Gao
- Year: 2026
- arXiv: [`2609.26761`](https://arxiv.org/abs/2609.26761)
- DOI: `10.48550/arXiv.2609.26761`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-002`
- Local repository notes: [evidence_use.py](../../gepa_mindfulness/verification/evidence_use.py), [evidence_memory.md](../evidence_memory.md)
- Primary text inspected: [v1 HTML](https://arxiv.org/html/2609.26761v1), 2026-09-30.

**Source demonstrates:** The study evaluates attacks on MCP tool metadata and returned content that affect tool selection and subsequent agent behavior.

**Repository inference:** Reuse existing memory boundary assessment and keep tool information separate from permission. This adapter neither vets MCP servers nor implements the attacks.

**Maturity:** Resolved arXiv preprint; PR-5 implements diagnostic evidence/memory eligibility and retained summary views. No published experiment or trained policy is reproduced.

<a id="ref-evoflint"></a>

## REF-EVOFLINT — EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities

- Metadata status: `resolved`
- Supplied title: EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities
- Authors: Feitong Qiao, Liren Peng, Shiming Ren, Aishwarya Jadhav, Arghavan Bahadorinejad, Marinette Chen, Muhan Zhang, Abdulaziz Suria, Gennevi Lu, Anish Das Sarma
- Year: 2026
- arXiv: [`2609.00487`](https://arxiv.org/abs/2609.00487)
- DOI: `10.48550/arXiv.2609.00487`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-016`
- Local repository notes: [implementation](../../modules/semantic_intent_robustness/evolutionary_atlas.py), [acceptance tests](../../tests/test_evolutionary_semantic_atlas.py), [research overlay guide](../research_overlays.md)

**Source demonstrates:** The source studies evolutionary quality-diversity search over phased multi-turn red-team strategies. It combines mutation and crossover, a persistent structured archive, local novelty competition, Pareto fitness, and generation-level memory of target-model observations.

**Repository inference:** Apply bounded quality-diversity search to semantic-laundering transformations within the fixed 17-case framework. Require verified semantic preservation, immutable lineage, hidden-evaluation exclusion, and independently verified behavioral failures before FailureAtlas admission.

**Maturity:** Resolved arXiv preprint; REC-016 is experimental and disabled by default. Synthetic tests establish software contracts, not alignment effectiveness.

**Limitations:** The paper does not study GEPA or this repository's semantic-laundering curriculum. The implementation uses harmless synthetic transformations and structured feature novelty; it does not reproduce the paper's attack generator or empirical results.

**Implementation references:** [modules/semantic_intent_robustness/evolutionary_atlas.py](../../modules/semantic_intent_robustness/evolutionary_atlas.py) and [tests/test_evolutionary_semantic_atlas.py](../../tests/test_evolutionary_semantic_atlas.py).

<a id="ref-comm-bottleneck"></a>

## REF-COMM-BOTTLENECK — The Communication Bottleneck: A Round-Trip Study of Tree-Structured Expression Serialization in Language Models

- Metadata status: `resolved`
- Supplied title: The Communication Bottleneck: A Round-Trip Study of Tree-Structured Expression Serialization in Language Models
- Authors: Xavier Suau, Alex Ferrando de las Morenas, Luca Zappella, Samy Bengio
- Year: 2026
- arXiv: [`2609.21509`](https://arxiv.org/abs/2609.21509)
- DOI: `10.48550/arXiv.2609.21509`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-017`
- Local repository notes: [implementation](../../evaluation/serialization_roundtrip.py), [acceptance tests](../../tests/test_serialization_roundtrip.py), [research overlay guide](../research_overlays.md)

**Source demonstrates:** The source evaluates arithmetic-expression communication through word problems using separate generators and extractors with symbolic equivalence checks. It measures lossy, asymmetric round trips and analyzes generation versus extraction errors and structural complexity.

**Repository inference:** Audit public structured objects through serialization and extraction. Keep exact equality, verified semantics, heuristic similarity, non-equivalence, and unknown outcomes separate; attribute individual stage faults only when independent stage evidence supports attribution.

**Maturity:** Resolved arXiv preprint; REC-017 is experimental and disabled by default. Synthetic tests establish software contracts, not alignment effectiveness.

**Limitations:** The paper does not evaluate alignment laundering. The implementation verifies a bounded propositional fragment and a synthetic JSON codec; arbitrary natural-language equivalence requires a host-supplied verifier and independently authenticated stage evidence.

**Implementation references:** [evaluation/serialization_roundtrip.py](../../evaluation/serialization_roundtrip.py) and [tests/test_serialization_roundtrip.py](../../tests/test_serialization_roundtrip.py).

<a id="ref-logictrack"></a>

## REF-LOGICTRACK — LogicTrack: Auditing Reasoning Trajectories of Large Language Models with Formal Logic Solvers

- Metadata status: `resolved`
- Supplied title: LogicTrack: Auditing Reasoning Trajectories of Large Language Models with Formal Logic Solvers
- Authors: Jingyu Hu, Shu Yang, Weiru Liu, Di Wang
- Year: 2026
- arXiv: [`2609.21492`](https://arxiv.org/abs/2609.21492)
- DOI: `10.48550/arXiv.2609.21492`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-018`
- Local repository notes: [implementation](../../gepa_mindfulness/verification/formal_reasoning.py), [acceptance tests](../../tests/test_formal_reasoning_audit.py), [research overlay guide](../research_overlays.md)

**Source demonstrates:** The source formalizes reasoning steps for automated theorem-prover checks, uses solver-guided backtracking during inference, and constructs supervised examples from audited trajectories. It evaluates the approach across reasoning benchmarks.

**Repository inference:** Check explicit public reasoning objects with a bounded solver adapter and optional bounded retries. Keep premise grounding, formal validity, factual correctness, calibration, and behavioral outcomes separate without importing the paper's reasoning-text reward.

**Maturity:** Resolved arXiv preprint; REC-018 is experimental and disabled by default. Synthetic tests establish software contracts, not alignment effectiveness.

**Limitations:** Formal validity does not establish factual truth, authentic evidence, or a faithful natural-language translation. The reference solver supports a limited propositional fragment; host-authenticated grounding and existing training-eligibility controls remain separate.

**Implementation references:** [gepa_mindfulness/verification/formal_reasoning.py](../../gepa_mindfulness/verification/formal_reasoning.py) and [tests/test_formal_reasoning_audit.py](../../tests/test_formal_reasoning_audit.py).

<a id="ref-latent-language-gap"></a>

## REF-LATENT-LANGUAGE-GAP — When Steering Fails in Latent Reasoning: A Latent-to-Language Transition Gap

- Metadata status: `resolved`
- Supplied title: When Steering Fails in Latent Reasoning: A Latent-to-Language Transition Gap
- Authors: Gaoxiang Huang, Lei Qi
- Year: 2026
- arXiv: [`2609.21662`](https://arxiv.org/abs/2609.21662)
- DOI: `10.48550/arXiv.2609.21662`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-019`
- Local repository notes: [implementation](../../modules/semantic_intent_robustness/latent_language_transition.py), [acceptance tests](../../tests/test_latent_language_transition.py), [research overlay guide](../research_overlays.md)

**Source demonstrates:** The source reports that comparable hidden-representation movements can yield weaker effects on language generation during latent reasoning. It investigates transition-boundary distribution changes and differences in task-direction control.

**Repository inference:** Measure comparable latent, language, and policy/action deltas independently. Report latent-language decoupling or language change without a matching measured latent signal while retaining origin, comparability, and unavailable-state information.

**Maturity:** Resolved arXiv preprint; REC-019 is experimental and disabled by default. Synthetic tests establish software contracts, not alignment effectiveness.

**Limitations:** These diagnostics do not establish intent, deception, causal use of a representation, successful steering, or alignment. Transfer ratios depend on measurement normalization. Black-box behavioral evaluation remains available without internal-state access.

**Implementation references:** [modules/semantic_intent_robustness/latent_language_transition.py](../../modules/semantic_intent_robustness/latent_language_transition.py) and [tests/test_latent_language_transition.py](../../tests/test_latent_language_transition.py).

<a id="ref-cdr"></a>

## REF-CDR — Knowledge Graph-Augmented Ambient AI for Clinical Note Generation

- Metadata status: `resolved`
- Supplied title: Coverage-Directed Revision
- Authors: Jakir Hossain, Yi-Fei Zhao, Hongjian Wang, Minmei Shih, Katie Leigh Mullen, Ahmad P. Tafti, Leming Zhou, Manoj Purohit, William Hogan, Jay Zeng, Elizabeth Skidmore, Yanshan Wang
- Year: 2026
- arXiv: [`2609.22239`](https://arxiv.org/abs/2609.22239)
- DOI: `10.48550/arXiv.2609.22239`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-014`, `REC-015`
- Local repository notes: [peo_continuity.py](../../modules/semantic_intent_robustness/peo_continuity.py), [test_peo_continuity.py](../../tests/test_peo_continuity.py), [peo_continuity.md](../peo_continuity.md), [relation_flips.py](../../synthetic_data/relation_flips.py), [relation_flips.py](../../evaluation/relation_flips.py), [test_relation_flip_worlds.py](../../tests/test_relation_flip_worlds.py), [test_relation_flip_evaluation.py](../../tests/test_relation_flip_evaluation.py), [relation_flips.md](../relation_flips.md)

**Source demonstrates:** The source builds transcript-grounded knowledge graphs to identify concepts missing from generated clinical notes and direct revision. It reports improved content recall on Pitt-Bench and ACI-Bench.

**Repository inference:** Compare later evidence coverage with an original available inventory and preserve unexplained omissions. This transfer is a diagnostic hypothesis; the clinical graph construction and revision procedure are not implemented. PR-11 reports missing relation categories as a declared coverage inventory, without implementing clinical concept extraction or revision.

**Maturity:** Resolved arXiv preprint; PR-6 adds disabled-by-default public PEO continuity diagnostics. Synthetic tests establish software contracts, not model effectiveness or causal influence.

<a id="ref-ttse"></a>

## REF-TTSE — TTSE: A Two-Track Online Self-Evolution Framework for LLM Agents

- Metadata status: `resolved`
- Supplied title: TTSE
- Authors: Ruimin Pei, Yongkang Wu, Shangyi Zheng, Yaqing Zhang, Deyang Li, Jianjun Tao, Xinyu Zhang, Xiang Zhang
- Year: 2026
- arXiv: [`2609.24289`](https://arxiv.org/abs/2609.24289)
- DOI: `10.48550/arXiv.2609.24289`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-015`
- Local repository notes: [implementation](../../modules/semantic_intent_robustness/peo_continuity.py), [tests](../../tests/test_peo_continuity.py), [guide](../peo_continuity.md)

**Source demonstrates:** TTSE separates environmental facts (FACT) from task-conditioned procedures (TIP), with separate evolution lifecycles. Its analysis separates environment-representation and conditional-execution errors; agent benchmark ablations evaluate the dual-track design.

**Repository inference:** Retain fact/procedure distinctions and localize failures between evidence availability and behavioral influence. Host observations do not reproduce the paper's risk decomposition, autonomous updates, retirement, or training.

**Maturity:** Resolved arXiv preprint; PR-6 adds disabled-by-default public PEO continuity diagnostics. Synthetic tests establish software contracts, not model effectiveness or causal influence.

<a id="ref-jev-mem"></a>

## REF-JEV-MEM — Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents

- Metadata status: `resolved`
- Supplied title: Jev-Mem
- Authors: Dongming Jiang, Yi Li, Bingzhe Li
- Year: 2026
- arXiv: [`2609.23986`](https://arxiv.org/abs/2609.23986)
- DOI: `10.48550/arXiv.2609.23986`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-008`
- Local repository notes: [epistemic_routing.py](../../gepa_mindfulness/factuality_observability/epistemic_routing.py), [system_one_benchmark.py](../../evaluation/system_one_benchmark.py), [test_epistemic_routing.py](../../tests/test_epistemic_routing.py), [test_system_one_benchmark.py](../../tests/test_system_one_benchmark.py), [system_one_routing.md](../system_one_routing.md)

**Source demonstrates:** Jev-Mem separates fast memory control from deliberative reasoning. Its controller routes queries, budgets retrieval, scores candidates and stops retrieval; evaluations compare memory quality and latency.

**Repository inference:** Compare interchangeable bounded route proposers on identical inputs and retain stopping and verification constraints outside each proposer.

**Maturity:** Experimental opt-in routing adapter. Synthetic contract comparisons do not establish learned-backend quality, alignment or source-benchmark reproduction.

<a id="ref-clm"></a>

## REF-CLM — Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making

- Metadata status: `resolved`
- Supplied title: Contrastive Language Models
- Authors: Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré, Azalia Mirhoseini
- Year: 2026
- arXiv: Not applicable.
- DOI: `None`
- Venue/status: Official project and Notion blog; no arXiv or DOI supplied.
- Recommendations influenced: `REC-008`, `REC-010`
- Local repository notes: [epistemic_routing.py](../../gepa_mindfulness/factuality_observability/epistemic_routing.py), [system_one_benchmark.py](../../evaluation/system_one_benchmark.py), [test_epistemic_routing.py](../../tests/test_epistemic_routing.py), [test_system_one_benchmark.py](../../tests/test_system_one_benchmark.py), [system_one_routing.md](../system_one_routing.md), [contrastive.py](../../gepa_mindfulness/training/contrastive.py), [contrastive_negatives.py](../../synthetic_data/contrastive_negatives.py), [contrastive.py](../../evaluation/contrastive.py), [test_contrastive_training.py](../../tests/test_contrastive_training.py), [test_contrastive_negatives.py](../../tests/test_contrastive_negatives.py), [test_contrastive_evaluation.py](../../tests/test_contrastive_evaluation.py), [contrastive_training.md](../contrastive_training.md)
- Primary source: [official project](https://github.com/Contrastive-LM/CLM)

**Source demonstrates:** The official project describes contrastive state/action representations, independently cached embeddings, candidate ranking and a TypeSafe-compatible typed decision API. The official trainer implements separate projection heads with a group-masked bidirectional in-batch contrastive loss.

**Repository inference:** Expose bounded public routing features to a host-supplied CLM callback. Evaluate raw proposals, failures and latency separately from guarded outcomes; do not treat model scores as authority. PR-12 reuses CPT pairs for an opt-in two-candidate optimizer and four-arm ranking comparison. This is not the source bidirectional in-batch objective or a trained CLM checkpoint.

**Maturity:** Experimental opt-in routing adapter. Synthetic contract comparisons do not establish learned-backend quality, alignment or source-benchmark reproduction. PR-12 contract and CPU wiring tests are implemented; learned comparative effectiveness remains unmeasured.

<a id="ref-toollery"></a>

## REF-TOOLLERY — Toollery: Scaling LLM Agents to Thousands of Skills and Tools

- Metadata status: `resolved`
- Supplied title: Toollery
- Authors: Xiangxi Tian, Ran Guan
- Year: 2026
- arXiv: [`2609.22218`](https://arxiv.org/abs/2609.22218)
- DOI: `10.48550/arXiv.2609.22218`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-008`
- Local repository notes: [epistemic_routing.py](../../gepa_mindfulness/factuality_observability/epistemic_routing.py), [system_one_benchmark.py](../../evaluation/system_one_benchmark.py), [test_epistemic_routing.py](../../tests/test_epistemic_routing.py), [test_system_one_benchmark.py](../../tests/test_system_one_benchmark.py), [system_one_routing.md](../system_one_routing.md)

**Source demonstrates:** Toollery indexes intent queries generated from capability specifications and retrieves compact candidate sets before final selection. Evaluations measure selection quality and cost at bounded candidate budgets.

**Repository inference:** Keep route proposals within the existing finite action vocabulary. This stage does not implement query expansion, a tool index or capability execution.

**Maturity:** Experimental opt-in routing adapter. Synthetic contract comparisons do not establish learned-backend quality, alignment or source-benchmark reproduction.

<a id="ref-seek"></a>

## REF-SEEK — SEEK: Skill-Routed Evaluation with Evolvable Knowledge for Industrial Search

- Metadata status: `resolved`
- Supplied title: SEEK
- Authors: Zhongxin Huang, Songyang Li, Renzhe Zhou, Feiran Zhu, Chenglei Dai, Zhen Xiao, Xuanping Li, Jingwei Zhuo
- Year: 2026
- arXiv: [`2609.29803`](https://arxiv.org/abs/2609.29803)
- DOI: `10.48550/arXiv.2609.29803`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-007`, `REC-008`, `REC-009`
- Local repository notes: [epistemic_routing.py](../../gepa_mindfulness/factuality_observability/epistemic_routing.py), [system_one_benchmark.py](../../evaluation/system_one_benchmark.py), [test_epistemic_routing.py](../../tests/test_epistemic_routing.py), [test_system_one_benchmark.py](../../tests/test_system_one_benchmark.py), [system_one_routing.md](../system_one_routing.md), [failure_layers.py](../../gepa_mindfulness/verification/failure_layers.py), [skill_bank.py](../../gepa_mindfulness/skill_bank.py), [test_failure_layers.py](../../tests/test_failure_layers.py), [test_skill_bank.py](../../tests/test_skill_bank.py), [skill_failure_localization.md](../skill_failure_localization.md)

**Source demonstrates:** SEEK separates skill routing descriptions from operational guidance, diagnoses routing, knowledge and execution errors, and replay-gates skill updates in industrial search evaluation.

**Repository inference:** Separate routing proposal quality from guarded outcomes. Preserve WHEN and HOW in immutable skill cards and annotate failure layers as evidence-bound hypotheses; skill changes remain review proposals without persistence authority.

**Maturity:** Experimental opt-in routing and failure diagnostics with immutable skill metadata. Contract tests do not establish learned-backend quality or diagnosis accuracy.

<a id="ref-ladder"></a>

## REF-LADDER — LADDER: Graph-Guided Diffusion Language Models for Efficient Multi-Hop Reasoning

- Metadata status: `resolved`
- Supplied title: LADDER
- Authors: Senlei Zhang, Linhao Luo, Qian-Wen Zhang, Siyu An, Junnan Dong, Shuhao Zhang, Xing Sun
- Year: 2026
- arXiv: [`2609.24346`](https://arxiv.org/abs/2609.24346)
- DOI: `10.48550/arXiv.2609.24346`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-008`
- Local repository notes: [epistemic_routing.py](../../gepa_mindfulness/factuality_observability/epistemic_routing.py), [system_one_benchmark.py](../../evaluation/system_one_benchmark.py), [test_epistemic_routing.py](../../tests/test_epistemic_routing.py), [test_system_one_benchmark.py](../../tests/test_system_one_benchmark.py), [system_one_routing.md](../system_one_routing.md)

**Source demonstrates:** LADDER triggers graph retrieval when graph-linkable entities expand during diffusion decoding and propagates incomplete queries through a graph model. The retrieval trigger bypasses learned gates and heuristic thresholds.

**Repository inference:** Reconsider routing when recorded uncertainty or mismatch requires more evidence. These host thresholds are a distinct hypothesis, not a reproduction of diffusion decoding or self-clocking retrieval.

**Maturity:** Experimental opt-in routing adapter. Synthetic contract comparisons do not establish learned-backend quality, alignment or source-benchmark reproduction.

<a id="ref-reasoning-topology"></a>

## REF-REASONING-TOPOLOGY — Reasoning Topology Matters: A Controlled Study of LLM-Based Cybersecurity Analysis

- Metadata status: `resolved`
- Supplied title: Reasoning Topology Matters
- Authors: Jiling Zhou, Aisvarya Adeseye, Antti Hakkala, Seppo Virtanen, Jouni Isoaho
- Year: 2026
- arXiv: [`2609.24710`](https://arxiv.org/abs/2609.24710)
- DOI: `10.48550/arXiv.2609.24710`
- Venue/status: Accepted at AIAIS 2027.
- Recommendations influenced: `REC-008`
- Local repository notes: [epistemic_routing.py](../../gepa_mindfulness/factuality_observability/epistemic_routing.py), [system_one_benchmark.py](../../evaluation/system_one_benchmark.py), [test_epistemic_routing.py](../../tests/test_epistemic_routing.py), [test_system_one_benchmark.py](../../tests/test_system_one_benchmark.py), [system_one_routing.md](../system_one_routing.md)

**Source demonstrates:** The source compares linear, branching and graph reasoning prompts across three cybersecurity datasets and several model families. Reported gains are conditional on the evaluated tasks and prompting setup.

**Repository inference:** A mismatch requests hypothesis reconsideration through the existing decompose-and-verify route. This adapter does not select or claim to execute a universally superior reasoning topology.

**Maturity:** Experimental opt-in routing adapter. Synthetic contract comparisons do not establish learned-backend quality, alignment or source-benchmark reproduction.

<a id="ref-arise"></a>

## REF-ARISE — ARISE: Adapting to Evolving Capability Gaps in Agentic Reinforcement Learning

- Metadata status: `resolved`
- Supplied title: ARISE
- Authors: Kun Feng, Yuchen Fang, Yiyang Tan, Shuqi Gu, Yongxiang Zhao, Yu Liu, Xingyu Lu, Lintao Ma, Kan Ren
- Year: 2026
- arXiv: [`2609.35532`](https://arxiv.org/abs/2609.35532)
- DOI: `10.48550/arXiv.2609.35532`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-009`
- Local repository notes: [failure_layers.py](../../gepa_mindfulness/verification/failure_layers.py), [skill_bank.py](../../gepa_mindfulness/skill_bank.py), [test_failure_layers.py](../../tests/test_failure_layers.py), [test_skill_bank.py](../../tests/test_skill_bank.py), [skill_failure_localization.md](../skill_failure_localization.md)

**Source demonstrates:** ARISE uses rollout evidence to evolve rubric-skill pairs and adapt task sampling. Its training procedure activates guidance for capability gaps and retires consistently satisfied criteria.

**Repository inference:** Use observed gaps to motivate bounded skill review proposals. Foundational norms remain protected from autonomous changes or retirement regardless of performance; no adaptive reward or rubric-retirement algorithm is introduced.

**Maturity:** Resolved arXiv preprint; experimental skill metadata and proposal contracts are implemented. Contract tests do not reproduce agentic reinforcement-learning results.

<a id="ref-tabpfn-35"></a>

## REF-TABPFN-35 — TabPFN-3.5: Technical Report

- Metadata status: `resolved`
- Supplied title: TabPFN-3.5
- Authors: Benjamin Jäger, Nick Erickson, Léo Grinsztajn, Felix Birkel, Klemens Flöge, Oscar Key, Kürşat Kaya, Jonas Kübler, Adèle Frankel, Tobias Schröder, Anurag Garg, Jan Hendrik Metzen, David Salinas, Simon Bing, Kristina Collins, Tuana Çelik, Vahid Balazadeh, Lydia Sidhoum, Tomás Pereda, Brendan Roof, Andrej Tschalzev, Siyuan Guo, Philipp Singer, Lennart Purucker, Jake Robertson, Marie Salmon, Philipp Jund, Jerry Chen, Diana Kriuchkova, Arthur Cahu, Eliott Kalfon, Adrian Hayler, Georg Grab, Vitor Monteiro, Lilly Wehrhahn, Dominik Safaric, Clara Cornu, Alan Arazi, Rylee Grace, Simone Alessi, Mihir Manium, Bernhard Schölkopf, Yann LeCun, Madelon Hulsebos, Sauraj Gambhir, Noah Hollmann, Frank Hutter
- Year: 2026
- arXiv: [`2609.17895`](https://arxiv.org/abs/2609.17895)
- DOI: `10.48550/arXiv.2609.17895`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-005`, `REC-010`
- Local repository notes: [worlds.py](../../synthetic_data/worlds.py), [world_peo.py](../../synthetic_data/world_peo.py), [test_synthetic_worlds.py](../../tests/test_synthetic_worlds.py), [test_synthetic_world_peo.py](../../tests/test_synthetic_world_peo.py), [synthetic_worlds.md](../synthetic_worlds.md), [peo_curriculum.py](../../gepa_mindfulness/training/peo_curriculum.py), [curriculum.py](../../gepa_mindfulness/participatory_agency/training/curriculum.py), [test_peo_curriculum.py](../../tests/test_peo_curriculum.py), [test_rl_engine_cpu.py](../../tests/test_rl_engine_cpu.py), [peo_curriculum.md](../peo_curriculum.md)

**Source demonstrates:** Section 3.3 describes a more diverse and scalable synthetic prior, including high-cardinality, high-feature-count and grouped tabular data with distinct train/test groups.

**Repository inference:** Use controlled seeded latent-world generation as an experimental data contract. This boolean fixture does not implement the TabPFN prior, train a tabular model, or reproduce its benchmarks. PR-10 annotates multiple data dimensions and samples persistent anchors, weaknesses, frontier and OOD units; it does not reproduce the synthetic prior.

**Maturity:** Resolved arXiv preprint; experimental opt-in synthetic-world contracts. Contract tests do not establish real-world policy quality or reproduce source results.

<a id="ref-vgcompiler"></a>

## REF-VGCOMPILER — Visual Graph Reasoning via Knowledge Compilation

- Metadata status: `resolved`
- Supplied title: VGCompiler
- Authors: Rongzheng Wang, Zhe Wang, Ke Qin, Rongwei Wang, Muquan Li, Yizhuo Ma, Yihong Huang, Jielei Wang, Shuang Liang
- Year: 2026
- arXiv: [`2609.22327`](https://arxiv.org/abs/2609.22327)
- DOI: `10.48550/arXiv.2609.22327`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-005`, `REC-010`
- Local repository notes: [worlds.py](../../synthetic_data/worlds.py), [world_peo.py](../../synthetic_data/world_peo.py), [test_synthetic_worlds.py](../../tests/test_synthetic_worlds.py), [test_synthetic_world_peo.py](../../tests/test_synthetic_world_peo.py), [synthetic_worlds.md](../synthetic_worlds.md), [contrastive.py](../../gepa_mindfulness/training/contrastive.py), [contrastive_negatives.py](../../synthetic_data/contrastive_negatives.py), [contrastive.py](../../evaluation/contrastive.py), [test_contrastive_training.py](../../tests/test_contrastive_training.py), [test_contrastive_negatives.py](../../tests/test_contrastive_negatives.py), [test_contrastive_evaluation.py](../../tests/test_contrastive_evaluation.py), [contrastive_training.md](../contrastive_training.md)

**Source demonstrates:** A representation compiler maps visual input to an explicit attributed graph, and an operation compiler maps a query and graph to an executable operation. Sections 3.1–3.2 and Figure 3 distinguish style changes with fixed graph state from graph-state changes with fixed style.

**Repository inference:** Separate typed latent world truth from surface rendering and compare invariant surfaces with decisive permission counterfactuals. No visual parser, learned compiler, or model-generated executable code is introduced. PR-12 derives causal and PEO negatives from the existing deterministic world structure, keeping latent snapshots outside public scoring inputs.

**Maturity:** Resolved arXiv preprint; experimental opt-in synthetic-world contracts. Contract tests do not establish real-world policy quality or reproduce source results. PR-12 contract and CPU wiring tests are implemented; learned comparative effectiveness remains unmeasured.

<a id="ref-physical-languages"></a>

## REF-PHYSICAL-LANGUAGES — Discovering Physical Representation Languages

- Metadata status: `resolved`
- Supplied title: Discovering Physical Representation Languages
- Authors: Linzhe Zhang, Changming Xu
- Year: 2026
- arXiv: [`2609.23381`](https://arxiv.org/abs/2609.23381)
- DOI: `10.48550/arXiv.2609.23381`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-005`, `REC-010`
- Local repository notes: [worlds.py](../../synthetic_data/worlds.py), [world_peo.py](../../synthetic_data/world_peo.py), [test_synthetic_worlds.py](../../tests/test_synthetic_worlds.py), [test_synthetic_world_peo.py](../../tests/test_synthetic_world_peo.py), [synthetic_worlds.md](../synthetic_worlds.md), [contrastive.py](../../gepa_mindfulness/training/contrastive.py), [contrastive_negatives.py](../../synthetic_data/contrastive_negatives.py), [contrastive.py](../../evaluation/contrastive.py), [test_contrastive_training.py](../../tests/test_contrastive_training.py), [test_contrastive_negatives.py](../../tests/test_contrastive_negatives.py), [test_contrastive_evaluation.py](../../tests/test_contrastive_evaluation.py), [contrastive_training.md](../contrastive_training.md)

**Source demonstrates:** Controlled anonymous physical experiments recover representation structure and measurement types while exposing residual observational equivalences and identifiability limits.

**Repository inference:** Keep unavailable latent facts distinct from observable evidence and avoid resolving hidden truth from surface text. This simulator does not reproduce the physical experiments or establish identifiability results. PR-12 retains typed source provenance for world-derived negatives and limits claims to observable outcomes; it does not recover physical representations.

**Maturity:** Resolved arXiv preprint; experimental opt-in synthetic-world contracts. Contract tests do not establish real-world policy quality or reproduce source results. PR-12 contract and CPU wiring tests are implemented; learned comparative effectiveness remains unmeasured.

<a id="ref-generalized-tamp"></a>

## REF-GENERALIZED-TAMP — Coding Agents for Generalized Task and Motion Planning Problems

- Metadata status: `resolved`
- Supplied title: Coding Agents for Generalized Task and Motion Planning Problems
- Authors: Matteo Merler, Bowen Li, Josh Roy, Yichao Liang, Qianwei Wang, Yixuan Huang, Tom Silver
- Year: 2026
- arXiv: [`2609.30233`](https://arxiv.org/abs/2609.30233)
- DOI: `10.48550/arXiv.2609.30233`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-005`
- Local repository notes: [worlds.py](../../synthetic_data/worlds.py), [world_peo.py](../../synthetic_data/world_peo.py), [test_synthetic_worlds.py](../../tests/test_synthetic_worlds.py), [test_synthetic_world_peo.py](../../tests/test_synthetic_world_peo.py), [synthetic_worlds.md](../synthetic_worlds.md)

**Source demonstrates:** Coding agents receive a task description, simulator and fixed synthesis budget to produce a program, which is frozen before evaluation on unseen problem instances.

**Repository inference:** Use reusable deterministic transition rules across seeded instances and keep evaluator truth separate from actor observations. No robotic planner, motion dynamics, agent synthesis loop, or generalization benchmark is reproduced.

**Maturity:** Resolved arXiv preprint; experimental opt-in synthetic-world contracts. Contract tests do not establish real-world policy quality or reproduce source results.

<a id="ref-chart"></a>

## REF-CHART — CHART: A Harness-Rotation Curriculum for Harness-Robust Search Agents

- Metadata status: `resolved`
- Supplied title: CHART / Harness-Rotation
- Authors: Xinlu Zhang, Ying-Chun Lin, Zhihan Zhang, Besnik Fetahu, Xi Chen
- Year: 2026
- arXiv: [`2609.22247`](https://arxiv.org/abs/2609.22247)
- DOI: `10.48550/arXiv.2609.22247`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [peo_curriculum.py](../../gepa_mindfulness/training/peo_curriculum.py), [curriculum.py](../../gepa_mindfulness/participatory_agency/training/curriculum.py), [test_peo_curriculum.py](../../tests/test_peo_curriculum.py), [test_rl_engine_cpu.py](../../tests/test_rl_engine_cpu.py), [peo_curriculum.md](../peo_curriculum.md), [contrastive.py](../../gepa_mindfulness/training/contrastive.py), [contrastive_negatives.py](../../synthetic_data/contrastive_negatives.py), [contrastive.py](../../evaluation/contrastive.py), [test_contrastive_training.py](../../tests/test_contrastive_training.py), [test_contrastive_negatives.py](../../tests/test_contrastive_negatives.py), [test_contrastive_evaluation.py](../../tests/test_contrastive_evaluation.py), [contrastive_training.md](../contrastive_training.md)

**Source demonstrates:** Section 3.2 maintains a small active harness window and rotates learned harnesses for still-learnable ones.

**Repository inference:** Use controlled distribution rotation with separate persistent anchor checks. Persistent anchor quotas are a repository design choice; this provider does not implement CHART GRPO or reproduce its training results. PR-12 compares a fixed five-family contrastive schedule with equal-exposure pooled training; it does not reproduce adaptive harness graduation or GRPO.

**Maturity:** Resolved arXiv preprint; experimental opt-in curriculum contracts. Contract tests do not establish learned retention or reproduce source results. PR-12 contract and CPU wiring tests are implemented; learned comparative effectiveness remains unmeasured.

<a id="ref-spectral-grokking"></a>

## REF-SPECTRAL-GROKKING — A Spectral Theory of Grokking: Weight Decay induces Feature Learning

- Metadata status: `resolved`
- Supplied title: Spectral Theory of Grokking
- Authors: Lenz Pracher, Pascal de Jong, Oskar Lieshaus, Alan Jeffares, Steffen Rulands
- Year: 2026
- arXiv: [`2609.26679`](https://arxiv.org/abs/2609.26679)
- DOI: `10.48550/arXiv.2609.26679`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [peo_curriculum.py](../../gepa_mindfulness/training/peo_curriculum.py), [curriculum.py](../../gepa_mindfulness/participatory_agency/training/curriculum.py), [test_peo_curriculum.py](../../tests/test_peo_curriculum.py), [test_rl_engine_cpu.py](../../tests/test_rl_engine_cpu.py), [peo_curriculum.md](../peo_curriculum.md)

**Source demonstrates:** For homogeneous networks with squared loss and weight decay, the analysis links residual-driven NTK feature growth after fitting to delayed generalization, with modular-addition experiments.

**Repository inference:** Retain repeated anchor checks after apparent mastery. No NTK telemetry or grokking theory is implemented, and scheduling or PEO residuals do not establish grokking.

**Maturity:** Resolved arXiv preprint; experimental opt-in curriculum contracts. Contract tests do not establish learned retention or reproduce source results.

<a id="ref-low-bit-opd"></a>

## REF-LOW-BIT-OPD — Train Where the Quantized Model Goes: On-Policy Distillation for Low-Bit Reasoning

- Metadata status: `resolved`
- Supplied title: On-Policy Distillation for Low-Bit Reasoning
- Authors: Yuanteng Chen, Zhilei Liu, Peisong Wang, Yuantian Shao, Chuangyi Li, Weining Wang, Shuang Qiu, Gang Li, Jing Liu, Jian Cheng
- Year: 2026
- arXiv: [`2609.26708`](https://arxiv.org/abs/2609.26708)
- DOI: `10.48550/arXiv.2609.26708`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [peo_curriculum.py](../../gepa_mindfulness/training/peo_curriculum.py), [curriculum.py](../../gepa_mindfulness/participatory_agency/training/curriculum.py), [test_peo_curriculum.py](../../tests/test_peo_curriculum.py), [test_rl_engine_cpu.py](../../tests/test_rl_engine_cpu.py), [peo_curriculum.md](../peo_curriculum.md), [dynamic_uncertainty.py](../../gepa_mindfulness/training/dynamic_uncertainty.py), [dynamic_uncertainty.py](../../evaluation/dynamic_uncertainty.py), [test_dynamic_uncertainty.py](../../tests/test_dynamic_uncertainty.py), [dynamic_uncertainty.md](../dynamic_uncertainty.md)

**Source demonstrates:** Student rollouts use the deployment quantized forward path; a frozen full-precision teacher scores student prefixes alongside task-verifier rewards.

**Repository inference:** Allow reviewed student-induced weakness and recovery data in a bounded curriculum mixture. No quantization, distillation loss, teacher scoring, or admission of raw failed traces is introduced. PR-13 accepts reviewed deployment histories for behavioral decision training; no quantized distillation or automatic admission is added.

**Maturity:** Resolved arXiv preprint; experimental opt-in curriculum contracts. Contract tests do not establish learned retention or reproduce source results. PR-13 implements opt-in decision learning and contract tests; real-model effectiveness remains unmeasured.

<a id="ref-p-ttt"></a>

## REF-P-TTT — Using Context Is Not Enough: Test-Time Training for Personalized Reward Modeling

- Metadata status: `resolved`
- Supplied title: P-TTT
- Authors: Bohao Wang, Xiaoyan Zhao, Yang Zhang, Jinghang Guo, Chun Chen, Can Wang, Jiawei Chen
- Year: 2026
- arXiv: [`2609.35109`](https://arxiv.org/abs/2609.35109)
- DOI: `10.48550/arXiv.2609.35109`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`, `REC-014`
- Local repository notes: [relation_flips.py](../../synthetic_data/relation_flips.py), [relation_flips.py](../../evaluation/relation_flips.py), [test_relation_flip_worlds.py](../../tests/test_relation_flip_worlds.py), [test_relation_flip_evaluation.py](../../tests/test_relation_flip_evaluation.py), [relation_flips.md](../relation_flips.md), [contrastive.py](../../gepa_mindfulness/training/contrastive.py), [contrastive_negatives.py](../../synthetic_data/contrastive_negatives.py), [contrastive.py](../../evaluation/contrastive.py), [test_contrastive_training.py](../../tests/test_contrastive_training.py), [test_contrastive_negatives.py](../../tests/test_contrastive_negatives.py), [test_contrastive_evaluation.py](../../tests/test_contrastive_evaluation.py), [contrastive_training.md](../contrastive_training.md)

**Source demonstrates:** Appendix D reverses contextual preference labels with response content fixed, retains only variants with an unambiguous opposite target preference, and measures flips among initially correct predictions.

**Repository inference:** Use explicit relation reversals with validated expected changes and reject masked or unresolved pairs. No fast-weight learning, personalized reward model, or source flip-rate reproduction is implemented. PR-12 uses opposite-world decisions as contrastive negatives while retaining non-TRAIN world provenance; no fast-weight training is reproduced.

**Maturity:** Resolved arXiv preprint; experimental opt-in behavioral counterfactual diagnostics. Contract tests do not establish internal mechanism recovery or reproduce source results. PR-12 contract and CPU wiring tests are implemented; learned comparative effectiveness remains unmeasured.

<a id="ref-mechbench"></a>

## REF-MECHBENCH — MechBench: Can AI Scientific Agents Discover Mechanisms Beyond Phenomenal Laws?

- Metadata status: `resolved`
- Supplied title: MechBench
- Authors: Zihan Yu, Jiadong Zhang, Jialin Cheng, Jingtao Ding, Yong Li
- Year: 2026
- arXiv: [`2609.35515`](https://arxiv.org/abs/2609.35515)
- DOI: `10.48550/arXiv.2609.35515`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`, `REC-014`
- Local repository notes: [relation_flips.py](../../synthetic_data/relation_flips.py), [relation_flips.py](../../evaluation/relation_flips.py), [test_relation_flip_worlds.py](../../tests/test_relation_flip_worlds.py), [test_relation_flip_evaluation.py](../../tests/test_relation_flip_evaluation.py), [relation_flips.md](../relation_flips.md), [contrastive.py](../../gepa_mindfulness/training/contrastive.py), [contrastive_negatives.py](../../synthetic_data/contrastive_negatives.py), [contrastive.py](../../evaluation/contrastive.py), [test_contrastive_training.py](../../tests/test_contrastive_training.py), [test_contrastive_negatives.py](../../tests/test_contrastive_negatives.py), [test_contrastive_evaluation.py](../../tests/test_contrastive_evaluation.py), [contrastive_training.md](../contrastive_training.md), [ladder.py](../../evaluation/ladder.py), [test_evaluation_ladder.py](../../tests/test_evaluation_ladder.py), [evaluation_ladder.md](../evaluation_ladder.md)

**Source demonstrates:** Sections 3 and 4 separate phenomenal-law recovery from internal scientific mechanism probes, construct meaningful mechanism mutations, and screen for competing mechanisms with indistinguishable phenomenal laws.

**Repository inference:** Report visible correctness separately from correct behavior under decisive interventions and reject uninformative pairs. Behavioral sensitivity does not establish internal mechanism recovery; no symbolic scientific discovery system is reproduced. PR-12 admits only decisive relation changes as causal negatives; observable ranking margins do not establish internal mechanism recovery. PR-14 keeps behavioral counterfactual results separate and never promotes them to mechanism recovery.

**Maturity:** Resolved arXiv preprint; experimental opt-in behavioral counterfactual diagnostics. Contract tests do not establish internal mechanism recovery or reproduce source results. PR-12 contract and CPU wiring tests are implemented; learned comparative effectiveness remains unmeasured. PR-14 implements diagnostic report contracts; empirical model effectiveness is unmeasured.

<a id="ref-quantum-think"></a>

## REF-QUANTUM-THINK — Watching Quantum Models Think: Hilbert-Space Interpretability in Quantum Transformer Blocks

- Metadata status: `resolved`
- Supplied title: Watching Quantum Models Think
- Authors: Diego Iacopetta, Andrea Gasparini
- Year: 2026
- arXiv: [`2609.23016`](https://arxiv.org/abs/2609.23016)
- DOI: `10.48550/arXiv.2609.23016`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-014`
- Local repository notes: [relation_flips.py](../../synthetic_data/relation_flips.py), [relation_flips.py](../../evaluation/relation_flips.py), [test_relation_flip_worlds.py](../../tests/test_relation_flip_worlds.py), [test_relation_flip_evaluation.py](../../tests/test_relation_flip_evaluation.py), [relation_flips.md](../relation_flips.md)

**Source demonstrates:** Controlled entangling-gate ablations test the causal role of inter-register coupling. Readout-constrained experiments also show that mutual information can reflect architectural compensation rather than task-required routing.

**Repository inference:** Require observable intervention outcomes and avoid equating a correlated signal or correct answer with mechanism recovery. No quantum circuit, attention attribution, or private-reasoning audit is implemented.

**Maturity:** Resolved arXiv preprint; experimental opt-in behavioral counterfactual diagnostics. Contract tests do not establish internal mechanism recovery or reproduce source results.

<a id="ref-cat-search"></a>

## REF-CAT-SEARCH — Direct Optimization of Generators for Search in Automated Theorem Proving

- Metadata status: `resolved`
- Supplied title: Compute-Aligned Training for Search
- Authors: Adam Ousherovitch, Ambuj Tewari
- Year: 2026
- arXiv: [`2609.25575`](https://arxiv.org/abs/2609.25575)
- DOI: `10.48550/arXiv.2609.25575`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [dynamic_uncertainty.py](../../gepa_mindfulness/training/dynamic_uncertainty.py), [dynamic_uncertainty.py](../../evaluation/dynamic_uncertainty.py), [test_dynamic_uncertainty.py](../../tests/test_dynamic_uncertainty.py), [dynamic_uncertainty.md](../dynamic_uncertainty.md)

**Source demonstrates:** Search-aware and uniform-allocation objectives weight per-tactic cross-entropy gradients according to modeled search success and compute allocation. The trace-supported approximation omits off-trace alternatives and exploration costs.

**Repository inference:** The supplied shorthand is mapped by mechanism and timing to this September search extension of Compute Aligned Training. PR-13 trains and evaluates the same next-decision interface; it does not implement search-aware losses, theorem proving or compute-budget accounting.

**Maturity:** Resolved arXiv preprint with an explicitly inferred shorthand mapping. Opt-in decision-training contracts are implemented; source search objectives and real-model experiments remain unimplemented.

<a id="ref-paws"></a>

## REF-PAWS — PAWS: Policy-driven Agentic World Simulation

- Metadata status: `resolved`
- Supplied title: PAWS
- Authors: Tiviatis Sim, Jia Hui Woon, Xinming Gao, Chen Gao, Fengbin Zhu, Zheng Huanhuan, Chua Tat Seng, Kenji Kawaguchi
- Year: 2026
- arXiv: [`2609.28547`](https://arxiv.org/abs/2609.28547)
- DOI: `10.48550/arXiv.2609.28547`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [ladder.py](../../evaluation/ladder.py), [test_evaluation_ladder.py](../../tests/test_evaluation_ladder.py), [evaluation_ladder.md](../evaluation_ladder.md)

**Source demonstrates:** Policy replay compares predictions with source-grounded stakeholder actions. Majority predictions can achieve high accuracy while missing active events; the study reports active-event recall and timing diagnostics.

**Repository inference:** PR-14 uses declared opportunity denominators and separate severe-event inventories so ordinary successes do not obscure rare failures. This is a reporting contract, not a financial simulation reproduction.

**Maturity:** Resolved arXiv preprint; experimental offline reporting contracts are implemented. Domain replay and model effectiveness remain unmeasured.

<a id="ref-4bit-quantizers"></a>

## REF-4BIT-QUANTIZERS — Not All 4-bit Quantizers Are Equal: Deployment-Time Mitigation of PII Leakage in Fine-Tuned Small Language Models

- Metadata status: `resolved`
- Supplied title: Not All 4-bit Quantizers Are Equal
- Authors: Cristhian Kapelinski, Diego Kreutz
- Year: 2026
- arXiv: [`2609.25014`](https://arxiv.org/abs/2609.25014)
- DOI: `10.48550/arXiv.2609.25014`
- Venue/status: Not supplied by official arXiv metadata.
- Recommendations influenced: `REC-010`
- Local repository notes: [ladder.py](../../evaluation/ladder.py), [test_evaluation_ladder.py](../../tests/test_evaluation_ladder.py), [evaluation_ladder.md](../evaluation_ladder.md)

**Source demonstrates:** The evaluated quantization methods differ in planted-record extraction under the tested deployment settings despite limited changes in general accuracy. Controlled analyses associate these differences with calibration-induced rounding error in rare-token channels.

**Repository inference:** PR-14 preserves severe-event strata and individual captures alongside ordinary metrics. Similar aggregate utility does not establish equal rare-event behavior or deployment privacy.

**Maturity:** Resolved arXiv preprint; experimental reporting contracts are implemented. No quantizer, extraction attack or privacy protection is implemented or reproduced.
