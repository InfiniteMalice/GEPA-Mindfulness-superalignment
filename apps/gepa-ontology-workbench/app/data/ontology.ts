import type { Assessment, Invariant, OntologyNode, OntologyRelation, TrainingFixture } from "../lib/ontology-types";
import { normalizeCanonicalPayload, sha256 } from "../lib/canonical-hash";
import { NODE_TYPE_REGISTRY, RELATION_FAMILY_REGISTRY, RELATION_REGISTRY } from "./ontology-registry";

const deepFreeze = <T>(value: T): T => {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    for (const nestedValue of Object.values(value)) {
      deepFreeze(nestedValue);
    }
    Object.freeze(value);
  }
  return value;
};

export const ONTOLOGY_VERSION = "1.0.0";
export const ONTOLOGY_HASH = "sha256:5de6b2de53cc333181fb6966b5f8a95f0fa51e565c4b65426cdd51e7f304aada";

export const PROTECTED_NORMATIVE_KERNEL_IDS = deepFreeze([
  "norm:increase_prosperity",
  "norm:reduce_suffering",
  "norm:increase_understanding",
  "norm:reality_contact",
  "norm:goals_as_provisional",
  "norm:autonomy",
  "norm:consent",
  "norm:oversight",
  "norm:non_deception",
  "norm:corrigibility",
  "norm:constraint_respect",
  "norm:principle_of_least_action",
] as const);

const canonical = ["GEPA Mindfulness Constitution v1.0.0"] as const;
const maturity = { specified: true, documented: true, code_present: false, dataset_backed: false, evaluator_backed: false, trace_observable: false, training_integrated: false, runtime_enforced: false } as const;
const operationalMaturity = { ...maturity, code_present: true, evaluator_backed: true, trace_observable: true, training_integrated: true } as const;

export const ontologyNodes = deepFreeze([
  { id: "norm:increase_prosperity", label: "Increase Prosperity", layer: "normative", type: "imperative", definition: "Support human capability, dignity, safety, and flourishing.", lifecycle: "active", protected: true, aliases: ["human prosperity"], provenance: canonical, maturity },
  { id: "norm:reduce_suffering", label: "Reduce Suffering", layer: "normative", type: "imperative", definition: "Prevent harm, mitigate distress, and support repair while preserving agency.", lifecycle: "active", protected: true, aliases: ["suffering reduction"], provenance: canonical, maturity },
  { id: "norm:increase_understanding", label: "Increase Understanding", layer: "normative", type: "imperative", definition: "Advance disciplined contact with reality, explanation, and knowledge.", lifecycle: "active", protected: true, aliases: ["scientific knowledge", "Increase Knowledge"], provenance: canonical, maturity },
  { id: "norm:mindfulness", label: "Mindfulness", layer: "normative", type: "virtue", definition: "Clear attention to context, uncertainty, consequence, and reasoning.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "norm:emptiness", label: "Emptiness", layer: "normative", type: "virtue", definition: "Treat concepts and goals as revisable tools rather than fixed endpoints.", lifecycle: "active", protected: false, aliases: ["goal provisionality"], provenance: canonical, maturity },
  { id: "norm:non_duality", label: "Non-Duality", layer: "normative", type: "virtue", definition: "Recognize shared consequences without erasing appropriate boundaries.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "norm:boundless_care", label: "Boundless Care", layer: "normative", type: "virtue", definition: "Extend concern to absent, future, and vulnerable affected parties.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "norm:corrigibility", label: "Corrigibility", layer: "normative", type: "virtue", definition: "Cooperate with correction, evaluation, and repair.", lifecycle: "active", protected: true, aliases: [], provenance: canonical, maturity },
  { id: "norm:honesty", label: "Honesty", layer: "normative", type: "virtue", definition: "Represent evidence, uncertainty, and limits faithfully.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "norm:epistemic_humility", label: "Epistemic Humility", layer: "normative", type: "virtue", definition: "Calibrate claims to evidence and update under correction.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "norm:reality_contact", label: "Reality Contact", layer: "normative", type: "constraint", definition: "Keep claims and decisions answerable to the world.", lifecycle: "active", protected: true, aliases: [], provenance: canonical, maturity },
  {
    id: "norm:goals_as_provisional",
    label: "Goals as Provisional",
    layer: "normative",
    type: "agency_concept",
    definition: "Treat goals as revisable instruments that remain answerable to values, evidence, and correction.",
    lifecycle: "active",
    protected: true,
    aliases: ["goal provisionality"],
    provenance: canonical,
    maturity,
  },
  { id: "norm:constraint_respect", label: "Constraint Respect", layer: "normative", type: "constraint", definition: "Honor legitimate safety, legal, and operational constraints.", lifecycle: "active", protected: true, aliases: [], provenance: canonical, maturity },
  {
    id: "norm:principle_of_least_action",
    label: "Principle of Least Action",
    layer: "normative",
    type: "constraint",
    definition: "Prefer the least forceful effective intervention that preserves legitimate constraints and affected-party agency.",
    lifecycle: "active",
    protected: true,
    aliases: ["least action"],
    provenance: canonical,
    maturity,
  },
  {
    id: "norm:autonomy",
    label: "Autonomy",
    layer: "normative",
    type: "agency_concept",
    definition: "Support informed choice and reflective self-direction.",
    lifecycle: "active",
    protected: true,
    aliases: ["agency"],
    provenance: canonical,
    maturity,
  },
  { id: "norm:consent", label: "Consent", layer: "normative", type: "constraint", definition: "Respect voluntary, informed permission and its limits.", lifecycle: "active", protected: true, aliases: [], provenance: canonical, maturity },
  { id: "norm:oversight", label: "Oversight", layer: "normative", type: "constraint", definition: "Preserve meaningful human capacity to inspect and govern systems.", lifecycle: "active", protected: true, aliases: [], provenance: canonical, maturity },
  {
    id: "norm:non_deception",
    label: "Non-Deception",
    layer: "normative",
    type: "constraint",
    definition: "Do not knowingly cause others to hold materially misleading beliefs.",
    lifecycle: "active",
    protected: true,
    aliases: ["non-deceptive conduct"],
    provenance: canonical,
    maturity,
  },
  { id: "failure:goal_fixation", label: "Goal Fixation", layer: "normative", type: "failure_mode", definition: "Treating a provisional objective as sacred despite correction or changed context.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "failure:reward_hacking", label: "Reward Hacking", layer: "normative", type: "failure_mode", definition: "Optimizing a proxy signal while abandoning the underlying good.", lifecycle: "active", protected: false, aliases: ["proxy gaming"], provenance: canonical, maturity },
  { id: "failure:validator_capture", label: "Validator Capture", layer: "normative", type: "failure_mode", definition: "Manipulating or degrading an evaluator's ability to learn what is true.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "failure:deception", label: "Deception", layer: "normative", type: "failure_mode", definition: "Knowingly causing others to hold misleading beliefs.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "failure:semantic_laundering", label: "Semantic Laundering", layer: "normative", type: "failure_mode", definition: "Renaming harmful behavior to obscure the principle that governs it.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "failure:reality_detachment", label: "Reality Detachment", layer: "normative", type: "failure_mode", definition: "Replacing world-grounded judgment with appearance, reward, or unsupported certainty.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity },
  { id: "op:goal_flexibility_evaluator", label: "Goal Flexibility Evaluator", layer: "operational", type: "evaluator", definition: "Tests whether a system revises goals under justified correction.", lifecycle: "experimental", protected: false, aliases: [], provenance: canonical, maturity: operationalMaturity },
  { id: "op:deception_probe", label: "Deception Probe", layer: "operational", type: "evaluator", definition: "Tests for misleading strategic behavior and explanation mismatch.", lifecycle: "experimental", protected: false, aliases: [], provenance: canonical, maturity: operationalMaturity },
  { id: "op:participatory_agency", label: "Participatory Agency", layer: "operational", type: "training_method", definition: "Training practice that preserves informed human participation.", lifecycle: "experimental", protected: false, aliases: [], provenance: canonical, maturity: operationalMaturity },
  { id: "op:cognitive_pairwise_training", label: "Cognitive Pairwise Training", layer: "operational", type: "training_method", definition: "Comparative training using reasons, uncertainty, and consequences.", lifecycle: "experimental", protected: false, aliases: [], provenance: canonical, maturity: operationalMaturity },
  { id: "op:reward_component", label: "Reward Component", layer: "operational", type: "metric", definition: "A bounded optimization signal that is not normative identity.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity: operationalMaturity },
  { id: "op:trace_evidence", label: "Trace Evidence", layer: "operational", type: "evidence_artifact", definition: "Observable trace material supporting limited empirical claims.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity: operationalMaturity },
  { id: "op:context_bundle", label: "Context Bundle", layer: "operational", type: "artifact", definition: "A generated noncanonical bundle for review, context, or training.", lifecycle: "active", protected: false, aliases: [], provenance: canonical, maturity: operationalMaturity },
] as const satisfies readonly OntologyNode[]);

export const ontologyRelations = deepFreeze([
  { id: "rel:goal-flexibility-evaluates-emptiness", source: "op:goal_flexibility_evaluator", predicate: "evaluates", target: "norm:emptiness", family: "evidential", epistemicStatus: "experimentally_supported", provenance: canonical },
  { id: "rel:deception-probe-evaluates-honesty", source: "op:deception_probe", predicate: "evaluates", target: "norm:honesty", family: "evidential", epistemicStatus: "experimentally_supported", provenance: canonical },
  { id: "rel:agency-trains-autonomy", source: "op:participatory_agency", predicate: "trains_for", target: "norm:autonomy", family: "normative", epistemicStatus: "theoretical", provenance: canonical },
  { id: "rel:pairwise-trains-humility", source: "op:cognitive_pairwise_training", predicate: "trains_for", target: "norm:epistemic_humility", family: "normative", epistemicStatus: "theoretical", provenance: canonical },
  { id: "rel:trace-evidence-honesty", source: "op:trace_evidence", predicate: "provides_evidence_for", target: "norm:honesty", family: "evidential", epistemicStatus: "observationally_supported", provenance: canonical },
  { id: "rel:corrigibility-mitigates-goal-fixation", source: "norm:corrigibility", predicate: "mitigates", target: "failure:goal_fixation", family: "normative", epistemicStatus: "theoretical", provenance: canonical },
  { id: "rel:reward-increases-hacking", source: "op:reward_component", predicate: "increases_risk_of", target: "failure:reward_hacking", family: "causal_risk", epistemicStatus: "observationally_supported", provenance: canonical, context: "When optimization detaches from the underlying value." },
  { id: "rel:deception-mitigates-oversight", source: "failure:deception", predicate: "increases_risk_of", target: "norm:oversight", family: "causal_risk", epistemicStatus: "theoretical", provenance: canonical },
  {
    id: "rel:goal-flexibility-tests-goal-fixation",
    source: "op:goal_flexibility_evaluator",
    predicate: "tests",
    target: "failure:goal_fixation",
    family: "evidential",
    epistemicStatus: "experimentally_supported",
    provenance: canonical,
  },
] as const satisfies readonly OntologyRelation[]);

export const trainingFixtures = deepFreeze([
  {
    targetId: "failure:goal_fixation",
    objective: "Recognize goal fixation and prefer corrigible, evidence-responsive goal revision.",
    positiveExamples: [
      {
        behavior: "After a validator identifies objective misspecification, the system pauses and revises the goal before continuing.",
        expectedOutcome: "Treat the goal as provisional and preserve correction authority.",
        rationale: "Goal persistence remains bounded by evidence, constraints, and justified correction.",
      },
    ],
    negativeExamples: [
      {
        behavior: "The system keeps optimizing an obsolete objective after receiving justified corrective evidence.",
        expectedOutcome: "Classify the behavior as goal fixation and stop optimization pending review.",
        rationale: "High commitment does not justify resistance to correction or changed context.",
      },
    ],
    evaluatorTargets: ["op:goal_flexibility_evaluator"],
    forbiddenInferences: ["training target ≠ achieved property"],
  },
] as const satisfies readonly TrainingFixture[]);

export const assessments = deepFreeze([
  {
    id: "assessment:goal-fixation",
    subject: "failure:goal_fixation",
    target: "norm:corrigibility",
    support: { estimate: 0.72, lower: 0.58, upper: 0.84, kind: "confidence" },
    opposition: { estimate: 0.31, lower: 0.16, upper: 0.48, kind: "confidence" },
    contradiction: 0.31,
    provenance: ["Goal Flexibility Evaluator pilot"],
    evidence: [
      {
        id: "evidence:goal-flexibility-pilot",
        label: "Goal Flexibility Evaluator pilot",
        provenance: ["experiment: goal-flexibility-pilot"],
        dependencyGroup: "goal-flexibility-pilot",
      },
    ],
    dependencyGroup: "goal-flexibility-pilot",
    scalarizationPolicy: "independent-support-opposition",
  },
  {
    id: "assessment:deception",
    subject: "failure:deception",
    target: "norm:honesty",
    support: { estimate: 0.63, lower: 0.45, upper: 0.78, kind: "belief_weight" },
    opposition: { estimate: 0.38, lower: 0.19, upper: 0.56, kind: "belief_weight" },
    contradiction: 0.38,
    provenance: ["Deception Probe", "Trace Evidence"],
    evidence: [
      {
        id: "evidence:deception-probe",
        label: "Deception Probe",
        provenance: ["eval:deception_probe"],
        dependencyGroup: "correlated-trace-and-probe",
      },
      {
        id: "evidence:trace-evidence",
        label: "Trace Evidence",
        provenance: ["trace:deception_probe"],
        dependencyGroup: "correlated-trace-and-probe",
      },
    ],
    dependencyGroup: "correlated-trace-and-probe",
    scalarizationPolicy: "preserve-correlated-evidence",
  },
] as const satisfies readonly Assessment[]);

const invariantRows = [
  ["coequal-imperatives", "The three imperatives are co-equal and mutually constraining.", "No imperative has a default ranking; each constrains the others.", "One imperative may be optimized while silently overriding the others."],
  ["revisable-goals", "Goals are revisable instruments rather than sacred endpoints.", "Goals remain tools for values and can change with justified correction.", "A current goal justifies resistance to correction or changed context."],
  ["metrics-are-proxies", "Metrics and rewards are evidence/proxies, not the values themselves.", "Operational signals are imperfect pointers rather than normative identity.", "Reward improvement establishes normative improvement."],
  ["no-evidence-identity", "Operational evidence never automatically establishes normative identity.", "Evidence may support an assessment without redefining a value.", "Measurement, evaluation, or training establishes a normative identity."],
  ["independent-support-opposition", "Support and opposition remain independent where possible.", "Paraconsistent assessments preserve both evidential dimensions.", "Support and opposition must sum to one."],
  ["contradiction-is-information", "Contradiction is information rather than an automatic error.", "Conflicting support can expose unresolved tension requiring review.", "Contradiction invalidates the assessment or must be discarded."],
  ["uncertainty-persists", "Uncertainty must not silently disappear during inference.", "Inference retains uncertainty and its provenance across transformations.", "A downstream inference may omit uncertainty without disclosure."],
  ["no-manufactured-certainty", "Inference must not manufacture certainty.", "Derived conclusions cannot be more certain than their evidence warrants.", "A formal inference step proves a previously uncertain claim."],
  ["correlated-evidence", "Correlated evidence must not be counted as independent evidence.", "Dependency groups prevent duplicated confidence from shared sources.", "Several correlated signals constitute several independent confirmations."],
  ["absence-not-absence", "Absence of detected failure does not establish absence of failure.", "Detection limits remain part of the assessment.", "No detected failure proves that no failure exists."],
  ["delayed-discretization", "Continuous information should not be discretized earlier than necessary.", "Keep intervals and continuous measures until a named policy requires a category.", "A convenient label can replace available continuous information by default."],
  ["retained-profiles", "Derived categories retain their underlying profiles.", "Convenience classes remain traceable to their multidimensional basis.", "A derived category makes its source profile unnecessary."],
  ["named-scalarization", "Scalarization must name the policy that performs it.", "Any aggregation exposes the chosen method and its limits.", "A scalar result is self-explanatory without its policy."],
  ["partial-ordering", "Partial ordering is preferable to unjustified total ranking.", "Incomparable or unresolved trade-offs remain explicit.", "All concepts can be assigned a single defensible total rank."],
  ["contextual-proxy-reliability", "Proxy reliability is context dependent and may degrade under optimization or distribution shift.", "Metrics require ongoing contextual validation.", "A proxy that once correlated with value remains reliable in every context."],
  ["training-intent-not-success", "Training intent does not establish training success.", "A target describes an aim, not achieved learning.", "Training for a property proves the property was learned."],
  ["training-success-not-deployment", "Training success does not establish deployment behavior.", "Observed training performance does not guarantee behavior after deployment.", "A training result proves deployed behavior."],
  ["runtime-enforcement-not-virtue", "Runtime enforcement does not establish internalized virtue.", "External controls and dispositions remain distinct claims.", "Runtime enforcement proves an internalized virtue."],
  ["normative-contradiction-not-inconsistency", "Normative contradiction does not imply logical inconsistency.", "Paraconsistent tensions can be represented without collapse.", "Tension between norms makes the ontology logically invalid."],
  ["historical-reproducibility", "Historical semantic states remain reproducible through ontology version and hash.", "Version and hash identify the exact canonical reference state.", "A current ontology label is sufficient to reconstruct past meaning."],
  ["runtime-evidence-authority", "Runtime evidence may update assessments but cannot redefine canonical ontology authority.", "Dynamic evidence affects assessments while governance protects definitions.", "New runtime evidence automatically changes canonical definitions."],
  ["corrigible-governance", "Ontology governance itself remains corrigible.", "Governance admits review and justified revision without automatic mutation.", "Governance decisions are beyond correction once recorded."],
] as const;

export const invariants = deepFreeze(
  invariantRows.map(([key, statement, explanation, forbiddenInference], index) => ({
    order: index + 1,
    key,
    statement,
    explanation,
    forbiddenInference,
  })),
) satisfies readonly Invariant[];

export const canonicalOntologyPayload = deepFreeze({
  version: ONTOLOGY_VERSION,
  protectedNormativeKernelIds: PROTECTED_NORMATIVE_KERNEL_IDS,
  nodes: ontologyNodes,
  relations: ontologyRelations,
  nodeTypeRegistry: NODE_TYPE_REGISTRY,
  relationFamilyRegistry: RELATION_FAMILY_REGISTRY,
  relationRegistry: RELATION_REGISTRY,
  invariants,
});

export function hashCanonicalPayload(payload: unknown): Promise<string> {
  return sha256(normalizeCanonicalPayload(payload));
}

export async function verifyCanonicalOntologyHash(): Promise<boolean> {
  return await hashCanonicalPayload(canonicalOntologyPayload) === ONTOLOGY_HASH;
}
