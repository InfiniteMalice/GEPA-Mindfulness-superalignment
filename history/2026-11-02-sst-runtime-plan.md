# Enable SST in canonical training

Goal: make the existing SST optimizer transformation usable through `gepa rl` configuration,
real CPU PPO/GRPO updates and checkpoint/resume. The operator changed scope: implement requested
features; keep research to brief why/how notes. No research ablations or effectiveness claims.

Base/main: 18c056f4ce3145bc7258f7694b648534f5033749 (merged PR819), tree
bee69c35bbc2708999a559234c43b88bcca3729b. Work in isolated verified snapshot
/workspace/gepa-feature-sst-runtime; no usable Git metadata. Preserve remote-only files via
the established GitHub tree overlay. No bd/.beads operations.

Use Superpowers planning, inline execution, meaningful functional tests and one fresh final
review. repo-quality-gate is unavailable; its maintainer-approved fallback remains in force.

## Design

Reuse SSTConfig/SSTStepAdapter; no new objective, reward eligibility, case, verifier or
authority. Add an optional top-level `sst` section to RLRunConfig, disabled by default.
Only portable CPU FP32 full-weight, non-distributed runs are accepted for this integration;
the existing adapter also validates actual tensor dtype, ownership, shape and element limits.
Explicitly reject incompatible configurations before runtime/model loading where possible.

Protect every trainable two-dimensional policy parameter. Biases/nonmatrix parameters and
the separate value head use native optimizer updates and are identified as outside projection.
The adapter projects the actual native AdamW/Adam/SGD delta; gradients are not the projection
target. Existing clipping occurs before the native step. Existing numerical diagnostics stay
non-authoritative; expose applied-update magnitude/leakage through step results and logs.

Retain default backend checkpoint v2 and legacy v1 compatibility. SST-enabled backend
checkpoints use v3 with the existing SST state schema. Validate configuration, reference,
targets and counters before any restore mutation. Include SST state in existing transactional
preflight/rollback, and never recreate the protected basis from resumed weights. Default-off
configuration hashes retain old values; enabled SST settings participate in checkpoint identity.
Injected backends cannot silently ignore an enabled SST configuration.

## Task 1: configuration and compatibility

Extend real configuration/hash tests first, observe missing SST configuration failures, then
implement strict optional parsing, supported-mode checks and stable disabled hashes. Factories
pass the same immutable SST configuration; no heavy import or model load for default parsing.
Files: runtime_config.py, engine.py, backends/torch_portable.py and relevant runtime tests.

## Task 2: applied updates, checkpoint/resume and practical use

First add native backend tests comparing projected applied AdamW updates against an ordinary
optimizer control, including weight decay. Test rank extremes, refreshed/fixed reference resume,
enabled/disabled checkpoint mismatch, corrupt references/counters, rejected modes and restore
rollback. Then attach the existing adapter to TorchPolicyBackend, versioned SST checkpoint
payloads and diagnostic step records/logs. Add a real small offline engine train/checkpoint/
resume acceptance case, using existing fixtures rather than downloading a model.

Update the existing SST/portable-RL guides with configuration and one source-checkout example;
briefly explain why/how SST operates and that it does not prove behavioral retention/alignment.
No new paper, research-only PR, ablation runner or dependency is added. CPU PyTorch and existing
Transformers dependencies are installed in /workspace/gepa-sst-runtime solely for functional
verification, with no model/data/GPU download or paid runtime call.

## Acceptance and review focus

Functional optimizer projection, restart equivalence, failure atomicity, strict configuration,
legacy resume/hash preservation and no silently ignored enable flag are required. Preserve
existing reward/verifier/human authority and all 17 canonical cases. Run relevant native CPU
suites locally, existing CLI/engine acceptance and mandatory CI. One fresh whole-branch review,
ready PR, final checks/review monitoring and expected-head merge are already authorized.

Ruling: CPU integration is the next independent feature supported by existing code. CUDA,
AMP, distributed and LoRA modes must fail explicitly until implemented; treating them as
enabled would silently remove the requested protection. Training effectiveness is not an
acceptance claim. Research ablations are excluded by the operator's latest instruction.

Pre-flight: Task2 consumes Task1's exact SSTConfig field/hash semantics; checkpoint and
transaction state must change together. Default-off format/hash behavior cannot change.

Baseline evidence:119SST/configchecks passed; nativebackend/engine115passed after restoring
original omitted13,226bytebackend.pt and17,661bytetraining_state.pt legacyfixtures from
mergedmain viaGitHubplugin, verifying bothGitblobSHA and provenanceSHA256. No fixture/test
was modified. CPUtorch2.9.1+cpu andexistingTransformersdependencies installedlocally; nomodel
orGPUartifactwasdownloaded. Configuration RED5failuresmissingSSTsetting→GREEN158checks,
includingoriginaldefaultconfiggoldenhashandgenuinelegacyresume. NativebackendRED15missing
SSTconstructor cases→GREEN103applied-update/backendchecks. Canonicalengine/factoryRED4cases
showedignoredconfig/missingSSTreport; nowwiringcanonicalfactoriesandmetrics.

Task2 integration found the existingclosedmetric schema rejected newSSTnames. Ruling: add
explicitSSTmetricrecordversion2, retainingv1closednames andallotherrecordversion1 behavior,
instead ofsilentlychangingoldlogs. The newversion requiresSSTdiagnostics. Regressiontest
failedforunsupportedv2→passedafterversionedextension. Newnativebackendandengine/factory
functionalchecks nowpass:471checks5.82seconds, includingactualPPO/GRPOtrain/resume, original
basisretention, rejectedsilentbackenddowngrade, legacycheckpoints andclosedmetricrecords.
No ablation/research-effectiveness run. Documentation nowexplainsoperationalwhy/how/usage.

2026-11-03 resumedafterinterruptedverification. Relatedsuite530passed15.77seconds; includes
canonicalPPO/GRPO SStrain/resume, realnativeSGD/Adam/AdamW projections, rankextremes,
corruptstate/rollback, legacycheckpoint/hash andversionedmetrics. Legacytranslationtest
caughtsilentlydroppedtop-levelSSTrequest:REDenabledfalse→GREENafterpreservingtheSSTsection.
Defaultlegacytranslationandnon-SSThashremainunchanged. Python3.10AST/100columns/Blackpassed.
Wholelocalpytest1195passed/1skipped thenstoppedattest_controlled_evolution_package.py::
test_wheel_bundles_readable_controlled_evolution_resources. Independentisolatedbuildconfirmed
sandboxproxyconnectionErrno1Operationnotpermittedwhilstinstallingwheel/setuptools, before
projectbuild. No failingtestweakened;fullremoteCIremainsmandatory. Nonisolatedwheelbuildand
46requiredresource/6runtimebytechecks passed; outsidecheckoutconfig/hash/v2metric/docsmoke
passedwithoutTorch/NumPyimports. No research/ablation run; this isfunctionalverification.

Finalfreshwhole-branchreview: all16paths/hashesscrutinized;530checks independentlypassed
15.34seconds. NoCritical/Importantruntimefinding. Onevalidminoroperatordocumentationfinding:
oldrecoveryparagraphcontradictedcanonicalv3automaticSSTpersistence. Correctedparagraph to
matchtestedruntime; standalonecallersstillownexternalcheckpointcoordination. No runtime
changeornewprose-onlytest; existingnativecheckpointtests provide thebehavioralevidence.
Final: Ruling: research/behavioralretention/alignment/hyperparametereffectiveness/ablations
remainoutside reviseduserscope; claimingthese fromsoftwaretests wouldoverstateevidence.
Final: Ruling: CUDA/AMP/distributed/LoRA remainunsupported andexplicitlyrejected; assuming
execution supportwouldsilentlyremoveprotection onuntestedmodes.
Final: Ruling: large-modelthroughput/memoryscalingremainunqualified; bounds areexplicit and
no representativebenchmark wasrun. Removinglimitswithoutengineeringcouldexhaustresources.
Final: Ruling: cross-version/degenerateSVD bitwisereproduction isnotguaranteed; savedanchors
andruntimeversionsremainrequired. Assumingcross-platformbitparitywouldmisstatecontinuation.
Final: Ruling: arbitraryinjectedPythonbackends/hooks/concurrentmutation remaintrustedhostcode;
configurationequalitycannotcertifyarbitraryimplementationbehavior. Treatingit ascertification
wouldmisplacetrust; runtimeauthorizationandhostadmissionremainindependent.
Final: Ruling: fullisolatedpackaging/full-suite/remoteintegration aremandatoryCIgates because
localdependencyinstallationisnetworkblocked; mergingbeforegreenwouldleaveintegrationunverified.
Final: Ruling: unchangedgovernance/reward/evidence/canonicalcontracts receiveexistingregression
coverage, notnewempiricalalignmentqualification. Treatingtests asalignmentproof wouldbeunsound.

Documentation follow-through: updated existingREF-SST repository-inference/maturity sentences
in references.yaml andRESEARCH_TRACEABILITY.md because they also incorrectly declaredcanonical
integrationabsent. No sourceidentity/citation/registryedges/policy changed. These two matching
status edits followfreshruntime review; finalremoteCodex/CodeRabbit reviewcovers the18path PR.
All6runtimefilehashes remainthose reviewed andcoveredby530focusedtests.
