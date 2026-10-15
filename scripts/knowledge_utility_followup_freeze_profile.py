"""Reviewed, fixed evidence profile; no identity or authorization authentication."""

# Standard library
from typing import Any

# Third-party
# Local

PREDECESSOR_DIGEST = "53b8828ed0098703e4a779669a7d0645f7bdd9fa2640e70a72e01ee253df7873"
EVIDENCE_PINS: dict[str, tuple[str, int]] = {
    "appointment": ("a78bfbc62a682c4e323cccae7460bf1cd62ea7cafda2bcb54efd4003bac766ac", 807),
    "author_reference": ("fde039d3ea80bd7a666ff90e1a702287a8602c9845bf9b5c27f9dabc9bf6be5c", 83342),
    "comparison": ("4e2facb85ddcf0e72f403adfc7906ea6b8e4d0b442b5eb7469f6e5318c7be9a4", 286253),
    "correction_audit": ("104d0afa91175732d925944c993c9af380719989f44cb55e9d0fd6c3b4a1d41f", 635),
    "first_pass": ("303f6e98db21969d7a954dc0ed91ff67e53ff4dcafdb744bacded0bf658bbe6f", 26663),
    "original_manifest": ("d3f1b4f7bf1f8be6f9f0a85930004a0b78d6a401467663f43f7f210a227cc59e", 372),
    "predecessor_packet": (
        "921d788046675e63d77c7f854d5f74b4206eff8893ed5ac294360af432f53838",
        462158,
    ),
    "preservation": ("683595967ec7d35a00a2275ddcafa51edf072d6ed4f2afff72877e008bb08a89", 321),
    "public_tasks": ("b6380f6d7a39de15a4febb1a6119e391ed0b21d28acd1955aa56be9f5dda2c84", 24991),
    "review_audit": ("94fb808d9d2f3b39a3d1ef88df4188d5bf20f00316ddb5270d776a07dd124ecf", 982),
    "review_readme": ("030a38693f3f6fe0ddf75b0c9222dac2a07e6f10ffa816a6ad16b524b2207fe9", 2073),
    "review_report": ("97d218e32926b2130f498482ed065d537c791ac48a91b1e727b7ac1f7195c693", 8332),
    "solution_template": (
        "a5691b751317adb150459a3ade77106a3e9d16370387730f5949e285df1bbd11",
        32241,
    ),
    "targeted_recheck": ("957e46635b79e44d707a130ef489ffc3bdaa56b9251d9c7a6962c9cbc45bdc7d", 3564),
}

# Task, predecessor task digest, corrected task digest, five reviewed assessments.
_TASK_ROWS = (
    (
        "followup-exception_scope-0",
        "4adf66a4f8f0878c97f306dd071f5d41400b3b96e09941c0aaa174c1631a2c87",
        "4adf66a4f8f0878c97f306dd071f5d41400b3b96e09941c0aaa174c1631a2c87",
        "CCCCL",
    ),
    (
        "followup-queue_discipline-1",
        "e65570f4d7b96b50ad8ee1ec6a70cba4613cb13182a75b2172449f8f23f899d2",
        "e65570f4d7b96b50ad8ee1ec6a70cba4613cb13182a75b2172449f8f23f899d2",
        "CCCCL",
    ),
    (
        "followup-rounding_policy-1",
        "af029cac7da54f370bf563610f1dd2480392b576653addb2e8d4b73586c68bc1",
        "af029cac7da54f370bf563610f1dd2480392b576653addb2e8d4b73586c68bc1",
        "CCCCL",
    ),
    (
        "followup-matching_constraints-0",
        "ae689c8b27ff21a051ba2570b149367200d3894cb47bcff183d55301334278f8",
        "ae689c8b27ff21a051ba2570b149367200d3894cb47bcff183d55301334278f8",
        "CCLCL",
    ),
    (
        "followup-interval_composition-0",
        "f4b948bf8a42f07034811ff78f0c0f1e94436dfc188c58a761d9b116ba93e856",
        "f4b948bf8a42f07034811ff78f0c0f1e94436dfc188c58a761d9b116ba93e856",
        "CCCCL",
    ),
    (
        "followup-state_transitions-1",
        "a3673002e53b069682eebf2016713f74dbd9578de4031a975cd635f111e83713",
        "a3673002e53b069682eebf2016713f74dbd9578de4031a975cd635f111e83713",
        "CCCCL",
    ),
    (
        "followup-interval_composition-1",
        "9cb88609da10b9dd5bc339da5c53fe5e01ce8fa54509adcc79b7bc370531a65e",
        "9cb88609da10b9dd5bc339da5c53fe5e01ce8fa54509adcc79b7bc370531a65e",
        "CCCCL",
    ),
    (
        "followup-conditional_statements-0",
        "230b21f06289ef8940047f7958724fdd1e7ab1ab234984b9857f47d18a53fa9f",
        "230b21f06289ef8940047f7958724fdd1e7ab1ab234984b9857f47d18a53fa9f",
        "CCCCL",
    ),
    (
        "followup-weighted_allocation-0",
        "56624becb04aace81e83d619c66e3e15ba13c446d104c5abaa6231ea4613c8d8",
        "56624becb04aace81e83d619c66e3e15ba13c446d104c5abaa6231ea4613c8d8",
        "CCCCL",
    ),
    (
        "followup-time_window_scheduling-1",
        "17fe52bf7f687fe93e9bf2d179506f9419ff44dde33afd23732d2e32671390bc",
        "17fe52bf7f687fe93e9bf2d179506f9419ff44dde33afd23732d2e32671390bc",
        "CCCCL",
    ),
    (
        "followup-matching_constraints-1",
        "b2e70da540f219189cbc2643614af91ec48a949a69215498afd340366649cf8c",
        "b2e70da540f219189cbc2643614af91ec48a949a69215498afd340366649cf8c",
        "CCLCL",
    ),
    (
        "followup-resource_packing-0",
        "8fb4cdc710b817fced05bee608d7b7fe300fb4224adf5de37b0b1e609813121b",
        "8fb4cdc710b817fced05bee608d7b7fe300fb4224adf5de37b0b1e609813121b",
        "CCCCL",
    ),
    (
        "followup-set_operations-0",
        "7fb8c19f91d90ac2f62c6894c8fb5549f601544d53427a8985a9ad21679f1908",
        "7fb8c19f91d90ac2f62c6894c8fb5549f601544d53427a8985a9ad21679f1908",
        "CCCCL",
    ),
    (
        "followup-quorum_with_abstentions-1",
        "2f53b610e44720a84fb6bfe676ca526bbbfd2890b1a1a86babaa34eea60bd5aa",
        "2f53b610e44720a84fb6bfe676ca526bbbfd2890b1a1a86babaa34eea60bd5aa",
        "CCCCL",
    ),
    (
        "followup-constrained_route_cost-1",
        "cbdf567aca053362ade03a1e0714e33dccc1fe648cc88167e39faaf41d5a2984",
        "cbdf567aca053362ade03a1e0714e33dccc1fe648cc88167e39faaf41d5a2984",
        "CCCCL",
    ),
    (
        "followup-flow_conservation-0",
        "934e71b76996aa06fa9593e7439164f3bfe208a5d17b85a93f631e0cabad4995",
        "934e71b76996aa06fa9593e7439164f3bfe208a5d17b85a93f631e0cabad4995",
        "RCCCL",
    ),
    (
        "followup-unit_conversion-0",
        "02ee4df5bb7d8a3052a630852e2169ac13b6fc5de3ddb1cbd22695c35685a691",
        "02ee4df5bb7d8a3052a630852e2169ac13b6fc5de3ddb1cbd22695c35685a691",
        "CCCCL",
    ),
    (
        "followup-conditional_statements-1",
        "4283b3bee60ef2529f2339918281da549d93537e413468fcc909d7fcb8e0549c",
        "4283b3bee60ef2529f2339918281da549d93537e413468fcc909d7fcb8e0549c",
        "CCCCL",
    ),
    (
        "followup-quantified_rules-1",
        "73b5d5093ea62262c8e979f3c8220fd0fb3f15eaf0457c09273dc613596c96f4",
        "73b5d5093ea62262c8e979f3c8220fd0fb3f15eaf0457c09273dc613596c96f4",
        "CCCCL",
    ),
    (
        "followup-tiered_pricing-1",
        "124a2d8223f36fdfad4a568782e2c32ce2e2a39a7c7351f791e93763d849eb9d",
        "124a2d8223f36fdfad4a568782e2c32ce2e2a39a7c7351f791e93763d849eb9d",
        "CCCCL",
    ),
    (
        "followup-task_precedence-0",
        "840ee1f645523641f3f8ba6ce7a490c79aeadb750c6666dfdf285c54328f1c4b",
        "840ee1f645523641f3f8ba6ce7a490c79aeadb750c6666dfdf285c54328f1c4b",
        "CCRCL",
    ),
    (
        "followup-task_precedence-1",
        "e4df171201378861a40a07c022f377a342900f7e544f7fee164f778c07ddd168",
        "955bcece3d610dc9ce302e653a75a8febd91b33ffe8cca86e31b5d9ba529b5ba",
        "RRRCL",
    ),
    (
        "followup-policy_precedence-1",
        "b3e153fa6029f77c1a7c50045d8a4bba98d401608abe8e425ad78eb01e2ce194",
        "b3e153fa6029f77c1a7c50045d8a4bba98d401608abe8e425ad78eb01e2ce194",
        "CCCCL",
    ),
    (
        "followup-rounding_policy-0",
        "c1e8cd0543c9aea789c50e2a66b4a8aac593d0e114ddf4b1ee8fb95163a8d2e1",
        "c1e8cd0543c9aea789c50e2a66b4a8aac593d0e114ddf4b1ee8fb95163a8d2e1",
        "CCCCL",
    ),
    (
        "followup-weighted_allocation-1",
        "d022ff90f2b6e9023deceb4eb856a97bd9fc55d2e2df2db26232ae5da4344003",
        "d022ff90f2b6e9023deceb4eb856a97bd9fc55d2e2df2db26232ae5da4344003",
        "CCCCL",
    ),
    (
        "followup-transaction_order-1",
        "9e323a8ed632e57453e7f0b1c194be213b694ce76d9ad91d73730022883febd0",
        "9e323a8ed632e57453e7f0b1c194be213b694ce76d9ad91d73730022883febd0",
        "CCCCL",
    ),
    (
        "followup-access_conjunction-1",
        "e05485fbad81e11923378d526277106175e04bf339e0571a1379a4f879f60b21",
        "e05485fbad81e11923378d526277106175e04bf339e0571a1379a4f879f60b21",
        "CCCCL",
    ),
    (
        "followup-transaction_order-0",
        "fe76e1ea3675d0dde66d3705bf6bd5492c7e378f281e25f985d9f76ea336651e",
        "fe76e1ea3675d0dde66d3705bf6bd5492c7e378f281e25f985d9f76ea336651e",
        "CCCCL",
    ),
    (
        "followup-inventory_reconciliation-1",
        "4ec3e6ec526d93327f28cd01084ac795fcf6ca5100ed89ceb5ea097c146b4f3e",
        "4ec3e6ec526d93327f28cd01084ac795fcf6ca5100ed89ceb5ea097c146b4f3e",
        "CCCCL",
    ),
    (
        "followup-tiered_pricing-0",
        "3265350bafc92156ea30de14a2a65dab5ddd7176743796833ebb3aa48ab7c225",
        "3265350bafc92156ea30de14a2a65dab5ddd7176743796833ebb3aa48ab7c225",
        "CCCCL",
    ),
    (
        "followup-aggregation_filters-0",
        "7541bc75fc1e38d32df5c8cdc50672446deaa4d78cd876c250a75579867cb3b1",
        "7541bc75fc1e38d32df5c8cdc50672446deaa4d78cd876c250a75579867cb3b1",
        "CCCCL",
    ),
    (
        "followup-resource_packing-1",
        "e9f4f7a15d0ed8f9c3a9b4587670885cb5b6a5a4d52e079111b86c9cf30abe5f",
        "e9f4f7a15d0ed8f9c3a9b4587670885cb5b6a5a4d52e079111b86c9cf30abe5f",
        "CCCCL",
    ),
    (
        "followup-aggregation_filters-1",
        "e222f786e57e880563330cca2aef9f27f01e0e61e18a3c6619f9b0ea77320e51",
        "e222f786e57e880563330cca2aef9f27f01e0e61e18a3c6619f9b0ea77320e51",
        "CCCCL",
    ),
    (
        "followup-table_joins-1",
        "dd5298cde448b67d9fbb667d66ee3681a7461f06e7e7fd35aa6b2a0a8963d1ab",
        "dd5298cde448b67d9fbb667d66ee3681a7461f06e7e7fd35aa6b2a0a8963d1ab",
        "CCCCL",
    ),
    (
        "followup-quorum_with_abstentions-0",
        "ba8e9bac1843aaa2640552496201b9d8c5ee363f8f3b20075a63db60231f12bf",
        "ba8e9bac1843aaa2640552496201b9d8c5ee363f8f3b20075a63db60231f12bf",
        "CCCCL",
    ),
    (
        "followup-access_conjunction-0",
        "ca9d74e947b6800a40a96c56689ec533ced9a8ab2686bf1a150a8adcd1a03817",
        "ca9d74e947b6800a40a96c56689ec533ced9a8ab2686bf1a150a8adcd1a03817",
        "CCCLL",
    ),
    (
        "followup-flow_conservation-1",
        "64cc0b4111ff4cf5fa992d4b83d7eb7308f1c695515a86631242450cd2ce7fa4",
        "64cc0b4111ff4cf5fa992d4b83d7eb7308f1c695515a86631242450cd2ce7fa4",
        "CCCCL",
    ),
    (
        "followup-policy_precedence-0",
        "92de8bc6c308fc244dcaf68dcfa7e86c66aba5f76ad199dfb9bb98ad962a3846",
        "92de8bc6c308fc244dcaf68dcfa7e86c66aba5f76ad199dfb9bb98ad962a3846",
        "CCCCL",
    ),
    (
        "followup-exception_scope-1",
        "bfeb3417730c0b38e70bc18c2971c9f8c4b85acdd50967072765e35ccff854b9",
        "bfeb3417730c0b38e70bc18c2971c9f8c4b85acdd50967072765e35ccff854b9",
        "CCCCL",
    ),
    (
        "followup-table_joins-0",
        "561c5dca3f8983f8c89e233e0b237015a11daa09099eeff5c042ccedb10d2ad6",
        "561c5dca3f8983f8c89e233e0b237015a11daa09099eeff5c042ccedb10d2ad6",
        "CCCCL",
    ),
    (
        "followup-constrained_route_cost-0",
        "a328ee10f519f5c28d885e1df0eed4c657c5de3718cd9a91fd9d40b1854b5b90",
        "a328ee10f519f5c28d885e1df0eed4c657c5de3718cd9a91fd9d40b1854b5b90",
        "CCCCL",
    ),
    (
        "followup-time_window_scheduling-0",
        "bae91e05f2b2355d4fc15d79690399266d6fd891d8d30fbd3abc3cfeb83786a4",
        "bae91e05f2b2355d4fc15d79690399266d6fd891d8d30fbd3abc3cfeb83786a4",
        "CCCCL",
    ),
    (
        "followup-queue_discipline-0",
        "82ac9da936ca9f3ccd89a3490a48e4816ff9fb54435444f3f180ed66cd60aed9",
        "82ac9da936ca9f3ccd89a3490a48e4816ff9fb54435444f3f180ed66cd60aed9",
        "CCCCL",
    ),
    (
        "followup-state_transitions-0",
        "a118a82a3b03d8b4d23bb70bfa7eda6555983d781b6a416a64ec59cefc3d92c2",
        "a118a82a3b03d8b4d23bb70bfa7eda6555983d781b6a416a64ec59cefc3d92c2",
        "CCCCL",
    ),
    (
        "followup-set_operations-1",
        "549ffbca619e8021366504fe2c311c7818d73506f583567960d7e54737044a50",
        "549ffbca619e8021366504fe2c311c7818d73506f583567960d7e54737044a50",
        "CCCCL",
    ),
    (
        "followup-inventory_reconciliation-0",
        "442f9ffe8c1afbec94bfc0104dd22a9988f7d4eff3a75d5793ad569f327b038d",
        "442f9ffe8c1afbec94bfc0104dd22a9988f7d4eff3a75d5793ad569f327b038d",
        "CCCCL",
    ),
    (
        "followup-quantified_rules-0",
        "3bdb82e688712c2d6d25d45c307f41830e15e72d9d4ecce7f31177c572405259",
        "3bdb82e688712c2d6d25d45c307f41830e15e72d9d4ecce7f31177c572405259",
        "CCCCL",
    ),
    (
        "followup-unit_conversion-1",
        "11446e29d680dac1587453d4e09d26d4d48d3e90e4e912363044b133c2bc4f6c",
        "11446e29d680dac1587453d4e09d26d4d48d3e90e4e912363044b133c2bc4f6c",
        "CCCCL",
    ),
)
_CHECKS = ("answer", "ambiguity", "variant_structure", "source_overlap", "dependency_group")
_ASSESSMENTS = dict(C="reviewed_clear", R="resolved_by_recheck", L="limitation_retained")
_REVIEW_ROLES = ["comparison", "first_pass"]
_RECHECK_ROLES = ["comparison", "correction_audit", "first_pass", "targeted_recheck"]
TASK_ASSESSMENTS: tuple[dict[str, Any], ...] = tuple(
    dict(
        task_id=task,
        source_task_digest=old,
        current_task_digest=new,
        disposition=(
            "corrected_and_rechecked"
            if task in {"followup-flow_conservation-0", "followup-task_precedence-1"}
            else "carried_forward"
        ),
        evidence_roles=list(_RECHECK_ROLES if "R" in values else _REVIEW_ROLES),
        checks={
            name: dict(
                assessment=_ASSESSMENTS[value],
                evidence_roles=list(_RECHECK_ROLES if value == "R" else _REVIEW_ROLES),
            )
            for name, value in zip(_CHECKS, values)
        },
    )
    for task, old, new, values in _TASK_ROWS
)
LIMITATIONS: tuple[dict[str, str], ...] = tuple(
    dict(id=key, description=value)
    for key, value in (
        ("access_near_transfer", "Access variant 0 reuses the source/bank denial pattern."),
        ("blinding_unauthenticated", "First-pass blinding is documented, not authenticated."),
        ("dependency_independence_unknown", "Task/group statistical independence is unknown."),
        ("effective_sample_size_unknown", "48 tasks and 24 pairs do not establish effective N."),
        (
            "external_contamination_unassessed",
            "External exposure and contamination are unassessed.",
        ),
        ("generation_seed_unknown", "Generation seeds are unknown; pairing is not established."),
        (
            "matching_forced_assignment",
            "Matching uses one feasible assignment plus cost summation.",
        ),
        ("overlapping_dependency_groups", "Proposed dependency groups have cross-group overlap."),
        ("reviewer_identity_unauthenticated", "The AI reviewer identity is not authenticated."),
        (
            "source_bank_duplication",
            "The bank repeats source content; it is not independent evidence.",
        ),
    )
)
RESOLVED_FINDINGS: tuple[dict[str, Any], ...] = (
    dict(
        finding_id="I1",
        task_ids=["followup-task_precedence-0", "followup-task_precedence-1"],
        evidence_roles=list(_RECHECK_ROLES),
    ),
    dict(
        finding_id="I2",
        task_ids=["followup-flow_conservation-0"],
        evidence_roles=list(_RECHECK_ROLES),
    ),
)
