# Retained operational freeze record

`accepted_freeze_record.json` is the exact `freeze_record` extracted from the packet frozen
offline on 2026-10-15 after explicit maintainer acceptance of the presented candidate and
all ten limitations. The decision's `authorization_ref` cites that human reply. The record
does not authenticate the human, reviewer, blinding, statistical independence, or study results.

The record contains the complete candidate, including 14 evidence-byte hashes, 48 task
assessments, two resolved findings, and ten limitations, plus the recorded maintainer decision.
The original evidence files remain external retained artifacts; this fixture contains their
bindings, not their contents. Before registration, the operator reproduced the candidate from
those original files and verified the frozen packet against the approved source and decision.

| Artifact | Canonical content digest |
| --- | --- |
| Frozen packet | `c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1` |
| Source draft v3 | `718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e` |
| Acceptance candidate | `e8a41152d1f647779a4facbfddc06c2defd61987377950bb345066df172561b9` |
| Maintainer decision | `4c012dc2300e2780af357651d62273bf991eeddcd32015a11050d1c9d68cc767` |

`tests/test_knowledge_utility_followup_registration.py` reconstructs the frozen content from
the deterministic draft generator and this record, checks the exact frozen digest, and exercises
the production registry without overrides. Its empty capture batch is test input, not a live
execution record. Existing synthetic freeze fixtures remain separate and unregistered.

These tests verify recorded content and scoring compatibility. They do not independently repeat
the original task review or authenticate the authorization reference. The registered packet
remains DEVELOPMENT with live execution, optimizer input, and authority disabled.
