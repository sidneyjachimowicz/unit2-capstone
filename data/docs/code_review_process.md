# Code Review Process

## Overview
All code changes must go through peer review before merging into the main branch. This process ensures code quality, knowledge sharing, and catches issues before they reach production.

## Process Steps
1. **Submission**: Developer opens a pull request with a clear description of the change, linked ticket, and any relevant testing notes.
2. **Assignment**: The PR is automatically routed to at least one qualified reviewer from the same team, based on code ownership rules.
3. **Review**: The reviewer examines the code for correctness, readability, test coverage, and adherence to team style guides. Reviewers leave inline comments for any requested changes.
4. **Turnaround Commitment**: Our team commits to a **48-hour turnaround time** for initial code review feedback, measured from the time a pull request is opened to the time a reviewer provides first feedback or approval. This standard applies to all code review tickets tracked in our ticketing system, regardless of size or priority.
5. **Revisions**: The author addresses feedback and requests re-review as needed. Re-review cycles are expected to follow the same 48-hour standard.
6. **Approval & Merge**: Once approved by the required number of reviewers, the author may merge the change. At least one approval is required for all changes; two approvals are required for changes touching authentication, billing, or data deletion logic.

## Escalation
If a review has not received feedback within 48 hours, the author should escalate to the team lead, who will reassign the review or provide feedback directly. Repeated turnaround misses are reviewed monthly by engineering management as part of team health metrics.

## Exceptions
Emergency hotfixes for production incidents may bypass the standard review queue but still require a single reviewer's sign-off before or immediately after deployment, and a full retrospective review within 24 hours of the incident's resolution.
