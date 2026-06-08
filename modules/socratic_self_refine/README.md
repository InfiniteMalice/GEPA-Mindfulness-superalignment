# Socratic Self-Refine

This package provides an optional, bounded refinement scaffold inspired by Socratic Self-Refine (SSR). It decomposes public reasoning into verifiable sub-question/sub-answer units, re-solves selected units in a controlled way, estimates self-consistency, and selectively repairs weak units instead of blindly rewriting a full answer.

SSR is disabled by default. It preserves the original trace and emits inspectable reports suitable for evaluation, inference-time review, and peer review. Policy checks can be run before and after repair so semantic-intent decisions, bounded-help modes, refusals, clarification requirements, memory trust boundaries, and high-stakes ambiguity handling are not bypassed.

This module does not add new 17-case categories and does not require hidden chain-of-thought disclosure.
