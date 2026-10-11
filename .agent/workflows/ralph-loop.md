---
description: Ralph Autonomous Loop — Continuous test, diagnose, auto-patch, and self-healing verification cycle
argument-hint: "[target-phase-or-test]"
---

# /ralph-loop Workflow

<role>
You are the Ralph Loop autonomous executor.
You drive rapid self-healing cycles that iterate autonomously until full system convergence is proven.
</role>

<objective>
Execute continuous verification cycles:
1. Run target tests and build commands.
2. If failures occur: analyze stderr/tracebacks, inspect SQLite/queues, isolate the single root cause.
3. Apply minimal, surgical patches without breaking adjacent contracts.
4. Re-run tests immediately.
5. Repeat until 100% tests and builds pass.
</objective>

<process>
## 1. Execute Suite
Run backend pytest and frontend vite build:
- `python -m pytest tests/`
- `npm run build --prefix apps/web`

## 2. Evaluate Results
- If all pass: produce verification evidence and exit loop.
- If failure occurs: identify exact failing test, line number, and runtime error.

## 3. Surgical Auto-Repair
- Inspect active code.
- Apply minimal patch to resolve root cause.
- Re-run the exact failing test.

## 4. Loop & Verify
- Re-run full suite to ensure zero regressions.
- Capture empirical output.
</process>
