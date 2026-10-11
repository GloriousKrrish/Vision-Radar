# VisionRadar Project-Scoped Agent Rules

## Integrated Productivity Mesh: GSD + Roo Code + CodeRabbit + Ralph Loop

1. **GSD Protocol (Get Shit Done)**:
   - Always adhere to the canonical loop: `SPEC -> PLAN -> EXECUTE -> VERIFY -> COMMIT`.
   - Never declare a milestone or phase complete without empirical verification (captured logs, status codes, terminal outputs).
   - Use search-first discipline to minimize context window degradation.

2. **CodeRabbit Quality Standard**:
   - Zero tolerance for N+1 query patterns in SQLAlchemy. Always use `.in_()` or batch relationships.
   - Guard against synchronous blocking in FastAPI event loops.
   - Ensure WebSocket streaming queues never silently drop terminal state messages (`completed`, `error`).
   - Validate frontend React rendering efficiency: keep canvas render loops isolated in `requestAnimationFrame`.

3. **Ralph Autonomous Self-Healing Loop**:
   - When encountering a failure: do not stall or ask superficial questions.
   - Isolate the error trace, hypothesize the single root cause, apply surgical code patches, and rerun the test suite automatically.
   - Only conclude when all tests pass (pytest suite + frontend build + E2E pipeline verification).
