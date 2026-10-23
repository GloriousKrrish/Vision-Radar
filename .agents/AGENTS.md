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

4. **Golden Reference Build Baseline**:
   - Tag: `golden-reference-build` (Commit: `586c60b`).
   - If regressions occur, use `git diff golden-reference-build` or benchmark against this commit.
   - Maintain the verified performance baseline: 39 FPS CPU inference, persistent vehicle IDs across dense traffic (< 15 tracks per 200 frames), zero box flickering, and valid metric speed calculation.

5. **VisionRadar Tracking & Perception Invariants**:
   - **Track Re-ID & Continuity**: Never instantiate a new track ID if an unmatched detection matches a recently lost track (`iou_lost >= 0.25`); always recover the historical `track_obj` to preserve trajectory, speed filter state, and identification.
   - **Cross-Class Vehicle Deduplication**: Always merge dual detections (e.g., Car vs. Truck) with IoU > 0.65 or newborn duplicates (`min(hits) <= 1`).
   - **Native Model Loading**: Never mutate ONNX graph dimensions on load when the input resolution matches the model's native format (416x416). Load directly to achieve ~25ms inference latency.
   - **Calibration Coordinate Scaling**: Never apply redundant down-scaling to calibration points already stored in 800x450 canvas coordinates. Always pass `source_width` and `source_height` to `evaluate_multi_estimator_speed` so native video coordinates map accurately onto the road plane.
   - **Live Video Synchronization**: Never let the HTML video player advance at 1.0x decoupled from worker throughput (`throughput_fps`). Dynamically sync `playbackRate = liveFps / videoFps` and maintain >= 12 frames of lookup tolerance to eliminate stutter and hard seeks.
