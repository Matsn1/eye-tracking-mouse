# Eye-tracking mouse: hands-free computer control with a webcam

Control the mouse using only your face: gaze moves the cursor, blinks click, opening your mouth opens the on-screen keyboard. It runs with a normal laptop webcam.

<!-- Add a demo video here -->

## Pipeline
1. **Landmarks:** MediaPipe Face Mesh with iris refinement gives 478 face landmarks per frame. The iris center (landmark 474) is used as the gaze point.
2. **Calibration:** the user looks at the four screen corners for 2 s each (with a beep cue). The averaged iris positions define the mapping range.
3. **Mapping:** iris position is normalized into [0, 1] using the calibrated range, clamped, and scaled to screen pixels.
4. **Smoothing (two stages):** a 5-frame moving average to remove landmark jitter, then an exponential low-pass filter (α = 0.05) so the cursor moves steadily instead of shaking.
5. **Gestures** (from landmark distances):
   - Eye openness = vertical distance between eyelid landmarks, compared with a threshold.
   - Left blink < 0.4 s → click. 0.4–1.2 s → toggle click-and-hold (drag). > 1.5 s → double click.
   - Right blink → right click (with a cooldown).
   - Both eyes closed → ignored (natural blink) and releases any held drag.
   - Mouth open > 0.2 s → opens the Windows on-screen keyboard, and drag is disabled while typing.

## Problems I hit
- The cursor shook at every frame → added the two-stage smoothing (moving average + exponential filter).
- Natural blinks caused clicks → only one-eye blinks trigger actions, and blink duration separates click / hold / double click.
- The usable gaze range is small and differs per person → per-user corner calibration.

## Files
- `eye_mouse.py`: Python, OpenCV, MediaPipe, PyAutoGUI
