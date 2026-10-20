package com.facecam.headtracker.network;

/**
 * Thread-safe immutable representation of a head pose measurement,
 * with state management and smoothing/damping calculations.
 */
public record HeadPoseState(
        float xMm,
        float yMm,
        float zMm,
        float yaw,
        float pitch,
        float roll,
        boolean isValid,
        long timestampMs
) {
    public static final HeadPoseState ZERO = new HeadPoseState(0f, 0f, 0f, 0f, 0f, 0f, false, 0L);

    /**
     * Timeout in milliseconds. If no valid packet arrives within this time,
     * the pose smoothly returns to 0 (neutral).
     */
    public static final long TIMEOUT_MS = 1500L;

    /**
     * Calculates the smoothed offset relative to the base reference pose.
     * Keeps last stable offset without sudden drops on occasional dropped frames.
     *
     * @param basePose the calibration/reference pose (recenter base)
     * @param lastOffset previous offset state (x, y, z, yaw, pitch, roll)
     * @param smoothingFactor lerp factor between 0.0 (no movement) and 1.0 (instant)
     * @param currentTimeMs current time in milliseconds
     * @return array of 6 floats: [dx, dy, dz, dYaw, dPitch, dRoll]
     */
    public float[] computeSmoothedOffset(HeadPoseState basePose, float[] lastOffset, float smoothingFactor, long currentTimeMs) {
        boolean timedOut = (currentTimeMs - this.timestampMs > TIMEOUT_MS);

        float targetX = 0f;
        float targetY = 0f;
        float targetZ = 0f;
        float targetYaw = 0f;
        float targetPitch = 0f;
        float targetRoll = 0f;

        if (this.isValid && !timedOut) {
            float baseX = basePose != null ? basePose.xMm : 0f;
            float baseY = basePose != null ? basePose.yMm : 0f;
            float baseZ = basePose != null ? basePose.zMm : 0f;
            float baseYaw = basePose != null ? basePose.yaw : 0f;
            float basePitch = basePose != null ? basePose.pitch : 0f;
            float baseRoll = basePose != null ? basePose.roll : 0f;

            targetX = this.xMm - baseX;
            targetY = this.yMm - baseY;
            targetZ = this.zMm - baseZ;
            targetYaw = this.yaw - baseYaw;
            targetPitch = this.pitch - basePitch;
            targetRoll = this.roll - baseRoll;
        } else if (!timedOut && lastOffset != null && lastOffset.length >= 6) {
            // Pojedyncza zgubiona klatka: utrzymaj poprzednią pozycję zamiast nagłego szarpnięcia do zera!
            return lastOffset.clone();
        }

        float[] result = new float[6];
        if (lastOffset == null || lastOffset.length < 6) {
            result[0] = targetX;
            result[1] = targetY;
            result[2] = targetZ;
            result[3] = targetYaw;
            result[4] = targetPitch;
            result[5] = targetRoll;
        } else {
            // LERP towards target
            float alpha = Math.clamp(smoothingFactor, 0.01f, 1.0f);
            result[0] = lastOffset[0] + (targetX - lastOffset[0]) * alpha;
            result[1] = lastOffset[1] + (targetY - lastOffset[1]) * alpha;
            result[2] = lastOffset[2] + (targetZ - lastOffset[2]) * alpha;
            result[3] = lastOffset[3] + (targetYaw - lastOffset[3]) * alpha;
            result[4] = lastOffset[4] + (targetPitch - lastOffset[4]) * alpha;
            result[5] = lastOffset[5] + (targetRoll - lastOffset[5]) * alpha;
        }

        return result;
    }
}
