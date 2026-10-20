package com.facecam.headtracker.render;

import org.joml.Matrix4f;

/**
 * Off-axis perspective projection calculator for head-tracked displays.
 * Simulates a virtual window ("fish tank VR" / dynamic perspective) based on the user's
 * head position relative to the screen plane.
 */
public class OffAxisProjection {
    // Physical screen dimensions for MacBook Air 13.3" (in millimeters)
    public static final float DEFAULT_SCREEN_WIDTH_MM = 286.5f;
    public static final float DEFAULT_SCREEN_HEIGHT_MM = 179.0f;

    // Default near and far clipping planes (in meters / blocks)
    public static final float DEFAULT_NEAR = 0.05f;
    public static final float DEFAULT_FAR = 1000.0f;

    // Minimum eye distance in mm to avoid division by zero or negative frustum values
    public static final float MIN_EYE_DISTANCE_MM = 50.0f;

    public static final float DEFAULT_BLOCK_SCALE_METERS = 0.10f; // 1 block = 10 cm = 100 mm

    private final float screenWidthMm;
    private final float screenHeightMm;
    private final float blockScaleMeters;

    public OffAxisProjection() {
        this(DEFAULT_SCREEN_WIDTH_MM, DEFAULT_SCREEN_HEIGHT_MM, DEFAULT_BLOCK_SCALE_METERS);
    }

    public OffAxisProjection(float screenWidthMm, float screenHeightMm) {
        this(screenWidthMm, screenHeightMm, DEFAULT_BLOCK_SCALE_METERS);
    }

    public OffAxisProjection(float screenWidthMm, float screenHeightMm, float blockScaleMeters) {
        this.screenWidthMm = screenWidthMm;
        this.screenHeightMm = screenHeightMm;
        this.blockScaleMeters = (blockScaleMeters > 0.0f) ? blockScaleMeters : DEFAULT_BLOCK_SCALE_METERS;
    }

    public float getScreenWidthMm() {
        return screenWidthMm;
    }

    public float getScreenHeightMm() {
        return screenHeightMm;
    }

    public float getBlockScaleMeters() {
        return blockScaleMeters;
    }

    /**
     * Gets screen width measured in Minecraft blocks based on blockScaleMeters.
     */
    public float getScreenWidthBlocks() {
        return screenWidthMm / (blockScaleMeters * 1000.0f);
    }

    /**
     * Gets screen height measured in Minecraft blocks based on blockScaleMeters.
     */
    public float getScreenHeightBlocks() {
        return screenHeightMm / (blockScaleMeters * 1000.0f);
    }

    /**
     * Converts physical eye coordinates (mm) to Minecraft block space based on block scale.
     *
     * @param eyeMm Eye position in millimeters
     * @param blockScaleMeters Size of 1 block in meters
     * @return Eye position in blocks
     */
    public static float mmToBlocks(float eyeMm, float blockScaleMeters) {
        float scaleM = (blockScaleMeters > 0.0f) ? blockScaleMeters : DEFAULT_BLOCK_SCALE_METERS;
        return eyeMm / (scaleM * 1000.0f);
    }

    /**
     * Converts physical eye coordinates (mm) to Minecraft block space using this instance's block scale.
     */
    public float mmToBlocks(float eyeMm) {
        return mmToBlocks(eyeMm, this.blockScaleMeters);
    }

    /**
     * Computes an asymmetric frustum projection matrix for the given eye position.
     *
     * In the asymmetric frustum formula at the near plane:
     *   scale = near / eyeZBlocks
     *   left = (-halfW_blocks - eyeXBlocks) * scale
     *   right = (halfW_blocks - eyeXBlocks) * scale
     *   bottom = (-halfH_blocks - eyeYBlocks) * scale
     *   top = (halfH_blocks - eyeYBlocks) * scale
     *
     * Note that the mm-to-block conversion factor cancels out algebraically:
     *   (-halfW_mm / factor - eyeX_mm / factor) * (near / (eyeZ_mm / factor))
     *   = (-halfW_mm - eyeX_mm) * (near / eyeZ_mm)
     *
     * Therefore, the frustum shape on the near clipping plane depends directly on the
     * physical proportions between the eye and screen dimensions:
     *   scale = near / max(eyeZmm, MIN_EYE_DISTANCE_MM)
     *   left = (-halfW_mm - eyeXmm) * scale
     *   right = (halfW_mm - eyeXmm) * scale
     *   bottom = (-halfH_mm - eyeYmm) * scale
     *   top = (halfH_mm - eyeYmm) * scale
     *
     * @param eyeXmm Eye X offset relative to the screen center in millimeters (+X = right)
     * @param eyeYmm Eye Y offset relative to the screen center in millimeters (+Y = up)
     * @param eyeZmm Eye distance in front of the screen plane in millimeters (+Z = towards user, > 0)
     * @param near Near clipping plane distance
     * @param far Far clipping plane distance
     * @param zZeroToOne Whether depth range is [0, 1] (Vulkan/Metal/DirectX) or [-1, 1] (OpenGL)
     * @return 4x4 asymmetric projection matrix
     */
    public Matrix4f computeFrustum(float eyeXmm, float eyeYmm, float eyeZmm, float near, float far, boolean zZeroToOne) {
        float ze = Math.max(eyeZmm, MIN_EYE_DISTANCE_MM);

        float halfW = screenWidthMm / 2.0f;
        float halfH = screenHeightMm / 2.0f;

        // Frustum window bounds on the near clipping plane
        float scale = near / ze;
        float left = (-halfW - eyeXmm) * scale;
        float right = (halfW - eyeXmm) * scale;
        float bottom = (-halfH - eyeYmm) * scale;
        float top = (halfH - eyeYmm) * scale;

        Matrix4f matrix = new Matrix4f();
        matrix.setFrustum(left, right, bottom, top, near, far, zZeroToOne);
        return matrix;
    }

    /**
     * Computes asymmetric frustum with custom block scale (for explicit scaling parameterization).
     *
     * @param eyeXmm Eye X offset in mm
     * @param eyeYmm Eye Y offset in mm
     * @param eyeZmm Eye distance in mm
     * @param blockScaleMeters Scale of 1 block in meters (e.g. 0.10f for 10 cm diorama)
     * @param near Near clipping plane distance
     * @param far Far clipping plane distance
     * @param zZeroToOne Depth range [0, 1] vs [-1, 1]
     * @return 4x4 asymmetric projection matrix
     */
    public Matrix4f computeFrustum(float eyeXmm, float eyeYmm, float eyeZmm, float blockScaleMeters, float near, float far, boolean zZeroToOne) {
        // As proven mathematically, the block scale cancels out in the frustum projection ratio.
        // We preserve this overload for API completeness and diorama calculations.
        return computeFrustum(eyeXmm, eyeYmm, eyeZmm, near, far, zZeroToOne);
    }

    /**
     * Computes asymmetric frustum with default OpenGL [-1, 1] depth range.
     */
    public Matrix4f computeFrustum(float eyeXmm, float eyeYmm, float eyeZmm, float near, float far) {
        return computeFrustum(eyeXmm, eyeYmm, eyeZmm, near, far, false);
    }

    /**
     * Computes asymmetric frustum with custom block scale and default OpenGL [-1, 1] depth range.
     */
    public Matrix4f computeFrustum(float eyeXmm, float eyeYmm, float eyeZmm, float blockScaleMeters, float near, float far) {
        return computeFrustum(eyeXmm, eyeYmm, eyeZmm, blockScaleMeters, near, far, false);
    }
}
