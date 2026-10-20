package com.facecam.headtracker.render;

import org.joml.Matrix4f;
import org.joml.Vector4f;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class OffAxisProjectionTest {

    @Test
    public void testSymmetricFrustumAtCenter() {
        OffAxisProjection proj = new OffAxisProjection(286.5f, 179.0f);
        float near = 0.05f;
        float far = 1000.0f;
        float distance = 500.0f; // 50 cm in front of screen center

        Matrix4f matrix = proj.computeFrustum(0.0f, 0.0f, distance, near, far, false);

        // For centered eye, left = -right, bottom = -top
        // In OpenGL frustum:
        // m20 = (right + left) / (right - left) = 0
        // m21 = (top + bottom) / (top - bottom) = 0
        assertEquals(0.0f, matrix.m20(), 1e-5f, "Center projection should have symmetric horizontal frustum (m20 == 0)");
        assertEquals(0.0f, matrix.m21(), 1e-5f, "Center projection should have symmetric vertical frustum (m21 == 0)");

        // Check m00: 2 * near / (right - left) = 2 * near / (width * near / distance) = 2 * distance / width
        float expectedM00 = (2.0f * distance) / 286.5f;
        assertEquals(expectedM00, matrix.m00(), 1e-4f, "m00 scaling should match 2 * distance / width");

        // Check m11: 2 * distance / height
        float expectedM11 = (2.0f * distance) / 179.0f;
        assertEquals(expectedM11, matrix.m11(), 1e-4f, "m11 scaling should match 2 * distance / height");
    }

    @Test
    public void testAsymmetricFrustumWhenLeaningRight() {
        OffAxisProjection proj = new OffAxisProjection(286.5f, 179.0f);
        float near = 0.05f;
        float far = 1000.0f;
        float distance = 500.0f;
        float eyeX = 50.0f; // 5 cm to the right

        Matrix4f matrix = proj.computeFrustum(eyeX, 0.0f, distance, near, far, false);

        // When leaning right (+X), frustum shifts, right + left = -2 * eyeX * (near / distance) != 0
        // m20 = (right + left) / (right - left) = -2 * eyeX / width
        float expectedM20 = (-2.0f * eyeX) / 286.5f;
        assertEquals(expectedM20, matrix.m20(), 1e-4f, "m20 should reflect horizontal asymmetry");
        assertTrue(matrix.m20() < 0, "m20 should be negative when moving right");

        // Vertical remains symmetric
        assertEquals(0.0f, matrix.m21(), 1e-5f);
    }

    @Test
    public void testAsymmetricFrustumWhenLeaningUp() {
        OffAxisProjection proj = new OffAxisProjection(286.5f, 179.0f);
        float near = 0.05f;
        float far = 1000.0f;
        float distance = 500.0f;
        float eyeY = 40.0f; // 4 cm up

        Matrix4f matrix = proj.computeFrustum(0.0f, eyeY, distance, near, far, false);

        // m21 = (top + bottom) / (top - bottom) = -2 * eyeY / height
        float expectedM21 = (-2.0f * eyeY) / 179.0f;
        assertEquals(expectedM21, matrix.m21(), 1e-4f, "m21 should reflect vertical asymmetry");
        assertTrue(matrix.m21() < 0, "m21 should be negative when moving up");

        // Horizontal remains symmetric
        assertEquals(0.0f, matrix.m20(), 1e-5f);
    }

    @Test
    public void testZZeroToOneFlag() {
        OffAxisProjection proj = new OffAxisProjection();
        float near = 0.05f;
        float far = 1000.0f;

        Matrix4f mGl = proj.computeFrustum(0.0f, 0.0f, 500.0f, near, far, false);
        Matrix4f mVk = proj.computeFrustum(0.0f, 0.0f, 500.0f, near, far, true);

        // In OpenGL (-1 to 1), near maps to -1: Vector4f(0, 0, -near, 1) * mGl -> z/w = -1
        Vector4f vNear = new Vector4f(0, 0, -near, 1.0f);
        Vector4f resGl = mGl.transform(vNear, new Vector4f());
        assertEquals(-1.0f, resGl.z / resGl.w, 1e-4f);

        // In ZZeroToOne (0 to 1), near maps to 0: Vector4f(0, 0, -near, 1) * mVk -> z/w = 0
        Vector4f resVk = mVk.transform(vNear, new Vector4f());
        assertEquals(0.0f, resVk.z / resVk.w, 1e-4f);
    }

    @Test
    public void testMinEyeDistanceClamp() {
        OffAxisProjection proj = new OffAxisProjection();
        // Zero or negative eye distance should clamp to MIN_EYE_DISTANCE_MM and not produce NaNs
        Matrix4f matrix = proj.computeFrustum(0.0f, 0.0f, 0.0f, 0.05f, 1000.0f);
        assertFalse(Float.isNaN(matrix.m00()));
        assertFalse(Float.isInfinite(matrix.m00()));
    }

    @Test
    public void testBlockScaleScreenDimensions() {
        // MacBook Air 13.3": 286.5 mm x 179.0 mm
        // blockScaleMeters = 0.10f (1 block = 100 mm)
        OffAxisProjection proj = new OffAxisProjection(286.5f, 179.0f, 0.10f);

        assertEquals(0.10f, proj.getBlockScaleMeters(), 1e-6f);
        // 286.5 mm / 100 mm = 2.865 blocks
        assertEquals(2.865f, proj.getScreenWidthBlocks(), 1e-4f);
        // 179.0 mm / 100 mm = 1.790 blocks
        assertEquals(1.790f, proj.getScreenHeightBlocks(), 1e-4f);

        // Test mmToBlocks conversion
        assertEquals(5.0f, proj.mmToBlocks(500.0f), 1e-4f); // 500 mm = 5 blocks at 0.10m scale
        assertEquals(1.0f, OffAxisProjection.mmToBlocks(100.0f, 0.10f), 1e-4f);
    }

    @Test
    public void testFrustumScaleInvarianceWithBlockScale() {
        // Proves that frustum shape is invariant under block scaling (units cancel out)
        OffAxisProjection projStandard = new OffAxisProjection(286.5f, 179.0f, 1.0f); // 1:1 scale (1m = 1 block)
        OffAxisProjection projDiorama = new OffAxisProjection(286.5f, 179.0f, 0.10f); // diorama scale (0.1m = 1 block)

        float eyeX = 40.0f;
        float eyeY = -30.0f;
        float eyeZ = 450.0f;
        float near = 0.05f;
        float far = 1000.0f;

        Matrix4f mStandard = projStandard.computeFrustum(eyeX, eyeY, eyeZ, near, far, false);
        Matrix4f mDiorama = projDiorama.computeFrustum(eyeX, eyeY, eyeZ, near, far, false);
        Matrix4f mExplicitOverload = projDiorama.computeFrustum(eyeX, eyeY, eyeZ, 0.10f, near, far, false);

        // All matrix elements should match exactly
        assertEquals(mStandard.m00(), mDiorama.m00(), 1e-6f);
        assertEquals(mStandard.m11(), mDiorama.m11(), 1e-6f);
        assertEquals(mStandard.m20(), mDiorama.m20(), 1e-6f);
        assertEquals(mStandard.m21(), mDiorama.m21(), 1e-6f);
        assertEquals(mStandard.m22(), mDiorama.m22(), 1e-6f);
        assertEquals(mStandard.m32(), mDiorama.m32(), 1e-6f);

        assertEquals(mStandard.m00(), mExplicitOverload.m00(), 1e-6f);
        assertEquals(mStandard.m20(), mExplicitOverload.m20(), 1e-6f);
    }
}
