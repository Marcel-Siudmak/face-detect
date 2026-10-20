package com.facecam.headtracker.mixin;

import com.facecam.headtracker.HeadTrackerMod;
import com.facecam.headtracker.config.HeadTrackerConfig;
import com.facecam.headtracker.config.TrackingMode;
import com.facecam.headtracker.network.HeadPoseState;
import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;
import org.joml.Matrix4f;
import org.joml.Quaternionf;
import org.joml.Vector3f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Mixin for Camera to apply 6-DoF leaning / peeking translation offsets and optional head rotation.
 * Injects after camera orientation and position have been calculated in update(DeltaTracker).
 */
@Mixin(Camera.class)
public abstract class CameraMixin {

    @Shadow
    private Vec3 position;

    @Shadow
    @Final
    private BlockPos.MutableBlockPos blockPosition;

    @Shadow
    @Final
    private Quaternionf rotation;

    @Shadow
    private net.minecraft.client.renderer.culling.Frustum cullFrustum;

    @Shadow
    private int matrixPropertiesDirty;

    @Shadow
    @Final
    private org.joml.Vector3f forwards;

    @Shadow
    @Final
    private org.joml.Vector3f up;

    @Shadow
    @Final
    private org.joml.Vector3f left;

    @Shadow
    private float depthFar;

    @Inject(method = "update", at = @At("RETURN"))
    private void onUpdatePost(DeltaTracker deltaTracker, CallbackInfo ci) {
        HeadTrackerConfig config = HeadTrackerConfig.getInstance();
        if (!config.isEnabled()) {
            return;
        }

        // Pobierz wygładzone przesunięcie na podstawie konfigurowalnego współczynnika wygładzania
        float[] offset = HeadTrackerMod.getReceiver().getSmoothedOffset(config.getSmoothingFactor());
        float dxMm = offset[0];
        float dyMm = offset[1];
        float dzMm = offset[2];
        float dYawDeg = offset[3];
        float dPitchDeg = offset[4];
        float dRollDeg = offset[5];

        boolean hasTranslation = Math.abs(dxMm) >= 0.1f || Math.abs(dyMm) >= 0.1f || Math.abs(dzMm) >= 0.1f;
        boolean hasRotation = config.isEnableRotation() &&
                (Math.abs(dYawDeg) >= 0.2f || Math.abs(dPitchDeg) >= 0.2f || Math.abs(dRollDeg) >= 0.2f);

        // 1. Obsługa rotacji głowy (tylko wtedy, gdy użytkownik JĄ WŁĄCZY w ustawieniach!)
        if (hasRotation) {
            float rotScale = config.getRotationScale();
            float clampedYaw = Math.clamp(dYawDeg * rotScale, -25.0f, 25.0f);
            float clampedPitch = Math.clamp(dPitchDeg * rotScale, -20.0f, 20.0f);
            float clampedRoll = Math.clamp(dRollDeg * rotScale, -25.0f, 25.0f);

            float yawRad = (float) Math.toRadians(clampedYaw);
            float pitchRad = (float) Math.toRadians(clampedPitch);
            float rollRad = (float) Math.toRadians(clampedRoll);

            if (Math.abs(clampedYaw) >= 0.1f) {
                this.rotation.rotateY(yawRad);
            }
            if (Math.abs(clampedPitch) >= 0.1f) {
                this.rotation.rotateX(pitchRad);
            }
            if (Math.abs(clampedRoll) >= 0.1f) {
                this.rotation.rotateLocalZ(rollRad);
            }

            new org.joml.Vector3f(0f, 0f, -1f).rotate(this.rotation, this.forwards);
            new org.joml.Vector3f(0f, 1f, 0f).rotate(this.rotation, this.up);
            new org.joml.Vector3f(-1f, 0f, 0f).rotate(this.rotation, this.left);

            this.matrixPropertiesDirty |= 3;
        }

        // 2. Obsługa translacji
        if (hasTranslation) {
            float dxBlocks;
            float dyBlocks;
            float dzBlocks;

            if (config.getMode() == TrackingMode.VIRTUAL_WINDOW) {
                // W trybie VIRTUAL_WINDOW:
                // Szyba okna (ekran) stoi w stałym punkcie świata gry (pozycja gracza).
                // Fizyczne oko użytkownika znajduje się w punkcie (eyeX, eyeY, eyeZ) względem środka szyby.
                // Skala okna: virtualWindowBlockScale (np. 0.10f m na blok = 1 blok to 10 cm w realu).
                float blockScale = Math.max(0.01f, config.getVirtualWindowBlockScale());
                // Przelicznik mm na bloki: (dxMm / 1000.0f) / blockScale
                float mmToBlocks = 0.001f / blockScale;

                // Gdy głowa przesuwa się w prawo (+dxMm):
                // Punkt obserwacji oka przesuwa się w prawo (+X lokalne), patrząc na obiekty za szybą z ich PRAWEJ strony!
                // W połączeniu z asymetrycznym frustum w GameRenderer daje to idealny widok prawego boku drzewa.
                dxBlocks = config.isEnableX() ? (dxMm * mmToBlocks) : 0.0f;
                dyBlocks = config.isEnableY() ? (dyMm * mmToBlocks) : 0.0f;
                // dzMm < 0 to przybliżenie do ekranu. W lokalnym układzie MC: +Z to odsunięcie w tył od szyby
                dzBlocks = config.isEnableZ() ? (dzMm * mmToBlocks) : 0.0f;
            } else {
                // Tryb HEAD_MIRROR:
                float tScale = config.getTranslationScale();
                float zScale = config.getZoomScale();

                dxBlocks = config.isEnableX() ? (dxMm * HeadTrackerMod.METERS_PER_MM * tScale) : 0.0f;
                dyBlocks = config.isEnableY() ? (dyMm * HeadTrackerMod.METERS_PER_MM * tScale) : 0.0f;
                dzBlocks = config.isEnableZ() ? (dzMm * HeadTrackerMod.METERS_PER_MM * zScale) : 0.0f;
            }

            Vector3f localDisp = new Vector3f(dxBlocks, dyBlocks, dzBlocks);
            localDisp.rotate(this.rotation);

            this.position = new Vec3(
                    this.position.x + localDisp.x,
                    this.position.y + localDisp.y,
                    this.position.z + localDisp.z
            );

            this.blockPosition.set(this.position.x, this.position.y, this.position.z);
        }

        // 3. Aktualizacja culling frustum z nową pozycją kamery (bez psucia geometrii i macierzy)
        if (this.cullFrustum != null) {
            this.cullFrustum.prepare(this.position.x, this.position.y, this.position.z);
        }
    }
}
