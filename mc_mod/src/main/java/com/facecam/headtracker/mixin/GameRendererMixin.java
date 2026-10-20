package com.facecam.headtracker.mixin;

import com.facecam.headtracker.HeadTrackerMod;
import com.facecam.headtracker.config.HeadTrackerConfig;
import com.facecam.headtracker.config.TrackingMode;
import com.facecam.headtracker.network.HeadPoseState;
import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.state.GameRenderState;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Mixin for GameRenderer to apply off-axis asymmetric frustum projection
 * in VIRTUAL_WINDOW mode.
 */
@Mixin(GameRenderer.class)
public abstract class GameRendererMixin {

    @Shadow
    @Final
    private Minecraft minecraft;

    @Shadow
    @Final
    private GameRenderState gameRenderState;

    @Shadow
    @Final
    private Camera mainCamera;

    /**
     * In VIRTUAL_WINDOW mode, natural motion parallax is handled via calibrated
     * camera position in CameraMixin, keeping projectionMatrix intact to avoid UBO/GPU stalls.
     */
    @Inject(method = "extractCamera", at = @At("RETURN"))
    private void onExtractCameraPost(DeltaTracker deltaTracker, float f, CallbackInfo ci) {
        // Retained for future shader/HUD adjustments if needed.
    }
}
