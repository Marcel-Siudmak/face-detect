package com.facecam.headtracker;

import com.facecam.headtracker.config.HeadTrackerConfig;
import com.facecam.headtracker.config.TrackingMode;
import com.facecam.headtracker.input.KeyBindings;
import com.facecam.headtracker.network.UdpReceiver;
import com.facecam.headtracker.render.OffAxisProjection;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HeadTrackerMod implements ClientModInitializer {
    public static final String MOD_ID = "headtracker";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    private static final UdpReceiver RECEIVER = new UdpReceiver();
    private static final OffAxisProjection OFF_AXIS_PROJECTION = new OffAxisProjection();

    // Movement scale: 0.001f = 1 block per 1000 mm (10 cm = 0.1 block)
    public static final float METERS_PER_MM = 0.001f;
    // Rotation scale: 0.25f = spokojny, subtelny obrót bez szarpania
    public static final float ROTATION_SCALE = 0.25f;
    // Smoothing factor for pose interpolation: 0.12f (wysoka stabilność, brak drgań)
    public static final float SMOOTHING_FACTOR = 0.12f;

    @Override
    public void onInitializeClient() {
        LOGGER.info("[HeadTracker] Initializing HeadTracker 6DoF mod for Minecraft 26.3...");

        // Load configuration from disk
        HeadTrackerConfig.load();

        // Start UDP Receiver
        RECEIVER.start();

        // Register Keybindings and Tick Handlers
        KeyBindings.register();

        // Register client shutdown hook to gracefully save config and close socket
        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            LOGGER.info("[HeadTracker] Client stopping, saving config and closing UDP receiver...");
            HeadTrackerConfig.save();
            RECEIVER.stop();
        });
    }

    public static HeadTrackerConfig getConfig() {
        return HeadTrackerConfig.getInstance();
    }

    public static UdpReceiver getReceiver() {
        return RECEIVER;
    }

    public static OffAxisProjection getOffAxisProjection() {
        return OFF_AXIS_PROJECTION;
    }

    public static boolean isEnabled() {
        return HeadTrackerConfig.getInstance().isEnabled();
    }

    public static void setEnabled(boolean value) {
        HeadTrackerConfig.getInstance().setEnabled(value);
        HeadTrackerConfig.save();
    }

    public static boolean isOffAxisEnabled() {
        return HeadTrackerConfig.getInstance().getMode() == TrackingMode.VIRTUAL_WINDOW;
    }

    public static void setOffAxisEnabled(boolean value) {
        HeadTrackerConfig.getInstance().setMode(value ? TrackingMode.VIRTUAL_WINDOW : TrackingMode.HEAD_MIRROR);
        HeadTrackerConfig.save();
    }
}
