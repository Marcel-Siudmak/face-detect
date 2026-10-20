package com.facecam.headtracker.config;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Configuration and persistence layer for HeadTracker mod.
 */
public class HeadTrackerConfig {
    private static final Logger LOGGER = LoggerFactory.getLogger("headtracker-config");
    private static final String FILE_NAME = "headtracker.json";

    private static final Gson GSON = new GsonBuilder()
            .setPrettyPrinting()
            .create();

    private static HeadTrackerConfig instance = new HeadTrackerConfig();

    // Configuration fields with default values
    public boolean enabled = true;
    public TrackingMode mode = TrackingMode.HEAD_MIRROR;
    public boolean enableX = true;
    public boolean enableY = true;
    public boolean enableZ = true;
    public boolean enableRotation = false; // Domyślnie wyłączona rotacja, by nie psuć stabilności
    public float translationScale = 1.0f; // Mnożnik (0.1f do 3.0f)
    public float zoomScale = 1.0f;        // Mnożnik osi Z (0.1f do 3.0f)
    public float rotationScale = 0.25f;   // Mnożnik rotacji (0.05f do 1.0f)
    public float smoothingFactor = 0.15f; // Interpolacja lerp (0.05f do 0.50f)
    public float virtualWindowBlockScale = 0.10f; // 1 blok = 10 cm w rzeczywistosci (zakres 0.02f do 0.50f)

    public HeadTrackerConfig() {
    }

    public static HeadTrackerConfig getInstance() {
        if (instance == null) {
            instance = new HeadTrackerConfig();
        }
        return instance;
    }

    public static void setInstance(HeadTrackerConfig newInstance) {
        instance = (newInstance != null) ? newInstance : new HeadTrackerConfig();
    }

    public static Path getConfigPath() {
        try {
            if (FabricLoader.getInstance() != null && FabricLoader.getInstance().getConfigDir() != null) {
                return FabricLoader.getInstance().getConfigDir().resolve(FILE_NAME);
            }
        } catch (Throwable ignored) {
            // Fallback for tests or environments outside FabricLoader
        }
        return Path.of("config", FILE_NAME);
    }

    /**
     * Loads configuration from disk, falling back to defaults if an error occurs or the file is missing.
     */
    public static HeadTrackerConfig load() {
        return load(getConfigPath());
    }

    /**
     * Loads configuration from a specific path.
     */
    public static HeadTrackerConfig load(Path path) {
        if (Files.exists(path)) {
            try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
                HeadTrackerConfig loaded = GSON.fromJson(reader, HeadTrackerConfig.class);
                if (loaded != null) {
                    loaded.validate();
                    instance = loaded;
                    LOGGER.info("[HeadTracker] Configuration loaded successfully from {}", path);
                    return instance;
                }
            } catch (Exception e) {
                LOGGER.error("[HeadTracker] Failed to load config from {}, falling back to defaults", path, e);
            }
        } else {
            LOGGER.info("[HeadTracker] Config file not found at {}, creating defaults", path);
        }

        instance = new HeadTrackerConfig();
        save(path);
        return instance;
    }

    /**
     * Saves current configuration to disk.
     */
    public static void save() {
        save(getConfigPath());
    }

    /**
     * Saves current configuration to a specific path.
     */
    public static void save(Path path) {
        try {
            Path parent = path.getParent();
            if (parent != null && !Files.exists(parent)) {
                Files.createDirectories(parent);
            }
            try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                GSON.toJson(getInstance(), writer);
                LOGGER.info("[HeadTracker] Configuration saved to {}", path);
            }
        } catch (IOException e) {
            LOGGER.error("[HeadTracker] Failed to save config to {}", path, e);
        }
    }

    /**
     * Validates and repairs config values if corrupted or out of range.
     */
    public void validate() {
        if (this.mode == null) {
            this.mode = TrackingMode.HEAD_MIRROR;
        }
        if (Float.isNaN(this.translationScale) || this.translationScale <= 0.0f) {
            this.translationScale = 1.0f;
        }
        if (Float.isNaN(this.zoomScale) || this.zoomScale <= 0.0f) {
            this.zoomScale = 1.0f;
        }
        if (Float.isNaN(this.rotationScale) || this.rotationScale < 0.0f) {
            this.rotationScale = 0.25f;
        }
        if (Float.isNaN(this.smoothingFactor) || this.smoothingFactor <= 0.0f) {
            this.smoothingFactor = 0.15f;
        }
        if (Float.isNaN(this.virtualWindowBlockScale) || this.virtualWindowBlockScale <= 0.0f) {
            this.virtualWindowBlockScale = 0.10f;
        }
    }

    // Getters and Setters
    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public TrackingMode getMode() {
        return mode;
    }

    public void setMode(TrackingMode mode) {
        this.mode = (mode != null) ? mode : TrackingMode.HEAD_MIRROR;
    }

    public boolean isEnableX() {
        return enableX;
    }

    public void setEnableX(boolean enableX) {
        this.enableX = enableX;
    }

    public boolean isEnableY() {
        return enableY;
    }

    public void setEnableY(boolean enableY) {
        this.enableY = enableY;
    }

    public boolean isEnableZ() {
        return enableZ;
    }

    public void setEnableZ(boolean enableZ) {
        this.enableZ = enableZ;
    }

    public boolean isEnableRotation() {
        return enableRotation;
    }

    public void setEnableRotation(boolean enableRotation) {
        this.enableRotation = enableRotation;
    }

    public float getTranslationScale() {
        return translationScale;
    }

    public void setTranslationScale(float translationScale) {
        this.translationScale = translationScale;
    }

    public float getZoomScale() {
        return zoomScale;
    }

    public void setZoomScale(float zoomScale) {
        this.zoomScale = zoomScale;
    }

    public float getRotationScale() {
        return rotationScale;
    }

    public void setRotationScale(float rotationScale) {
        this.rotationScale = rotationScale;
    }

    public float getSmoothingFactor() {
        return smoothingFactor;
    }

    public void setSmoothingFactor(float smoothingFactor) {
        this.smoothingFactor = smoothingFactor;
    }

    public float getVirtualWindowBlockScale() {
        return virtualWindowBlockScale;
    }

    public void setVirtualWindowBlockScale(float virtualWindowBlockScale) {
        this.virtualWindowBlockScale = virtualWindowBlockScale;
    }
}
