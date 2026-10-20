package com.facecam.headtracker.config;

/**
 * Tracking mode for 6DoF head tracking in Minecraft.
 */
public enum TrackingMode {
    HEAD_MIRROR("Kopiowanie ruchów (Head Mirror)"),
    VIRTUAL_WINDOW("Wirtualne okno (Virtual Window)");

    private final String displayName;

    TrackingMode(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }

    @Override
    public String toString() {
        return displayName;
    }
}
