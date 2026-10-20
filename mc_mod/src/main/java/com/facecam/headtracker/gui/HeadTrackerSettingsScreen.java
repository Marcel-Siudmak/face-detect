package com.facecam.headtracker.gui;

import com.facecam.headtracker.HeadTrackerMod;
import com.facecam.headtracker.config.HeadTrackerConfig;
import com.facecam.headtracker.config.TrackingMode;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.AbstractSliderButton;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.CycleButton;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.CommonComponents;
import net.minecraft.network.chat.Component;
import net.minecraft.util.Mth;

import java.util.Locale;

/**
 * Settings screen for HeadTracker 6DoF mod.
 */
public class HeadTrackerSettingsScreen extends Screen {

    private final Screen parent;
    private final HeadTrackerConfig config;

    public HeadTrackerSettingsScreen(Screen parent) {
        super(Component.literal("Ustawienia Head Tracker 6DoF"));
        this.parent = parent;
        this.config = HeadTrackerConfig.getInstance();
    }

    @Override
    protected void init() {
        int centerX = this.width / 2;
        int startY = 38;
        int rowHeight = 24;
        int fullWidth = 310;
        int halfWidth = 150;
        int gap = 10;

        int leftX = centerX - fullWidth / 2;
        int rightX = leftX + halfWidth + gap;

        // Row 1: Global Enable/Disable & Tracking Mode
        CycleButton<Boolean> masterToggle = CycleButton.onOffBuilder(config.isEnabled())
                .create(leftX, startY, halfWidth, 20, Component.literal("Śledzenie głowy"), (button, value) -> {
                    config.setEnabled(value);
                    HeadTrackerMod.setEnabled(value);
                });
        addRenderableWidget(masterToggle);

        CycleButton<TrackingMode> modeToggle = CycleButton.<TrackingMode>builder(mode -> Component.literal(mode.getDisplayName()), config.getMode())
                .withValues(TrackingMode.HEAD_MIRROR, TrackingMode.VIRTUAL_WINDOW)
                .create(rightX, startY, halfWidth, 20, Component.literal("Tryb"), (button, value) -> {
                    config.setMode(value);
                    HeadTrackerMod.setOffAxisEnabled(value == TrackingMode.VIRTUAL_WINDOW);
                });
        addRenderableWidget(modeToggle);

        // Row 2: Oś X & Oś Y
        CycleButton<Boolean> toggleX = CycleButton.onOffBuilder(config.isEnableX())
                .create(leftX, startY + rowHeight, halfWidth, 20, Component.literal("Oś X (Poziom)"), (button, value) -> {
                    config.setEnableX(value);
                });
        addRenderableWidget(toggleX);

        CycleButton<Boolean> toggleY = CycleButton.onOffBuilder(config.isEnableY())
                .create(rightX, startY + rowHeight, halfWidth, 20, Component.literal("Oś Y (Pion)"), (button, value) -> {
                    config.setEnableY(value);
                });
        addRenderableWidget(toggleY);

        // Row 3: Oś Z & Obrót kamery
        CycleButton<Boolean> toggleZ = CycleButton.onOffBuilder(config.isEnableZ())
                .create(leftX, startY + rowHeight * 2, halfWidth, 20, Component.literal("Oś Z (Głębia)"), (button, value) -> {
                    config.setEnableZ(value);
                });
        addRenderableWidget(toggleZ);

        CycleButton<Boolean> toggleRot = CycleButton.onOffBuilder(config.isEnableRotation())
                .create(rightX, startY + rowHeight * 2, halfWidth, 20, Component.literal("Obrót kamery"), (button, value) -> {
                    config.setEnableRotation(value);
                });
        addRenderableWidget(toggleRot);

        // Row 4: Slider Czułość translacji (0.1x - 3.0x)
        ConfigSlider translationSlider = new ConfigSlider(
                leftX, startY + rowHeight * 3 + 4, fullWidth, 20,
                Component.literal("Czułość translacji"),
                0.1, 3.0, config.getTranslationScale(), "x",
                val -> config.setTranslationScale((float) val)
        );
        addRenderableWidget(translationSlider);

        // Row 5: Slider Czułość przybliżenia (0.1x - 3.0x)
        ConfigSlider zoomSlider = new ConfigSlider(
                leftX, startY + rowHeight * 4 + 6, fullWidth, 20,
                Component.literal("Czułość przybliżenia (Z)"),
                0.1, 3.0, config.getZoomScale(), "x",
                val -> config.setZoomScale((float) val)
        );
        addRenderableWidget(zoomSlider);

        // Row 6: Slider Wygładzanie (0.05 - 0.50)
        ConfigSlider smoothSlider = new ConfigSlider(
                leftX, startY + rowHeight * 5 + 8, fullWidth, 20,
                Component.literal("Wygładzanie (Smoothing)"),
                0.05, 0.50, config.getSmoothingFactor(), "",
                val -> config.setSmoothingFactor((float) val)
        );
        addRenderableWidget(smoothSlider);

        // Row 7: Slider Skala okna (1 blok) (0.02m - 0.50m)
        ConfigSlider windowScaleSlider = new ConfigSlider(
                leftX, startY + rowHeight * 6 + 10, fullWidth, 20,
                Component.literal("Skala okna (1 blok)"),
                0.02, 0.50, config.getVirtualWindowBlockScale(), "m",
                val -> config.setVirtualWindowBlockScale((float) val)
        );
        addRenderableWidget(windowScaleSlider);

        // Row 8: Przycisk Wycentruj (Recenter)
        Button recenterButton = Button.builder(Component.literal("Wycentruj pozycję (J)"), btn -> {
            HeadTrackerMod.getReceiver().recenter();
            if (this.minecraft != null && this.minecraft.player != null) {
                this.minecraft.player.sendOverlayMessage(Component.literal("§b[HeadTracker] Pozycja wycentrowana"));
            }
        }).bounds(leftX, startY + rowHeight * 7 + 14, fullWidth, 20).build();
        addRenderableWidget(recenterButton);

        // Footer: Gotowe / Zapisz
        Button doneButton = Button.builder(CommonComponents.GUI_DONE, btn -> onClose())
                .bounds(centerX - 100, Math.min(startY + rowHeight * 8 + 22, this.height - 28), 200, 20)
                .build();
        addRenderableWidget(doneButton);
    }

    @Override
    public void extractRenderState(GuiGraphicsExtractor extractor, int mouseX, int mouseY, float partialTick) {
        super.extractRenderState(extractor, mouseX, mouseY, partialTick);
        extractor.centeredText(this.font, this.title, this.width / 2, 16, 0xFFFFFFFF);
    }

    @Override
    public void onClose() {
        HeadTrackerConfig.save();
        if (this.minecraft != null && this.minecraft.gui != null) {
            this.minecraft.gui.setScreen(this.parent);
        }
    }

    /**
     * Slider widget mapping [min, max] range to [0.0, 1.0] internally.
     */
    private static class ConfigSlider extends AbstractSliderButton {
        private final Component prefix;
        private final double min;
        private final double max;
        private final String unit;
        private final java.util.function.DoubleConsumer onApply;

        public ConfigSlider(int x, int y, int width, int height,
                            Component prefix, double min, double max,
                            double currentValue, String unit,
                            java.util.function.DoubleConsumer onApply) {
            super(x, y, width, height, CommonComponents.EMPTY, (Mth.clamp(currentValue, min, max) - min) / (max - min));
            this.prefix = prefix;
            this.min = min;
            this.max = max;
            this.unit = unit;
            this.onApply = onApply;
            updateMessage();
        }

        @Override
        protected void updateMessage() {
            double currentActual = getActualValue();
            String formatted = String.format(Locale.ROOT, "%.2f%s", currentActual, unit);
            setMessage(Component.empty().append(prefix).append(": ").append(Component.literal(formatted)));
        }

        @Override
        protected void applyValue() {
            onApply.accept(getActualValue());
        }

        private double getActualValue() {
            return min + (this.value * (max - min));
        }
    }
}
