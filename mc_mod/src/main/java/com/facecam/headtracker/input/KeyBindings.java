package com.facecam.headtracker.input;

import com.facecam.headtracker.HeadTrackerMod;
import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.KeyMapping;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;

public class KeyBindings {
    public static final KeyMapping.Category CATEGORY = KeyMapping.Category.register(
            Identifier.fromNamespaceAndPath(HeadTrackerMod.MOD_ID, "keybinds")
    );

    public static KeyMapping toggleKey;
    public static KeyMapping recenterKey;
    public static KeyMapping openSettingsKey;

    public static void register() {
        toggleKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.headtracker.toggle",
                InputConstants.Type.KEYBOARD,
                InputConstants.KEY_H,
                CATEGORY
        ));

        recenterKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.headtracker.recenter",
                InputConstants.Type.KEYBOARD,
                InputConstants.KEY_J,
                CATEGORY
        ));

        openSettingsKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.headtracker.settings",
                InputConstants.Type.KEYBOARD,
                InputConstants.KEY_K,
                CATEGORY
        ));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (toggleKey.consumeClick()) {
                boolean newState = !HeadTrackerMod.isEnabled();
                HeadTrackerMod.setEnabled(newState);

                if (client.player != null) {
                    String msg = newState ? "§a[HeadTracker] Włączony" : "§c[HeadTracker] Wyłączony";
                    client.player.sendOverlayMessage(Component.literal(msg));
                }
            }

            while (recenterKey.consumeClick()) {
                HeadTrackerMod.getReceiver().recenter();

                if (client.player != null) {
                    client.player.sendOverlayMessage(Component.literal("§b[HeadTracker] Pozycja wycentrowana"));
                }
            }

            while (openSettingsKey.consumeClick()) {
                client.gui.setScreen(new com.facecam.headtracker.gui.HeadTrackerSettingsScreen(client.gui.screen()));
            }
        });
    }
}
