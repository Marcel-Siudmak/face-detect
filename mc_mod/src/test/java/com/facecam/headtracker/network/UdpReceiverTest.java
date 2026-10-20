package com.facecam.headtracker.network;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

public class UdpReceiverTest {
    private UdpReceiver receiver;
    private final int testPort = 14242;

    @BeforeEach
    public void setup() {
        receiver = new UdpReceiver(testPort);
        receiver.start();
    }

    @AfterEach
    public void tearDown() {
        if (receiver != null) {
            receiver.stop();
        }
    }

    @Test
    public void testReceiveValidPacket() throws Exception {
        byte[] packetBytes = new byte[32];
        ByteBuffer bb = ByteBuffer.wrap(packetBytes).order(ByteOrder.LITTLE_ENDIAN);
        // Magic 'FACE'
        bb.put((byte) 'F');
        bb.put((byte) 'A');
        bb.put((byte) 'C');
        bb.put((byte) 'E');
        // x, y, z, yaw, pitch, roll
        bb.putFloat(12.5f);
        bb.putFloat(-5.0f);
        bb.putFloat(450.0f);
        bb.putFloat(1.2f);
        bb.putFloat(-0.8f);
        bb.putFloat(0.1f);
        // flags: bit 0 = 1 (valid)
        bb.putInt(1);

        try (DatagramSocket socket = new DatagramSocket()) {
            DatagramPacket packet = new DatagramPacket(
                    packetBytes,
                    packetBytes.length,
                    InetAddress.getByName("127.0.0.1"),
                    testPort
            );
            socket.send(packet);
        }

        // Wait up to 500ms for packet to be processed
        long start = System.currentTimeMillis();
        while (System.currentTimeMillis() - start < 500) {
            HeadPoseState state = receiver.getCurrentState();
            if (state.isValid() && Math.abs(state.xMm() - 12.5f) < 0.001f) {
                break;
            }
            Thread.sleep(10);
        }

        HeadPoseState state = receiver.getCurrentState();
        assertTrue(state.isValid(), "State should be valid");
        assertEquals(12.5f, state.xMm(), 0.001f);
        assertEquals(-5.0f, state.yMm(), 0.001f);
        assertEquals(450.0f, state.zMm(), 0.001f);
        assertEquals(1.2f, state.yaw(), 0.001f);
        assertEquals(-0.8f, state.pitch(), 0.001f);
        assertEquals(0.1f, state.roll(), 0.001f);
        assertTrue(receiver.isConnected(), "Receiver should report connected");

        // Test Recenter
        receiver.recenter();
        HeadPoseState base = receiver.getBasePose();
        assertEquals(12.5f, base.xMm(), 0.001f);

        // Smoothed offset should be 0 immediately after recenter
        float[] offset = receiver.getSmoothedOffset(1.0f);
        assertEquals(0.0f, offset[0], 0.001f);
        assertEquals(0.0f, offset[1], 0.001f);
        assertEquals(0.0f, offset[2], 0.001f);
    }

    @Test
    public void testTimeoutDamping() {
        HeadPoseState oldPose = new HeadPoseState(100f, 100f, 100f, 0f, 0f, 0f, true, 1000L);
        // Time is way past timeout (e.g. 3000L > 1000L + 1500L)
        float[] prevOffset = new float[]{100f, 100f, 100f, 0f, 0f, 0f};
        float[] smoothed = oldPose.computeSmoothedOffset(HeadPoseState.ZERO, prevOffset, 0.5f, 3000L);

        // Target should be 0 because active = false due to timeout. Smoothed should damp towards 0:
        // 100 + (0 - 100) * 0.5 = 50
        assertEquals(50.0f, smoothed[0], 0.001f);
        assertEquals(50.0f, smoothed[1], 0.001f);
        assertEquals(50.0f, smoothed[2], 0.001f);
    }
}
