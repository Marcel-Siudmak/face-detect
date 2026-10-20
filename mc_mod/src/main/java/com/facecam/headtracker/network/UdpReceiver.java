package com.facecam.headtracker.network;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Asynchronous background UDP receiver for head-pose tracking packets.
 * Listens on port 4242 and updates thread-safe atomic state.
 */
public class UdpReceiver {
    private static final Logger LOGGER = LoggerFactory.getLogger(UdpReceiver.class);
    public static final int DEFAULT_PORT = 4242;
    public static final int PACKET_SIZE = 32;

    // Magic 'FACE' in Little-Endian byte array: ['F', 'A', 'C', 'E']
    // Bytes: 0x46, 0x41, 0x43, 0x45
    private static final byte[] MAGIC = new byte[]{(byte) 'F', (byte) 'A', (byte) 'C', (byte) 'E'};

    private final int port;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicReference<HeadPoseState> currentState = new AtomicReference<>(HeadPoseState.ZERO);
    private final AtomicReference<HeadPoseState> basePose = new AtomicReference<>(HeadPoseState.ZERO);

    // Cached offset for smoothing across rendering ticks
    private final float[] currentOffset = new float[6];

    private DatagramSocket socket;
    private Thread receiverThread;

    public UdpReceiver() {
        this(DEFAULT_PORT);
    }

    public UdpReceiver(int port) {
        this.port = port;
    }

    /**
     * Starts listening for UDP packets in a daemon background thread.
     */
    public synchronized void start() {
        if (running.get()) {
            LOGGER.warn("[HeadTracker] UDP Receiver is already running.");
            return;
        }

        try {
            socket = new DatagramSocket(port);
            socket.setReceiveBufferSize(65536);
            running.set(true);

            receiverThread = new Thread(this::listenLoop, "HeadTracker-UDP-Receiver");
            receiverThread.setDaemon(true);
            receiverThread.start();
            LOGGER.info("[HeadTracker] UDP Receiver listening on port {}", port);
        } catch (SocketException e) {
            LOGGER.error("[HeadTracker] Failed to bind UDP socket on port {}", port, e);
        }
    }

    /**
     * Stops the receiver and closes the socket.
     */
    public synchronized void stop() {
        if (!running.get()) {
            return;
        }
        running.set(false);

        if (socket != null && !socket.isClosed()) {
            socket.close();
        }

        if (receiverThread != null) {
            receiverThread.interrupt();
            receiverThread = null;
        }
        LOGGER.info("[HeadTracker] UDP Receiver stopped.");
    }

    private void listenLoop() {
        byte[] buffer = new byte[PACKET_SIZE];
        DatagramPacket packet = new DatagramPacket(buffer, buffer.length);

        while (running.get() && !Thread.currentThread().isInterrupted()) {
            try {
                socket.receive(packet);
                if (packet.getLength() == PACKET_SIZE) {
                    processPacket(buffer);
                }
            } catch (SocketException e) {
                if (!running.get()) {
                    break;
                }
                LOGGER.error("[HeadTracker] UDP Socket error", e);
            } catch (Exception e) {
                if (running.get()) {
                    LOGGER.error("[HeadTracker] Error while processing UDP packet", e);
                }
            }
        }
    }

    private void processPacket(byte[] data) {
        // Check magic header: 'FACE'
        if (data[0] != MAGIC[0] || data[1] != MAGIC[1] || data[2] != MAGIC[2] || data[3] != MAGIC[3]) {
            return;
        }

        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        bb.position(4); // Skip magic 4 bytes

        float xMm = bb.getFloat();
        float yMm = bb.getFloat();
        float zMm = bb.getFloat();
        float yaw = bb.getFloat();
        float pitch = bb.getFloat();
        float roll = bb.getFloat();
        int flags = bb.getInt();

        boolean isValid = (flags & 1) != 0;
        long timestamp = System.currentTimeMillis();

        HeadPoseState newState = new HeadPoseState(xMm, yMm, zMm, yaw, pitch, roll, isValid, timestamp);
        currentState.set(newState);

        // Auto-recenter on the very first valid packet so camera doesn't jump
        if (isValid && basePose.get() == HeadPoseState.ZERO) {
            basePose.set(newState);
            LOGGER.info("[HeadTracker] Auto-calibrated initial base pose: x={}, y={}, z={}", xMm, yMm, zMm);
        }
    }

    /**
     * Centers the head tracking by setting current pose as the base reference pose.
     */
    public void recenter() {
        HeadPoseState latest = currentState.get();
        if (latest != null && latest.isValid()) {
            basePose.set(latest);
            LOGGER.info("[HeadTracker] Recentered base pose to: x={}, y={}, z={}", latest.xMm(), latest.yMm(), latest.zMm());
        } else {
            basePose.set(HeadPoseState.ZERO);
            LOGGER.info("[HeadTracker] Recentered base pose to zero (no valid tracker signal).");
        }
    }

    public HeadPoseState getCurrentState() {
        return currentState.get();
    }

    public HeadPoseState getBasePose() {
        return basePose.get();
    }

    /**
     * Returns smoothed offset [dx, dy, dz, dYaw, dPitch, dRoll] with damping and timeout handling.
     */
    public synchronized float[] getSmoothedOffset(float smoothingFactor) {
        HeadPoseState current = currentState.get();
        HeadPoseState base = basePose.get();
        long now = System.currentTimeMillis();

        float[] updated = current.computeSmoothedOffset(base, currentOffset, smoothingFactor, now);
        System.arraycopy(updated, 0, currentOffset, 0, 6);
        return currentOffset.clone();
    }

    public boolean isConnected() {
        HeadPoseState current = currentState.get();
        return current != null && current.isValid() && (System.currentTimeMillis() - current.timestampMs() <= HeadPoseState.TIMEOUT_MS);
    }
}
