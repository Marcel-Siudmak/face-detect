import socket
import unittest
from head_tracker_udp import HeadTrackerUDPClient, unpack_pose, PACKET_SIZE, MAGIC_HEADER


class TestIntegrationFlow(unittest.TestCase):
    """End-to-End integration test simulating the HeadTracker UDP stream flow.

    Tests streaming packets representing different head movements:
    - Neutral position
    - Leaning right (positive X)
    - Leaning left (negative X)
    - Leaning forward / closer to screen (smaller Z)
    - Head rotation (yaw, pitch, roll)
    - Face tracking lost (valid=False)
    """

    def setUp(self):
        # Create a receiver socket on localhost with an ephemeral port
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", 0))
        self.receiver.settimeout(2.0)
        self.port = self.receiver.getsockname()[1]
        self.client = HeadTrackerUDPClient(host="127.0.0.1", port=self.port)

    def tearDown(self):
        self.client.close()
        self.receiver.close()

    def test_multi_packet_stream_flow(self):
        # Stream of 10 realistic head pose packets
        test_stream = [
            # 1. Neutral starting position
            {"x": 0.0, "y": 95.0, "z": 500.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "valid": True},
            # 2. Moving right slightly
            {"x": 25.0, "y": 95.0, "z": 500.0, "yaw": 3.0, "pitch": 0.0, "roll": -1.0, "valid": True},
            # 3. Leaning right further (peeking right)
            {"x": 60.0, "y": 92.0, "z": 480.0, "yaw": 8.0, "pitch": 1.0, "roll": -2.5, "valid": True},
            # 4. Returning towards center
            {"x": 20.0, "y": 94.0, "z": 490.0, "yaw": 2.0, "pitch": 0.0, "roll": -0.5, "valid": True},
            # 5. Leaning left (peeking left)
            {"x": -50.0, "y": 96.0, "z": 475.0, "yaw": -7.0, "pitch": -1.0, "roll": 2.0, "valid": True},
            # 6. Returning to neutral
            {"x": 0.0, "y": 95.0, "z": 500.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "valid": True},
            # 7. Leaning forward (closer to screen)
            {"x": 5.0, "y": 90.0, "z": 380.0, "yaw": 0.5, "pitch": 4.0, "roll": 0.0, "valid": True},
            # 8. Moving up and tilted
            {"x": -10.0, "y": 120.0, "z": 420.0, "yaw": -2.0, "pitch": -5.0, "roll": 3.0, "valid": True},
            # 9. Face lost / occlusion (valid=False)
            {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "valid": False},
            # 10. Face recovered back at neutral
            {"x": 2.0, "y": 95.5, "z": 505.0, "yaw": 0.2, "pitch": -0.1, "roll": 0.0, "valid": True},
        ]

        self.assertEqual(len(test_stream), 10, "Test stream must contain exactly 10 packets")

        received_packets = []

        # Send all 10 packets through the UDP client
        for idx, p in enumerate(test_stream):
            success = self.client.send_pose(
                x_mm=p["x"],
                y_mm=p["y"],
                z_mm=p["z"],
                yaw=p["yaw"],
                pitch=p["pitch"],
                roll=p["roll"],
                is_valid=p["valid"],
            )
            self.assertTrue(success, f"Failed to send packet {idx}")

            data, addr = self.receiver.recvfrom(1024)
            self.assertEqual(len(data), PACKET_SIZE, f"Packet {idx} has invalid size")

            magic, x, y, z, yaw, pitch, roll, flags = unpack_pose(data)
            received_packets.append({
                "magic": magic,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "flags": flags,
            })

        # Verify each received packet
        for idx, (expected, actual) in enumerate(zip(test_stream, received_packets)):
            self.assertEqual(actual["magic"], MAGIC_HEADER, f"Packet {idx} magic mismatch")
            is_valid_flag = bool(actual["flags"] & 1)
            self.assertEqual(is_valid_flag, expected["valid"], f"Packet {idx} valid flag mismatch")

            if expected["valid"]:
                self.assertAlmostEqual(actual["x"], expected["x"], places=3, msg=f"Packet {idx} X mismatch")
                self.assertAlmostEqual(actual["y"], expected["y"], places=3, msg=f"Packet {idx} Y mismatch")
                self.assertAlmostEqual(actual["z"], expected["z"], places=3, msg=f"Packet {idx} Z mismatch")
                self.assertAlmostEqual(actual["yaw"], expected["yaw"], places=3, msg=f"Packet {idx} Yaw mismatch")
                self.assertAlmostEqual(actual["pitch"], expected["pitch"], places=3, msg=f"Packet {idx} Pitch mismatch")
                self.assertAlmostEqual(actual["roll"], expected["roll"], places=3, msg=f"Packet {idx} Roll mismatch")
            else:
                self.assertEqual(actual["flags"] & 1, 0, f"Packet {idx} should have valid bit cleared")

        # Specific directional assertions:
        # Step 3 vs Step 1: Leaned right -> X increased
        self.assertGreater(received_packets[2]["x"], received_packets[0]["x"])
        # Step 5 vs Step 1: Leaned left -> X decreased (negative)
        self.assertLess(received_packets[4]["x"], received_packets[0]["x"])
        # Step 7 vs Step 1: Leaned forward -> Z decreased (closer to screen)
        self.assertLess(received_packets[6]["z"], received_packets[0]["z"])
        # Step 9: Face lost -> flags bit 0 is 0
        self.assertEqual(received_packets[8]["flags"] & 1, 0)


if __name__ == "__main__":
    unittest.main()
