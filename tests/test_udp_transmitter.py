import socket
import unittest
from head_tracker_udp import HeadTrackerUDPClient, unpack_pose, PACKET_SIZE, MAGIC_HEADER


class TestHeadTrackerUDPClient(unittest.TestCase):
    def setUp(self):
        # Create a receiver socket on localhost with an ephemeral port
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", 0))
        self.receiver.settimeout(1.0)
        self.port = self.receiver.getsockname()[1]
        self.client = HeadTrackerUDPClient(host="127.0.0.1", port=self.port)

    def tearDown(self):
        self.client.close()
        self.receiver.close()

    def test_packet_structure_and_values(self):
        test_x = 12.345
        test_y = -67.891
        test_z = 543.210
        test_yaw = 15.678
        test_pitch = -5.432
        test_roll = 1.234

        sent = self.client.send_pose(
            x_mm=test_x,
            y_mm=test_y,
            z_mm=test_z,
            yaw=test_yaw,
            pitch=test_pitch,
            roll=test_roll,
            is_valid=True,
        )
        self.assertTrue(sent)

        data, addr = self.receiver.recvfrom(1024)
        self.assertEqual(len(data), PACKET_SIZE)
        self.assertEqual(len(data), 32)

        magic, x, y, z, yaw, pitch, roll, flags = unpack_pose(data)

        self.assertEqual(magic, MAGIC_HEADER)
        self.assertEqual(magic, b"FACE")
        self.assertAlmostEqual(x, test_x, places=4)
        self.assertAlmostEqual(y, test_y, places=4)
        self.assertAlmostEqual(z, test_z, places=4)
        self.assertAlmostEqual(yaw, test_yaw, places=4)
        self.assertAlmostEqual(pitch, test_pitch, places=4)
        self.assertAlmostEqual(roll, test_roll, places=4)
        self.assertEqual(flags & 1, 1)

    def test_invalid_face_flag(self):
        sent = self.client.send_pose(
            x_mm=0.0,
            y_mm=0.0,
            z_mm=0.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            is_valid=False,
        )
        self.assertTrue(sent)

        data, addr = self.receiver.recvfrom(1024)
        magic, x, y, z, yaw, pitch, roll, flags = unpack_pose(data)

        self.assertEqual(magic, b"FACE")
        self.assertEqual(flags & 1, 0)


if __name__ == "__main__":
    unittest.main()
