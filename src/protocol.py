import socket
import struct
from typing import Optional, Tuple

MAGIC_HEADER = b"FACE"
PACKET_FORMAT = "<4sffffffi"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # 32 bytes


def pack_pose(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    yaw: float,
    pitch: float,
    roll: float,
    is_valid: bool = True,
    flags: int = 0,
) -> bytes:
    """Serializes pose data into a 32-byte binary packet.

    Format: <4sffffffi (little-endian)
    - 4s: magic header b"FACE"
    - 6x float32: x_mm, y_mm, z_mm, yaw, pitch, roll
    - 1x int32: flags (bit 0 = is_valid)
    """
    flag_val = flags | (1 if is_valid else 0)
    return struct.pack(
        PACKET_FORMAT,
        MAGIC_HEADER,
        float(x_mm),
        float(y_mm),
        float(z_mm),
        float(yaw),
        float(pitch),
        float(roll),
        int(flag_val),
    )


def unpack_pose(data: bytes) -> Tuple[bytes, float, float, float, float, float, float, int]:
    """Unpacks a 32-byte binary pose packet.

    Returns: (magic, x_mm, y_mm, z_mm, yaw, pitch, roll, flags)
    """
    if len(data) != PACKET_SIZE:
        raise ValueError(f"Invalid packet size: expected {PACKET_SIZE} bytes, got {len(data)}")
    return struct.unpack(PACKET_FORMAT, data)


class HeadTrackerUDPClient:
    """Non-blocking UDP transmitter client for head tracking poses."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4242):
        self.host = host
        self.port = port
        self.dest_addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def send_pose(
        self,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_mm: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        is_valid: bool = True,
        flags: int = 0,
    ) -> bool:
        """Encodes and sends pose packet via non-blocking UDP."""
        try:
            packet = pack_pose(
                x_mm=x_mm,
                y_mm=y_mm,
                z_mm=z_mm,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                is_valid=is_valid,
                flags=flags,
            )
            self.sock.sendto(packet, self.dest_addr)
            return True
        except Exception:
            return False

    def close(self):
        """Closes the underlying UDP socket."""
        try:
            self.sock.close()
        except Exception:
            pass


class HeadTrackerUDPReceiver:
    """Non-blocking UDP receiver for head tracking poses."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4242):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    def receive_latest_pose(self) -> Optional[dict]:
        """Drains the UDP buffer and returns the most recent pose packet received, or None."""
        latest_data = None
        while True:
            try:
                data, _ = self.sock.recvfrom(PACKET_SIZE * 4)
                if len(data) == PACKET_SIZE:
                    latest_data = data
            except BlockingIOError:
                break
            except Exception:
                break

        if latest_data is None:
            return None

        magic, x, y, z, yaw, pitch, roll, flags = unpack_pose(latest_data)
        if magic != MAGIC_HEADER:
            return None

        return {
            "x_mm": x,
            "y_mm": y,
            "z_mm": z,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "is_valid": bool(flags & 1),
            "flags": flags,
        }

    def close(self):
        """Closes the receiver UDP socket."""
        try:
            self.sock.close()
        except Exception:
            pass
