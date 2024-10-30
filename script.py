import asyncio
import logging
import cv2
import numpy as np
import av
from pyrtmp import StreamClosedException
from pyrtmp.rtmp import RTMPProtocol, SimpleRTMPController, SimpleRTMPServer
from pyrtmp.session_manager import SessionManager
from threading import Thread
import struct
import threading
import os
import time
import pyaudio
import pyvirtualcam

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTMPController(SimpleRTMPController):
    def __init__(self):
        self.buffer = bytearray()
        self.codec = av.codec.CodecContext.create('h264', 'r')
        self.frame = None
        self.frame_count = 0
        self.max_buffer_size = 10 * 1024 * 1024  # 10 MB
        self.sps = None
        self.pps = None
        self.frame_lock = threading.Lock()
        self.frame_saved = False
        self.frame_save_path = "saved_frame.jpg"
        super().__init__()

    async def on_video_message(self, session, message) -> None:
        logger.debug(f"Received video message of size: {len(message.payload)} bytes")
        self.parse_rtmp_message(message.payload)
        await super().on_video_message(session, message)

    def parse_rtmp_message(self, payload):
        if len(payload) < 5:
            logger.warning("Payload too short")
            return

        frame_type = payload[0] >> 4
        codec_id = payload[0] & 0x0F
        
        if codec_id != 7:  # 7 is AVC (H.264)
            logger.warning(f"Unsupported codec ID: {codec_id}")
            return

        avc_packet_type = payload[1]
        composition_time = struct.unpack('>I', b'\x00' + payload[2:5])[0]

        logger.debug(f"Frame type: {frame_type}, AVC packet type: {avc_packet_type}, Composition time: {composition_time}")

        if avc_packet_type == 0:  # AVC sequence header
            logger.info("AVC sequence header received")
            self.parse_avc_sequence_header(payload[5:])
        elif avc_packet_type == 1:  # AVC NALU
            logger.debug("AVC NALU received")
            self.process_h264_data(payload[5:])
        else:
            logger.warning(f"Unsupported AVC packet type: {avc_packet_type}")

    def parse_avc_sequence_header(self, data):
        logger.debug(f"AVC sequence header: {data[:16].hex()}")
        if len(data) < 11:
            logger.warning("AVC sequence header too short")
            return

        sps_size = struct.unpack('>H', data[6:8])[0]
        self.sps = data[8:8+sps_size]
        
        pps_size = struct.unpack('>H', data[8+sps_size+1:8+sps_size+3])[0]
        self.pps = data[8+sps_size+3:8+sps_size+3+pps_size]

        logger.debug(f"SPS size: {sps_size}, PPS size: {pps_size}")
        logger.debug(f"SPS: {self.sps.hex()}")
        logger.debug(f"PPS: {self.pps.hex()}")

    def process_h264_data(self, data):
        logger.debug(f"Processing H.264 data of size: {len(data)} bytes")
        self.buffer.extend(data)
        self.try_decode_frames()

    def try_decode_frames(self):
        self.analyze_buffer()
        try:
            while len(self.buffer) > 4:
                nal_unit = self.extract_nal_unit()
                if nal_unit:
                    logger.debug(f"Extracted NAL unit of size: {len(nal_unit)} bytes")
                    self.decode_nal_unit(nal_unit)
                else:
                    break
        except Exception as e:
            logger.error(f"Error in try_decode_frames: {e}")
            logger.exception(e)
            self.buffer = bytearray()

    def extract_nal_unit(self):
        if len(self.buffer) < 4:
            return None
        
        nal_length = struct.unpack('>I', self.buffer[:4])[0]
        if len(self.buffer) < nal_length + 4:
            return None
        
        nal_unit = self.buffer[4:nal_length+4]
        self.buffer = self.buffer[nal_length+4:]
        return nal_unit

    def decode_nal_unit(self, nal_unit):
        try:
            nal_type = nal_unit[0] & 0x1F
            logger.debug(f"NAL unit type: {nal_type}")

            if nal_type == 7:  # SPS
                self.sps = nal_unit
            elif nal_type == 8:  # PPS
                self.pps = nal_unit
            elif nal_type == 5:  # IDR frame
                if self.sps and self.pps:
                    complete_nal = b'\x00\x00\x00\x01' + self.sps + b'\x00\x00\x00\x01' + self.pps + b'\x00\x00\x00\x01' + nal_unit
                    packet = av.packet.Packet(complete_nal)
                    frames = self.codec.decode(packet)
                    self.process_frames(frames)
            else:
                packet = av.packet.Packet(b'\x00\x00\x00\x01' + nal_unit)
                frames = self.codec.decode(packet)
                self.process_frames(frames)

        except av.AVError as e:
            logger.error(f"Error decoding NAL unit: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in decode_nal_unit: {e}")
            logger.exception(e)

    def process_frames(self, frames):
        for frame in frames:
            logger.debug(f"Decoded frame of size: {frame.width}x{frame.height}")
            numpy_frame = frame.to_ndarray(format='bgr24')
            processed_frame = self.process_frame(numpy_frame)
            with self.frame_lock:
                self.frame = processed_frame
                self.frame_count += 1
            logger.debug(f"Processed frame {self.frame_count} of shape: {processed_frame.shape}")
            
            if not self.frame_saved:
                logger.info(f"Attempting to save frame {self.frame_count}")
                self.save_frame(processed_frame)

    def save_frame(self, frame):
        try:
            cv2.imwrite(self.frame_save_path, frame)
            logger.info(f"Successfully saved frame to {self.frame_save_path}")
            self.frame_saved = True
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            logger.exception(e)

    def process_frame(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def analyze_buffer(self):
        logger.debug(f"Buffer size: {len(self.buffer)} bytes")
        logger.debug(f"First 16 bytes: {self.buffer[:16].hex()}")
        logger.debug(f"Last 16 bytes: {self.buffer[-16:].hex()}")

    async def on_stream_closed(self, session: SessionManager, exception: StreamClosedException) -> None:
        logger.info("Stream closed")
        await super().on_stream_closed(session, exception)

class SimpleServer(SimpleRTMPServer):
    def __init__(self):
        super().__init__()

    async def create(self, host: str, port: int):
        loop = asyncio.get_event_loop()
        self.server = await loop.create_server(
            lambda: RTMPProtocol(controller=RTMPController()),
            host=host,
            port=port,
        )

def display_processed_frame(controller):
    while True:
        with controller.frame_lock:
            current_frame_count = controller.frame_count

        logger.debug(f"Current frame count: {current_frame_count}")
        
        if controller.frame_saved:
            logger.info("Frame has been saved. Exiting display thread.")
            break

        time.sleep(1)  # Wait for 1 second before checking again

async def main():
    server = SimpleServer()
    controller = RTMPController()
    await server.create(host="0.0.0.0", port=1935)
    logger.info("Server started on rtmp://0.0.0.0:1935")
    await server.start()

    # Start the virtual camera
    with pyvirtualcam.Camera(width=1920, height=1080, fps=30) as cam:
        logger.info(f'Using virtual camera: {cam.device}')
        
        # Wait for a few seconds to allow some frames to be processed
        await asyncio.sleep(5)
        logger.info("Starting display thread")

        display_thread = Thread(target=display_processed_frame, args=(controller,), daemon=True)
        display_thread.start()
        logger.info("Frame display thread started")

        try:
            while True:
                await asyncio.sleep(1)
                if not display_thread.is_alive():
                    logger.info("Display thread has finished")
                    break
                logger.debug(f"Current frame count: {controller.frame_count}")

                # Send the current frame to the virtual camera
                with controller.frame_lock:
                    if controller.frame is not None:
                        cam.send(controller.frame)
                        cam.sleep_until_next_frame()
        except Exception as e:
            logger.error(f"Server error: {e}")
            logger.exception(e)
        finally:
            await server.close()
            logger.info("Server closed")

if __name__ == "__main__":
    asyncio.run(main())
