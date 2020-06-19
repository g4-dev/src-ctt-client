import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import asyncio
import websockets
import http.client
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
from dotenv import load_dotenv
import os
from  websockets.exceptions import WebSocketException

logging.basicConfig(level=20)


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(
        self,
        callback=None,
        device=None,
        input_rate=RATE_PROCESS,
        file=None,
        curr_wf="render.wav",
    ):
        def proxy_callback(in_data, frame_count, time_info, status):
            # pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None:
            callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()
        self.curr_wf = wave.open(curr_wf, "wb")
        self.curr_wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        self.curr_wf.setsampwidth(2)
        self.curr_wf.setframerate(self.sample_rate)

        kwargs = {
            "format": self.FORMAT,
            "channels": self.CHANNELS,
            "rate": self.input_rate,
            "input": True,
            "frames_per_buffer": self.block_size_input,
            "stream_callback": proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs["input_device_index"] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, "rb")

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(), input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate
    )

    def write_wav(self, data):
        logging.info("write wav")
        self.curr_wf.writeframes(data)

    def close_wav(self):
        self.curr_wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(
        self, aggressiveness=3, device=None, input_rate=None, file=None, curr_wf=None
    ):
        super().__init__(
            device=device, input_rate=input_rate, file=file, curr_wf=curr_wf
        )
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


class cttApi(object):
    """Communicate with ctt api"""

    def __init__(self):
        load_dotenv(dotenv_path=ARGS.env)
        self.auth = ""
        self.protocol = os.getenv("API_PROTOCOL")
        self.headers = {
            "Content-Type": "application/json",
            "Origin": os.getenv("API_USER"),
        }
        self.transcript = {}

    def connect(self):
        if self.protocol == "http":
            self.conn = http.client.HTTPConnection(os.getenv("API_URL"))
        else:
            self.conn = http.client.HTTPSConnection(os.getenv("API_URL"))
        self.conn.request(
            "POST",
            "/login",
            json.dumps(
                {"name": os.getenv("API_USER"), "token": os.getenv("API_TOKEN")}
            ),
            headers=self.headers,
        )

        response = self.conn.getresponse()
        jwt_token = json.loads(response.read().decode())["token"]
        self.auth = {"Authorization": "Bearer {}".format(jwt_token)}
        self.headers = {**self.headers, **self.auth}

    def progress_transcript(self):
        self.conn.request(
            "POST",
            "/transcripts",
            json.dumps({}),
            headers=self.headers,
        )
        response = self.conn.getresponse()
        self.transcript = json.loads(response.read().decode())
        return self.transcript

    def done_transcript(self, audio_path, text_path):
        print("Trigger send of transcript")
        # /transcripts/save-audio
        mp_encoder = MultipartEncoder(
            fields={
                "text_file": (
                    self.transcript["transcript"]["uuid"] + ".txt",
                    open(text_path, "rb"),
                    "text/plain",
                ),
                "audio_file": (
                    self.transcript["transcript"]["uuid"] + ".wav",
                    open(audio_path, "rb"),
                    "audio/wave",
                ),
            }
        )
        encoder_headers = {"Content-Type": mp_encoder.content_type}
        full_headers = {**self.headers, **encoder_headers}
        self.conn.request(
            "POST",
            "/transcripts/done",
            body=mp_encoder,  # Build multipart form-data to send files
            headers=full_headers,
        )
        response = self.conn.getresponse()
        return json.loads(response.read().decode())

    async def ws(self, text):
        extra_headers = {**self.auth, **{"uuid": self.transcript["transcript"]["uuid"]}}
        async with websockets.connect(
            "ws://{}".format(ARGS.websocket), extra_headers=extra_headers
        ) as websocket:
            await websocket.send(text)
            response = await websocket.recv()
            print("ws : ", response)

    def canceled_transcript(self):
        self.conn.request(
            "PATCH",
            "/transcripts/"+self.transcript["transcript"]["uuid"],
            headers=self.headers,
        )
        response = self.conn.getresponse()
        print("Canceled transcript")
        return json.loads(response.read().decode())

class Transcriber(object):
    """Wrapper for transcriber process and his components"""
    MAX_RETRY = 4
    def __init__(self, ARGS):
        self.retries = 0
        # First Connect to ctt api
        self.api = cttApi()
        self.api.connect()
        # Load DeepSpeech model
        if os.path.isdir(ARGS.model):
            model_dir = ARGS.model
            ARGS.model = os.path.join(model_dir, "output_graph.tflite")
            ARGS.lm = os.path.join(model_dir, ARGS.lm)
            ARGS.trie = os.path.join(model_dir, ARGS.trie)

        print("Initializing model...")
        logging.info("ARGS.model: %s", ARGS.model)
        self.model = deepspeech.Model(ARGS.model, ARGS.beam_width)
        if ARGS.lm and ARGS.trie:
            logging.info("ARGS.lm: %s", ARGS.lm)
            logging.info("ARGS.trie: %s", ARGS.trie)
            self.model.enableDecoderWithLM(
                ARGS.lm, ARGS.trie, ARGS.lm_alpha, ARGS.lm_beta
            )

        print("Connected to call2Text api")
        transcript_uuid = self.api.progress_transcript()["transcript"]["uuid"]
        self.curr_wf = os.path.join(ARGS.savewav, "{}.wav".format(transcript_uuid))
        self.curr_txt = os.path.join(ARGS.savetext, "{}.txt".format(transcript_uuid))
        # Opening file after defining path
        self.transcript_text = open(self.curr_txt, "w")
        self.transcript_text.close()
        print("creating wav at : %s" % self.curr_wf)

        # Start audio with VAD
        self.vad_audio = VADAudio(
            aggressiveness=ARGS.vad_aggressiveness,
            device=ARGS.device,
            input_rate=ARGS.rate,
            file=ARGS.file,
            curr_wf=self.curr_wf,
        )
        print("Listening (ctrl-C to exit)...")

    def transcribe(self, ARGS):
        frames = self.vad_audio.vad_collector()

        # Stream from microphone to DeepSpeech using VAD
        spinner = None
        if not ARGS.nospinner:
            spinner = Halo(spinner="line")

        stream_context = self.model.createStream()
        wav_data = bytearray()
        for frame in frames:
            if frame is not None:
                if spinner:
                    spinner.start()
                logging.debug("streaming frame")
                self.model.feedAudioContent(
                    stream_context, np.frombuffer(frame, np.int16)
                )
                if ARGS.savewav:
                    wav_data.extend(frame)
            else:
                if spinner:
                    spinner.stop()
                logging.debug("end utterence")
                if ARGS.savewav:
                    self.vad_audio.write_wav(wav_data)
                    wav_data = bytearray()
                text = self.model.finishStream(stream_context)
                print("Recognized: %s" % text)
                self.write_text(text)
                # Send to websocket
                stream_context = self.model.createStream()
                asyncio.get_event_loop().run_until_complete(
                    self.api.ws("{}".format(text))
                )

    def write_text(self, text):
        self.transcript_text = open(self.curr_txt, "a")
        self.transcript_text.write("{}\n".format(text))
        self.transcript_text.close()

    def retry(self):
        if self.retries > self.MAX_RETRY: return False
        self.retries +=1
        return True

def main(ARGS):
    transcriber = Transcriber(ARGS)
    try:
        transcriber.transcribe(ARGS)
    except WebSocketException:
        if False == transcriber.retry(): transcriber.api.canceled_transcript()
        transcriber.api.connect()
        transcriber.api.done_transcript(transcriber.curr_wf, transcriber.curr_txt)
    except ConnectionError:
        print("Retrying update api...")
        if False == transcriber.retry() : transcriber.api.canceled_transcript()
        transcriber.api.connect()
        transcriber.api.done_transcript(transcriber.curr_wf, transcriber.curr_txt)
    except KeyboardInterrupt:
        transcriber.vad_audio.close_wav()
        transcriber.api.done_transcript(transcriber.curr_wf, transcriber.curr_txt)
        pass

if __name__ == "__main__":
    BEAM_WIDTH = 500
    DEFAULT_SAMPLE_RATE = 16000
    LM_ALPHA = 0.65
    LM_BETA = 1.45

    import argparse

    parser = argparse.ArgumentParser(
        description="Stream from microphone to DeepSpeech using VAD"
    )

    parser.add_argument(
        "-v",
        "--vad_aggressiveness",
        type=int,
        default=3,
        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3",
    )
    parser.add_argument("--nospinner", action="store_true", help="Disable spinner")
    # Saved files after processing
    parser.add_argument(
        "-w",
        "--savewav",
        help="Save .wav files of utterences to given directory",
        default="files/waves",
    )
    parser.add_argument(
        "-txt",
        "--savetext",
        help="Save .txt files of utterences to given directory",
        default="files/texts",
    )
    parser.add_argument(
        "-f", "--file", help="Read from .wav file instead of microphone"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)",
    )
    parser.add_argument(
        "-l",
        "--lm",
        default="lm.binary",
        help="Path to the language model binary file. Default: lm.binary",
    )
    parser.add_argument(
        "-t",
        "--trie",
        default="trie",
        help="Path to the language model trie file created with native_client/generate_trie. Default: trie",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=None,
        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.",
    )
    parser.add_argument(
        "-la",
        "--lm_alpha",
        type=float,
        default=LM_ALPHA,
        help=f"The alpha hyperparameter of the CTC decoder. Language Model weight. Default: {LM_ALPHA}",
    )
    parser.add_argument(
        "-lb",
        "--lm_beta",
        type=float,
        default=LM_BETA,
        help=f"The beta hyperparameter of the CTC decoder. Word insertion bonus. Default: {LM_BETA}",
    )
    parser.add_argument(
        "-bw",
        "--beam_width",
        type=int,
        default=BEAM_WIDTH,
        help=f"Beam width used in the CTC decoder when building candidate transcriptions. Default: {BEAM_WIDTH}",
    )
    # ctt
    parser.add_argument(
        "-ws",
        "--websocket",
        help="Choose an url to send your text",
        default="localhost:8081/transcripts/socket",
    )
    parser.add_argument("-e", "--env", help="Precise en env file", default=".env")

    # Bootstrap transcriber
    ARGS = parser.parse_args()
    if ARGS.savewav:
        os.makedirs(ARGS.savewav, exist_ok=True)
    if ARGS.savetext:
        os.makedirs(ARGS.savetext, exist_ok=True)
    main(ARGS)
