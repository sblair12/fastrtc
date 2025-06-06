import asyncio
import fractions
import functools
import inspect
import io
import json
import logging
import tempfile
import traceback
import warnings
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict, cast

import av
import librosa
import numpy as np
from fastapi import WebSocket
from gradio.data_classes import GradioModel, GradioRootModel
from numpy.typing import NDArray
from pydub import AudioSegment

logger = logging.getLogger(__name__)


AUDIO_PTIME = 0.020


class AudioChunk(TypedDict):
    start: int
    end: int


class WebRTCData(GradioModel):
    webrtc_id: str
    textbox: str = ""
    audio: Any | None = None


class WebRTCModel(GradioRootModel):
    root: WebRTCData | str


class AdditionalOutputs:
    def __init__(self, *args) -> None:
        self.args = args


class CloseStream:
    def __init__(self, msg: str = "Stream closed") -> None:
        self.msg = msg


class DataChannel(Protocol):
    def send(self, message: str) -> None: ...


def create_message(
    type: Literal[
        "send_input",
        "end_stream",
        "fetch_output",
        "stopword",
        "error",
        "warning",
        "log",
    ],
    data: list[Any] | str,
) -> str:
    return json.dumps({"type": type, "data": data})


current_channel: ContextVar[DataChannel | None] = ContextVar(
    "current_channel", default=None
)


@dataclass
class Context:
    webrtc_id: str
    websocket: WebSocket | None = None


current_context: ContextVar[Context | None] = ContextVar(
    "current_context", default=None
)


def get_current_context() -> Context:
    if not (ctx := current_context.get()):
        raise RuntimeError("No context found")
    return ctx


def _send_log(message: str, type: str) -> None:
    async def _send(channel: DataChannel) -> None:
        channel.send(
            json.dumps(
                {
                    "type": type,
                    "message": message,
                }
            )
        )

    if channel := current_channel.get():
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(_send(channel), loop)
        except RuntimeError:
            asyncio.run(_send(channel))


def Warning(  # noqa: N802
    message: str = "Warning issued.",
):
    """
    Send a warning message that is deplayed in the UI of the application.

    Parameters
    ----------
    audio : str
        The warning message to send

    Returns
    -------
    None
    """
    _send_log(message, "warning")


class WebRTCError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        _send_log(message, "error")


def split_output(
    data: tuple | Any,
) -> tuple[Any, AdditionalOutputs | CloseStream | None]:
    if isinstance(data, AdditionalOutputs):
        return None, data
    if isinstance(data, CloseStream):
        return None, data
    if isinstance(data, tuple):
        # handle the bare audio case
        if 2 <= len(data) <= 3 and isinstance(data[1], np.ndarray):
            return data, None
        if not len(data) == 2:
            raise ValueError(
                "The tuple must have exactly two elements: the data and an instance of AdditionalOutputs."
            )
        if not isinstance(data[-1], AdditionalOutputs | CloseStream):
            raise ValueError(
                "The last element of the tuple must be an instance of AdditionalOutputs."
            )
        return data[0], cast(AdditionalOutputs | CloseStream, data[1])
    return data, None


async def player_worker_decode(
    next_frame: Callable,
    queue: asyncio.Queue,
    thread_quit: asyncio.Event,
    channel: Callable[[], DataChannel | None] | None,
    set_additional_outputs: Callable | None,
    quit_on_none: bool = False,
    sample_rate: int = 48000,
    frame_size: int = int(48000 * AUDIO_PTIME),
):
    audio_samples = 0
    audio_time_base = fractions.Fraction(1, sample_rate)
    audio_resampler = av.AudioResampler(  # type: ignore
        format="s16",
        layout="stereo",
        rate=sample_rate,
        frame_size=frame_size,
    )
    first_sample_rate = None
    while not thread_quit.is_set():
        try:
            # Get next frame
            frame, outputs = split_output(
                await asyncio.wait_for(next_frame(), timeout=60)
            )
            if (
                isinstance(outputs, AdditionalOutputs)
                and set_additional_outputs
                and channel
                and channel()
            ):
                set_additional_outputs(outputs)
                cast(DataChannel, channel()).send(create_message("fetch_output", []))

            if frame is None:
                if isinstance(outputs, CloseStream):
                    await queue.put(outputs)
                if quit_on_none:
                    await queue.put(None)
                    break
                continue

            if not isinstance(frame, tuple) and not isinstance(frame[1], np.ndarray):
                raise WebRTCError(
                    "The frame must be a tuple containing a sample rate and a numpy array."
                )

            if len(frame) == 2:
                sample_rate, audio_array = frame
                layout = "mono"
            elif len(frame) == 3:
                sample_rate, audio_array, layout = frame
            else:
                raise ValueError(f"frame must be of length 2 or 3, got: {len(frame)}")

            logger.debug(
                "received array with shape %s sample rate %s layout %s",
                audio_array.shape,  # type: ignore
                sample_rate,
                layout,  # type: ignore
            )
            format = "s16" if audio_array.dtype == "int16" else "fltp"  # type: ignore
            if first_sample_rate is None:
                first_sample_rate = sample_rate

            if format == "s16":
                audio_array = audio_to_float32(audio_array)

            if first_sample_rate != sample_rate:
                audio_array = librosa.resample(
                    audio_array, target_sr=first_sample_rate, orig_sr=sample_rate
                )

            if audio_array.ndim == 1:
                audio_array = audio_array.reshape(1, -1)

            # Convert to audio frame and

            # This runs in the same timeout context
            frame = av.AudioFrame.from_ndarray(  # type: ignore
                audio_array,  # type: ignore
                format="fltp",
                layout=layout,  # type: ignore
            )
            frame.sample_rate = first_sample_rate
            for processed_frame in audio_resampler.resample(frame):
                processed_frame.pts = audio_samples
                processed_frame.time_base = audio_time_base
                audio_samples += processed_frame.samples
                await queue.put(processed_frame)
            if isinstance(outputs, CloseStream):
                await queue.put(outputs)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "Timeout in frame processing cycle after %s seconds - resetting", 60
            )
            continue
        except Exception as e:
            import traceback

            exec = traceback.format_exc()
            print("traceback %s", exec)
            print("Error processing frame: %s", str(e))
            if isinstance(e, WebRTCError):
                raise e
            else:
                continue


def audio_to_bytes(audio: tuple[int, NDArray[np.int16 | np.float32]]) -> bytes:
    """
    Convert an audio tuple containing sample rate and numpy array data into bytes.

    Parameters
    ----------
    audio : tuple[int, np.ndarray]
        A tuple containing:
            - sample_rate (int): The audio sample rate in Hz
            - data (np.ndarray): The audio data as a numpy array

    Returns
    -------
    bytes
        The audio data encoded as bytes, suitable for transmission or storage

    Example
    -------
    >>> sample_rate = 44100
    >>> audio_data = np.array([0.1, -0.2, 0.3])  # Example audio samples
    >>> audio_tuple = (sample_rate, audio_data)
    >>> audio_bytes = audio_to_bytes(audio_tuple)
    """
    audio_buffer = io.BytesIO()
    segment = AudioSegment(
        audio[1].tobytes(),
        frame_rate=audio[0],
        sample_width=audio[1].dtype.itemsize,
        channels=1,
    )
    segment.export(audio_buffer, format="mp3")
    return audio_buffer.getvalue()


def audio_to_file(audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
    """
    Save an audio tuple containing sample rate and numpy array data to a file.

    Parameters
    ----------
    audio : tuple[int, np.ndarray]
        A tuple containing:
            - sample_rate (int): The audio sample rate in Hz
            - data (np.ndarray): The audio data as a numpy array

    Returns
    -------
    str
        The path to the saved audio file

    Example
    -------
    >>> sample_rate = 44100
    >>> audio_data = np.array([0.1, -0.2, 0.3])  # Example audio samples
    >>> audio_tuple = (sample_rate, audio_data)
    >>> file_path = audio_to_file(audio_tuple)
    >>> print(f"Audio saved to: {file_path}")
    """
    bytes_ = audio_to_bytes(audio)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(bytes_)
    return f.name


def audio_to_float32(
    audio: NDArray[np.int16 | np.float32] | tuple[int, NDArray[np.int16 | np.float32]],
) -> NDArray[np.float32]:
    """
    Convert an audio tuple containing sample rate (int16) and numpy array data to float32.

    Parameters
    ----------
    audio : np.ndarray
        The audio data as a numpy array

    Returns
    -------
    np.ndarray
        The audio data as a numpy array with dtype float32

    Example
    -------
    >>> audio_data = np.array([0.1, -0.2, 0.3])  # Example audio samples
    >>> audio_float32 = audio_to_float32(audio_data)
    """
    if isinstance(audio, tuple):
        warnings.warn(
            UserWarning(
                "Passing a (sr, audio) tuple to audio_to_float32() is deprecated "
                "and will be removed in a future release. Pass only the audio array."
            ),
            stacklevel=2,  # So that the warning points to the user's code
        )
        _sr, audio = audio

    if audio.dtype == np.int16:
        # Divide by 32768.0 so that the values are in the range [-1.0, 1.0).
        # 1.0 can actually never be reached because the int16 range is [-32768, 32767].
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float32:
        return audio  # type: ignore
    else:
        raise TypeError(f"Unsupported audio data type: {audio.dtype}")


def audio_to_int16(
    audio: NDArray[np.int16 | np.float32] | tuple[int, NDArray[np.int16 | np.float32]],
) -> NDArray[np.int16]:
    """
    Convert an audio tuple containing sample rate and numpy array data to int16.

    Parameters
    ----------
    audio : np.ndarray
        The audio data as a numpy array

    Returns
    -------
    np.ndarray
        The audio data as a numpy array with dtype int16

    Example
    -------
    >>> audio_data = np.array([0.1, -0.2, 0.3], dtype=np.float32)  # Example audio samples
    >>> audio_int16 = audio_to_int16(audio_data)
    """
    if isinstance(audio, tuple):
        warnings.warn(
            UserWarning(
                "Passing a (sr, audio) tuple to audio_to_float32() is deprecated "
                "and will be removed in a future release. Pass only the audio array."
            ),
            stacklevel=2,  # So that the warning points to the user's code
        )
        _sr, audio = audio

    if audio.dtype == np.int16:
        return audio  # type: ignore
    elif audio.dtype == np.float32:
        # Convert float32 to int16 by scaling to the int16 range.
        # Multiply by 32767 and not 32768 so that int16 doesn't overflow.
        return (audio * 32767.0).astype(np.int16)
    else:
        raise TypeError(f"Unsupported audio data type: {audio.dtype}")


def aggregate_bytes_to_16bit(chunks_iterator):
    """
    Aggregate bytes to 16-bit audio samples.

    This function takes an iterator of chunks and aggregates them into 16-bit audio samples.
    It handles incomplete samples and combines them with the next chunk.

    Parameters
    ----------
    chunks_iterator : Iterator[bytes]
        An iterator of byte chunks to aggregate

    Returns
    -------
    Iterator[NDArray[np.int16]]
    """
    leftover = b""
    for chunk in chunks_iterator:
        current_bytes = leftover + chunk

        n_complete_samples = len(current_bytes) // 2
        bytes_to_process = n_complete_samples * 2

        to_process = current_bytes[:bytes_to_process]
        leftover = current_bytes[bytes_to_process:]

        if to_process:
            audio_array = np.frombuffer(to_process, dtype=np.int16).reshape(1, -1)
            yield audio_array


async def async_aggregate_bytes_to_16bit(chunks_iterator):
    """
    Aggregate bytes to 16-bit audio samples.

    This function takes an iterator of chunks and aggregates them into 16-bit audio samples.
    It handles incomplete samples and combines them with the next chunk.

    Parameters
    ----------
    chunks_iterator : Iterator[bytes]
        An iterator of byte chunks to aggregate

    Returns
    -------
    Iterator[NDArray[np.int16]]
        An iterator of 16-bit audio samples
    """
    leftover = b""

    async for chunk in chunks_iterator:
        current_bytes = leftover + chunk

        n_complete_samples = len(current_bytes) // 2
        bytes_to_process = n_complete_samples * 2

        to_process = current_bytes[:bytes_to_process]
        leftover = current_bytes[bytes_to_process:]

        if to_process:
            audio_array = np.frombuffer(to_process, dtype=np.int16).reshape(1, -1)
            yield audio_array


def webrtc_error_handler(func):
    """Decorator to catch exceptions and raise WebRTCError with stacktrace."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, WebRTCError):
                raise e
            else:
                raise WebRTCError(str(e)) from e

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, WebRTCError):
                raise e
            else:
                raise WebRTCError(str(e)) from e

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


async def wait_for_item(queue: asyncio.Queue, timeout: float = 0.1) -> Any:
    """
    Wait for an item from an asyncio.Queue with a timeout.

    This function attempts to retrieve an item from the queue using asyncio.wait_for.
    If the timeout is reached, it returns None.

    This is useful to avoid blocking `emit` when the queue is empty.
    """

    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        return None


RTCConfigurationCallable = (
    Callable[[], dict[str, Any]]
    | Callable[[], Coroutine[dict[str, Any], Any, dict[str, Any]]]
    | Callable[[str | None, str | None, str | None], dict[str, Any]]
    | Callable[
        [str | None, str | None, str | None],
        Coroutine[dict[str, Any], Any, dict[str, Any]],
    ]
    | dict[str, Any]
)
