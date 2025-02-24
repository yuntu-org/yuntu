"""Auxiliar utilities for Audio classes and methods."""
from typing import Optional
import os
import glob
import io
import hashlib
import wave
import numpy as np
import librosa
import soundfile
import shutil

SAMPWIDTHS = {
    'PCM_16': 2,
    'PCM_32': 4,
    'PCM_U8': 1,
    'FLOAT': 4
}

def s3_glob(path):
    """Glob using s3fs.

    Parameters
    ----------
    path : str
        Aws s3 path.

    Returns
    -------
        path_names : list
            A list of path names.

    """
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    return [os.path.join("s3://", x) for x in s3.glob(path)]

def s3_walk(path):
    """Walk s3 path.

    Parameters
    ----------
    path : str
        Aws s3 path.

    Returns
    -------
        path_names : list
            A list of path names.

    """
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    return [(os.path.join("s3://", x[0]),x[1],x[2]) for x in s3.walk(path)]

def ag_glob(path):
    """Agnostic glob.

    Parameters
    ----------
    path : str
        Any path, remote or local.

    Returns
    -------
        path_names : list
            A list of path names.

    """
    if path[:5] == "s3://":
        return s3_glob(path)
    return glob.glob(path)

def ag_walk(path):
    """Agnostic walk.

    Parameters
    ----------
    path : str
        Any path, remote or local.

    Returns
    -------
        path_names : list
            A list of path names.

    """
    if path[:5] == "s3://":
        return s3_walk(path)
    return os.walk(path)

def media_open_s3(path):
    """Open file from s3 path.

    Parameters
    ----------
    path : str
        Aws s3 path.

    Returns
    -------
        file : file
            File object to access data.

    """
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    bucket = path.replace("s3://", "").split("/")[0]
    key = path.replace(f"s3://{bucket}/", "")
    return s3.open('{}/{}'.format(bucket, key))

def media_copy_s3(source_path, target_path):
    """Copy file from s3 source to target.

    Parameters
    ----------
    source_path : str
        Aws s3 source path.
    target_path : str
        A target path.

    Returns
    -------
        target_path : str
            The destination path

    """
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    if source_path[:5] == "s3://" and target_path[:5] == "s3://":
        return s3.copy(source_path, target_path)
    elif source_path[:5] == "s3://":
        return s3.download(source_path, target_path)
    return s3.upload(source_path, target_path)

def media_copy(source_path, target_path):
    """Copy file from any source to target.

    Parameters
    ----------
    source_path : str
        Any source path.
    target_path : str
        A target path.

    Returns
    -------
        target_path : str
            The destination path

    """
    if source_path[:5] == "s3://" or target_path[:5] == "s3://":
        return media_copy_s3(source_path, target_path)
    return shutil.copy(source_path, target_path)

def media_open(path, mode='rb'):
    """Open file from any path.

    Parameters
    ----------
    path : str
        Any path, remote or local.
    mode : str
        Open mode.

    Returns
    -------
        file : file
            File object to access data.

    """
    if path[:5] == "s3://":
        return media_open_s3(path)
    return open(path, mode)

def media_exists(path):
    """Check if path exists.

    Parameters
    ----------
    path : str
        Any path, remote or local

    Returns
    -------
        exists : bool
            Wether the path exists or not.

    """
    if path[:5] == "s3://":
        from s3fs.core import S3FileSystem
        s3 = S3FileSystem()
        return s3.exists(path)
    return os.path.exists(path)

def load_media_info_s3(path):
    """Return basic media info of s3 locations.

    Parameters
    ----------
    path : str
        Aws s3 path.

    Returns
    -------
        samplerate : int
            Recording samplerate.
        channels : int
            Number of channels in file.
        duration : float
            Recording duration.
        subtype : str
            Wav subtype.
        filesize : int
            Size of file.

    """
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    info = s3.info(path)
    if "size" in info:
        if info["size"] is not None:
            filesize = info["size"]
    elif "Size" in info:
        if info["Size"] is not None:
            filesize = info["Size"]
    else:
        raise ValueError(f"Could not retrieve size of file {path}")

    audio_info = soundfile.info(media_open_s3(path))

    return audio_info.samplerate, audio_info.channels, audio_info.duration, audio_info.subtype, filesize

def load_media_info(path):
    """Return basic media info of any audio path.

    Parameters
    ----------
    path : str
        Any path, remote or local.

    Returns
    -------
        samplerate : int
            Recording samplerate.
        channels : int
            Number of channels in file.
        duration : float
            Recording duration.
        subtype : str
            Wav subtype.
        filesize : int
            Size of file.

    """
    if not isinstance(path, io.BytesIO):
        if path[:5] == "s3://":
            return load_media_info_s3(path)
    audio_info = soundfile.info(path)
    filesize = media_size(path)
    return audio_info.samplerate, audio_info.channels, audio_info.duration, audio_info.subtype, filesize

def binary_md5(path, blocksize=65536):
    """Hash file by blocksize.

    Parameters
    ----------
    path : str
        Any path, remote or local.
    blocksize : int
        Size of chunks to read at once for hash generation.

    Returns
    -------
    hash : str
        A hash key for the file.

    """
    if path is None:
        raise ValueError("Path is None.")
    if not media_exists(path):
        raise ValueError("Path does not exist.")

    hasher = hashlib.md5()
    with media_open(path, "rb") as media:
        buf = media.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = media.read(blocksize)
    return hasher.hexdigest()

def media_size(path):
    """Return media size or None.

    Parameters
    ----------
    path : str
        Any path, remote or local.

    Returns
    -------
    media_size : int
        Size of file in bytes.

    """
    if isinstance(path, io.BytesIO):
        return len(path.getvalue())

    if path is not None:
        if os.path.isfile(path):
            return os.path.getsize(path)
    return None

def read_info(path, timeexp):
    """Read recording information form file.

    Parameters
    ----------
    path : str
        Any path, remote or local.
    timeexp : float
        Audio time expansion to interpret signal.

    Returns
    -------
    media_info : dict
        A dictionary holding media information from headers.

    """
    samplerate, nchannels, duration, subtype, filesize = load_media_info(path)
    media_info = {}
    media_info["samplerate"] = samplerate
    media_info["nchannels"] = nchannels
    media_info["sampwidth"] = SAMPWIDTHS[subtype]
    media_info["length"] = int(duration*samplerate)
    media_info["filesize"] = filesize
    media_info["duration"] = float(duration) / timeexp
    return media_info

def hash_file(path, alg="md5"):
    """Produce hash from audio recording.

    Parameters
    ----------
    path : str
        Any path, remote or local.
    alg : str
        Hash algorithm (only MD5 for the moment).

    Returns
    -------
    hash : str
        A hash key for the file.

    Raises
    ------

    NotImplementedError
        If the specified algorithm is not implemented.

    """
    if alg == "md5":
        return binary_md5(path)
    raise NotImplementedError("Algorithm "+alg+" is not implemented.")

def read_media(path,
               samplerate,
               offset=0.0,
               duration=None,
               **kwargs):
    """Read data as audio and return signal.

    Parameters
    ----------
    path : str
        Any path, remote or local.
    samplerate : int
        Audio samplerate to use for reading.
    offset : float
        Time offset to start reading.

    Returns
    -------
    signal : np.array
        Array containing audio data.

    """
    if isinstance(path, str):
        if path[:5] == "s3://":
            path = media_open_s3(path)
    return librosa.load(path,
                        sr=samplerate,
                        offset=offset,
                        duration=duration,
                        mono=True,
                        **kwargs)

def write_media(path,
                signal,
                samplerate,
                nchannels,
                media_format="wav"):
    """Write media.

    Parameters
    ----------
    signal : np.array
        Array containing audio signal.
    samplerate : int
        Write with this samplerate.
    nchannels : int
        Number of channels within new audio file.
    media_format : str
        Audio format to use for new file.

    """
    if media_format not in ["wav", "flac", "ogg"]:
        raise NotImplementedError("Writer for " + media_format
                                  + " not implemented.")
    if nchannels > 1:
        signal = np.transpose(signal, (1, 0))
    soundfile.write(path,
                    signal,
                    samplerate,
                    format=media_format)


def get_channel(signal, channel, nchannels):
    """Return correct channel in any case.

    Parameters
    ----------
    signal : np.array
        Array containing audio signal.
    channel : int
        Target channel to retrieve.
    nchannels : int
        Number of channels in data.

    Returns
    -------
    signal : np.array
        Array containing audio data from desired channel.

    """
    if nchannels > 1:
        return np.squeeze(signal[[channel], :])
    return signal


def resample(
        array: np.array,
        original_sr: int,
        target_sr: int,
        res_type: Optional[str] = 'kaiser_best',
        fix: Optional[bool] = True,
        scale: Optional[bool] = False,
        **kwargs):
    """Resample audio.

    Parameters
    ----------
    array : np.array
        Array containing original data.
    original_sr : int
        Original samplerate.
    target_sr : int
        Target samplerate.
    res_type : str
        Resampling method.
    fix : bool
        Adjust size of resulting audio file.
    scale : bool
        Scale data to have equivalent total energy.

    Returns
    -------
    signal : np.array
        Array containing resampled audio data.

    """
    return librosa.core.resample(
        array,
        orig_sr=original_sr,
        target_sr=target_sr,
        res_type=res_type,
        fix=fix,
        scale=scale,
        **kwargs)


def channel_mean(signal, keepdims=False):
    """Return channel mean.

    Parameters
    ----------
    signal : np.array
        Array containing audio data.
    keepdims : bool
        Keep array's original dimensions.

    Returns
    -------
    signal : np.array
        Array containing mean of array by channel.

    """
    return np.mean(signal,
                   axis=0,
                   keepdims=keepdims)
