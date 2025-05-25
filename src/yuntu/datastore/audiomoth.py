"""Prebuilt datastore for AudioMoth recorders"""

import os
from collections import OrderedDict
import struct
import re
import pytz
from datetime import datetime
import pandas as pd

from yuntu.datastore.base import Storage
from yuntu.core.audio.utils import hash_file, media_open
from yuntu.core.database.recordings import ULTRASONIC_SAMPLERATE_THRESHOLD

RIFF_ID_LENGTH = 4
LENGTH_OF_COMMENT = 256
PCM_FORMAT = 1

RIFF_FORMAT = f'{RIFF_ID_LENGTH}s'
uint32_t = 'I'
uint16_t = 'H'


class CustomStruct(OrderedDict):
    endian = '<'

    def __init__(self, objects):
        super().__init__(objects)

        self.struct = struct.Struct(self.collapse())
        self.size = self.struct.size

    def collapse(self):
        fmt = self.endian
        for value in self.values():
            if isinstance(value, CustomStruct):
                fmt += value.collapse().replace(self.endian, '')
            else:
                fmt += value

        return fmt

    def unpack(self, buffer):
        results = {}

        index = 0
        for key, value in self.items():
            if isinstance(value, CustomStruct):
                size = value.size
            else:
                size = struct.calcsize(value)

            subbuffer = buffer[index:index + size]

            if isinstance(value, CustomStruct):
                unpacked = value.unpack(subbuffer)
            else:
                unpacked = struct.unpack(self.endian + value, subbuffer)[0]

            results[key] = unpacked
            index += size

        return results


chunk_t = CustomStruct([
    ('id', RIFF_FORMAT),
    ('size', uint32_t)
])

icmt_t = CustomStruct([
    ('icmt', chunk_t),
    ('comment', f'{LENGTH_OF_COMMENT}s')
])

wavFormat_t = CustomStruct([
    ('format', uint16_t),
    ('numberOfChannels', uint16_t),
    ('samplesPerSecond', uint32_t),
    ('bytesPerSecond', uint32_t),
    ('bytesPerCapture', uint16_t),
    ('bitsPerSample', uint16_t),
])

wavHeader_t = CustomStruct([
    ('riff', chunk_t),
    ('format', RIFF_FORMAT),
    ('fmt', chunk_t),
    ('wavFormat', wavFormat_t),
    ('list', chunk_t),
    ('info', RIFF_FORMAT),
    ('icmt', icmt_t),
    ('data', chunk_t),
])


def read_am_header(path):
    with media_open(path, 'rb') as buffer:
        data = wavHeader_t.unpack(buffer.read(wavHeader_t.size))

    return data


id_regex = re.compile(r'AudioMoth ([0-9A-Z]{16})')


def get_am_id(comment):
    match = id_regex.search(comment)
    return match.group(1)


date_regex = re.compile(r'((\d{2}:\d{2}:\d{2}) (\d{2}\/\d{2}\/\d{4}) \((UTC((-|\+)(\d+))?)\))')


def get_am_datetime(comment):
    match = date_regex.search(comment)
    timezone = match.group

    raw = match.group(1)
    time = match.group(2)
    date = match.group(3)

    tz = match.group(4)

    try:
        offset_direction = match.group(6)
        offset = match.group(7)

        if len(offset) != 4:
            offset = '{:02d}00'.format(int(offset))

        new_tz = tz.replace(match.group(5), offset_direction + offset)
        raw = raw.replace(tz, new_tz)
        tz = new_tz
    except Exception:
        pass

    if "(UTC)" in raw:
        datetime_format = '%H:%M:%S %d/%m/%Y (%Z)'
    else:
        datetime_format = '%H:%M:%S %d/%m/%Y (%Z%z)'

    return {
        'raw': raw,
        'time': time,
        'date': date,
        'tz': tz,
        'datetime': datetime.strptime(raw, datetime_format),
        'format': datetime_format
    }


gain_regex = re.compile(r'gain setting (\d)')
gain_regex_alt = re.compile(r'at (\w*) gain')

def get_am_gain(comment):
    match = gain_regex.search(comment)
    if match is None:
        match = gain_regex_alt.search(comment)
    if match is not None:
        return match.group(1)
    return None


battery_regex = re.compile(r'battery state was (\d.\dV)')
battery_regex_alt = re.compile(r'battery state was greater than (\d.\d)')

def get_am_battery_state(comment):
    match = battery_regex.search(comment)
    if match is None:
        match = battery_regex_alt.search(comment)
    if match is not None:
        return match.group(1)
    return None

temperature_regex = re.compile(r'temperature was (\d{2}.\dC)')
temperature_regex_alt = re.compile(r'temperature was (\d{1}.\dC)')

def get_am_temperture(comment):
    match = temperature_regex.search(comment)
    if match is None:
        match = temperature_regex_alt.search(comment)
    if match is not None:
        return match.group(1)
    return None

class AudioMothStorage(Storage):

    def get_recording_dataframe(self, with_annotations=False):
        data = []
        for datum in self.iter():
            try:
                rec_meta = self.prepare_datum(datum)
            except:
                print(f"Warning: Could not read header of {datum}, ignoring...")
                rec_meta = None

            if rec_meta is not None:
                media_info = rec_meta.pop('media_info')
                rec_meta.update(media_info)

                if with_annotations:
                    annotations = [self.prepare_annotation(datum, annotation)
                                   for annotation in self.iter_annotations(datum)]
                    annotations = [x for x in annotations if x is not None]
                    rec_meta["annotations"] = annotations

                data.append(rec_meta)

        return pd.DataFrame(data)

    def iter_annotations(self, datum):
        return []

    def get_metadata(self):
        meta = {"type": "AudioMothStorage"}
        meta["dir_path"] = self.dir_path

        return meta

    def prepare_datum(self, datum):
        header = read_am_header(datum)

        nchannels = header['wavFormat']['numberOfChannels']
        samplerate = header['wavFormat']['samplesPerSecond']
        sampwidth = header['wavFormat']['bitsPerSample']
        filesize = header['riff']['size'] + 4
        length = int(8 * (filesize - wavHeader_t.size) /
                     (nchannels * sampwidth))

        media_info = {
            'nchannels': nchannels,
            'sampwidth': sampwidth,
            'samplerate': samplerate,
            'length': length,
            'filesize': filesize,
            'duration': length / samplerate
        }

        spectrum = 'ultrasonic' if samplerate > ULTRASONIC_SAMPLERATE_THRESHOLD else 'audible'

        comment = header['icmt']['comment'].decode('utf-8').rstrip('\x00')
        battery = get_am_battery_state(comment)
        gain = get_am_gain(comment)
        am_id = get_am_id(comment)
        temperature = get_am_temperture(comment)

        metadata = {
            'am_id': am_id,
            'gain': gain,
            'battery': battery,
            'comment': comment,
            'temperature': temperature
        }

        datetime_info = get_am_datetime(comment)
        datetime_ = datetime_info['datetime']

        if "(UTC)" in datetime_info["raw"]:
            time_zone = "UTC"
        else:
            time_zone = datetime_.tzinfo.tzname(datetime_)

        timezone_utc = pytz.timezone("UTC")

        return {
            'path': datum,
            'hash': hash_file(datum),
            'timeexp': 1,
            'media_info': media_info,
            'metadata': metadata,
            'spectrum': spectrum,
            'time_raw': datetime_info['raw'],
            'time_format': datetime_info['format'],
            'time_zone': time_zone,
            'time_utc': datetime_.astimezone(timezone_utc)
        }

    def prepare_annotation(self, datum, annotation):
        pass
