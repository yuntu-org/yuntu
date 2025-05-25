'''Prebuilt storage for all GUANO compatible files'''
import pytz
from datetime import datetime
from guano import GuanoFile
import pandas as pd

from yuntu.datastore.base import Storage
from yuntu.core.audio.utils import hash_file, read_info
from yuntu.core.database.recordings import ULTRASONIC_SAMPLERATE_THRESHOLD

class GuanoStorage(Storage):

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
        meta = {"type": "GuanoStorage"}
        meta["dir_path"] = self.dir_path

        return meta

    def prepare_datum(self, datum):
        guano_info = {key:value for key,value in GuanoFile(datum).items()}
        media_info = read_info(datum, timeexp=1.0)
        samplerate = media_info['samplerate']

        spectrum = 'ultrasonic' if samplerate > ULTRASONIC_SAMPLERATE_THRESHOLD else 'audible'

        timezone_utc = pytz.timezone("UTC")
        datetime_ = guano_info.pop("Timestamp")
        longitude, latitude = guano_info["Loc Position"]

        time_zone = datetime_.tzinfo.tzname(datetime_)
        time_format = '%H:%M:%S %d/%m/%Y (%Z)'
        time_raw = datetime_.strftime(format=time_format)

        return {
            'path': datum,
            'hash': hash_file(datum),
            'timeexp': 1,
            'media_info': media_info,
            'metadata': guano_info,
            'spectrum': spectrum,
            'latitude': latitude,
            'longitude': longitude,
            'time_raw': time_raw,
            'time_format': time_format,
            'time_zone': time_zone,
            'time_utc': datetime_.astimezone(timezone_utc)
        }

    def prepare_annotation(self, datum, annotation):
        pass