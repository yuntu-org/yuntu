"""DataStore Module.

A datastore is a utility class meant to facilitate
the import of large volumes of recordings into a yuntu
collection.
"""
from abc import ABC
from abc import abstractmethod
import os
import glob
import pickle
import pandas as pd
from pony.orm import db_session
from yuntu.core.audio.utils import read_info, hash_file


class Datastore(ABC):
    _size = None

    def __init__(self):
        self._metadata = None
        self.base_dir = '.'

    def get_abspath(self, path):
        if self.base_dir is None:
            return path
        return os.path.join(self.base_dir, path)

    @abstractmethod
    def iter(self):
        """Return an iterator of the data to import."""

    def iter_annotations(self, datum):
        """Return an iterator of the annotations of the corresponding datum."""
        return []

    @abstractmethod
    def prepare_datum(self, datum):
        """Prepare a datastore datum for collection insertion."""

    def prepare_annotation(self, datum, annotation):
        """Prepare a datastore annotation for collection insertion."""
        pass

    @abstractmethod
    def get_metadata(self):
        """Return self's metadata"""

    def pickle(self):
        """Pickle instance."""
        return pickle.dumps(self)

    @property
    def metadata(self):
        """Datastore metadata"""
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

    @property
    @abstractmethod
    def size(self):
        """Return datastore total elements"""

    def create_datastore_record(self, collection):
        """Register this datastore into the collection."""
        return collection.db_manager.models.datastore(metadata=self.metadata)

    @db_session
    def insert_into(self, collection):
        datastore_record = self.create_datastore_record(collection)
        datastore_record.flush()
        datastore_id = datastore_record.id

        recording_inserts = 0
        annotation_inserts = 0
        for datum in self.iter():
            meta = self.prepare_datum(datum)
            meta['path'] = self.get_abspath(meta['path'])
            meta['datastore'] = datastore_record
            recording = collection.insert(meta)[0]

            for annotation in self.iter_annotations(datum):
                annotation_meta = self.prepare_annotation(datum, annotation)
                annotation_meta['recording'] = recording
                collection.annotate([annotation_meta])
                annotation_inserts += 1

            recording_inserts += 1

        return datastore_id, recording_inserts, annotation_inserts

    def get_recording_dataframe(self, with_annotations=False):
        data = []
        for datum in self.iter():
            rec_meta = self.prepare_datum(datum)
            media_info = rec_meta.pop('media_info')
            rec_meta.update(media_info)

            if with_annotations:
                annotations = [self.prepare_annotation(datum, annotation)
                               for annotation in self.iter_annotations(datum)]
                rec_meta["annotations"] = annotations

            data.append(rec_meta)

        return pd.DataFrame(data)

    def get_annotation_dataframe(self):
        data = []
        for datum in self.iter():
            for annotation in self.iter_annotations(datum):
                data.append(self.prepare_annotation(datum, annotation))
        return pd.DataFrame(data)


class DataBaseDatastore(Datastore, ABC):

    def __init__(self, db_config, query, mapping, base_dir=None, tqdm=None):
        super().__init__()
        self.db_config = db_config
        self.query = query

        if base_dir is not None:
            self.base_dir = base_dir

        self.mapping = mapping
        self.tqdm = tqdm

    def get_metadata(self):
        meta = {"type": "DataBaseDatastore"}
        meta["db_config"] = self.db_config
        meta["query"] = self.query
        meta["mapping"] = self.mapping
        meta["base_dir"] = self.base_dir
        return meta


class Storage(Datastore):

    def __init__(self, dir_path, tqdm=None):
        super().__init__()
        self.dir_path = dir_path
        self.tqdm = tqdm

    def get_metadata(self):
        meta = {"type": "Storage"}
        meta["dir_path"] = self.dir_path
        return meta

    def iter(self):
        if self.tqdm is not None:
            for fname in self.tqdm(glob.glob(os.path.join(self.dir_path,
                                                          '*.WAV'))):
                yield fname
        else:
            for fname in glob.glob(os.path.join(self.dir_path, '*.WAV')):
                yield fname

    def prepare_datum(self, datum):
        timeexp = 1
        media_info = read_info(datum, timeexp=1)
        metadata = {}
        spectrum = 'ultrasonic' if media_info["samplerate"] > 100000 else 'audible'

        return {
            'path': datum,
            'hash': hash_file(datum),
            'timeexp': timeexp,
            'media_info': media_info,
            'metadata': metadata,
            'spectrum': spectrum,
        }

    @property
    def size(self):
        if self._size is None:
            self._size = len(glob.glob(os.path.join(self.dir_path, '*.WAV')))
        return self._size

    def create_datastore_record(self, collection):
        """Register this datastore into the collection."""
        return collection.db_manager.models.storage(metadata=self.metadata,
                                                    dir_path=self.dir_path)


class RemoteStorage(Storage):

    def __init__(self, dir_path, metadata_url=None, auth=None):
        super().__init__(dir_path)
        self.metadata_url = metadata_url
        self.auth = auth

    def get_metadata(self):
        meta = {
            "type": "RemoteStorage",
            "dir_path": self.dir_path,
            "metadata_url": self.metadata_url
        }
        return meta

    def create_datastore_record(self, collection):
        """Register this datastore into the collection."""
        return collection.db_manager.models.remote_storage(metadata=self.metadata,
                                                           dir_path=self.dir_path)