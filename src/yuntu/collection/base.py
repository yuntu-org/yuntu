"""Base classes for collection."""
import os
import json
import pandas as pd
import shapely.wkt

from yuntu.core.database.base import DatabaseManager
from yuntu.core.database.timed import TimedDatabaseManager
from yuntu.core.database.spatial import SpatialDatabaseManager
from yuntu.core.database.spatiotemporal import SpatioTemporalDatabaseManager
from yuntu.core.database.spatial import build_query_with_geom
from yuntu.core.audio.audio import Audio
from yuntu.core.annotation.annotation import Annotation
from yuntu.datastore.copy import CopyDatastore

def _parse_annotation(annotation):
    return {
        'type': annotation.type,
        'id': annotation.id,
        'labels': annotation.labels,
        'metadata': annotation.metadata,
        'geometry': {
            'wkt': annotation.geometry
        }
    }


class Collection:
    """Base class for all collections.

    Parameters
    ----------
    db_config: dict
        Database parameters.
    base_path: str
        A string that specifies a directory path to prepend to each file
        path. This is useful when paths are relative and a copy of the data
        exists in a given directory.
    """

    db_config = {
        'provider': 'sqlite',
        'config': {
            'filename': ':memory:',
            'create_db': True
        }
    }
    audio_class = Audio
    annotation_class = Annotation
    db_manager_class = DatabaseManager

    def __init__(self, db_config=None, base_path=""):
        """Initialize collection.

        Create a new collection handler according to database or source
        configuration.

        Parameters
        ----------
        db_config: dict
            Database parameters.
        base_path: str
            A string that specifies a directory path to prepend to each file
            path. This is useful when paths are relative and a copy of the data
            exists in a given directory.
        """
        self.base_path = base_path
        if self.base_path != "":
            if self.base_path[:5] != "s3://":
                if not os.path.isabs(self.base_path):
                    self.base_path = os.path.abspath(self.base_path)

        if db_config is not None:
            self.db_config = db_config

        if self.db_config["provider"] == "sqlite":
            if self.base_path != "":
                filename = self.db_config["config"]["filename"]
                if not os.path.isabs(filename):
                    filename = os.path.join(self.base_path, filename)
                self.db_config["config"]["filename"] = filename

        self.db_manager = self.get_db_manager()

    def __getitem__(self, key):
        queryset = self.recordings()
        if isinstance(key, int):
            return self.build_audio(queryset[key:key + 1][0])

        return [self.build_audio(recording) for recording in queryset[key]]

    def __iter__(self):
        for recording in self.recordings():
            yield self.build_audio(recording)

    def __len__(self):
        return self.recordings().count()

    def get(self, key, with_metadata=True):
        """Get Audio object by key.

        Fetch one Audio object using the entry's Id as key.

        Parameters
        ----------
        key : int, str
            Recording Id.
        with_metadata: bool
            Wether to include metadata in the response or not.

        Returns
        -------
        audio : Audio
            An audio object.
        """
        record = self.recordings(lambda rec: rec.id == key).get()
        return self.build_audio(record, with_metadata=with_metadata)

    def get_recording_dataframe(
            self,
            query=None,
            limit=None,
            offset=0,
            with_metadata=False,
            with_annotations=False,
            **kwargs):
        """Get audio dataframe from query.

        Fetch recording entries and build a pandas dataframe compatible with
        AudioAccessor.

        Parameters
        ----------
        query : callable
            A function that conforms to Pony's query syntax.
        limit : int
            The number of maximum entries to return.
        offset : int
            Skip this many entries and return the rest up to limit.
        with_metadata : bool
            Wether to include metadata in the response or not.
        with_annotations : bool
            Wether to include annotations in the response or not.

        Returns
        -------
        recording_dataframe : pandas.DataFrame
            A dataframe holding recordings.
        """
        if limit is None:
            query_slice = slice(offset, None)
        else:
            query_slice = slice(offset, offset + limit)
        recordings = self.recordings(query=query, **kwargs).order_by(lambda r: r.id)[query_slice]

        records = []
        for recording in recordings:
            data = recording.to_dict()
            media_info = data.pop('media_info')
            data.update(media_info)
            data["path"] = self.get_abspath(data["path"])

            if not with_metadata:
                data.pop('metadata')

            if with_annotations:
                data['annotations'] = [
                    _parse_annotation(annotation)
                    for annotation in recording.annotations]

            records.append(data)

        return pd.DataFrame(records)

    def get_annotation_dataframe(
            self,
            query=None,
            limit=None,
            offset=0,
            with_metadata=None):
        """Get annotation dataframe from query.

        Fetch recording entries and build a pandas dataframe compatible with
        AnnotationAccessor.

        Parameters
        ----------
        query: callable
            A function that conforms to Pony's query syntax.
        limit: int
            The number of maximum entries to return.
        offset: int
            Skip this many entries and return the rest up to limit.
        with_metadata: bool
            Wether to include metadata in the response or not.

        Returns
        -------
        annotation_dataframe: pandas.DataFrame
            A dataframe holding annotations.
        """
        if limit is None:
            query_slice = slice(offset, None)
        else:
            query_slice = slice(offset, offset + limit)
        annotations = self.annotations(query=query).order_by(lambda a: a.id)[query_slice]

        records = []
        for annotation in annotations:
            data = annotation.to_dict()
            labels = data.pop('labels')

            if not with_metadata:
                data.pop('metadata')

            data['labels'] = labels

            for label in labels:
                data[label['key']] = label['value']

            records.append(data)

        return pd.DataFrame(records)

    def get_db_manager(self):
        """Get database manager.

        Use config to retrieve a database connection manager.

        Parameters
        ----------
            None

        Returns
        -------
        db_manager: DatabaseManager
            Data base manager according to configuration.
        """
        return self.db_manager_class(**self.db_config)

    def get_abspath(self, path):
        """Get absolute path from entry.

        Compute absolute path using base path configuration.

        Parameters
        ----------
        path: str
            Recording path.

        Returns
        -------
        absolute_path: str
            Absolute path according to configs.
        """
        if path[:5] != "s3://":
            if not os.path.isabs(path):
                return os.path.join(self.base_path, path)
        return path

    def insert(self, meta_arr):
        """Directly insert new media entries without a datastore.

        Insert an array of entries to collection.

        Parameters
        ----------
        meta_arr: list
            A list of dictionaries specifying new entries to insert.

        Returns
        -------
        insertions: list
            Newly inserted entries.
        """
        if not isinstance(meta_arr, (list, tuple)):
            meta_arr = [meta_arr]
        return self.db_manager.insert(meta_arr)

    def annotate(self, meta_arr):
        """Insert annotations to database.

        Insert an array of entries to collection.

        Parameters
        ----------
        meta_arr: list
            A list of dictionaries specifying new entries to insert.

        Returns
        -------
        insertions: list
            Newly inserted entries.
        """
        return self.db_manager.insert(meta_arr, model="annotation")

    def update_recordings(self, query, set_obj):
        """Update matching recordings.

        Update entries that match query according to specification.

        Parameters
        ----------
        query: callable
            A function that conforms to Pony's query syntax.
        set_obj: dict
            A dictionary that has field names as keys and new values as values.

        Returns
        -------
        updates: list
            Update results.
        """
        return self.db_manager.update(query, set_obj, model="recording")

    def update_annotations(self, query, set_obj):
        """Update matching annotations.
        Update entries that match query according to specification.

        Parameters
        ----------
        query: callable
            A function that conforms to Pony's query syntax.
        set_obj: dict
            A dictionary that has field names as keys and new values as values.

        Returns
        -------
        updates: list
            Update results.
        """
        return self.db_manager.update(query, set_obj, model="annotation")

    def delete_recordings(self, query):
        """Delete matching recordings.

        Delete entries that match query.

        Parameters
        ----------
        query: callable
            A function that conforms to Pony's query syntax.

        Returns
        -------
        deletes: list
            Delete results.
        """
        return self.db_manager.delete(query, model='recording')

    def delete_annotations(self, query):
        """Delete matching annotations.

        Delete entries that match query.

        Parameters
        ----------
        query: callable
            A function that conforms to Pony's query syntax.

        Returns
        -------
        deletes: list
            Delete results.
        """
        return self.db_manager.delete(query, model='annotation')

    @property
    def recordings_model(self):
        """Recordings model.

        Recordings model for this collection.

        """
        return self.db_manager.models.recording

    @property
    def annotations_model(self):
        """Annotations model.

        Annotations model for this collection.

        """
        return self.db_manager.models.annotation

    def annotations(self, query=None, iterate=True):
        """Retrieve annotations from database.

        Fetch annotations according to query.

        Parameters
        ----------
        query: callable
            A function that conforms to Pony's query syntax.
        iterate: bool
            Wether to return as list or as an iterator.
        Returns
        -------
        annotations: list, iterator
            A list of annotations as pony entities.
        """
        matches = self.db_manager.select(query, model="annotation")
        if iterate:
            return matches
        return list(matches)

    def recordings(self, query=None, iterate=True):
        """Retrieve recording objects from database

        Fetch recordings according to query.

        Parameters
        ----------
        query: callable
            A function that conforms to Pony's query syntax.
        iterate: bool
            Wether to return as list or as an iterator.
        Returns
        -------
        recordings: list, iterator
            A list of recording entries as pony entities.
        """
        matches = self.db_manager.select(query, model="recording")
        if iterate:
            return matches
        return list(matches)

    def build_audio(self, recording, with_metadata=True):
        """Build audio object.

        Build audio object from database entry.

        Parameters
        ----------
        recording: pony.Entity
            An entity that conforms to Recording class for this collection.
        with_metadata: bool
            Wether to include metadata in the response or not.

        Returns
        -------
        audio: Audio
        """
        annotations = []
        for annotation in recording.annotations:
            data = annotation.to_dict()
            annotation = self.annotation_class.from_record(data)
            annotations.append(annotation)

        metadata = recording.metadata if with_metadata else None

        path = self.get_abspath(recording.path)

        return self.audio_class(
            path=path,
            id=recording.id,
            media_info=recording.media_info,
            timeexp=recording.timeexp,
            metadata=metadata,
            annotations=annotations,
            lazy=True)

    def pull(self, datastore):
        """Pull data from datastore and insert into collection.

        Pull data from a datastore and insert to collection.

        Parameters
        ----------
        datastore: Datastore
            A yuntu datastore that specifies a data source to incorporate.

        """

        datastore.insert_into(self)

    def materialize(self, out_name, query=None, out_dir="", tqdm=None):
        """Create materialized collection.

        Create an sqlite copy of collection's database as well as a directory
        holding a copy of the corresponding data.

        Parameters
        ----------
        out_name: str
            Name for the new materialized collection.
        query: callable
            A function that conforms to Pony's query syntax.
        out_dir: str
            A target directory to copy data.
        tqdm: module
            A tqdm module to use for reporting progress.

        Returns
        -------
        collection_path: str
            The absolute path for a newly copy of the corresponding data and
            database.
        """
        target_path = os.path.join(out_dir, out_name)
        copystore = CopyDatastore(collection=self,
                                  query=query,
                                  target_path=target_path,
                                  tqdm=tqdm)
        col_type = "simple"
        if isinstance(self.db_manager, TimedDatabaseManager):
            col_type = "timed"

        target_config = {
            "col_type": col_type,
            "db_config": {
                "provider": "sqlite",
                "config": {
                    "filename": "db.sqlite",
                    "create_db": True
                }
            }
        }

        col_config_path = os.path.join(target_path, f"col_config.json")

        with open(col_config_path, "w") as f:
            json.dump(target_config, f)

        target_col = self.__class__(db_config=target_config["db_config"],
                                    base_path=target_path)

        _, _, _, = copystore.insert_into(target_col)

        return target_path

class TimedCollection(Collection):
    """Time aware collection."""
    db_manager_class = TimedDatabaseManager

class SpatialCollection(Collection):
    """Geographic aware collection."""
    db_manager_class = SpatialDatabaseManager

    def recordings(self, query=None, wkt=None, method='intersects', iterate=True):
        """Retrieve audio objects."""
        if wkt is not None:
            geom_query = build_query_with_geom(provider=self.db_manager.provider, wkt=wkt, method=method)
            matches = self.db_manager.select(geom_query, model="recording")
            if query is not None:
                matches = matches.filter(query)
        else:
            matches = self.db_manager.select(query, model="recording")

        if iterate:
            return matches
        return list(matches)

    def get_recording_dataframe(
            self,
            query=None,
            limit=None,
            offset=0,
            with_metadata=False,
            with_annotations=False,
            with_geometry=False,
            **kwargs):
        if limit is None:
            query_slice = slice(offset, None)
        else:
            query_slice = slice(offset, offset + limit)
        recordings = self.recordings(query=query, **kwargs).order_by(lambda r: r.id)[query_slice]

        records = []
        for recording in recordings:
            data = recording.to_dict()
            media_info = data.pop('media_info')
            data.update(media_info)
            data["path"] = self.get_abspath(data["path"])

            if not with_metadata:
                data.pop('metadata')

            if not with_geometry:
                data.pop('geometry')
            else:
                data["geometry"] = shapely.wkt.loads(data["geometry"])

            if with_annotations:
                data['annotations'] = [
                    _parse_annotation(annotation)
                    for annotation in recording.annotations]

            records.append(data)

        return pd.DataFrame(records)


class SpatioTemporalCollection(SpatialCollection):
    """Geographic aware collection."""
    db_manager_class = SpatioTemporalDatabaseManager
