"""Media module.

This module defines the base class for all media objects in yuntu.
A media object is any object that holds information on an acoustic event.
This could be the full wav array, the zero crossing rate or the spectrogram.
These media objects can all be stored and read from the filesystem.
"""
import os

from abc import ABC
from abc import abstractmethod
from urllib.parse import urlparse

from yuntu.utils import download_file
from yuntu.core.windows import Window
from yuntu.core.annotation.annotated_object import AnnotatedObjectMixin


class Media(ABC, AnnotatedObjectMixin):
    """Media class.

    This is the base class for all media objects in yuntu.
    """

    window_class = Window
    mask_class = None

    # Plotting variables
    plot_title = 'Media Object'
    plot_xlabel = ''
    plot_ylabel = ''

    # pylint: disable=super-init-not-called, unused-argument
    def __init__(
            self,
            path=None,
            lazy=False,
            array=None,
            window=None,
            **kwargs):
        """Construct a media object."""
        self.path = path

        if window is None:
            # pylint: disable=abstract-class-instantiated
            window = self.window_class()

        if not isinstance(window, Window):
            window = Window.from_dict(window)

        self.window = window

        if array is not None:
            self._array = array
        elif not lazy:
            self._array = self._load_array()

        super().__init__(**kwargs)

    def _load_array(self):
        if self.is_remote():
            tmpfile = self.remote_load()
            array = self.load(path=tmpfile)
            tmpfile.close()
            return array

        return self.load()

    def force_load(self):
        self._array = self._load_array()

    @property
    def array(self):
        """Get media contents."""
        if self.is_empty():
            self._array = self._load_array()
        return self._array

    @property
    def path_ext(self):
        """Get extension of media file."""
        _, ext = os.path.splitext(self.path)
        return ext

    def to_dict(self):
        """Return a dictionary holding all media metadata."""
        data = {
            'type': self.__class__.__name__,
            'window': self.window.to_dict(),
            **super().to_dict()
        }

        if self.path is not None:
            data['path'] = self.path

        return data

    def is_remote(self):
        if self.path is None:
            return False

        if os.path.exists(self.path):
            return False

        if self.path[:5] == 's3://':
            return False

        parsed = urlparse(self.path)

        if not parsed.scheme:
            return False

        if parsed.scheme == 'file':
            return False

        return True

    def remote_load(self):
        parsed = urlparse(self.path)

        if parsed.scheme in ['http', 'https']:
            return download_file(self.path)

        message = (
            'Remote loading is not implemented for '
            f'scheme {parsed.scheme}')
        raise NotImplementedError(message)

    def path_exists(self, path=None):
        """Determine if the media file exists in the filesystem."""
        if path is None:
            path = self.path

        if path is None:
            return False

        if not isinstance(path, str):
            return True

        if "s3://" == path[:5]:
            from s3fs.core import S3FileSystem
            s3 = S3FileSystem()
            return s3.exists(path)

        return os.path.exists(path)

    def clean(self):
        """Clear media contents and free memory."""
        del self._array

    def is_empty(self):
        """Check if array has not been loaded yet."""
        return not hasattr(self, '_array') or self._array is None

    def load(self, path=None):
        if self.path_exists(path):
            return self.load_from_path(path)

        return self.compute()

    def compute(self):
        message = 'This class does not implement a compute method'
        raise NotImplementedError(message)

    def load_from_path(self, path=None):
        if path is None:
            path = self.path

        message = 'This class does not implement a load from path method'
        raise NotImplementedError(message)

    @abstractmethod
    def write(self, path=None, **kwargs):
        """Write media object into filesystem."""

    def get_plot_title(self):
        return self.plot_title

    def get_plot_xlabel(self):
        return self.plot_xlabel

    def get_plot_ylabel(self):
        return self.plot_ylabel

    @abstractmethod
    def plot(self, ax=None, **kwargs):  # pylint: disable=invalid-name
        """Plot a representation of the media object."""
        ax = super().plot(ax=ax, **kwargs)

        title = kwargs.get('title', False)
        if title:
            if not isinstance(title, str):
                title = self.get_plot_title()
            ax.set_title(title)

        xlabel = kwargs.get('xlabel', False)
        if xlabel:
            if not isinstance(xlabel, str):
                xlabel = self.get_plot_xlabel()
            ax.set_xlabel(xlabel)

        ylabel = kwargs.get('ylabel', False)
        if ylabel:
            if not isinstance(ylabel, str):
                ylabel = self.get_plot_ylabel()
            ax.set_ylabel(ylabel)

        if kwargs.get('window', False):
            window_kwargs = kwargs.get('window_kwargs', {})
            ax = self.window.plot(ax=ax, **window_kwargs)

        return ax

    def normalized(self, method='minmax', **kwargs):
        if method == 'minmax':
            array = self.array

            minimum = kwargs.get('minimum', None)
            if minimum is None:
                minimum = array.min()

            maximum = kwargs.get('maximum', None)
            if maximum is None:
                maximum = array.max()

            return (array - minimum) / (maximum - minimum)

        if method == 'z':
            array = self.array

            mean = kwargs.get('mean', None)
            if mean is None:
                mean = array.mean()

            std = kwargs.get('std', None)
            if std is None:
                std = array.std()

            return (array - mean) / std

        message = f'Normalization method {method} is not implemented'
        raise NotImplementedError(message)

    def copy(self, **kwargs):
        """Copy media element."""
        data = self._copy_dict(**kwargs)
        cls = type(self)
        return cls(**data)

    def _copy_dict(self, with_array=True, **kwargs):
        data = {
            'annotations': self.annotations,
            'window': self.window.copy(),
            'path': self.path,
        }

        if not self.is_empty() and with_array:
            data['array'] = self.array.copy(**kwargs)

        return data

    def __copy__(self):
        """Copy media element."""
        return self.copy()

    # pylint: disable=no-self-use
    def _has_trivial_window(self):
        return True

    # pylint: disable=no-self-use
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Use numpy universal functions on media array."""
        modified_inputs = tuple([
            inp.array
            if isinstance(inp, Media) else inp
            for inp in inputs
        ])
        modified_kwargs = {
            key:
                value.array
                if isinstance(value, Media)
                else value
            for key, value in kwargs.items()
        }

        return getattr(ufunc, method)(*modified_inputs, **modified_kwargs)

    def __enter__(self):
        """Behaviour for context manager"""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Behaviour for context manager"""
        self.clean()


NUMPY_METHODS = [
    'all',
    'any',
    'argmax',
    'argmin',
    'argpartition',
    'argsort',
    'astype',
    'byteswap',
    'choose',
    'clip',
    'compress',
    'conj',
    'conjugate',
    'cumprod',
    'cumsum',
    'diagonal',
    'dot',
    'dump',
    'dumps',
    'fill',
    'flatten',
    'getfield',
    'item',
    'itemset',
    'max',
    'mean',
    'min',
    'newbyteorder',
    'nonzero',
    'partition',
    'prod',
    'ptp',
    'put',
    'ravel',
    'repeat',
    'reshape',
    'resize',
    'round',
    'searchsorted',
    'setfield',
    'setflags',
    'sort',
    'squeeze',
    'std',
    'sum',
    'swapaxes',
    'take',
    'tobytes',
    'tofile',
    'tolist',
    'tostring',
    'trace',
    'transpose',
    'var',
    'view',
    '__abs__',
    '__add__',
    '__and__',
    '__bool__',
    '__contains__',
    '__delitem__',
    '__divmod__',
    '__eq__',
    '__float__',
    '__floordiv__',
    '__ge__',
    '__getitem__',
    '__gt__',
    '__iadd__',
    '__iand__',
    '__ifloordiv__',
    '__ilshift__',
    '__imatmul__',
    '__imod__',
    '__imul__',
    '__index__',
    '__int__',
    '__invert__',
    '__ior__',
    '__ipow__',
    '__irshift__',
    '__isub__',
    '__iter__',
    '__itruediv__',
    '__ixor__',
    '__le__',
    '__len__',
    '__lshift__',
    '__lt__',
    '__matmul__',
    '__mod__',
    '__mul__',
    '__ne__',
    '__neg__',
    '__or__',
    '__pos__',
    '__pow__',
    '__radd__',
    '__rand__',
    '__rdivmod__',
    '__repr__',
    '__rfloordiv__',
    '__rlshift__',
    '__rmatmul__',
    '__rmod__',
    '__rmul__',
    '__ror__',
    '__rpow__',
    '__rrshift__',
    '__rshift__',
    '__rsub__',
    '__rtruediv__',
    '__rxor__',
    '__setitem__',
    '__str__',
    '__sub__',
    '__truediv__',
    '__xor__'
]


NUMPY_PROPERTIES = [
    'T',
    'data',
    'dtype',
    'flags',
    'flat',
    'imag',
    'real',
    'size',
    'itemsize',
    'nbytes',
    'ndim',
    'shape',
    'strides',
    'ctypes',
    'base',
]


def _build_method(method_name):
    def class_method(self, *args, **kwargs):
        return getattr(self.array, method_name)(*args, **kwargs)
    return class_method


def _build_property(property_name):
    @property
    def class_property(self):
        return getattr(self.array, property_name)
    return class_property


for meth in NUMPY_METHODS:
    setattr(Media, meth, _build_method(meth))

for prop in NUMPY_PROPERTIES:
    setattr(Media, prop, _build_property(prop))
