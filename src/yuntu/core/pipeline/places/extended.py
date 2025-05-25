"""Place nodes that behave like frames."""
import os
from copy import copy
from collections import OrderedDict
import numpy as np
import dask.array as dask_array
from dask.delayed import Delayed
import dask.bag as dask_bag
import dask.dataframe as dd
import pandas as pd
from yuntu.core.pipeline.base import Node
from yuntu.core.pipeline.places.base import Place
from yuntu.core.pipeline.places.base import DynamicPlace
from yuntu.core.pipeline.places.base import PickleablePlace
from yuntu.core.pipeline.places.base import ScalarPlace
from yuntu.core.pipeline.places.base import BoolPlace
from yuntu.core.pipeline.places.base import DictPlace
from yuntu.core.pipeline.transitions.base import Transition


class NumpyArrayMixin:
    """Mixin that adds numpy array like behaviour."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Produce UfuncTransition from method and inputs."""
        op_inputs = []
        arg_count = 0
        for item in inputs:
            if isinstance(item, Node):
                op_inputs.append(item)
            else:
                input_class = _guess_place_class(item)
                op_inputs.append(input_class(name=f"{ufunc.__name__}" +
                                             f"_input_{arg_count}",
                                             pipeline=self.pipeline,
                                             data=item,
                                             is_output=False,
                                             persist=False,
                                             keep=False))
            arg_count += 1
        op_nin = len(op_inputs)

        op_outputs = []
        arg_count = 0
        for out in kwargs.get('out', ()):
            if isinstance(out, Node):
                op_outputs.append(out)
            else:
                input_class = _guess_place_class(out)
                op_outputs.append(input_class(name=f"{ufunc.__name__}" +
                                              f"_input_ufunc_out_{arg_count}",
                                              pipeline=self.pipeline,
                                              data=out,
                                              is_output=False,
                                              persist=False,
                                              keep=False))
            arg_count += 1
        op_nout = len(op_outputs)

        extra_args = []
        kwarg_keys = [key for key in kwargs if key != "out"]
        for key in kwarg_keys:
            if isinstance(kwargs[key], Node):
                extra_args.append(kwargs[key])
            else:
                input_class = _guess_place_class(kwargs[key])
                extra_args.append(input_class(name=f"{key}",
                                              pipeline=self.pipeline,
                                              data=kwargs[key]))

        def wrapper(*args):
            inputs_ = args[0:op_nin]
            outputs_ = args[op_nin:op_nin+op_nout]
            other_args = args[op_nin+op_nout:]

            inputs_ = tuple(inputs_)
            outputs_ = tuple(outputs_)
            if len(outputs_) == 0:
                outputs_ = None
            wrapper_kwargs = {}
            wrapper_kwargs["out"] = outputs_
            for i in range(len(kwarg_keys)):
                wrapper_kwargs[kwarg_keys[i]] = other_args[i]
            return getattr(ufunc, method)(*inputs_, **wrapper_kwargs)

        transition_name = ufunc.__name__

        outputs = []
        if len(op_outputs) == 0:
            outputs.append(NumpyArrayPlace(name=f"{transition_name}_output",
                                           pipeline=self.pipeline,
                                           is_output=False,
                                           persist=False,
                                           keep=False))
        else:
            for out in op_outputs:
                new_out = copy(out)
                new_out.set_pipeline(self.pipeline)
                outputs.append(new_out)

        inputs = op_inputs+op_outputs+extra_args
        inp_sig = []
        out_sig = []

        for inp in inputs:
            inp_sig.append(inp.__class__)

        for out in outputs:
            out_sig.append(out.__class__)

        signature = (tuple(inp_sig), tuple(out_sig))

        transition = Transition(name=transition_name,
                                operation=wrapper,
                                pipeline=self.pipeline,
                                inputs=op_inputs+op_outputs+extra_args,
                                outputs=outputs,
                                signature=signature)

        if len(transition.outputs) == 1:
            return transition.outputs[0]
        return transition.outputs


class NumpyArrayPlace(PickleablePlace, NumpyArrayMixin):
    data_class = np.ndarray


# PANDAS EXTENSIONS
class PandasSeriesMixin:
    """Mixin that adds pandas series like behaviour to nodes."""


class PandasDataFrameMixin:
    """Mixin that adds pandas dataframe like behaviour to nodes."""


class PandasSeriesGroupByMixin:
    """Mixin that adds pandas series groupby like behaviour to nodes."""


class PandasDataFrameGroupByMixin:
    """Mixin that adds pandas dataframe groupby like behaviour to nodes."""


class PandasSeriesPlace(PickleablePlace, PandasSeriesMixin):
    data_class = pd.Series

    def validate(self, data):
        if data is None:
            return True
        return (super().validate(data)
                and isinstance(data, self.data_class))

    def write(self, path=None, data=None):
        if path is None:
            if self.pipeline is None:
                raise ValueError("Can not infer output path for node without a"
                                 "pipeline")
            self.pipeline.init_dirs()
            path = self.get_persist_path()
        if data is None:
            data = self.data
        if not self.validate(data):
            message = "Data is invalid."
            raise ValueError(message)
        data.to_pickle(path, None)

    def read(self, path=None):
        if path is None:
            path = self.get_persist_path()
        if not os.path.exists(path):
            message = "No pickled data at path."
            raise ValueError(message)
        data = pd.read_pickle(path, None)
        return data


class PandasDataFramePlace(PickleablePlace, PandasDataFrameMixin):
    data_class = pd.DataFrame

    def validate(self, data):
        if data is None:
            return True
        return (super().validate(data)
                and isinstance(data, self.data_class))

    def write(self, path=None, data=None):
        if path is None:
            if self.pipeline is None:
                raise ValueError("Can not infer output path for node without a"
                                 "pipeline")
            self.pipeline.init_dirs()
            path = self.get_persist_path()
        if data is None:
            data = self.data
        if not self.validate(data):
            message = "Data is invalid."
            raise ValueError(message)
        data.to_pickle(path, None)

    def read(self, path=None):
        if path is None:
            path = self.get_persist_path()
        if not os.path.exists(path):
            message = "No pickled data at path."
            raise ValueError(message)
        data = pd.read_pickle(path, None)
        return data


# DASK EXTENSIONS #
class DaskArrayMixin:
    """Mixin that adds dask array behaviour to nodes."""


class DaskSeriesMixin:
    """Mixin that adds dask series behaviour to nodes."""


class DaskBagMixin:
    """Mixin that adds dask series behaviour to nodes."""


class DaskDelayedMixin:
    """Mixin that adds dask series behaviour to nodes."""


class DaskDataFrameMixin:
    """Mixin that adds dask dataframe behaviour to nodes."""


class DaskSeriesGroupByMixin:
    """Mixin that adds dask series group by behaviour to nodes."""


class DaskDataFrameGroupByMixin:
    """Mixin that adds dask dataframe group by behaviour to nodes."""


class DaskArrayPlace(DynamicPlace, DaskArrayMixin):
    """Dask array input."""
    data_class = dask_array.core.Array

    @property
    def data(self):
        if hasattr(self._result, "compute"):
            return self._result.compute()
        return self._result


class DaskSeriesPlace(Place, DaskSeriesMixin):
    """Dask series input."""
    data_class = dd.Series

    def write(self, path=None, data=None):
        if path is None:
            if self.pipeline is None:
                raise ValueError("Can not infer output path for node without a"
                                 "pipeline")
            self.pipeline.init_dirs()
            path = self.get_persist_path()
        if data is None:
            data = self.data
        if not isinstance(data, pd.Series):
            if not self.validate(data):
                message = "Data is invalid."
                raise ValueError(message)
        return data.to_csv(path)

    def read(self, path=None):
        if path is None:
            path = self.get_persist_path()
        if not os.path.exists(path):
            message = "No persisted data at path."
            raise ValueError(message)
        return dd.read_csv(path)

    def get_persist_path(self):
        if self.key is None:
            base_name = self.name
        else:
            base_name = self.key
        return os.path.join(self.pipeline.persist_dir, base_name+".csv")

    def set_value(self, value):
        """Set result value manually."""
        if not isinstance(value, pd.Series):
            if not self.validate(value):
                raise ValueError("Value is incompatible with node type "
                                 f"{type(self)}")
        self._result = value


class DaskBagPlace(DynamicPlace, DaskBagMixin):
    """Dask bag input."""
    data_class = dask_bag.core.Bag

    @property
    def data(self):
        if hasattr(self._result, "compute"):
            return self._result.compute()
        return self._result


class DaskDelayedPlace(DynamicPlace, DaskDelayedMixin):
    """Dask delayed input."""
    data_class = Delayed

    @property
    def data(self):
        if hasattr(self._result, "compute"):
            return self._result.compute()
        return self._result


class DaskDataFramePlace(Place, DaskDataFrameMixin):
    """Dask dataframe input."""
    data_class = dd.DataFrame

    def write(self, path=None, data=None):
        if path is None:
            if self.pipeline is None:
                raise ValueError("Can not infer output path for node without a"
                                 "pipeline")
            self.pipeline.init_dirs()
            path = self.get_persist_path()
        if data is None:
            data = self.data
        if not isinstance(data, pd.DataFrame):
            if not self.validate(data):
                message = "Data is invalid."
                raise ValueError(message)
        return data.to_parquet(path, compression="GZIP")

    def read(self, path=None):
        if path is None:
            path = self.get_persist_path()
        if not os.path.exists(path):
            message = "No persisted data at path."
            raise ValueError(message)
        try:
            df = dd.read_parquet(path)
        except:
            df = dd.read_parquet(path, engine="pyarrow")
        return df

    def get_persist_path(self):
        if self.key is None:
            base_name = self.name
        else:
            base_name = self.key
        return os.path.join(self.pipeline.persist_dir, base_name+".parquet")

    def set_value(self, value):
        """Set result value manually."""
        if not isinstance(value, pd.DataFrame):
            if not self.validate(value):
                raise ValueError("Value is incompatible with node type "
                                 f"{type(self)}")
        self._result = value


class DaskSeriesGroupByPlace(DynamicPlace, DaskSeriesGroupByMixin):
    """Dask series groupby input."""
    data_class = pd.api.typing.SeriesGroupBy


class DaskDataFrameGroupByPlace(DynamicPlace, DaskSeriesMixin):
    """Dask delayed input."""
    data_class = pd.api.typing.DataFrameGroupBy


PLACES = OrderedDict()
PLACES['dynamic'] = DynamicPlace
PLACES['bool'] = BoolPlace
PLACES['scalar'] = ScalarPlace
PLACES['numpy_array'] = NumpyArrayPlace
PLACES['dict'] = DictPlace
PLACES['pandas_dataframe'] = PandasDataFramePlace
PLACES['pandas_series'] = PandasSeriesPlace
PLACES['dask_array'] = DaskArrayPlace
PLACES['dask_bag'] = DaskBagPlace
PLACES['dask_dataframe'] = DaskDataFramePlace
PLACES['dask_delayed'] = DaskDelayedPlace
PLACES['dask_series'] = DaskSeriesPlace
PLACES['pickleable'] = PickleablePlace


def _guess_place_class(data):
    """Determine the closest fit amongst input nodes."""
    if data is None:
        return PLACES['pickleable']
    for pname in PLACES:
        if pname not in ['pickleable', 'dynamic']:
            if PLACES[pname]().validate(data):
                return PLACES[pname]
    return PickleablePlace


def place(data=None,
          ptype=None,
          name=None,
          pipeline=None,
          parent=None,
          is_output=False,
          persist=False,
          keep=False):
    """Return input node according to data input class."""
    if ptype is not None:
        if ptype not in PLACES:
            raise KeyError(f"Name {ptype} is not a place name.")
        place_class = PLACES[ptype]
    else:
        place_class = _guess_place_class(data)

    return place_class(name=name,
                       data=data,
                       pipeline=pipeline,
                       parent=parent,
                       is_output=is_output,
                       persist=persist,
                       keep=keep)


NUMPY_ARRAY_METHOD_SPECS = {
    'all': {'out_sig':  (NumpyArrayPlace,)},
    'any': {'out_sig':  (NumpyArrayPlace,)},
    'argmax': {'out_sig':  (NumpyArrayPlace,)},
    'argmin': {'out_sig':  (NumpyArrayPlace,)},
    'argpartition': {'out_sig':  (NumpyArrayPlace,)},
    'argsort': {'out_sig':  (NumpyArrayPlace,)},
    'astype': {'out_sig':  (NumpyArrayPlace,)},
    'byteswap': {'out_sig':  (NumpyArrayPlace,)},
    'choose': {'out_sig':  (NumpyArrayPlace,)},
    'clip': {'out_sig':  (NumpyArrayPlace,)},
    'compress': {'out_sig':  (NumpyArrayPlace,)},
    'conj': {'out_sig':  (NumpyArrayPlace,)},
    'conjugate': {'out_sig':  (NumpyArrayPlace,)},
    'cumprod': {'out_sig':  (NumpyArrayPlace,)},
    'cumsum': {'out_sig':  (NumpyArrayPlace,)},
    'diagonal': {'out_sig':  (NumpyArrayPlace,)},
    'dot': {'out_sig':  (NumpyArrayPlace,)},
    'dump': {'out_sig':  (NumpyArrayPlace,)},
    'dumps': {'out_sig':  (NumpyArrayPlace,)},
    'fill': {'out_sig':  (NumpyArrayPlace,)},
    'flatten': {'out_sig':  (NumpyArrayPlace,)},
    'getfield': {'out_sig':  (NumpyArrayPlace,)},
    'item': {'out_sig':  (NumpyArrayPlace,)},
    'itemset': {'out_sig':  (NumpyArrayPlace,)},
    'max': {'out_sig':  (NumpyArrayPlace,)},
    'mean': {'out_sig':  (NumpyArrayPlace,)},
    'min': {'out_sig':  (NumpyArrayPlace,)},
    'newbyteorder': {'out_sig':  (NumpyArrayPlace,)},
    'nonzero': {'out_sig':  (NumpyArrayPlace,)},
    'partition': {'out_sig':  (NumpyArrayPlace,)},
    'prod': {'out_sig':  (NumpyArrayPlace,)},
    'ptp': {'out_sig':  (NumpyArrayPlace,)},
    'put': {'out_sig':  (NumpyArrayPlace,)},
    'ravel': {'out_sig':  (NumpyArrayPlace,)},
    'repeat': {'out_sig':  (NumpyArrayPlace,)},
    'reshape': {'out_sig':  (NumpyArrayPlace,)},
    'resize': {'out_sig':  (NumpyArrayPlace,)},
    'round': {'out_sig':  (NumpyArrayPlace,)},
    'searchsorted': {'out_sig':  (NumpyArrayPlace,)},
    'setfield': {'out_sig':  (NumpyArrayPlace,)},
    'setflags': {'out_sig':  (NumpyArrayPlace,)},
    'sort': {'out_sig':  (NumpyArrayPlace,)},
    'squeeze': {'out_sig':  (NumpyArrayPlace,)},
    'std': {'out_sig':  (NumpyArrayPlace,)},
    'sum': {'out_sig':  (NumpyArrayPlace,)},
    'swapaxes': {'out_sig':  (NumpyArrayPlace,)},
    'take': {'out_sig':  (NumpyArrayPlace,)},
    'tobytes': {'out_sig':  (NumpyArrayPlace,)},
    'tofile': {'out_sig':  (NumpyArrayPlace,)},
    'tolist': {'out_sig':  (NumpyArrayPlace,)},
    'tostring': {'out_sig':  (NumpyArrayPlace,)},
    'trace': {'out_sig':  (NumpyArrayPlace,)},
    'transpose': {'out_sig':  (NumpyArrayPlace,)},
    'var': {'out_sig':  (NumpyArrayPlace,)},
    'view': {'out_sig':  (NumpyArrayPlace,)},
    '__abs__': {'out_sig':  (NumpyArrayPlace,)},
    '__add__': {'out_sig':  (NumpyArrayPlace,)},
    '__and__': {'out_sig':  (NumpyArrayPlace,)},
    '__bool__': {'out_sig':  (NumpyArrayPlace,)},
    '__contains__': {'out_sig':  (NumpyArrayPlace,)},
    '__delitem__': {'out_sig':  (NumpyArrayPlace,)},
    '__divmod__': {'out_sig':  (NumpyArrayPlace,)},
    # '__eq__': {'out_sig':  (NumpyArrayPlace,)},
    '__float__': {'out_sig':  (NumpyArrayPlace,)},
    '__floordiv__': {'out_sig':  (NumpyArrayPlace,)},
    # '__ge__': {'out_sig':  (NumpyArrayPlace,)},
    '__getitem__': {'out_sig':  (NumpyArrayPlace,)},
    # '__gt__': {'out_sig':  (NumpyArrayPlace,)},
    '__iadd__': {'out_sig':  (NumpyArrayPlace,)},
    '__iand__': {'out_sig':  (NumpyArrayPlace,)},
    '__ifloordiv__': {'out_sig':  (NumpyArrayPlace,)},
    # '__ilshift__': {'out_sig':  (NumpyArrayPlace,)},
    '__imatmul__': {'out_sig':  (NumpyArrayPlace,)},
    '__imod__': {'out_sig':  (NumpyArrayPlace,)},
    '__imul__': {'out_sig':  (NumpyArrayPlace,)},
    '__index__': {'out_sig':  (NumpyArrayPlace,)},
    '__int__': {'out_sig':  (NumpyArrayPlace,)},
    '__invert__': {'out_sig':  (NumpyArrayPlace,)},
    '__ior__': {'out_sig':  (NumpyArrayPlace,)},
    '__ipow__': {'out_sig':  (NumpyArrayPlace,)},
    # '__irshift__': {'out_sig':  (NumpyArrayPlace,)},
    '__isub__': {'out_sig':  (NumpyArrayPlace,)},
    '__iter__': {'out_sig':  (NumpyArrayPlace,)},
    '__itruediv__': {'out_sig':  (NumpyArrayPlace,)},
    '__ixor__': {'out_sig':  (NumpyArrayPlace,)},
    # '__le__': {'out_sig':  (NumpyArrayPlace,)},
    '__len__': {'out_sig':  (NumpyArrayPlace,)},
    # '__lshift__': {'out_sig':  (NumpyArrayPlace,)},
    # '__lt__': {'out_sig':  (NumpyArrayPlace,)},
    '__matmul__': {'out_sig':  (NumpyArrayPlace,)},
    '__mod__': {'out_sig':  (NumpyArrayPlace,)},
    '__mul__': {'out_sig':  (NumpyArrayPlace,)},
    # '__ne__': {'out_sig':  (NumpyArrayPlace,)},
    '__neg__': {'out_sig':  (NumpyArrayPlace,)},
    '__or__': {'out_sig':  (NumpyArrayPlace,)},
    '__pos__': {'out_sig':  (NumpyArrayPlace,)},
    '__pow__': {'out_sig':  (NumpyArrayPlace,)},
    '__radd__': {'out_sig':  (NumpyArrayPlace,)},
    '__rand__': {'out_sig':  (NumpyArrayPlace,)},
    '__rdivmod__': {'out_sig':  (NumpyArrayPlace,)},
    # '__repr__': {'out_sig':  (NumpyArrayPlace,)},
    '__rfloordiv__': {'out_sig':  (NumpyArrayPlace,)},
    # '__rlshift__': {'out_sig':  (NumpyArrayPlace,)},
    '__rmatmul__': {'out_sig':  (NumpyArrayPlace,)},
    '__rmod__': {'out_sig':  (NumpyArrayPlace,)},
    '__rmul__': {'out_sig':  (NumpyArrayPlace,)},
    '__ror__': {'out_sig':  (NumpyArrayPlace,)},
    '__rpow__': {'out_sig':  (NumpyArrayPlace,)},
    # '__rrshift__': {'out_sig':  (NumpyArrayPlace,)},
    # '__rshift__': {'out_sig':  (NumpyArrayPlace,)},
    '__rsub__': {'out_sig':  (NumpyArrayPlace,)},
    '__rtruediv__': {'out_sig':  (NumpyArrayPlace,)},
    '__rxor__': {'out_sig':  (NumpyArrayPlace,)},
    '__setitem__': {'out_sig':  (NumpyArrayPlace,)},
    # '__str__': {'out_sig':  (NumpyArrayPlace,)},
    '__sub__': {'out_sig':  (NumpyArrayPlace,)},
    '__truediv__': {'out_sig':  (NumpyArrayPlace,)},
    '__xor__': {'out_sig':  (NumpyArrayPlace,)}
}

DASK_BAG_METHOD_SPECS = {
    # 'all': {'out_sig':  (BoolPlace, )},
    # 'any': {'out_sig':  (BoolPlace, )},
    # 'compute': {'out_sig':  (DaskBagPlace, )},
    'count': {'out_sig':  (ScalarPlace, )},
    'distinct': {'out_sig':  (DaskBagPlace, )},
    'filter': {'out_sig':  (DaskBagPlace, )},
    'flatten': {'out_sig':  (DaskBagPlace, )},
    'fold': {'out_sig':  (DaskBagPlace, )},
    'foldby': {'out_sig':  (DaskBagPlace, )},
    'frequencies': {'out_sig':  (DaskBagPlace, )},
    # 'groupby': {'out_sig':  (DaskBagGroupByPlace, )},
    'join': {'out_sig':  (DaskBagPlace, )},
    'map': {'out_sig':  (DaskBagPlace, )},
    'map_partitions': {'out_sig':  (DaskBagPlace, )},
    'max': {'out_sig':  (DaskBagPlace, )},
    'mean': {'out_sig':  (DaskBagPlace, )},
    'min': {'out_sig':  (DaskBagPlace, )},
    'pluck': {'out_sig':  (DaskBagPlace, )},
    'product': {'out_sig':  (DaskBagPlace, )},
    'reduction': {'out_sig':  (DaskBagPlace, )},
    'random_sample': {'out_sig':  (DaskBagPlace, )},
    'remove': {'out_sig':  (DaskBagPlace, )},
    'repartition': {'out_sig':  (DaskBagPlace, )},
    'starmap': {'out_sig':  (DaskBagPlace, )},
    'std': {'out_sig':  (DaskBagPlace, )},
    'sum': {'out_sig':  (DaskBagPlace, )},
    'take': {'out_sig':  (DaskBagPlace, )},
    'to_dataframe': {'out_sig':  (DaskDataFramePlace, )},
    'to_delayed': {'out_sig':  (DaskDelayedPlace, )},
    'to_textfiles': {'out_sig':  (PickleablePlace, )},
    'to_avro': {'out_sig':  (PickleablePlace, )},
    'topk': {'out_sig':  (DaskBagPlace, )},
    # 'var': {'out_sig':  (DaskBagPlace, )},
    # 'visualize': {'out_sig':  (DaskBagPlace, )}
}

DASK_SERIES_METHOD_SPECS = {
    'add': {'out_sig': (DaskSeriesPlace,)},
    'align': {'out_sig': (DaskSeriesPlace, PickleablePlace)},
    'all': {'out_sig': (DaskSeriesPlace,)},
    'any': {'out_sig': (DaskSeriesPlace,)},
    'append': {'out_sig': (DaskSeriesPlace,)},
    'apply': {'out_sig': (DaskSeriesPlace,)},
    'astype': {'out_sig': (DaskSeriesPlace,)},
    'autocorr': {'out_sig': (ScalarPlace,)},
    'between': {'out_sig': (DaskSeriesPlace,)},
    'bfill': {'out_sig': (DaskSeriesPlace,)},
    # 'clear_divisions': {'out_sig': (None,)},
    'clip': {'out_sig': (DaskSeriesPlace,)},
    'clip_lower': {'out_sig': (DaskSeriesPlace,)},
    'clip_upper': {'out_sig': (DaskSeriesPlace,)},
    # 'compute',
    'copy': {'out_sig':  ((ScalarPlace, DaskSeriesPlace),)},
    'corr': {'out_sig':  (ScalarPlace,)},
    'count': {'out_sig':  (ScalarPlace,)},
    'cov': {'out_sig':  (ScalarPlace,)},
    'cummax': {'out_sig':  (DaskSeriesPlace,)},
    'cummin': {'out_sig':  (DaskSeriesPlace,)},
    'cumprod': {'out_sig':  (DaskSeriesPlace,)},
    'cumsum': {'out_sig':  (DaskSeriesPlace,)},
    'describe': {'out_sig':  (DaskSeriesPlace,)},
    'diff': {'out_sig':  (DaskSeriesPlace,)},
    'div': {'out_sig':  (DaskSeriesPlace,)},
    'drop_duplicates': {'out_sig':  (DaskSeriesPlace,)},
    'dropna': {'out_sig':  (DaskSeriesPlace,)},
    'eq': {'out_sig':  (DaskSeriesPlace,)},
    'explode': {'out_sig':  (DaskSeriesPlace,)},
    'ffill': {'out_sig':  (DaskSeriesPlace,)},
    'fillna': {'out_sig':  (DaskSeriesPlace,)},
    'first': {'out_sig':  (DaskSeriesPlace,)},
    'floordiv': {'out_sig':  (DaskSeriesPlace,)},
    'ge': {'out_sig':  (DaskSeriesPlace,)},
    'get_partition': {'out_sig':  (DaskSeriesPlace,)},
    'groupby': {'out_sig':  (DaskSeriesGroupByPlace,)},
    'gt': {'out_sig':  (DaskSeriesPlace,)},
    'head': {'out_sig':  (DaskSeriesPlace,)},
    'idxmax': {'out_sig':  (DaskSeriesPlace,)},
    'idxmin': {'out_sig':  (DaskSeriesPlace,)},
    'isin': {'out_sig':  (DaskSeriesPlace,)},
    'isna': {'out_sig':  (DaskSeriesPlace,)},
    'isnull': {'out_sig':  (DaskSeriesPlace,)},
    # 'iteritems': {'out_sig':  (DaskSeriesPlace,)},
    'last': {'out_sig':  (DaskSeriesPlace,)},
    'le': {'out_sig':  (DaskSeriesPlace,)},
    'lt': {'out_sig':  (DaskSeriesPlace,)},
    'map': {'out_sig':  (DaskSeriesPlace,)},
    'map_overlap': {'out_sig':  (DaskSeriesPlace,)},
    'map_partitions': {'out_sig':  (DaskSeriesPlace,)},
    'mask': {'out_sig':  (DaskSeriesPlace,)},
    'max': {'out_sig':  (DaskSeriesPlace,)},
    'mean': {'out_sig':  (DaskSeriesPlace,)},
    'memory_usage': {'out_sig':  (ScalarPlace,)},
    'memory_usage_per_partition': {'out_sig':  (DaskSeriesPlace,)},
    'min': {'out_sig':  (DaskSeriesPlace,)},
    'mod': {'out_sig':  (DaskSeriesPlace,)},
    'mul': {'out_sig':  (DaskSeriesPlace,)},
    'ne': {'out_sig':  (DaskSeriesPlace,)},
    'nlargest': {'out_sig':  (DaskSeriesPlace,)},
    'notnull': {'out_sig':  (DaskSeriesPlace,)},
    'nsmallest': {'out_sig':  (DaskSeriesPlace,)},
    'nunique': {'out_sig':  (ScalarPlace,)},
    'nunique_approx': {'out_sig':  (ScalarPlace,)},
    # 'persist': {'out_sig':  (DaskSeriesPlace,)},
    # 'pipe': {'out_sig':  (PickleablePlace,)},
    'pow': {'out_sig':  (DaskSeriesPlace,)},
    'prod': {'out_sig':  (DaskSeriesPlace,)},
    'quantile': {'out_sig':  (DaskSeriesPlace,)},
    'radd': {'out_sig':  (DaskSeriesPlace,)},
    'random_split': {'out_sig':  (DaskSeriesPlace,)},
    'rdiv': {'out_sig':  (DaskSeriesPlace,)},
    'reduction': {'out_sig':  (DaskSeriesPlace,)},
    'repartition': {'out_sig':  (DaskSeriesPlace,)},
    'replace': {'out_sig':  (DaskSeriesPlace,)},
    'rename': {'out_sig':  (DaskSeriesPlace,)},
    'resample': {'out_sig':  (DaskSeriesPlace,)},
    'reset_index': {'out_sig':  (DaskSeriesPlace,)},
    # 'rolling': {'out_sig':  (DaskSeriesPlace,)},
    'round': {'out_sig':  (DaskSeriesPlace,)},
    'sample': {'out_sig':  (DaskSeriesPlace,)},
    'sem': {'out_sig':  (DaskSeriesPlace,)},
    'shift': {'out_sig':  (DaskSeriesPlace,)},
    'std': {'out_sig':  (DaskSeriesPlace,)},
    'sub': {'out_sig':  (DaskSeriesPlace,)},
    'sum': {'out_sig':  (DaskSeriesPlace,)},
    'to_bag': {'out_sig':  (DaskBagPlace,)},
    'to_csv': {'out_sig':  (PickleablePlace,)},
    'to_dask_array': {'out_sig':  (DaskArrayPlace,)},
    'to_delayed': {'out_sig':  (DaskDelayedPlace,)},
    'to_frame': {'out_sig':  (DaskDataFramePlace,)},
    'to_hdf': {'out_sig':  (PickleablePlace,)},
    'to_string': {'out_sig':  (ScalarPlace,)},
    'to_timestamp': {'out_sig':  (DaskDataFramePlace,)},
    'truediv': {'out_sig':  (DaskSeriesPlace,)},
    'unique': {'out_sig':  (DaskSeriesPlace,)},
    'value_counts': {'out_sig':  (DaskSeriesPlace,)},
    'var': {'out_sig':  (DaskSeriesPlace,)},
    # 'visualize': {'out_sig':  (DaskSeriesPlace,)},
    'where': {'out_sig':  (DaskSeriesPlace,)}
}

DASK_DATAFRAME_METHOD_SPECS = {
    'add': {'out_sig': (DaskDataFramePlace,)},
    'append': {'out_sig': (DaskDataFramePlace,)},
    'apply': {'out_sig': (DaskDataFramePlace,)},
    'assign': {'out_sig': (DaskDataFramePlace,)},
    'astype': {'out_sig': (DaskDataFramePlace,)},
    'categorize': {'out_sig': (DaskDataFramePlace,)},
    # 'compute': {'out_sig': (DaskDataFramePlace,)},
    'corr': {'out_sig': (DaskDataFramePlace,)},
    'count': {'out_sig': ((DaskSeriesPlace, DaskDataFramePlace),)},
    'cov': {'out_sig': (DaskDataFramePlace,)},
    'cummax': {'out_sig': (DaskDataFramePlace,)},
    'cummin': {'out_sig': (DaskDataFramePlace,)},
    'cumprod': {'out_sig': (DaskDataFramePlace,)},
    'cumsum': {'out_sig': (DaskDataFramePlace,)},
    'describe': {'out_sig': (DaskDataFramePlace,)},
    'div': {'out_sig': (DaskDataFramePlace,)},
    'drop': {'out_sig': (DaskDataFramePlace,)},
    'drop_duplicates': {'out_sig': (DaskDataFramePlace,)},
    'dropna': {'out_sig': (DaskDataFramePlace,)},
    'explode': {'out_sig': (DaskDataFramePlace,)},
    'fillna': {'out_sig': (DaskDataFramePlace,)},
    'floordiv': {'out_sig': (DaskDataFramePlace,)},
    'get_partition': {'out_sig': (DaskDataFramePlace,)},
    'groupby': {'out_sig': (DaskDataFrameGroupByPlace,)},
    'head': {'out_sig': (DaskDataFramePlace,)},
    'isna': {'out_sig': (DaskDataFramePlace,)},
    'isnull': {'out_sig': (DaskDataFramePlace,)},
    # 'iterrows': {'out_sig': (DaskDataFramePlace,)},
    # 'itertuples': {'out_sig': (DaskDataFramePlace,)},
    'join': {'out_sig': (DaskDataFramePlace,)},
    'map_partitions': {'out_sig': (DaskDataFramePlace,)},
    'mask': {'out_sig': (DaskDataFramePlace,)},
    'max': {'out_sig': ((DaskSeriesPlace, DaskDataFramePlace),)},
    'mean': {'out_sig': ((DaskSeriesPlace, DaskDataFramePlace),)},
    'memory_usage': {'out_sig': (DaskSeriesPlace,)},
    'memory_usage_per_partition': {'out_sig': (DaskSeriesPlace,)},
    'merge': {'out_sig': (DaskDataFramePlace,)},
    'min': {'out_sig': ((DaskSeriesPlace, DaskDataFramePlace),)},
    'mod': {'out_sig': (DaskDataFramePlace,)},
    'mul': {'out_sig': (DaskDataFramePlace,)},
    'nlargest': {'out_sig': (DaskDataFramePlace,)},
    'pop': {'out_sig': (DaskSeriesPlace,)},
    'pow': {'out_sig': (DaskDataFramePlace,)},
    'prod': {'out_sig': ((DaskSeriesPlace, DaskDataFramePlace),)},
    'quantile': {'out_sig': (DaskDataFramePlace,)},
    'query': {'out_sig': (DaskDataFramePlace,)},
    'radd': {'out_sig': (DaskDataFramePlace,)},
    'random_split': {'out_sig': (DaskDataFramePlace,)},
    'rdiv': {'out_sig': (DaskDataFramePlace,)},
    'rename': {'out_sig': (DaskDataFramePlace,)},
    'repartition': {'out_sig': (DaskDataFramePlace,)},
    'replace': {'out_sig': (DaskDataFramePlace,)},
    'reset_index': {'out_sig': (DaskDataFramePlace,)},
    'rfloordiv': {'out_sig': (DaskDataFramePlace,)},
    'rmod': {'out_sig': (DaskDataFramePlace,)},
    'rmul': {'out_sig': (DaskDataFramePlace,)},
    'rpow': {'out_sig': (DaskDataFramePlace,)},
    'rsub': {'out_sig': (DaskDataFramePlace,)},
    'rtruediv': {'out_sig': (DaskDataFramePlace,)},
    'sample': {'out_sig': (DaskDataFramePlace,)},
    'set_index': {'out_sig': (DaskDataFramePlace,)},
    'std': {'out_sig': (DaskDataFramePlace,)},
    'sub': {'out_sig': (DaskDataFramePlace,)},
    'sum': {'out_sig': ((DaskSeriesPlace, DaskDataFramePlace),)},
    'tail': {'out_sig': (DaskDataFramePlace,)},
    'to_bag': {'out_sig': (DaskBagPlace,)},
    'to_csv': {'out_sig': (PickleablePlace,)},
    'to_dask_array': {'out_sig': (DaskArrayPlace,)},
    'to_delayed': {'out_sig': (DaskDelayedPlace,)},
    'to_hdf': {'out_sig': (PickleablePlace,)},
    'to_json': {'out_sig': (PickleablePlace,)},
    'to_parquet': {'out_sig': (PickleablePlace,)},
    'to_records': {'out_sig': (DaskArrayPlace,)},
    'truediv': {'out_sig': (DaskDataFramePlace,)},
    'var': {'out_sig': ((DaskSeriesPlace, DaskDataFramePlace),)},
    # 'visualize': {'out_sig': (DaskDataFramePlace,)},
    'where': {'out_sig': (DaskDataFramePlace,)}
}

DASK_DATAFRAME_GROUPBY_METHOD_SPECS = {
    'aggregate': {'out_sig':  (DaskDataFramePlace,)},
    'apply': {'out_sig':  (DaskDataFramePlace,)},
    'count': {'out_sig':  (DaskDataFramePlace,)},
    'cumcount': {'out_sig':  (DaskSeriesPlace,)},
    'cumprod': {'out_sig':  (DaskDataFramePlace,)},
    'cumsum': {'out_sig':  (DaskDataFramePlace,)},
    'get_group': {'out_sig':  (DaskDataFrameGroupByPlace,)},
    'max': {'out_sig':  (DaskDataFramePlace,)},
    'mean': {'out_sig':  (DaskDataFramePlace,)},
    'min': {'out_sig':  (DaskDataFramePlace,)},
    'size': {'out_sig':  (DaskDataFramePlace,)},
    'std': {'out_sig':  (DaskDataFramePlace,)},
    'sum': {'out_sig':  (DaskDataFramePlace,)},
    'var': {'out_sig':  (DaskDataFramePlace,)},
    'cov': {'out_sig':  (DaskDataFramePlace,)},
    'corr': {'out_sig':  (DaskDataFramePlace,)},
    'first': {'out_sig':  (DaskDataFramePlace,)},
    'last': {'out_sig':  (DaskDataFramePlace,)},
    'idxmin': {'out_sig':  (DaskSeriesPlace,)},
    'idxmax': {'out_sig':  (DaskSeriesPlace,)}
}

DASK_SERIES_GROUPBY_METHOD_SPECS = {
    'aggregate': {'out_sig':  (DaskSeriesPlace,)},
    'apply': {'out_sig':  (DaskSeriesPlace,)},
    'count': {'out_sig':  (DaskSeriesPlace,)},
    'cumcount': {'out_sig':  (DaskSeriesPlace,)},
    'cumprod': {'out_sig':  (DaskSeriesPlace,)},
    'cumsum': {'out_sig':  (DaskSeriesPlace,)},
    'get_group': {'out_sig':  (DaskSeriesGroupByPlace,)},
    'max': {'out_sig':  (DaskSeriesPlace,)},
    'mean': {'out_sig':  (DaskSeriesPlace,)},
    'min': {'out_sig':  (DaskSeriesPlace,)},
    'nunique': {'out_sig':  (DaskSeriesPlace,)},
    'size': {'out_sig':  (DaskSeriesPlace,)},
    'std': {'out_sig':  (DaskSeriesPlace,)},
    'sum': {'out_sig':  (DaskSeriesPlace,)},
    'var': {'out_sig':  (DaskSeriesPlace,)},
    'first': {'out_sig':  (DaskSeriesPlace,)},
    'last': {'out_sig':  (DaskSeriesPlace,)},
    'idxmin': {'out_sig':  (DaskSeriesPlace,)},
    'idxmax': {'out_sig':  (DaskSeriesPlace,)}
}


def _build_method(method_name, op_class, out_sig):
    def class_method(self, *args, **kwargs):
        return _build_method_op(self,
                                method_name,
                                args,
                                kwargs,
                                op_class,
                                out_sig)
    return class_method


def _build_method_op(self, method_name, args, kwargs, op_class, out_sig):
    op_args = []
    arg_count = 0
    for item in args:
        if isinstance(item, Node):
            op_args.append(item)
        else:
            input_class = _guess_place_class(item)
            op_args.append(input_class(name=f"{method_name}" +
                                            f"_input_{arg_count}",
                                       pipeline=self.pipeline,
                                       data=item))
        arg_count += 1
    n_args = len(op_args)

    op_kwargs = []
    kwarg_keys = [key for key in kwargs]
    non_op_kwargs = {}
    for key in kwarg_keys:
        if isinstance(kwargs[key], Node):
            op_kwargs.append(kwargs[key])
        else:
            non_op_kwargs[key] = kwargs[key]

    def wrapper(*node_args):
        data = node_args[0]
        args_ = tuple(node_args[1:n_args+1])
        kwargs_ = node_args[n_args+1:]
        wrapper_kwargs = {}
        if len(kwargs_) > 0:
            for i in range(len(kwargs_)):
                wrapper_kwargs[kwarg_keys[i]] = kwargs_[i]
        wrapper_kwargs.update(non_op_kwargs)

        return getattr(data, method_name)(*args_, **wrapper_kwargs)

    transition_name = method_name

    inputs = [self] + op_args + op_kwargs

    outputs = []
    for out_spec in out_sig:
        if isinstance(out_spec, tuple):
            out_class = out_spec[0]
        else:
            out_class = out_spec
        curr_len = len(outputs)
        outputs.append(out_class(name=(f"{transition_name}" +
                                       f"_output_{curr_len}"),
                                 pipeline=self.pipeline,
                                 is_output=False,
                                 persist=False,
                                 keep=False))

    inp_sig = []

    for inp in inputs:
        inp_sig.append(inp.__class__)

    signature = (tuple(inp_sig), out_sig)

    transition = op_class(name=transition_name,
                          operation=wrapper,
                          pipeline=self.pipeline,
                          inputs=inputs,
                          outputs=outputs,
                          signature=signature)

    if len(transition.outputs) == 1:
        return transition.outputs[0]
    return transition.outputs


for meth in NUMPY_ARRAY_METHOD_SPECS:
    setattr(NumpyArrayMixin,
            meth,
            _build_method(meth,
                          Transition,
                          NUMPY_ARRAY_METHOD_SPECS[meth]['out_sig']))


for meth in DASK_BAG_METHOD_SPECS:
    setattr(DaskBagMixin,
            meth,
            _build_method(meth,
                          Transition,
                          DASK_BAG_METHOD_SPECS[meth]['out_sig']))


for meth in DASK_SERIES_METHOD_SPECS:
    setattr(DaskSeriesMixin,
            meth,
            _build_method(meth,
                          Transition,
                          DASK_SERIES_METHOD_SPECS[meth]['out_sig']))


for meth in DASK_DATAFRAME_METHOD_SPECS:
    setattr(DaskDataFrameMixin,
            meth,
            _build_method(meth,
                          Transition,
                          DASK_DATAFRAME_METHOD_SPECS[meth]['out_sig']))


for meth in DASK_DATAFRAME_GROUPBY_METHOD_SPECS:
    (setattr(DaskDataFrameGroupByMixin,
     meth,
     _build_method(meth,
                   Transition,
                   DASK_DATAFRAME_GROUPBY_METHOD_SPECS[meth]['out_sig'])))


for meth in DASK_SERIES_GROUPBY_METHOD_SPECS:
    setattr(DaskSeriesGroupByMixin,
            meth,
            _build_method(meth,
                          Transition,
                          DASK_SERIES_GROUPBY_METHOD_SPECS[meth]['out_sig']))
