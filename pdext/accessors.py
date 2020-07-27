
import os
import pickle
import logging
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from pdext.utils import _collection, _is_collection

pd._config.config.register_option('io.label_encoder_path', None)



@pd.api.extensions.register_series_accessor('list')
class ListAccessor:
    def __init__(self, series):
        self._validate(series)
        self._series = series
        
    def __getitem__(self, index):
        return self._series.apply(lambda x: x[index])
    
    def __setitem__(self, index, value):
        return self._series.apply(self._setter, index=index, value=value)
    
    @staticmethod
    def _setter(lst, index, value):
        if callable(value):
            lst[index] = value(lst[index])
        else:
            lst[index] = value
        return lst
    
    @staticmethod
    def _validate(series):
        if not series.apply(lambda x: hasattr(x, '__getitem__')).all():
            raise AttributeError('Can only use `list` accessor with values that implement `__getitem__`.')
            
    def len(self):
        return self._series.apply(lambda x: len(x))
    
    def split(self, columns=None):
        frame = self._series.apply(pd.Series)
        if columns is not None:
            return frame.set_axis(columns, 1)
        return frame
    
    def join(self, sep=''):
        return self._series.apply(lambda x: sep.join(x))    


@pd.api.extensions.register_series_accessor('dict')
class DictAccessor:
    def __init__(self, series):
        self._validate(series)
        self._series = series
                
    def __getitem__(self, key):
        return self._series.apply(lambda x: x[key])
    
    def __setitem__(self, key, value):
        return self._series.apply(self._setter, key=key, value=value)
        
    def get(self, key):
        return self._series.apply(lambda x: x.get(key, np.NaN))
    
    def iget(self, index):
        return self._series.apply(self._igetter, index=index)

    # def keys(self):
    #     return self._series.apply(lambda x: x.keys())

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self._series.apply(lambda x: getattr(x, attr)(*args, **kwargs))
    
    @staticmethod
    def _validate(series):
        if not series.apply(lambda x: hasattr(x, '__getitem__')).all():
            raise AttributeError('Can only use `dict` accessor with values that implement `__getitem__`.')
    
    def _setter(self, dict_, key, value):
        if callable(value):
            dict_[key] = value(dict_[key])
        else:
            dict_[key] = value
        return dict_
    
    @staticmethod
    def _igetter(dict_, index):
        keys = list(dict_.keys())
        try:
            return dict_[keys[index]]
        except IndexError:
            return np.NaN


@pd.api.extensions.register_dataframe_accessor('label_encoder')
class LabelEncoderAccessor:
    def __init__(self, frame):
        self._frame = frame
        
    def _store(self, encoder, col, path):
        filepath = self._get_filepath(path, col)
        if not os.path.exists(filepath):
            with open(filepath, 'wb') as file:
                pickle.dump(encoder, file)
                
    def _load(self, col, path):
        filepath = self._get_filepath(path, col)
        with open(filepath, 'rb') as file:
            return pickle.load(file)
        
    @staticmethod
    def _get_filepath(path, col):
        filename = 'label_encoder-' + col + '.pickle' 
        return os.path.join(path, filename)

    def _get_encoder(self, path, col):
        try:
            return self._load(col, path)
        except Exception:        
            return LabelEncoder()

    @staticmethod
    def _validate_path(path, notna=False):
        default_path = pd.options.io.label_encoder_path
        if path is None:
            if default_path is not None:
                return default_path
            raise ValueError('Must specify `path` argument or set a default option using `pd.options.io.label_encoder_path`.')
        return path
        
    def encode(self, columns, path=None):
        path = self._validate_path(path)
        for col in _collection(columns):
            encoder = self._get_encoder(path, col)
            if hasattr(encoder, 'classes_'):
                self._frame[col] = encoder.transform(self._frame[col])
            else:
                self._frame[col] = encoder.fit_transform(self._frame[col])
                if path is not None:
                    self._store(encoder, col, path)
        return self._frame
    
    def decode(self, columns, path=None):
        path = self._validate_path(path, notna=True)
        for col in _collection(columns):
            encoder = self._load(col, path)
            self._frame[col] = encoder.inverse_transform(self._frame[col])
        return self._frame


@pd.api.extensions.register_dataframe_accessor('plots')
class PlotAccessorExt(pd.plotting._core.PlotAccessor):
    
    @functools.wraps(pd.DataFrame.plot)
    def __call__(self, *args, **kwargs):
        cols = self._parent.select_dtypes(np.number).columns
        kwargs.setdefault('label', cols)
        kwargs_it = self._iter_kwargs(kwargs)

        for col in cols:
            plt.figure(figsize=(16, 4))
            self._parent[col].plot(*args, **next(kwargs_it))
            plt.legend()
    
    def _iter_kwargs(self, kwargs):
        pos = 0
        while True:
            yield {key: self._get_single_kwarg(val, pos) for key, val in kwargs.items()}
            pos += 1
            
    @staticmethod
    def _get_single_kwarg(val, pos):
        if not _is_collection(val):
            return val
        try:
            return val[pos]
        except IndexError:
            return None


@pd.api.extensions.register_dataframe_accessor('applyext')
class ApplyExtension:
    def __init__(self, frame):
        self._frame = frame
    
    def __call__(self, func, subset=None):
        self._validate(func, subset)
        subset = _collection(subset)
        for col, func_ in self._iterargs(func, subset):
            self._frame[col] = func_(self._frame[col])
        return self._frame

    def _validate(self, func, subset):
        if subset is None and not isinstance(func, dict):
            raise ValueError('Must specify target subset as `func` keys or `subset`.')

    def _iterargs(self, func, subset):
        if not isinstance(func, dict):
            for col in subset:
                yield col, func
        else:
            for cols, func_ in func.items():
                for col in _collection(cols):
                    yield col, func_
    

@pd.api.extensions.register_dataframe_accessor('astypeext')
class AstypeExtension:
    def __init__(self, frame):
        self._frame = frame
        
    def __call__(self, dtype, subset=None, **kwargs):
        dtypes, subset = self._validate_args(dtype, subset)
        self._convert(dtypes, subset, **kwargs) 
        return self._frame
    
    def _convert(self, dtypes, subset, **kwargs):
        for dtype, col in zip(dtypes, subset):
            if col in self._frame:
                try:
                    self._frame[col] = self._frame[col].astype(dtype, **kwargs)
                except ValueError as e:
                    if isinstance(dtype, str) and dtype.startswith('int'):
                        base = dtype[3:] 
                        nullable_dtype = getattr(pd, f'Int{base}Dtype')
                        self._frame[col] = self._frame[col].astype(nullable_dtype(), **kwargs)
                    elif str(dtype) == "<class 'int'>":
                        self._frame[col] = self._frame[col].astype(pd.Int64Dtype(), **kwargs)
                    else:
                        raise e
            
    @staticmethod
    def _unpack_dict(dtype_map):
        dtype = []
        subset = []
        for subset_, dtype_ in dtype_map.items():
            if isinstance(subset_, tuple):
                for col in subset_:
                    subset.append(col)
                    dtype.append(dtype_)
            else:
                subset.append(subset_)
                dtype.append(dtype_)
        return dtype, subset

    def _validate_args(self, dtype, subset):
        if subset is None:
            if isinstance(dtype, dict):
                return self._unpack_dict(dtype)
            else:
                subset = self._frame.columns
                return [dtype]*len(subset), subset
        elif not _is_collection(subset):
            return [dtype], [subset]
        else:
            if _is_collection(dtype):
                if len(dtype) != len(subset):
                    raise ValueError("`subset` and `dtype` lengths must match")
                return dtype, subset
            else:
                return [dtype]*len(subset), subset


@pd.api.extensions.register_dataframe_accessor('eqext')
@pd.api.extensions.register_series_accessor('eqext')
class EqualExtension:
    "Considers all N/A values equal."
    
    def __init__(self, frame):
        self.frame = frame
        
    def __call__(self, other):
        return self.frame.eq(other) | (self.frame.isna() & other.isna())


@pd.api.extensions.register_dataframe_accessor('neext')
@pd.api.extensions.register_series_accessor('neext')
class NotEqualExtension:
    "Considers all N/A values equal."
    
    def __init__(self, frame):
        self.frame = frame
        
    def __call__(self, other):
        return (~self.frame.eq(other)) & (self.frame.notna() | other.notna())


@pd.api.extensions.register_series_accessor('range')
@pd.api.extensions.register_dataframe_accessor('range')
class RangeAccessor:
    def __new__(cls, series):
        return series.agg([min, max])


@pd.api.extensions.register_index_accessor('is_consecutive')
class ConsecutiveDateAccessor:
    def __init__(self, index):
        self._index = index

    def __call__(self):
        if isinstance(self._index, pd.DatetimeIndex):
            return self._index.notna().all() and self._index.to_series().diff().nunique() <= 1
        raise NotImplementedError(f'`is_consecutive` is not implemented for an instance of `{type(self._index).__name__}`')    


@pd.api.extensions.register_dataframe_accessor('idx')
@pd.api.extensions.register_series_accessor('idx')
class IdxAccessor:
    """
    Allows to apply pd.DataFrame (with multiindex unless single level specified) 
    or pd.Series (else) methods on an object's index.

    Examples
    --------

    index = pd.date_range('2020', periods=10)
    s = pd.Series(range(10), index=index)
    (
        s.idx.to_frame('date')
        .idx.assign(
            day_name=lambda x: x['date'].dt.day_name(), 
            month=lambda x: x['date'].dt.month)
    )

    --------
    index = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=['x', 'y', 'z'])
    data = {'var_1': [15, 20, 25], 'var_2': [.1, .2, .3]}
    df = pd.DataFrame(data, index=index)

    df.idx.mul([2, 5, 1]).idx.prod(1)
    """      
    def __init__(self, frame):
        self._frame = frame
        self._level = None
        self._accessor = None
        self._copy = True
        
    def __call__(self, level=None, copy=True):
        self._level = self._validate_level(level)
        self._copy = copy
        return self
        
    def __getitem__(self, level):
        """
        Return a single MultiIndex level.
        """
        if not hasattr(self._frame.index, 'levels'):
            raise TypeError(f'`__getitem__` is not supported for an instance of {type(self._frame.index).__name__}.')
        return self._frame.index.get_level_values(level)
        
    def __getattr__(self, attr):
        if ('ipython' in attr) or ('repr' in attr):
            raise AttributeError
        elif self._is_accessor(attr):
            self._accessor = attr
            return self
        else:
            try:
                caller_obj = self._get_caller_obj()
                target_method = self._get_target_method(caller_obj, attr)
                return self.apply_on_index(target_method)
            finally:
                del self._frame.idx      # cleanup
        
    def _validate_level(self, level):
        if level is not None:
            if not hasattr(self._frame.index, 'levels'):
                raise ValueError(f'Unexpected parameter `level` value for an instance of `{type(self._frame.index).__name__}`.') 

            elif _is_collection(level):
                return [self._frame.index._get_level_number(level_) for level_ in level]
            return self._frame.index._get_level_number(level)            
        return level
        
    def apply_on_index(self, target_method):
        @functools.wraps(target_method)
        def _method_wrapper(*args, **kwargs):
            result = target_method(*args, **kwargs)
            return self._update_index(result)
        return _method_wrapper
    
    def _get_target_method(self, caller_obj, attr):
        if self._accessor is not None:      # target method is an accessor method
            caller_obj = getattr(caller_obj, self._accessor)  
        return getattr(caller_obj, attr)

    def _update_index(self, index_new):
        frame_new = self._frame.copy() if self._copy else self._frame
        if self._level is not None:
            # append remaining levels
            index_old = self._frame.index.to_frame()
            remaining_levels = index_old.drop(index_old.columns[self._level], 1)
            index_new = pd.concat([remaining_levels, index_new], 1)
            if index_old.shape == index_new.shape:
                index_new = index_new.reindex_like(index_old)
            index_new = pd.MultiIndex.from_frame(index_new)

        if isinstance(index_new, pd.DataFrame):
                index_new = pd.MultiIndex.from_frame(index_new)
        frame_new.index = index_new
        return frame_new
            
    def _is_accessor(self, attr):
        type_ = self._get_caller_type()
        return attr in getattr(type_, '_accessors')
        
    def _get_caller_obj(self):
        index = self._frame.index
        if hasattr(index, 'levels'):
            caller_obj = index.to_frame()
            if self._level is not None:
                return caller_obj.iloc[:, self._level]
            return caller_obj
        return pd.Series(index)
    
    def _get_caller_type(self):
        if hasattr(self._frame.index, 'levels'):
            if self._level is None or _is_collection(self._level):
                return pd.DataFrame
        return pd.Series