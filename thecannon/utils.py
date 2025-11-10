#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General utility functions.
"""

__all__ = ["short_hash", "wrapper"]

# import functools
import logging
import os
import pickle
# import signal
import sys
from tempfile import mkstemp
from time import time
import numpy as np
import inspect

# Adjustment to be compatible with python 3.12
try:
    from collections import Iterable
except:
    from collections.abc import Iterable
from hashlib import md5
from multiprocessing.pool import Pool
from multiprocessing import Lock, TimeoutError, Value

logger = logging.getLogger(__name__)


# Initialize global counter for incrementing between threads.
_counter = Value("i", 0)
_counter_lock = Lock()


def _init_pool(args):
    global _counter
    _counter = args


class wrapper(object):
    """
    A generic wrapper with a progressbar, which can be used either in serial or
    in parallel.

    Parameters
    ----------

    f
        The function to apply.

    args: list
        Additional arguments to supply to the function ``f``.

    kwds: dict
        Keyword arguments to supply to the function ``f``.

    N: int
        The number of items that will be iterated over.

    message: str, optional
        An information message to log before showing the progressbar.

    size: int, optional
        The width of the progressbar in characters.

    Returns
    -------
    generator
    """

    def __init__(self, f, args, kwds, N, message=None, size=100):
        self.f = f
        self.args = list(args if args is not None else [])
        self.kwds = kwds if kwds is not None else {}
        self._init_progressbar(N, message)

    def _init_progressbar(self, N, message=None):
        """
        Initialise a progressbar.

        Parameters
        ----------
        N: int
            The number of items that will be iterated over.

        essage: str, optional
            An information message to log before showing the progressbar.
        """

        self.N = int(N)

        try:
            rows, columns = os.popen("stty size", "r").read().split()

        except:
            logger.debug("Couldn't get screen size. Progressbar may look odd.")
            self.W = 100

        else:
            self.W = min(100, int(columns) - (12 + 21 + 2 * len(str(self.N))))

        self.t_init = time()
        self.message = message
        if 0 >= self.N:
            return None

        if message is not None:
            logger.info(message.rstrip())

        sys.stdout.flush()
        with _counter_lock:
            _counter.value = 0

    def _update_progressbar(self):
        """
        Increment the progressbar by one iteration.
        """

        if 0 >= self.N:
            return None

        global _counter, _counter_lock
        with _counter_lock:
            _counter.value += 1

        index = _counter.value

        increment = max(1, int(self.N / float(self.W)))

        eta_minutes = ((time() - self.t_init) / index) * (self.N - index) / 60.0

        if index >= self.N:
            status = "({0:.0f}s)                         ".format(time() - self.t_init)

        elif (
            float(index) / self.N >= 0.05 and eta_minutes > 1
        ):  # MAGIC fraction for when we can predict ETA
            status = "({0}/{1}; ~{2:.0f}m until finished)".format(
                index, self.N, eta_minutes
            )

        else:
            status = "({0}/{1})                          ".format(index, self.N)

        sys.stdout.write(
            ("\r[{done: <" + str(self.W) + "}] {percent:3.0f}% {status}").format(
                done="=" * int(index / increment),
                percent=100.0 * index / self.N,
                status=status,
            )
        )
        sys.stdout.flush()

        if index >= self.N:
            sys.stdout.write("\r\n")
            sys.stdout.flush()

    def __call__(self, x):
        try:
            result = self.f(*(list(x) + self.args), **self.kwds)
        except:
            logger.exception("Exception within wrapped function")
            raise

        self._update_progressbar()
        return result


def short_hash(contents):
    """
    Return a short hash string of some iterable content.

    Parameters
    ----------

    contents: str
        The contents to calculate a hash for.

    Returns
    -------
    str:
        A concatenated string of 10-character length hashes for all items in the
        contents provided.
    """
    if not isinstance(contents, Iterable):
        contents = [contents]
    return "".join(
        [str(md5(str(item).encode("utf-8")).hexdigest())[:10] for item in contents]
    )


def _unpack_value(value):
    """
    Unpack contents if it is pickled to a temporary file.

    Parameters
    ----------
    value:
        A non-string variable or a string referring to a pickled file path.

    Returns
    -------
    The original value, or the unpacked contents if a valid path was given.
    """

    if isinstance(value, (str,)) and os.path.exists(value):
        with open(value, "rb") as fp:
            contents = pickle.load(fp)
        return contents
    return value


def _pack_value(value, protocol=-1):
    """
    Pack contents to a temporary file.

    Parameters
    ----------
    alue:
        The contents to temporarily pickle.

    protocol: int, optional
        The pickling protocol to use.

    Returns
    -------
    str
        A temporary filename where the contents are stored.
    """

    _, temporary_filename = mkstemp()
    with open(temporary_filename, "wb") as fp:
        pickle.dump(value, fp, protocol)
    return temporary_filename


def slog(c, e):
    """The scaled log transform.

    Parameters
    ----------
    c : numeric
        Dynamic range pivot, > 0.
    e : numeric
        Small number safety factor.

    Returns
    -------
    callable
        A function that accepts a single value, and computes the scaled log transform.
    """
    try:
        assert c > 0, "c must be >0"
    except AssertionError as err:
        raise ValueError(err)
    
    return lambda x: np.log((x / c) + e)

def slog_inv(c, e):
    """The inverse scaled log transform.

    Parameters
    ----------
    c : _type_
        Dynamic range pivot, > 0.
    e : _type_
        Small number safety factor

    Returns
    -------
    callable
        A function that accepts a single value, and computes the inverse scaled log transform.
    """
    try:
        assert c > 0, "c must be >0"
    except AssertionError as err:
        raise ValueError(err)
    
    return lambda x: c * (np.exp(x) - e)

def rst(m, c):
    """The rational saturating transform.

    Parameters
    ----------
    m : numeric
        The saturation ceiling, >0.
    c : numeric
        The knee, >0.

    Returns
    -------
    callable
        A function that accepts a single value, and computes the rational saturating transform.
    """
    try:
        assert m > 0, "m must be > 0."
        assert c > 0, "c must be > 0."
    except AssertionError as e:
        raise ValueError(e)
    
    return lambda x: m * (x / (x + c))

def rst_inv(m, c):
    """The inverse rational saturating transform.

    Parameters
    ----------
    m : numeric
        The saturation ceiling, >0.
    c : numeric
        The knee, >0.

    Returns
    -------
    callable
        A function that accepts a single value, and computes the inverse rational saturating transform.
    """
    try:
        assert m > 0, "m must be > 0."
        assert c > 0, "c must be > 0."
    except AssertionError as e:
        raise ValueError(e)
    
    return lambda x: x * c / (m - x)


class TransformFunc(object):
    """A class for describing a label transform within TheCannon.

    Parameters
    ----------
    object : _type_
        _description_
    """

    _forward = None
    _inverse = None
    _min = -np.inf
    _max = np.inf

    def __init__(self, *args,
                 forward=None,
                 inverse=None,
                 min=-np.inf, 
                 max=np.inf, 
                 **kwargs):

        self.min = min
        self.max = max
        self.set_funcs(forward, inverse)

    @property
    def max(self):
        return self._max
    
    @max.setter
    def max(self, m):
        if m is None: self._max = np.inf
        try:
            self._max = float(m)
        except TypeError as e:
            raise e
        
    @property
    def min(self):
        return self._min
    
    @min.setter
    def min(self, m):
        if m is None: self._min = -np.inf
        try:
            self._min = float(m)
        except TypeError as e:
            raise e

    @property
    def forward(self):
        return self._forward
    
    @forward.setter
    def forward(self, fnc):
        raise RuntimeError("You cannot set forward directly - please use set_funcs")
    
    @property
    def inverse(self):
        return self._inverse
    
    @inverse.setter
    def inverse(self, fnc):
        raise RuntimeError("You cannot set inverse directly - please use set_funcs")
    
    def set_funcs(self, forward, inverse):
        """Set the forward and inverse functions.

        Parameters
        ----------
        forward : callable
            The forward transformation function. Should be a callable accepting a 
            single numeric argument.
        inverse : callable
            The inverse transformation function. Should be a callable accepting a 
            single numeric argument.
        """

        if forward is None and inverse is None:
            self._forward = lambda x: x
            self._inverse = lambda x: x
            return
        
        # Input checking
        try:
            assert callable(forward), "Forward is not callable"
            assert callable(inverse), "Inverse is not callable"
        except AssertionError as e:
            raise ValueError(e)
        
        # We can't possibly check that the functions run over all 
        # possible values - as a best effort, we make sure they 
        # run successfully over the object min and max, and that 
        # the inverse comes back to the original value.
        try:
            interim_vals = [forward(_) for _ in (self.min, self.max)]
            assert not np.any([np.isnan(_) for _ in interim_vals]), "Forward function returned a NaN"
            vals = [inverse(_) for _ in interim_vals]
            assert not np.any([np.isnan(_) for _ in vals]), "Inverse function returned a NaN"
            assert np.allclose(vals, [self.min, self.max]), "Forward, then inverse, did not return original values!"
        except (ValueError, TypeError, RuntimeError) as e:
            raise ValueError(e)
        except AssertionError as e:
            raise ValueError(e)
        
        self._forward = forward
        self._inverse = inverse

class TransformSlog(TransformFunc):

    def __init__(self, c, e, *args, min=1e-12, max=np.inf, **kwargs):
        try:
            min > 0, "min must be >0 for scaled log transform"
        except AssertionError as err:
            raise ValueError(err)

        super().__init__(min=min, max=max, forward=slog(c, e), inverse=slog_inv(c, e))

class TransformRst(TransformFunc):

    def __init__(self, m, c, *args, min=-1e12, max=1e12, **kwargs):
        try:
            assert np.isfinite(min) and np.isfinite(max), "Rational scaled transform does not support non-finite values"
        except AssertionError as e:
            raise ValueError(e)

        super().__init__(min=min, max=max, forward=rst(m, c), inverse=rst_inv(m, c))
