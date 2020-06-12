


def _is_collection(obj):
    if hasattr(obj, '__iter__') and not isinstance(obj, str) and not isinstance(obj, type):
        return True
    return False


def _collection(obj):
    if not _is_collection(obj):
        return (obj,)
    return obj