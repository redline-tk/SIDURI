import hashlib

def stable_seed(*parts):
    """Deterministic seed, independent of PYTHONHASHSEED (unlike hash())."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)
