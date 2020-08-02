""" Project initialization and common objects. """

import logging
import os
import sys
import re

logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

def parse_system(system):
    """
    Parse a system string into a pair of projectiles and a beam energy.

    """
    match = re.fullmatch('([A-Z]?[a-z])([A-Z]?[a-z])([0-9]+)', system)
    return match.group(1, 2), int(match.group(3))


class lazydict(dict):
    """
    A dict that populates itself on demand by calling a unary function.

    """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __missing__(self, key):
        self[key] = value = self.function(key, *self.args, **self.kwargs)
        return value
