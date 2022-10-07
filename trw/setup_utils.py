from typing import List
import os
import re

def get_project_name():
    """
    :return: the name of the project. This is done by finding the directories in './src/'
    """
    possible_names = []
    for d in os.listdir('./src/'):
        if '.' not in d:
            possible_names.append(d)
    assert len(possible_names) == 1, 'Only one name allowed. Found possible names={}'.format(possible_names)
    return possible_names[0]


def read_requirements(path: str) -> List[str]:
    """
    Load the requirements and support `-r` option to load dependent requirement files
    """
    with open(path, 'r', encoding='utf-8') as f:
        all_reqs = f.read().split('\n')

    # find embedded requirements inside (e.i., `-r <other requirements>`)
    # "pip install -r <file>" handles nested requirements, so do that too here
    root = os.path.dirname(path)
    sub_reqs = []

    filtered_reqs = []
    for x in all_reqs:
        m = re.findall(r'^-r\s+(\S+)', x)
        if len(m) == 1:
            sub_reqs += read_requirements(os.path.join(root, m[0]))
        elif len(x) > 0:
            filtered_reqs.append(x)
    return filtered_reqs + sub_reqs
