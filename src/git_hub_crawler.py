from __future__ import absolute_import, division, print_function, unicode_literals

'''
Created on 26.11.2017

@author: amiller
'''

import os
import errno
import stat
import shutil
import tempfile
import git
import pytest
import ast

from collections import Counter
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.unlink, os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO) # 0777
        func(path)
    else:
        raise


class TempRepo(git.Repo):
    @staticmethod
    def clone_from_to_temp(url, **kwargs):
        directory_name = tempfile.mkdtemp()
        return TempRepo.clone_from(url, directory_name, **kwargs)

    @property
    def directory(self):
        return os.path.abspath(os.path.join(self.git_dir, '..'))

    def __exit__(self, type, value, traceback):
        print('woot: ' + self.directory)
        shutil.rmtree(self.directory, ignore_errors=False, onerror=handle_remove_readonly)

        return git.Repo.__exit__(self, type, value, traceback)





class RepoCrawler():
    def __init__(self):
        self.pairs = []
        self.list_data = []


    def do_lists(self, parent, keyword, child_list):
    #    list_data[(parent, keyword)] += child_list + [None]
        for child in child_list:
            if child is None:
                self.list_data.append(((str(type(parent)), keyword), None))
            else:
                self.list_data.append(((str(type(parent)), keyword), str(type(child))))

    def walk2(self, node):
        parent = node
        for keyword, child in ast.iter_fields(node):
            if isinstance(child, ast.AST):
                self.pairs.append(((str(type(parent)), keyword), str(type(child))))
                self.walk2(child)
            elif isinstance(child, list):
                self.do_lists(parent, keyword, child)
                for item in child:
                    if isinstance(item, ast.AST):
                        self.walk2(item)

    def add_and_crawl(self, directory):
        for root, dir, files in os.walk(directory):
            for file in files:
        #for file in os.listdir(directory):
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    print(path)

                    with open(path, 'r') as python_file:
                        python_code = python_file.read()
                    try:
                        python_ast = ast.parse(python_code)
                        self.walk2(python_ast)
                    except SyntaxError:
                        pass  # not all modules can be parsed (e.g. no python3 compatibility)

    def get_probabilities(self):
        pairs = pd.DataFrame(self.pairs)
        pairs.columns = ['parent', 'child']
        
        frequency = pd.crosstab(pairs.parent, pairs.child)
        
        list_data = pd.DataFrame(self.list_data)
        list_data.columns = ['parent', 'child']
        frequency_list_data = pd.crosstab(list_data.parent, list_data.child)
        frequency = frequency.astype(np.float)
        frequency.values[:] = frequency.values / frequency.values.sum(axis=1, keepdims=True)
        return frequency


if __name__ == '__main__':
    with TempRepo.clone_from_to_temp('https://github.com/django/django', branch='master') as repo:
        #print(repo.directory)
        #pytest.main([repo.directory])
        crawler = RepoCrawler()
        crawler.add_and_crawl(repo.directory)
        probs = crawler.get_probabilities()
        print(probs.columns)
        plt.imshow(probs.values)
        plt.show()
        