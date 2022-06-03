from task import tasks
from algorithms import algorithms

task = 'img'
algorithm = 'APGM'

assert task in tasks.keys()
assert algorithm in algorithms.keys()

tasks[task](algorithms[algorithm])