import os

for fn in os.listdir(os.path.dirname(__file__)):
    if fn.endswith('.py'):
        exec ("import %s" % fn[:-3]) 

