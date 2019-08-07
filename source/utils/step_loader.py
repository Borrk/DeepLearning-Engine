import json
import os
import glob

import importlib

def load_step( model_name, step_name):
    model=importlib.import_module( model_name )
    step = getattr(model, step_name)

    return step

    