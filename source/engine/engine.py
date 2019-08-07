import json
import os
import glob
import importlib
from utils.step_loader import *

class ai_engine(object):
    """ the engine """
    
    global_cache={}
    steps = []

    def __init__(self ):
        self.global_cache['error_type'] = None
        self.global_cache['error_msg'] = ''


    def load_steps( self, config_file ):
        print("Loading steps...")

        # clear steps
        self.dispose()

        file = open(config_file, "r")
        engine_cfg = json.load(file)
        options = engine_cfg['step_options']

        # initialize and config steps
        for step_cfg in engine_cfg['steps']:
            if step_cfg['active'] == True:
                step = load_step( step_cfg['module'], step_cfg['name'] )

                # get step options
                step_options_name = step_cfg['options']
                step_options = options[step_options_name]

                # initialize step
                step.__init__( step, output_channel=self.global_cache, name= step_cfg['name'] )
                step.IParseConfig(step, config_json= step_options )
                self.steps.append( step )

                print( "step initialized: " + step.name )

    def run(self):
        print("Running engine...")
        if len(self.steps) < 1:
            raise Exception( "No steps created")
       
       # run step by step
        for step in self.steps:
            step.IRun(step)
            if self.global_cache['error_type'] == 'fatal':
                raise Exception( self.global_cache['error_msg'] )
    
        self.global_cache['error_type'] = None
        self.global_cache['error_msg'] = ''

    def run_from_file(self, config_file):
        self.load_steps( config_file)
        self.run()

    def dispose(self):
        print("Disposing steps...")

        for step in self.steps:
            step.IDispose(step)

        self.steps.clear()


