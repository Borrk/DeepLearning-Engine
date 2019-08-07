from engine.engine import ai_engine
import os

### test time parse
import time
import datetime

## main entry
path = './config/config.json'
eng = ai_engine()
eng.load_steps( path )
eng.run()
eng.dispose()
