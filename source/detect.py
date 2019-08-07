from engine.engine import ai_engine
import os

path = './config/detect_config.json'
eng = ai_engine()
eng.load_steps( path )
eng.run()
eng.dispose()