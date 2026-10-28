!pip install -q ultralytics lap
import glob
DATA = glob.glob("/kaggle/input/**/GMDCSA24", recursive=True)[0]
!python src/simulate_night.py --src "{DATA}" --dst /kaggle/working/data/night_test --limit 2
