import copy
import argparse
import json
import numpy             as np

from neural_mixture.utils_ML       import *
from neural_mixture.utils_OpenFOAM import *

def main():
    
    ap = argparse.ArgumentParser(
        prog="feature_engineering",
        description=(
            "Generate the dataset to be used to train the RFR model. \n"
            "You can take a look at a sample of the Json file in the examples directory. \n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", help="Path to JSON file containing the cases dict.")

    args = ap.parse_args()
    with open(args.config) as fh:
        setup_dict = json.load(fh)

    

if __name__ == "__main__":
    main()