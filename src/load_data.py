import os

from base import PROJECT_ROOT
from preprocess import process_data

def save_data(filename):
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', filename)
    PROC_PATH = os.path.join(PROJECT_ROOT, 'data',
                             "processed_" + os.path.basename(DATA_PATH))
    try:
        pre = os.path.basename(DATA_PATH)
        post = os.path.basename(PROC_PATH)

        print("\nOpening '{}'...".format(pre))
        print(DATA_PATH)
        with open(DATA_PATH) as f:
            data = pd.read_json(f)

        print("opened.")

        data = process_data(data)

        print("Writing processed data to '{}'...".format(post))

        with open(PROC_PATH, "w") as p:
            data.to_json(p)

        print("Finished processing '{}' into '{}'.".format(pre, post))
    except:
        print("Failed to process {} to file.".format(DATA_PATH))

def load_data(filename):
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', filename)
    with open(DATA_PATH) as f:
            data = pd.read_json(f)

    return data

# save_data("train.json")
# save_data("test.json")

