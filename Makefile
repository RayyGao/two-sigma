TRAIN="./data/train.json"
TEST="./data/test.json"
PROC_TRAIN="./data/processed_train.json"
PROC_TEST="./data/processed_test.json"

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

data-load:
	[ -f $(TRAIN) ] || wget "https://www.dropbox.com/s/7hw2we4w3dv6c17/train_data.json?dl=0" > $(TRAIN)
	[ -f $(TEST) ] || wget "https://www.dropbox.com/s/sdlcsynz5easl8r/test_data.json?dl=0" > $(TEST)

data-build:
	make data-load
	[ -f $(PROC_TRAIN) ] || python -c "import load_data; load_data.save_data('train.json')"
	[ -f $(PROC_TEST) ] || python -c "import load_data; load_data.save_data('test.json')"

all:
	make clean-pyc data-load data-build
