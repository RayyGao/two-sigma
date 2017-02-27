TRAIN="./data/train.json"
TEST="./data/test.json"
PROC_TRAIN="./data/processed_train.json"
PROC_TEST="./data/processed_test.json"

all:
	make clean-pyc data-load data-build

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

data-load:
	[ -f $(TRAIN) ] || wget -O $(TRAIN) "https://www.dropbox.com/s/7phk9vkqq47y7hh/train.json?dl=0"
	[ -f $(TEST) ] || wget -O $(TEST) "https://www.dropbox.com/s/vgabbhzlyv3lmwe/test.json?dl=0"

data-build:
	[ -f $(PROC_TRAIN) ] || python -c "from src import load_data; load_data.save_data('train.json')"
	[ -f $(PROC_TEST) ] || python -c "from src import load_data; load_data.save_data('test.json')"
