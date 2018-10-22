import json

from src import dataset, features, models

config = json.load("./config.json")
number = config["number"]

# Calculate interim data
if config["calculate_interim"]:
    interim_data = dataset.make_dataset.main(number,
                                             save=config["save_interim"])
else:
    interim_data = None

# Calculate processed data
if config["calculate_processed"]:
    processed_data = features.build_features.main(number, interim_data,
                                                  save=config["save_processed"])
else:
    processed_data = None

# Train model
models.train_predict.main(number, processed_data)
