from src import dataset, features, models

number = 1
interim_data = dataset.make_dataset.main(number)
processed_data = features.build_features.main(number, interim_data, save=True)
models.train_predict.main(number, processed_data)
