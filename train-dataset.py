import argparse
from azureml.data.dataset_factory import TabularDatasetFactory
import pandas as pd
from sklearn.linear_model import LogisticRegression
from azureml.core.run import Run
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core import Dataset
from azureml.core import Model
from azureml.core import Workspace

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="")
    parser.add_argument('--max_iter',default=100, help="")

    args = parser.parse_args()

    run = Run.get_context()

    #ds = Dataset.Tabular.from_delimited_files(path = [(datastore, 'train-dataset/tabular/cleandata.csv')])

    ds = Dataset.Tabular.from_delimited_files(path=['https://mlstrg233868.blob.core.windows.net/azureml-blobstore-8044eec8-5725-4250-800a-2905ee00c009/train-dataset/tabular/cleandata.csv'])

    df = ds.to_pandas_dataframe()

    x_col = ['Avg_Sleepiness', 'People_Queried_About_Sleep', 'raw_Crashes_per_Year','Avg_Crash_Severity','Percent_Morning_Crashes','Percent_Evening_Crashes','Crashes_per_Year']
    y_col = ['State']
    x_df = df.loc[:, x_col]
    y_df = df.loc[:, y_col]

    x_train, x_test = train_test_split(x_df, shuffle=False)
    y_train, y_test = train_test_split(y_df, shuffle=False)

    model = LogisticRegression(C=args.C,max_iter=args.max_iter).fit(x_train,y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()