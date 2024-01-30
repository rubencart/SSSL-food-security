import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    df = pd.read_csv("data/predicting_food_crises_data.csv")

    som_df = df[df.country == "Somalia"]
    som_df = som_df[df.fews_ipc.notnull()]

    som_df.year_month.unique()
    datetime.datetime.strptime("2009_10", "%Y_%m").strftime("%Y-%m")

    col = som_df["year_month"].apply(
        lambda x: (
            datetime.datetime.strptime(x, "%Y_%m") + relativedelta(months=1)
        ).strftime("%Y-%m-%d")
    )
    som_df = som_df.assign(ymd=col.values)

    som_13_df = som_df[som_df.ymd >= "2013-05-01"]
    som_13_df.to_csv("data/predicting_food_crises_data_somalia_from2013-05-01.csv")
