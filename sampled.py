print("1")
# Import libraries.
from dataclasses import replace
import pandas as pd
import phoenix as px

# Download curated datasets and load them into pandas DataFrames.
samples_df = pd.read_csv(
    "./data/samples.csv",
)

# Define schemas that tell Phoenix which columns of your DataFrames correspond to features, predictions, actuals (i.e., ground truth), embeddings, etc.
sample_schema = px.Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="prediction_ts",
    prediction_label_column_name="predicted_action",
    actual_label_column_name="actual_action",
    embedding_feature_column_names={
        "image_embedding": px.EmbeddingColumnNames(
            vector_column_name="image_vector",
            link_to_data_column_name="url",
        ),
    },
)

# Define your production and training datasets.
samples_ds = px.Dataset(samples_df, sample_schema)

# Launch Phoenix.
session = px.launch_app(samples_ds)

# View the Phoenix UI in the browser
session.url
