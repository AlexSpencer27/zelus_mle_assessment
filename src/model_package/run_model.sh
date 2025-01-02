# Define paths
MODEL_PATH="expected_runs_model.pkl"
DATA_PATH="data.parquet"
TEAM="Ireland"
OVERS=5
MATCH_ORDER="oldest"

# Run the Docker container with appropriate arguments
docker run --rm schnoodfam/zelus_mle_assessment:latest \
    --model $MODEL_PATH \
    --data $DATA_PATH \
    --batting-team $TEAM \
    --end-over $OVERS \
    --match-order $MATCH_ORDER
