import pandas as pd
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


users = pd.read_parquet("data/processed/users_1m.parquet")
movies = pd.read_parquet("data/processed/movies_1m.parquet")
ratings1m = pd.read_parquet("data/processed/ratings_1m.parquet")
ratings20m = pd.read_parquet("data/processed/ratings_20m.parquet")
tags = pd.read_parquet("data/processed/tags_20m.parquet")

for name, df in [("users_1m", users), ("movies_1m", movies), ("ratings_1m", ratings1m), ("ratings_20m", ratings20m), ("tags_20m", tags)]:
    print(f"\n=== {name} ===")
    print(df.dtypes)
    print(df.head(5))
