import duckdb
import pandas as pd

db_path = "QQ_Quant_DB/quant_lab.duckdb"
con = duckdb.connect(db_path, read_only=True)

# List tables
print("Tables:")
tables = con.execute("SHOW TABLES").fetchdf()
print(tables)

# Inspect features_cn schema
if "features_cn" in tables["name"].values:
    print("\nSchema of features_cn:")
    schema = con.execute("DESCRIBE features_cn").fetchdf()
    print(schema)

# Sample data
if "features_cn" in tables["name"].values:
    print("\nSample of features_cn:")
    sample = con.execute("SELECT * FROM features_cn LIMIT 5").fetchdf()
    print(sample)

con.close()
