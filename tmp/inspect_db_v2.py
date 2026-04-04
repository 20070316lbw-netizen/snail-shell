import duckdb
import pandas as pd

db_path = "QQ_Quant_DB/quant_lab.duckdb"
con = duckdb.connect(db_path, read_only=True)

# Inspect features_cn schema
print("\nSchema of features_cn:")
schema = con.execute("DESCRIBE features_cn").fetchdf()
print(schema.to_string())

# Inspect prices_cn schema
print("\nSchema of prices_cn:")
schema = con.execute("DESCRIBE prices_cn").fetchdf()
print(schema.to_string())

# Sample of features_cn
print("\nSample of features_cn:")
sample = con.execute("SELECT * FROM features_cn LIMIT 2").fetchdf()
print(sample.to_string())

# Sample of prices_cn
print("\nSample of prices_cn:")
sample = con.execute("SELECT * FROM prices_cn LIMIT 2").fetchdf()
print(sample.to_string())

con.close()
