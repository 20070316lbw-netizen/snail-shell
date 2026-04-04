import duckdb
import pandas as pd

db_path = "QQ_Quant_DB/quant_lab.duckdb"
con = duckdb.connect(db_path, read_only=True)

with open("tmp/db_full_schema.txt", "w", encoding="utf-8") as f:
    # Inspect features_cn schema
    f.write("Schema of features_cn:\n")
    schema = con.execute("DESCRIBE features_cn").fetchdf()
    f.write(schema.to_string() + "\n\n")

    # Inspect prices_cn schema
    f.write("Schema of prices_cn:\n")
    schema = con.execute("DESCRIBE prices_cn").fetchdf()
    f.write(schema.to_string() + "\n\n")

    # Sample of features_cn
    f.write("Sample of features_cn:\n")
    sample = con.execute("SELECT * FROM features_cn LIMIT 2").fetchdf()
    f.write(sample.to_string() + "\n\n")

    # Sample of prices_cn
    f.write("Sample of prices_cn:\n")
    sample = con.execute("SELECT * FROM prices_cn LIMIT 2").fetchdf()
    f.write(sample.to_string() + "\n\n")

con.close()
