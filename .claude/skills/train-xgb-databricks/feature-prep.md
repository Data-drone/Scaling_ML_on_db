# Feature Preparation Reference

## Type Detection from Spark Schema

```python
from pyspark.sql.types import (IntegerType, LongType, FloatType,
                                DoubleType, StringType, BooleanType)

numeric_types = (IntegerType, LongType, FloatType, DoubleType)
cat_types = (StringType,)

target_col = "label"  # adjust to your target column

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, numeric_types) and f.name != target_col]
cat_cols = [f.name for f in df.schema.fields
            if isinstance(f.dataType, cat_types)]
bool_cols = [f.name for f in df.schema.fields
             if isinstance(f.dataType, BooleanType)]

# Booleans: cast to int
for col in bool_cols:
    df = df.withColumn(col, df[col].cast("integer"))
    numeric_cols.append(col)
```

## Handling Categoricals

XGBoost CPU `hist` method does not natively handle string features.

**Low-cardinality (< 50 unique values):** Ordinal encode.

```python
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                          handleInvalid="keep")
            for c in cat_cols]

pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

# Use indexed columns instead of originals
feature_cols = numeric_cols + [f"{c}_idx" for c in cat_cols]
```

**High-cardinality (>= 50 unique):** Drop the column. High-cardinality ordinal encoding creates misleading ordinal relationships.

```python
high_card = [c for c in cat_cols
             if df.select(c).distinct().count() >= 50]
cat_cols = [c for c in cat_cols if c not in high_card]
```

## Imbalance Handling

For binary classification with imbalanced classes:

```python
class_counts = df.groupBy(target_col).count().collect()
counts = {row[target_col]: row["count"] for row in class_counts}
scale_pos_weight = counts[0] / counts[1]  # negative / positive
```

This tells XGBoost to weight the minority class higher during training.

## Train/Test Split

**Single-node (pandas):**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Distributed (Ray Data):**
```python
train_ds, test_ds = ray_dataset.train_test_split(test_size=0.2, seed=42)
```

## Memory Estimation

```python
import psutil

estimated_gb = (row_count * feature_count * 8) / 1e9  # 8 bytes per float64
available_gb = psutil.virtual_memory().available / 1e9

if estimated_gb > available_gb * 0.8:
    print(f"WARNING: {estimated_gb:.1f} GB > 80% of {available_gb:.1f} GB available")
    print("Consider distributed training (Ray) or a larger node")
```

**GPU memory estimation:**
```python
gpu_mem_needed_gb = estimated_gb * 6  # XGBoost GPU needs ~6x raw data for histograms
```

## Spark to Pandas Conversion

For single-node training, convert Spark DataFrame to pandas:

```python
import time
load_start = time.time()
pdf = df.select(feature_cols + [target_col]).toPandas()
print(f"Loaded {len(pdf):,} rows in {time.time() - load_start:.1f}s")
print(f"Memory: {pdf.memory_usage(deep=True).sum() / 1e9:.2f} GB")

X = pdf[feature_cols]
y = pdf[target_col]
```
