from pathlib import Path
import pandas as pd
import json


class Logger:

    def __init__(self):
        self.records = []
        self.metadata = {}

    # ---------------------------------------------------------

    def record(self, **kwargs):
        self.records.append(kwargs)

    # ---------------------------------------------------------

    def clear(self):
        self.records.clear()
        self.metadata = {}

    # ---------------------------------------------------------

    def dataframe(self):
        return pd.DataFrame(self.records)

    # ---------------------------------------------------------

    def summary(self):
        if len(self.records) == 0:
            return pd.DataFrame()
        df = self.dataframe()
        return df.describe(include="all")

    # ---------------------------------------------------------

    def set_metadata(self, md: dict):
        self.metadata = dict(md)

    # ---------------------------------------------------------

    def save(self, filename):
        filename = Path(filename)
        df = self.dataframe()
        filename.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False)

        try:
            meta_file = filename.with_suffix(filename.suffix + ".meta.json")
            safe_meta = json.loads(json.dumps(self.metadata, default=str))
            with open(meta_file, "w", encoding="utf-8") as fh:
                json.dump(safe_meta, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Warning: failed to write metadata file:", e)

        print()
        print("=" * 60)
        print("Logger")
        print("=" * 60)
        print(f"Saved {len(df)} observations")
        print(filename.resolve())
        if self.metadata:
            print(f"Saved metadata to {meta_file.resolve()}")
        print("=" * 60)
        return df