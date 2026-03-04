from sklearn.model_selection import train_test_split

def writer_disjoint_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    writers = df["writer_id"].unique()
    train_writers, temp_writers = train_test_split(
        writers,
        test_size=(1 - train_ratio),
        random_state=seed
    )
    val_writers, test_writers = train_test_split(
        temp_writers,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )
    train_df = df[df["writer_id"].isin(train_writers)].reset_index(drop=True)
    val_df   = df[df["writer_id"].isin(val_writers)].reset_index(drop=True)
    test_df  = df[df["writer_id"].isin(test_writers)].reset_index(drop=True)

    return train_df, val_df, test_df