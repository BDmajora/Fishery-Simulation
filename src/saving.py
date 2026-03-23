import os

def save_csv(df, folder, filename):
    os.makedirs(folder, exist_ok=True)  # ensure folder exists
    path = os.path.join(folder, filename)  # full file path
    df.to_csv(path, index=False)  # save DataFrame to CSV without row index
    return path  # return path to saved file
