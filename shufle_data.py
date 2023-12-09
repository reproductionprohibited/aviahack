
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

def train_test(name):
    df = pd.read_csv(f'{name}.csv')
    df = shuffle(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    features = df[['x', 'y', 'z', 'v_x', 'v_y', 'v_z']].values
    target = df['pressure'].values
    train_x, test_x, train_Y, test_Y = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return train_x, train_Y, test_x, test_Y