import openfoamparser_mai as Ofpp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

points=Ofpp.parse_internal_field(r'data\0.3M\150\C')
p = Ofpp.parse_internal_field(r'data\0.3M\150\p')
u = Ofpp.parse_internal_field(r'data\0.3M\150\U')

x = []
y = []
z = []
for point in points:
    x.append(point[0])
    y.append(point[1])
    z.append(point[2])
    
v_x = []
v_y = []
v_z = []
for v in u:
    v_x.append(v[0])
    v_y.append(v[1])
    v_z.append(v[2])
    
df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'v_x': v_x, 'v_y': v_y, 'v_z': v_z, 'pressure': p})
df = shuffle(df)

def train_val_test(df):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df