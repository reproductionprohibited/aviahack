import torch 
from pathlib import Path
from typing import Union
import os
import numpy as np
import openfoamparser_mai as Ofpp
import pandas as pd
from torch_geometric.data import Data 
from torch_geometric.nn import GCNConv 
import torch.nn.functional as F 
from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import StandardScaler 
import argparse

def _face_center_position(points: list, mesh: Ofpp.FoamMesh) -> list:
    vertecis = [mesh.points[p] for p in points]
    vertecis = np.array(vertecis)
    return list(vertecis.mean(axis=0))

def pressure_field_on_surface(solver_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 surface_name: str = 'Surface') -> None:
    """Поле давлений на поверхности тела:
    'Nodes' - List[x: float, y: float, z:float], 
    'Faces' - List [List[int]], 
    'Elements' - List [Dict{Faces: List[int],
                            Pressure: float,
                            Velocity: List[float],
                            VelocityModule: float,
                            Position: List[float]}
                            ], 
    'Surfaces' - List[
                    Tuple[Surface_name: str, 
                    List[Dict{ParentElementID: int,
                              ParentFaceId: int,
                              Position: List[float]}]
                    ]

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с расчетом.
        p (np.ndarray): Поле давления.
        surface_name (str): Имя для поверхности.
    """
    
    # Step 0: parse mesh and scale vertices
    mesh_bin = Ofpp.FoamMesh(solver_path )

    # Step I: compute TFemFace_Surface
    domain_names = ["motorBike_0".encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))

        body_faces = []
        for f, b in f_b_set:
            try:
                position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'CentrePosition': {'X': position[0], 'Y': position[1], 'Z': position[2]},
                    'PressureValue': p[b]
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,
                'Item2': body_faces}) 
        
        return surfaces


def parse_data(cat: str) -> pd.DataFrame:
    vels = {
        '0.6M': [203.4030334563726, 17.795459554216844, 0],
        '0.7M': [230.09319108031895, 61.65328473387146, 0],
        '150': [150.0, 0.0, 0.0],
        '183': [183.33333333333334, 0.0, 0.0],
        '216': [216.66666666666669, 0.0, 0.0],
        '250': [250.0, 0.0, 0.0], 
        '280': [280.0, 0.0, 0.0],
        '303': [303.3333333333333, 0.0, 0.0],
        '326': [326.6666666666667, 0.0, 0.0],
        '350': [350.0, 0.0, 0.0]
    }
    
    PATH_TO_CASE = f'agard/{cat}/'
    END_TIME = '150'
    base_path = Path(PATH_TO_CASE)
    time_path = base_path / Path(END_TIME)
    p_path = time_path / Path('p')
    p = Ofpp.parse_internal_field(p_path)
    surface = pressure_field_on_surface(base_path, p)

    x = []
    y = []
    z = []
    p = []

    for s in surface[0]['Item2']:
        x.append(s['CentrePosition']['X'])
        y.append(s['CentrePosition']['Y'])
        z.append(s['CentrePosition']['Z'])
        p.append(s['PressureValue'])

    v_x, v_y, v_z = vels['0.6M']

    df = pd.DataFrame({'x': x, 'y':y, 'z': z, 'v_x': v_x, 'v_y': v_y, 'v_z': v_z, 'pressure':p})

    return df


def create_edge_index(features: np.ndarray, k=5) -> torch.tensor: 
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(features[:, :3]) 
    _, indices = nbrs.kneighbors(features[:, :3]) 
 
    edges = [] 
    for i in range(len(features)): 
        for j in indices[i]: 
            if i != j: 
                edges.append([i, j]) 
 
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

class GCN(torch.nn.Module): 
    def __init__(self): 
        super(GCN, self).__init__() 
        self.conv1 = GCNConv(6, 64) 
        self.conv2 = GCNConv(64, 128) 
        self.conv3 = GCNConv(128, 64) 
        self.conv4 = GCNConv(64, 1) 
 
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 
 
        x = F.leaky_relu(self.conv1(x, edge_index)) 
        x = F.leaky_relu(self.conv2(x, edge_index)) 
        x = F.leaky_relu(self.conv3(x, edge_index)) 
        x = self.conv4(x, edge_index) 
 
        return x
    

def predict(agard_df):
    
    input_data = agard_df[['x', 'y', 'z', 'v_x', 'v_y', 'v_z']]
    
    scaler = StandardScaler() 
    features = scaler.fit_transform(input_data[['x', 'y', 'z', 'v_x', 'v_y', 'v_z']].values)
    print(type(features))    
    edge_index_test = create_edge_index(features) 
     
    x_tensor_train = torch.tensor(features, dtype=torch.float) 
    
    model = GCN()
    model.load_state_dict(torch.load("model_state_dict.pt"))
    model.eval()
    
    test_out = model(Data(x=x_tensor_train, edge_index=edge_index_test))
    
    return test_out 
    

def main():
    parser = argparse.ArgumentParser(description='Data Parsing Script')

    parser.add_argument('path', type=str, help=' Путь к папке с данными о конкретной скорости (пример: agard150.0)')

    args = parser.parse_args()
    
    agard_df = parse_data(args.path)
    
    model = GCN()
    
    model.load_state_dict(torch.load('model_state_dict.pt'))
    model.eval()

    input_data = agard_df[['x', 'y', 'z', 'v_x', 'v_y', 'v_z']].values
    
    pressures = predict(agard_df).detach()
    
    np.savetxt("predictions.txt", pressures)
    
    
if __name__ == "__main__":
    main()