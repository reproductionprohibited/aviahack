from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import openfoamparser_mai as Ofpp


class VelParser:
    def get_mesh_data(
        self,
        path_to_vel: str,
    ) -> Dict[str, List[np.ndarray]]:
        parsepath = Path(path_to_vel)
        mesh = Ofpp.FoamMesh(parsepath)
        data = {
            'faces': mesh.faces,
            'neighbour': mesh.neighbour,
            'owner': mesh.owner,
            'points': mesh.points,
            'boundary': mesh.boundary,
        }
        return data

    def get_vel_data(
        self,
        path_to_vel: str,
    ) -> Dict[str, List[np.ndarray]]:
        parsepath = Path(path_to_vel)
        children = parsepath.iterdir()
        time_paths = sorted(list(filter(lambda x: 'vel' in str(x), children)))[:-2]
        veldata = {}
        for time_path in time_paths:
            time_path = Path(time_path)
            p = Ofpp.parse_internal_field(str(time_path) + '/' + 'p')
            t = Ofpp.parse_internal_field(str(time_path) + '/' + 'T')
            u = Ofpp.parse_internal_field(str(time_path) + '/' + 'U')
            rho = Ofpp.parse_internal_field(str(time_path) + '/' + 'rho')

            key = str(time_path).split('/')[-1]
            veldata[key] = [p, t, u, rho]

        return veldata

    def get_vels(
        self,
        model: str,
        base_path: str,
        dimpath: str,
    ) -> List[str]:
        parsepath = Path(
            '/'.join([
                base_path,
                model,
                dimpath,
            ])
        )
        children = parsepath.iterdir()
        vels = list(filter(lambda x: 'vel' in str(x), children))
        return vels
