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

    def isvalidvelpath(self, string: str) -> bool:
        try:
            float_value = float(string[3:])
        except ValueError:
            return False

        return True

    def get_vel_data(
        self,
        vel: str = 'vel3.0',
        base_path: str = 'data',
        model: str = 'data_wage',
        dimpath: str = 'low_dim',
    ) -> Dict[str, List[np.ndarray]]:
        parsepath = Path(f'{base_path}/{model}/{dimpath}/{vel}')
        children = parsepath.iterdir()
        time_paths = sorted(list(filter(lambda x: self.is_string_float(str(x)), children)))[:-2]
        veldata = {}
        for time_path in time_paths:
            print(time_path)
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
    
    def get_all_dim_data(
        self,
        base_path: str = 'data',
        model: str = 'data_wage',
        dimpath: str = 'low_dim',
    ):
        parsepath = Path(
            '/'.join([
                base_path,
                model,
                dimpath,
            ])
        )
        dimdata = {}
        # return sorted(list(parsepath.iterdir()))[1:]
        for velpath in sorted(list(parsepath.iterdir()))[1:]:
            velpath = Path(velpath)
            key = str(velpath).split('/')[-1]
            dimdata[key] = self.get_vel_data(
                vel=key,
                base_path=base_path,
                model=model,
                dimpath=dimpath,
            )
        mesh: Ofpp.FoamMesh = Ofpp.FoamMesh(parsepath / Path('vel3.0'))

        dimdata['mesh'] = {
            'faces': mesh.faces,
            'boundary': mesh.boundary,
            'neighbour': mesh.neighbour,
            'owner': mesh.owner,
            'points': mesh.points,
        }
        return dimdata
