from functools import wraps
from pathlib import Path
from typing import Union, List, Dict
import os
import json
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import openfoamparser_mai as Ofpp
import pyvista


class VelParser:
    def get_vel_data(
        self,
        vel: str,
        model: str,
        base_path: str = 'data',
        dimpath: str = 'low_dim',
    ) -> Dict[str, List[np.ndarray]]:
        parsepath = Path(
            '/'.join([
                base_path,
                model,
                dimpath,
                vel,
            ])
        )
        time_paths = sorted(list(parsepath.iterdir()))[1:-2]
        veldata = {}
        for time_path in time_paths:
            time_path = Path(time_path)
            p = Ofpp.parse_internal_field(time_path / Path('p'))
            t = Ofpp.parse_internal_field(time_path / Path('T'))
            u = Ofpp.parse_internal_field(time_path / Path('U'))
            rho = Ofpp.parse_internal_field(time_path / Path('rho'))

            key = str(time_path).split('/')[-1]
            veldata[key] = [p, t, u, rho]
        return veldata

    def get_all_dim_data(
        self,
        model: str,
        base_path: str = 'data',
        dimpath: str = 'low_dim',
    ) -> Dict[str, Union(Dict[str, List[np.ndarray]]) | List[np.ndarray]]:
        parsepath = Path(
            '/'.join([
                base_path,
                model,
                dimpath,
            ])
        )
        dimdata = {}

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
