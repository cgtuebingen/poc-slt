import trimesh
import typing

def as_mesh(scene_or_mesh: typing.Union[trimesh.Scene, trimesh.Trimesh]) -> typing.Optional[trimesh.Trimesh]:
    if isinstance(scene_or_mesh, trimesh.Scene):
        if (len(scene_or_mesh.geometry) == 0 or scene_or_mesh == []):
            return None  # empty scene ==> empty mesh
        # mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces) for m in scene_or_mesh.geometry.values()])
        return trimesh.util.concatenate(scene_or_mesh.geometry.values())
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    # else:
    #     raise "unexpected input given to as_mesh: " #+ str(type(scene_or_mesh))