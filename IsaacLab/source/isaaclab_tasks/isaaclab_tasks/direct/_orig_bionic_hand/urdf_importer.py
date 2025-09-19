import omni.kit.commands
from isaacsim.core.utils.extensions import get_extension_path_from_name


class ImportURDF:
    def __init__(self, urdf_path: str, dest_path: str):
        self.urdf_path = urdf_path
        self.dest_path = dest_path

    def import_urdf_to_usd(self):
        # Setting up import configuration:
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False  
        import_config.convex_decomp = False       
        import_config.import_inertia_tensor = True  
        import_config.fix_base = True              
        import_config.distance_scale = 1.0          
        import_config.make_default_prim = True      
        import_config.self_collision = True         
        import_config.density = 0.0                 

        # Import the URDF and convert it to USD at the destination path
        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self.urdf_path,
            import_config=import_config,
            get_articulation_root=False,
            dest_path=self.dest_path
        )
        return status, prim_path