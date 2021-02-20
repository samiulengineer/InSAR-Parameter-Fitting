import numpy as np
from pl_igscmaes.data.generate_ps import generate_ps

dbs = [
    'bagdad.tsx.sm_dsc',
    'chino.tsx.sm_dsc',
    'cortez.tsx.sm_dsc',
    'elabra.tsx.sm_dsc',
    'escondida.tsx.sm_dsc',
    'grasbergmine.tsx.sm_dsc',
    'miami.tsx.sm_dsc',
]

processing_root = config['processing_root']

rule all:
    input: 
        expand("{processing_root}/ps/{db}/ps.json", db=dbs, processing_root=processing_root)

rule generate_ps:
    output:
        "{processing_root}/ps/{db}/ps.json"
    run:
        generate_ps(wildcards.db, processing_root+'/ps', str(output))
    
