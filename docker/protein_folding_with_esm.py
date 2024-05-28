from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import gradio as gr
from gradio_molecule3d import Molecule3D

reps =    [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "stick",
      "color": "whiteCarbon",
      "residue_range": "",
      "around": 0,
      "byres": False,
      "visible": False
    }
]

def read_mol(molpath):
    with open(molpath, "r") as fp:
        lines = fp.readlines()
    mol = ""
    for l in lines:
        mol += l
    return mol


def molecule(input_pdb):

    mol = read_mol(input_pdb)

    x = (
        """<!DOCTYPE html>
        <html>
        <head>    
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <style>
    body{
        font-family:sans-serif
    }
    .mol-container {
    width: 100%;
    height: 600px;
    position: relative;
    }
    .mol-container select{
        background-image:None;
    }
    </style>
     <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js" integrity="sha512-STof4xm1wgkfm7heWqFJVn58Hm3EtS31XFaagaa8VMReCXAkQnJZ+jEy8PCC/iT18dFy95WcExNHFTqLyp72eQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>  
    <div id="container" class="mol-container"></div>
  
            <script>
               let pdb = `"""
        + mol
        + """`  
      
             $(document).ready(function () {
                let element = $("#container");
                let config = { backgroundColor: "white" };
                let viewer = $3Dmol.createViewer(element, config);
                viewer.addModel(pdb, "pdb");
                viewer.getModel(0).setStyle({}, { cartoon: { colorscheme:"whiteCarbon" } });
                viewer.zoomTo();
                viewer.render();
                viewer.zoom(0.8, 2000);
              })
        </script>
        </body></html>"""
    )

    return f"""<iframe style="width: 100%; height: 600px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model = model.cuda()

model.esm = model.esm.half()

import torch

torch.backends.cuda.matmul.allow_tf32 = True

model.trunk.set_chunk_size(64)

def fold_protein(test_protein):
    tokenized_input = tokenizer([test_protein], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.cuda()
    with torch.no_grad():
        output = model(tokenized_input)
    pdb = convert_outputs_to_pdb(output)
    with open("output_structure.pdb", "w") as f:
        f.write("".join(pdb))
    html = molecule("output_structure.pdb")
    return html, "output_structure.pdb"

iface = gr.Interface(
    title="everything-ai-proteinfold",
    fn=fold_protein,
    inputs=gr.Textbox(
            label="Protein Sequence",
            info="Find sequences examples below, and complete examples with images at: https://github.com/AstraBert/proteinviz/tree/main/examples.md; if you input a sequence, you're gonna get the static image and the 3D model to explore and play with",
            lines=5,
            value=f"Paste or write amino-acidic sequence here",
        ),
    outputs=[gr.HTML(label="Protein 3D model"), Molecule3D(label="Molecular 3D model", reps=reps)], 
)

iface.launch(server_name="0.0.0.0", share=False)