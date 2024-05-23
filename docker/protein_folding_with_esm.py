from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from proteins_viz import *
import gradio as gr

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
    image = take_care("output_structure.pdb")
    return image

iface = gr.Interface(
    title="everything-ai-proteinfold",
    fn=fold_protein,
    inputs="text",
    outputs="image", 
)

iface.launch(server_name="0.0.0.0", share=False)
