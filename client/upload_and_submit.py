import torch
import numpy as np
import json
import time
import uuid
import requests
import subprocess
import os
from models import CNN4, CNN2

def load_model(model_path, model_class):
    state_dict = torch.load(model_path, map_location='cpu')
    model = model_class()
    model.load_state_dict(state_dict)
    return model.state_dict()

def model_state_to_vector(state_dict):
    vecs, shapes, keys = [], [], []
    for k, v in state_dict.items():
        arr = v.cpu().numpy().reshape(-1)
        vecs.append(arr)
        shapes.append(v.size())
        keys.append(k)
    return np.concatenate(vecs), keys, shapes

def vector_to_state_dict(vector, keys, shapes):
    state = {}
    pos = 0
    for k, shape in zip(keys, shapes):
        size = np.prod(shape)
        arr = vector[pos:pos+size].reshape(shape)
        state[k] = torch.tensor(arr)
        pos += size
    return state

def clip_and_add_noise(delta, clip_norm, noise_sigma):
    norm = np.linalg.norm(delta)
    if norm > clip_norm:
        delta *= (clip_norm / norm)
    noise = np.random.normal(0, noise_sigma, delta.shape)
    return delta + noise

def ipfs_add(file_path):
    with open(file_path, 'rb') as f:
        r = requests.post("http://127.0.0.1:5001/api/v0/add", files={'file': f})
    text = r.text.strip()
    return text.split()[1]

if __name__ == "__main__":
    # CONFIGURATION - CHANGE THIS FOR YOUR MODELS
    MODEL_CONFIG = {
        'covid': {'class': CNN4, 'local': './local_models/covid_local.pth'},
        'skin': {'class': CNN2, 'local': './local_models/skin_local.pth'}
    }
    
    model_name = 'covid'  # CHANGE TO 'skin' for skin model
    sender_id = f"client-{model_name}-1"
    model_class = MODEL_CONFIG[model_name]['class']
    local_path = MODEL_CONFIG[model_name]['local']
    global_path = './local_models/global_model.pth'
    
    print(f"🔄 Processing {model_name} model...")
    
    # Load models
    local_state = load_model(local_path, model_class)
    global_state = load_model(global_path, model_class)
    
    # Compute weight delta
    local_vec, keys, shapes = model_state_to_vector(local_state)
    global_vec, _, _ = model_state_to_vector(global_state)
    delta = local_vec - global_vec
    
    # Apply Differential Privacy
    clip_norm = 1.0
    noise_sigma = 0.5
    delta_dp = clip_and_add_noise(delta, clip_norm, noise_sigma)
    
    # Save delta file
    fname = f"update_{sender_id}_r1_{uuid.uuid4().hex[:8]}.pth"
    out_path = f"/tmp/{fname}"
    torch.save(vector_to_state_dict(delta_dp, keys, shapes), out_path)
    
    # Upload to IPFS
    cid = ipfs_add(out_path)
    print(f"✅ IPFS CID: {cid}")
    
    # Submit to Fabric blockchain
    tx_id = f"tx-{uuid.uuid4().hex[:8]}"
    payload = {
        "sender": sender_id,
        "cid": cid,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": 1,
        "metadata": json.dumps({
            "model": model_name,
            "epsilon": 8.0,
            "clip_norm": clip_norm,
            "param_count": len(delta)
        })
    }
    
    subprocess.run(["node", "submit_update.js", tx_id, json.dumps(payload)], check=True)
    print(f"✅ Blockchain Transaction: {tx_id}")
    print(f"✅ SUCCESS: {model_name.upper()} model update completed!")
