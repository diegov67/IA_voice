#from rvcgui import vc_single, get_output_path
from my_utils import load_audio
import traceback
from config import Config
from fairseq import checkpoint_utils
import soundfile as sf

from vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from infer_pack.modelsv2 import SynthesizerTrnMs768NSFsid_nono, SynthesizerTrnMs768NSFsid

import os
import torch


def vc_single(
    sid,
    input_audio,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    crepe_hop_length,
    output_path=None,
):          
    global tgt_sr, net_g, vc ,version, config    
    vc = VC(tgt_sr)
    print("output_path vc_single: ",output_path)
    version = "v1"
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = 1 #cpt.get("f0", 1)
        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )  
     
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            version,
            crepe_hop_length,
            None,
        )
        print(
            "npy: ", times[0], "s, f0: ", times[1], "s, infer: ", times[2], "s", sep=""
        )

        if output_path is not None:
            sf.write(output_path, audio_opt, tgt_sr, format='WAV')
        print("completada la conversion")
        return "Success", (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def get_vc(weight_root, sid):
    global hubert_model,is_half
    #config = Config()
    
    hubert_model = None
    device = "cpu"
    print(device)
    is_half = "false"
    
    
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:        
        if hubert_model != None:  
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()            
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return
    person = (weight_root)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")    
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q    
    net_g.eval().to("cpu")
    # if config.is_half:
    #     net_g = net_g.half()
    # else:
    #     net_g = net_g.float()
    net_g = net_g.float()
    #vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]    
    return

    
def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    #hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.to("cpu")
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model = hubert_model.float()
    hubert_model.eval()
    
def get_output_path(file_path):
    print("ingresa get_output_path")
    print(file_path)
    if not os.path.exists(file_path):
        # change the file extension to .wav
        
        return file_path  # File path does not exist, return as is

    # Split file path into directory, base filename, and extension
    dir_name, file_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(file_name)

    # Initialize index to 1
    index = 1

    # Increment index until a new file path is found
    while True:
        new_file_name = f"{file_name}_RVC_{index}{file_ext}"
        new_file_path = os.path.join(dir_name, new_file_name)
        if not os.path.exists(new_file_path):
            # change the file extension to .wav
            new_file_path = os.path.splitext(new_file_path)[0] + ".wav"
            print("salida", new_file_path)
            return new_file_path  # Found new file path, return it
        index += 1
    
def selected_model(choice):        
    models_dir = "./models"    
    model_dir = os.path.join(models_dir, choice)
    pth_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) 
                 and f.endswith(".pth") and not (f.startswith("G_") or f.startswith("D_"))
                 and os.path.getsize(os.path.join(model_dir, f)) < 200*(1024**2)]
    
    if pth_files:
        global pth_file_path
        pth_file_path = os.path.join(model_dir, pth_files[0])
        npy_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) 
                     and f.endswith(".index")]
        if npy_files:
            npy_files_dir = [os.path.join(model_dir, f) for f in npy_files]
            if len(npy_files_dir) == 1:
                index_file = npy_files_dir[0]
                print(f".pth file directory: {pth_file_path}")
                print(f".index file directory: {index_file}")                
            else:
                print(f"Incomplete set of .index files found in {model_dir}")
        else:
            print(f"No .index files found in {model_dir}")
        get_vc(pth_file_path, 0)
        global model_loaded
        model_loaded = True
    else:
        print(f"No eligible .pth files found in {model_dir}")


# Funcion que llama al proces de Voice Change


def enlace(sid, input_audio, f0_pitch, f0_file, f0_method, file_index, file_big_npy, index_rate, modelo):            
    
    selected_model(modelo)
    
    print("sid: ", sid, "input_audio: ", input_audio, "f0_pitch: ", f0_pitch, "f0_file: ", f0_file, "f0_method: ", f0_method,
          "file_index: ", file_index, "file_big_npy: ", "index_rate: ", index_rate, "output_file: ")
    crepe_hop_length = 128
    output_file = get_output_path(input_audio)
    try:
        result, audio_opt = vc_single(sid, input_audio, f0_pitch, f0_file, f0_method, file_index, index_rate, crepe_hop_length, output_file)
            # output_label.configure(text=result + "\n saved at" + output_file)
        print(os.path.join(output_file))
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(output_file)                           
            message = result
        else: 
            message = result            
        print(message)
        print("Conversion Completa")
    except Exception as e:
            print(e)
            message = "Voice conversion failed", e
            print(message)
    
    