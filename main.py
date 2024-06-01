import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
from btp_dataset import BtpDataset
from utils import time_series_to_plot
from tensorboardX import SummaryWriter
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator
from models.convolutional_models import CausalConvGenerator, CausalConvDiscriminator
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
import DTW.soft_dtw_cuda as soft_dtw
import DTW.dtw_cuda as dtw
import numpy as np

dtwc = dtw.DTW(True, normalize=False, bandwidth=1)
sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
sdtw_eval = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=1)
# dataset_folder = os.path.join("..","Resultados", "ROT_X2_", "ROT_X2_005", "generated_features")
dataset_folder = os.path.join("..","OVS","Investigation","Extracted Features", "Evaluation")
# dataset_folder = os.path.join("..","OnlineSignatureVerification","Investigation","Extracted Features", "Evaluation")
# dataset_folder = os.path.join("ROT_X2_", "ROT_X2_005", "generated_features")
training_guide = "training_guide.txt"

def dtr(x, y, len_x, len_y):
        return sdtw(x[None, :int(len_x)], y[None, :int(len_y)])[0]/((len_x + len_y))

def dtr_eval(x, y, len_x, len_y):
        return sdtw_eval(x[None, :int(len_x)], y[None, :int(len_y)])[0]/(64*(len_x + len_y))

def dte(x, y, shape1, shape2):
    # Your dte calculation logic here
    return dtwc(x[None,], y[None,])[0].detach().cpu().numpy()[0] / (64*(shape1 + shape2))


def get_eer(y_true, y_scores, result_folder : str = None, generate_graph : bool = False, n_epoch : int = None):
    fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_scores, pos_label=1)
    fnr = 1 - tpr

    far = fpr
    frr = fnr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # as a sanity check the value should be close to
    eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer + eer2)/2
    # eer = min(eer, eer2)

    if generate_graph:
        frr_list = np.array(frr)
        far_list = np.array(far)

        plt.plot(threshold, frr_list, 'b', label="FAR")
        plt.plot(threshold, far_list, 'r', label="FRR")
        plt.legend(loc="upper right")

        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.plot(eer_threshold, eer, 'ro')
        plt.text(eer_threshold + 0.05, eer+0.05, s="EER = " + "{:.5f}".format(eer))
        #plt.text(eer_threshold + 1.05, eer2+1.05, s="EER = " + "{:.5f}".format(eer2))
        plt.savefig(result_folder + os.sep + "Epoch" + str(n_epoch) + ".png")
        plt.cla()
        plt.clf()

    return eer, eer_threshold

import math
def inference(files : str, gen, features_path : str):
    """ Calcula score de dissimilaridade entre um grupo de assinaturas

    Args:
        files (str): nome dos arquivos separados por espaço seguido da label (0: original, 1: falsificada)
        distances (Dict[str, Dict[str, float]]): ex: distances[a][b] = distância entre assinatura a e b.
            distances[a] -> dicionário em que cada chave representa a distância entre a e a chave em questão.
            Esse dicionário contém todas as combinações possíveis entre chaves.
            distances[a][b] == distances[b][a].
    Raises:
        ValueError: caso o formato dos arquivos seja diferente do 4vs1 ou 1vs1.

    Returns:
        Tuple[float, str, int]: score de dissimilaridade, usuário de referência, predição (default: math.nan)
    """
    references = ['u0001_g_0000v08.pt','u0002_g_0001v14.pt','u0003_g_0002v17.pt','u0004_g_0003v09.pt','u0005_g_0004v16.pt','u0006_g_0005v14.pt','u0007_g_0006v10.pt','u0008_g_0007v11.pt','u0009_g_0008v23.pt','u0010_g_0009v14.pt','u0011_g_0010v07.pt','u0012_g_0011v14.pt','u0013_g_0012v07.pt','u0014_g_0013v01.pt','u0015_g_0014v08.pt','u0016_g_0015v15.pt','u0017_g_0016v08.pt','u0018_g_0017v08.pt','u0019_g_0018v01.pt','u0020_g_0019v08.pt','u0021_g_0020v08.pt','u0022_g_0021v11.pt','u0023_g_0022v13.pt','u0024_g_0023v01.pt','u0025_g_0024v22.pt','u0026_g_0025v14.pt','u0027_g_0026v16.pt','u0028_g_0027v16.pt','u0029_g_0028v07.pt','u0030_g_0029v10.pt','u0031_g_0030v23.pt','u0032_g_0031v09.pt','u0033_g_0032v24.pt','u0034_g_0033v15.pt','u0035_g_0034v19.pt','u0036_g_0035v11.pt','u0037_g_0036v17.pt','u0038_g_0037v13.pt','u0039_g_0038v21.pt','u0040_g_0039v07.pt','u0041_g_0040v01.pt','u0042_g_0041v17.pt','u0043_g_0042v19.pt','u0044_g_0043v23.pt','u0045_g_0044v06.pt','u0046_g_0045v08.pt','u0047_g_0046v24.pt','u0048_g_0047v03.pt','u0049_g_0048v20.pt','u0050_g_0049v08.pt','u0051_g_0050v07.pt','u0052_g_0051v12.pt','u0053_g_0052v18.pt','u0054_g_0053v06.pt','u0055_g_0054v15.pt','u0056_g_0055v13.pt','u0057_g_0056v23.pt','u0058_g_0057v12.pt','u0059_g_0058v05.pt','u0060_g_0059v16.pt','u0061_g_0060v16.pt','u0062_g_0061v11.pt','u0063_g_0062v07.pt','u0064_g_0063v16.pt','u0065_g_0064v14.pt','u0066_g_0065v13.pt','u0067_g_0066v18.pt','u0068_g_0067v02.pt','u0069_g_0068v24.pt','u0070_g_0069v16.pt','u0071_g_0070v10.pt','u0072_g_0071v05.pt','u0073_g_0072v16.pt','u0074_g_0073v12.pt','u0075_g_0074v09.pt','u0076_g_0075v07.pt','u0077_g_0076v12.pt','u0078_g_0077v12.pt','u0079_g_0078v22.pt','u0080_g_0079v23.pt','u0081_g_0080v03.pt','u0082_g_0081v14.pt','u0083_g_0082v03.pt','u0084_g_0083v17.pt','u0085_g_0084v01.pt','u0086_g_0085v11.pt','u0087_g_0086v04.pt','u0088_g_0087v10.pt','u0089_g_0088v18.pt','u0090_g_0089v03.pt','u0091_g_0090v22.pt','u0092_g_0091v11.pt','u0093_g_0092v19.pt','u0094_g_0093v06.pt','u0095_g_0094v10.pt','u0096_g_0095v20.pt','u0097_g_0096v05.pt','u0098_g_0097v20.pt','u0099_g_0098v10.pt','u0100_g_0099v10.pt']
    refs_dict = {}
    for r in references:
        refs_dict[r.split('_')[0]] = r


    tokens = files.split(" ")
    user_key = tokens[0].split("_")[0]
    # user_key = tokens[0]

    result = math.nan
    refs = []
    sign = ""

    if len(tokens) == 3: result = int(tokens[2]); refs.append(tokens[0]); sign = tokens[1]
    elif len(tokens) == 6: result = int(tokens[5]); refs = tokens[0:4]; sign = tokens[4]
    else: raise ValueError("Arquivos de comparação com formato desconhecido")

    # refs = [refs_dict[refs[0].split('_')[0]]] 

    s_avg = 0
    s_min = 0
    
    dists_query = []
    refs_dists = []
    
    # Obtém a distância entre todas as referências:
    # Isto é: (r1,r2), (r1,r3), (r1,r4), (r2,r3), (r2,r4) e (r3,r4); Obs: o DTW é simétrico, dtw(a,b)==dtw(b,a)
    with torch.no_grad():
        query = load_tensor(sign.split('.')[0] + '.pt')
        for i in range(0,len(refs)):
            x = load_tensor(refs[i].split('.')[0] + '.pt', features_path).cuda()

            noise = torch.randn(opt.batchSize, seq_len, nz, device=device)
            deltas = ref.repeat(opt.batchSize,1,1)
            noise = torch.cat((noise, deltas), dim=2)

            #Generate sequence given noise w/ deltas and deltas
            out_seqs = netG(noise)
            x = out_seqs[0]

            dists_query.append(dtr_eval(x, query, x.shape[0], query.shape[0]).detach().cpu())

            for j in range(i+1, len(refs)):
                y = load_tensor(refs[j].split('.')[0] + '.pt', features_path).cuda()

                refs_dists.append(dtr_eval(x,y,x.shape[0],y.shape[0]))

    refs_dists = np.array(refs_dists)
    dists_query = np.array(dists_query)
    

    """Cálculos de dissimilaridade a partir daqui"""
    dk = 1
    if len(refs_dists) > 1:
        dk = np.mean(refs_dists)
    dk_sqrt = dk ** (1.0/2.0) 

    s_avg = np.mean(dists_query)/dk_sqrt
    s_min = min(dists_query)/dk_sqrt

    return (s_avg + s_min), user_key, result

def evaluate(comparison_file : str, n_epoch : int, result_folder : str, features_path : str, gen):
    """ Avaliação da rede conforme o arquivo de comparação

    Args:
        comparison_file (str): path do arquivo que contém as assinaturas a serem comparadas entre si, bem como a resposta da comparação. 0 é positivo (original), 1 é negativo (falsificação).
        n_epoch (int): número que indica após qual época de treinamento a avaliação está sendo realizada.
        result_folder (str): path onde salvar os resultados.
    """
    
    lines = []
    with open(comparison_file, "r") as fr:
        lines = fr.readlines()

    os.makedirs(result_folder, exist_ok=True)

    file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
    print("\n\tAvaliando " + file_name)
    comparison_folder = result_folder + os.sep + file_name
    if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

    users = {}

    for line in tqdm(lines, "Calculando distâncias..."):
        distance, user_id, true_label = inference(line, gen, features_path=features_path)
        
        if user_id not in users: 
            users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": []}
        else:
            users[user_id]["distances"].append(distance)
            users[user_id]["true_label"].append(true_label)

    # Nesse ponto, todas as comparações foram feitas
    buffer = "user, eer_local, threshold, mean_eer, var_th, amp_th, th_range\n"
    local_buffer = ""
    global_true_label = []
    global_distances = []

    eers = []

    local_ths = []

    # Calculo do EER local por usuário:
    for user in tqdm(users, desc="Obtendo EER local..."):
        global_true_label += users[user]["true_label"]
        global_distances  += users[user]["distances"]

        # if "Task" not in comparison_file:
        if 0 in users[user]["true_label"] and 1 in users[user]["true_label"]:
            eer, eer_threshold = get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])
            th_range_local = np.max(np.array(users[user]["distances"])[np.array(users[user]["distances"]) < eer_threshold])

            local_ths.append(eer_threshold)
            eers.append(eer)
            local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + ", 0, 0, 0, " + "{:.5f}".format(eer_threshold -th_range_local) + " (" + "{:.5f}".format(th_range_local) + "~" + "{:.5f}".format(eer_threshold) + ")\n"

    print("Obtendo EER global...")
    
    # Calculo do EER global
    eer_global, eer_threshold_global = get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

    local_eer_mean = np.mean(np.array(eers))
    local_ths = np.array(local_ths)
    local_ths_var  = np.var(local_ths)
    local_ths_amp  = np.max(local_ths) - np.min(local_ths)
    
    th_range_global = np.max(np.array(global_distances)[np.array(global_distances) < eer_threshold_global])

    buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + ")\n" + local_buffer

    with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
        fw.write(buffer)

    ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
    print (ret_metrics)

    return ret_metrics

def get_epoch():
    # references = ['u0001_g_0000v08.pt','u0002_g_0001v14.pt','u0003_g_0002v17.pt','u0004_g_0003v09.pt','u0005_g_0004v16.pt','u0006_g_0005v14.pt','u0007_g_0006v10.pt','u0008_g_0007v11.pt','u0009_g_0008v23.pt','u0010_g_0009v14.pt','u0011_g_0010v07.pt','u0012_g_0011v14.pt','u0013_g_0012v07.pt','u0014_g_0013v01.pt','u0015_g_0014v08.pt','u0016_g_0015v15.pt','u0017_g_0016v08.pt','u0018_g_0017v08.pt','u0019_g_0018v01.pt','u0020_g_0019v08.pt','u0021_g_0020v08.pt']
    references = ['u0001_g_0000v08.pt','u0002_g_0001v14.pt','u0003_g_0002v17.pt','u0004_g_0003v09.pt','u0005_g_0004v16.pt','u0006_g_0005v14.pt','u0007_g_0006v10.pt','u0008_g_0007v11.pt','u0009_g_0008v23.pt','u0010_g_0009v14.pt','u0011_g_0010v07.pt','u0012_g_0011v14.pt','u0013_g_0012v07.pt','u0014_g_0013v01.pt','u0015_g_0014v08.pt','u0016_g_0015v15.pt','u0017_g_0016v08.pt','u0018_g_0017v08.pt','u0019_g_0018v01.pt','u0020_g_0019v08.pt','u0021_g_0020v08.pt','u0022_g_0021v11.pt','u0023_g_0022v13.pt','u0024_g_0023v01.pt','u0025_g_0024v22.pt','u0026_g_0025v14.pt','u0027_g_0026v16.pt','u0028_g_0027v16.pt','u0029_g_0028v07.pt','u0030_g_0029v10.pt','u0031_g_0030v23.pt','u0032_g_0031v09.pt','u0033_g_0032v24.pt','u0034_g_0033v15.pt','u0035_g_0034v19.pt','u0036_g_0035v11.pt','u0037_g_0036v17.pt','u0038_g_0037v13.pt','u0039_g_0038v21.pt','u0040_g_0039v07.pt','u0041_g_0040v01.pt','u0042_g_0041v17.pt','u0043_g_0042v19.pt','u0044_g_0043v23.pt','u0045_g_0044v06.pt','u0046_g_0045v08.pt','u0047_g_0046v24.pt','u0048_g_0047v03.pt','u0049_g_0048v20.pt','u0050_g_0049v08.pt','u0051_g_0050v07.pt','u0052_g_0051v12.pt','u0053_g_0052v18.pt','u0054_g_0053v06.pt','u0055_g_0054v15.pt','u0056_g_0055v13.pt','u0057_g_0056v23.pt','u0058_g_0057v12.pt','u0059_g_0058v05.pt','u0060_g_0059v16.pt','u0061_g_0060v16.pt','u0062_g_0061v11.pt','u0063_g_0062v07.pt','u0064_g_0063v16.pt','u0065_g_0064v14.pt','u0066_g_0065v13.pt','u0067_g_0066v18.pt','u0068_g_0067v02.pt','u0069_g_0068v24.pt','u0070_g_0069v16.pt','u0071_g_0070v10.pt','u0072_g_0071v05.pt','u0073_g_0072v16.pt','u0074_g_0073v12.pt','u0075_g_0074v09.pt','u0076_g_0075v07.pt','u0077_g_0076v12.pt','u0078_g_0077v12.pt','u0079_g_0078v22.pt','u0080_g_0079v23.pt','u0081_g_0080v03.pt','u0082_g_0081v14.pt','u0083_g_0082v03.pt','u0084_g_0083v17.pt','u0085_g_0084v01.pt','u0086_g_0085v11.pt','u0087_g_0086v04.pt','u0088_g_0087v10.pt','u0089_g_0088v18.pt','u0090_g_0089v03.pt','u0091_g_0090v22.pt','u0092_g_0091v11.pt','u0093_g_0092v19.pt','u0094_g_0093v06.pt','u0095_g_0094v10.pt','u0096_g_0095v20.pt','u0097_g_0096v05.pt','u0098_g_0097v20.pt','u0099_g_0098v10.pt','u0100_g_0099v10.pt']
    random.shuffle(references)

    epoch = []
    
    for file in references:
        batch = []
        file_prefix = file.split("v")[0]
        for i in range(0,25):
            new_file = file_prefix + 'v{:02d}'.format(i) + '.pt'
            if new_file != file:
                batch.append(new_file)
        
        random.shuffle(batch)

        batch = [file] + batch
        aux = []
        for f in batch:
            aux.append(f.replace('v', 'f').replace('g', 's'))

        random.shuffle(aux)

        epoch += batch + aux

    return epoch

def get_batch(epoch, batch_size):
    assert (len(epoch)) % batch_size == 0
    batch = epoch[:batch_size]
    epoch = epoch[batch_size:]
    
    return batch, epoch
    
def load_tensor(tensor_name, features_path = None):
    if features_path is not None:
        return torch.load(os.path.join(features_path,tensor_name))
        
    return torch.load(os.path.join(dataset_folder,tensor_name))
    
def load_batch(batch):
    max_size = -1
    lens = []
    with torch.no_grad():
        loaded_batch = [torch.load(os.path.join(dataset_folder,batch[0]))]
        lens.append(loaded_batch[0].shape[0])

        for sig in batch[1:]:
            query = load_tensor(sig)

            max_size = max(max_size,query.shape[0])

            loaded_batch.append(query)
            lens.append(query.shape[0])
        
        for i, ts in enumerate(loaded_batch):
            loaded_batch[i] = F.pad(ts, (0,0,0,max_size-ts.shape[0]), mode='constant', value=0.0)

    return torch.stack(loaded_batch,dim=0), lens

def discriminator_loss(real, fake, th=torch.tensor(0.14373250305652618)):
    anchor = real[0]
    pos_loss = 0
    neg_loss = 0
    for r in real[1:]:
        pos_loss += F.relu(dtr_eval(anchor, r, anchor.shape[0], r.shape[0]) - th)
    for f in fake:
        neg_loss += F.relu(th - dtr_eval(anchor, f, anchor.shape[0], f.shape[0]))

    return pos_loss + neg_loss


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="btp", help='dataset to use (only btp for now)')
parser.add_argument('--dataset_path', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--nz', type=int, default=64, help='dimensionality of the latent vector z')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='checkpoints', help='folder to save checkpoints')
parser.add_argument('--imf', default='images', help='folder to save images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=5, help='number of epochs after which saving checkpoints') 
parser.add_argument('--tensorboard_image_every', default=5, help='interval for displaying images on tensorboard') 
parser.add_argument('--delta_condition', action='store_true', help='whether to use the mse loss for deltas')
parser.add_argument('--delta_lambda', type=int, default=10, help='weight for the delta condition')
parser.add_argument('--alternate', action='store_true', help='whether to alternate between adversarial and mse loss in generator')
parser.add_argument('--dis_type', default='cnn', choices=['cnn','lstm'], help='architecture to be used for discriminator to use')
parser.add_argument('--gen_type', default='lstm', choices=['cnn','lstm'], help='architecture to be used for generator to use')
opt = parser.parse_args()

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H_%M")
run_name = f"{opt.run_tag}_{date}" if opt.run_tag != '' else date
log_dir_name = os.path.join(opt.logdir, run_name)
writer = SummaryWriter(log_dir_name)
writer.add_text('Options', str(opt), 0)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.imf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")

# if opt.dataset == "btp":
#     dataset = BtpDataset(opt.dataset_path)
# assert dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = opt.nz
#Retrieve the sequence length as first dimension of a sequence in the dataset
# seq_len = dataset[0].size(0)
#An additional input is needed for the delta
in_dim = opt.nz*2

if opt.dis_type == "lstm": 
    netD = LSTMDiscriminator(in_dim=1, hidden_dim=256).to(device)
if opt.dis_type == "cnn":
    netD = CausalConvDiscriminator(input_size=64, n_layers=8, n_channel=25, kernel_size=8, dropout=0).to(device)
if opt.gen_type == "lstm":
    netG = LSTMGenerator(in_dim=in_dim, out_dim=64, hidden_dim=256).to(device)
if opt.gen_type == "cnn":
    netG = CausalConvGenerator(noise_size=in_dim, output_size=1, n_layers=8, n_channel=10, kernel_size=8, dropout=0.2).to(device)
    
assert netG
assert netD

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))    
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

print("|Discriminator Architecture|\n", netD)
print("|Generator Architecture|\n", netG)

# criterion = nn.BCELoss().to(device)
delta_criterion = nn.MSELoss().to(device)

#Generate fixed noise to be used for visualization
# fixed_noise = torch.randn(opt.batchSize, seq_len, nz, device=device)

# if opt.delta_condition:
#     #Sample both deltas and noise for visualization
#     deltas = dataset.sample_deltas(opt.batchSize).unsqueeze(2).repeat(1, seq_len, 1)
#     fixed_noise = torch.cat((fixed_noise, deltas), dim=2)

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)

for e in tqdm(range(opt.epochs)):
    bla = True
    refm = None
    lenrfm = 0

    epoch = get_epoch()

    i = 0
    print("\n\n>>Epoch: " + str(e))
    dists = []
    dists2 = []
    dists3 = []

    disc_loss = []
    gen_loss = []
    delt_loss = []

    pbar = tqdm(total=(len(epoch)//(opt.batchSize)), position=0, leave=True, desc="Epoch " + str(e))
    # with torch.autograd.set_detect_anomaly(True):
    while epoch != []:
        batch_names, epoch = get_batch(epoch=epoch, batch_size=opt.batchSize)
        data,lens = load_batch(batch_names)

        genuines = data[:6]
        forgeries = data[25:5]

        ref = data[0]

        niter = opt.epochs * len(data) + i

    
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        #Train with real data
        netD.zero_grad()
        real = genuines.to(device)
        batch_size, seq_len = real.size(0), real.size(1)

        # output_real = netD(real)
        output_real = real

        # train with skilled forgeries
        # forgeries = forgeries.to(device)
        # batch_size, seq_len = forgeries.size(0), forgeries.size(1)

        # output_forgeries = netD(forgeries)
        # errD_forgeries = discriminator_loss(output_real, output_forgeries)
        # errD_forgeries.backward(retain_graph=True)

        #Train with fake data
        noise = torch.randn(batch_size, seq_len, nz, device=device)
        if opt.delta_condition:
            #Sample a delta for each batch and concatenate to the noise for each timestep
            deltas = ref.repeat(batch_size,1,1)
            noise = torch.cat((noise, deltas), dim=2)

        output_fake = netG(noise)
        output = netD(output_fake.detach())
        
        errD_fake = discriminator_loss(output_real, output)
        disc_loss.append(errD_fake.item())
        errD_fake.backward()
        # errD_fake.backward()

        # errD = errD_fake #+ errD_forgeries
        optimizerD.step()
        
        #Visualize discriminator gradients
        # for name, param in netD.named_parameters():
        #     writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

        # del output_real
        # del output_forgeries

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        output_fake = netD(output_fake)
        output_real2 = netD(real[0].unsqueeze(0))
        
        # errG = discriminator_loss(output_real2, output_fake)
        errG = discriminator_loss(torch.cat((output_real2, output_fake), dim=0), torch.tensor([]))
        gen_loss.append(errG.item())
        errG.backward()
        D_G_z2 = output.mean().item()
        i+=1

        if opt.delta_condition:
            #If option is passed, alternate between the losses instead of using their sum
            if opt.alternate:
                optimizerG.step()
                netG.zero_grad()

            num_syn = 1
            noise = torch.randn(num_syn, seq_len, nz, device=device)
            deltas = ref.repeat(num_syn,1,1)
            noise = torch.cat((noise, deltas), dim=2)
            out_seqs = netG(noise)
            delta_loss = 0

            count = 0
            for syn in out_seqs:
                # for index, sig in enumerate(genuines[1:]):
                #     delta_loss += F.relu(dtr(syn, deltas[0], syn.shape[0], lens[0]) + 1 - dtr(sig, syn, lens[index], syn.shape[0]))
                #     count += 1
                for index, sig in enumerate(forgeries[1:]):
                    delta_loss += F.relu(dtr(syn, deltas[0], syn.shape[0], lens[0]) + 1 - dtr(sig, syn, lens[index], syn.shape[0]))
                    count += 1
            
            delta_loss /= count
            delt_loss.append(delta_loss.detach().cpu().numpy()[0])
            if delta_loss <= 0:
                print("Loss menor que 0")
            delta_loss.backward()
        
        if bla: 
            refm = output_fake[0]
            bla = False

        optimizerG.step()
        
        #Visualize generator gradients
        for name, param in netG.named_parameters():
            writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
        
        ###########################
        # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
        ###########################

        #Report metrics
        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
        #       % (epoch, opt.epochs, i, len(dataloader),
        #          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
        # if opt.delta_condition:
        #     writer.add_scalar('MSE of deltas of generated sequences', delta_loss.item(), niter)
        #     print(' DeltaMSE: %.4f' % (delta_loss.item()/opt.delta_lambda), end='')
        # writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
        # writer.add_scalar('GeneratorLoss', errG.item(), niter)
        # writer.add_scalar('D of X', D_x, niter) 
        # writer.add_scalar('D of G of z', D_G_z1, niter)
        
        with torch.no_grad():
            res = dte(ref, output_fake[0], lens[0], lens[0]) * 2
            dists.append(batch_names[0] + ':\t\t' + str(res))

            res2 = dte(refm, ref, lens[0], lens[0]) * 2
            dists2.append(batch_names[0] + ':\t\t' + str(res2))

            v = random.randint(1,24)
            res3 = dte(data[v], output_fake[0], lens[v], lens[0]) * 2
            dists3.append(batch_names[0] + ':\t\t' + str(res3))
        pbar.update(1)
        # break #!!!
    pbar.close()
    ##### End of the epoch #####
    # real_plot = time_series_to_plot(dataset.denormalize(real_display))
    # if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
    #     writer.add_image("Real", real_plot, epoch)
    
    # fake = netG(fixed_noise)
    # fake_plot = time_series_to_plot(dataset.denormalize(fake))
    # torchvision.utils.save_image(fake_plot, os.path.join(opt.imf, opt.run_tag+'_epoch'+str(epoch)+'.jpg'))
    # if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
    #     writer.add_image("Fake", fake_plot, epoch)
                             
    # Checkpoint
    dists = sorted(dists)
    print("Dela sob medida")
    for s in dists:
        print(s)

    dists2 = sorted(dists2)
    print("Delta de um mesmo usuário para todos")
    for s in dists2:
        print(s)

    dists3 = sorted(dists3)
    print("Delta sob medida, mas comparando com alguma assinatura original aleatória do usuário do delta")
    for s in dists3:
        print(s)

    UPDATED_LOWER_BOUND = '.' + os.sep + "updated_protocol_lower_bound.txt"
    evaluate(UPDATED_LOWER_BOUND, e, result_folder=opt.outf, features_path=dataset_folder, gen=netG)
    MCYT_SKILLED_1VS1 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "skilled" + os.sep + "Comp_MCYT_skilled_stylus_1vs1.txt"
    
    if (e+1) % 10 == 0: evaluate(MCYT_SKILLED_1VS1, e, result_folder=opt.outf, features_path=dataset_folder, gen=netG)
    MCYT_RANDOM_1VS1 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "random" + os.sep + "Comp_MCYT_random_stylus_1vs1.txt"
    # if (e+1) % 5 == 0: evaluate(MCYT_RANDOM_1VS1, e, result_folder=opt.outf, features_path=dataset_folder, gen=netG)
    

    if (e % opt.checkpoint_every == 0) or (e == (opt.epochs - 1)):
        torch.save(netG, '%s/%s_netG_epoch_%d.pth' % (opt.outf, opt.run_tag, e))
        torch.save(netD, '%s/%s_netD_epoch_%d.pth' % (opt.outf, opt.run_tag, e))

    print("\n\n")
    print("Discriminator Loss:\t\t" + str(np.mean(np.array(disc_loss))))
    print("Generator Loss....:\t\t" + str(np.mean(np.array(gen_loss))))
    print("Delta Loss.......:\t\t" + str(np.mean(np.array(delt_loss))))