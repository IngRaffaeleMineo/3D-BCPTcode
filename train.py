#
from multiprocessing.sharedctypes import Value
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import argparse
from pathlib import Path
from tqdm import tqdm
from utils.dataset import get_loader
from utils.models.models import get_model
from utils.trainer import Trainer
from utils.saver import Saver
import glob
from utils import utils
import torch
import GPUtil

#torch.autograd.set_detect_anomaly(False) non è detto che se c'è quale nan bisogna bloccare subito l'allenamento

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, default='data\\MY_DATASET')
    parser.add_argument('--split_path', type=Path, default='data\\MYDATASET.json')
    parser.add_argument('--num_fold', type=int, default=0)
    parser.add_argument('--inner_loop', type=int, default=0)
    parser.add_argument('--cache_rate', type=float, default=1.0)

    parser.add_argument('--dataset3d', type=int, choices=[0,1], default=1)
    parser.add_argument('--dataset2d', type=int, choices=[0,1], default=0)
    parser.add_argument('--resize', type=int, nargs=3, default=[-1,256,256]) #(-1,224,224) se i3d
    parser.add_argument('--pad', type=int, nargs=3, default=[60,-1,-1])
    parser.add_argument('--datasetGrid', type=int, choices=[0,1], default=0)
    parser.add_argument('--datasetGridPatches', type=int, default=16)
    parser.add_argument('--datasetGridStride', type=int, default=4)
    parser.add_argument('--mean', type=float, nargs=3, default=[0.43216, 0.394666, 0.37645])
    parser.add_argument('--std', type=float, nargs=3, default=[0.22803, 0.22145, 0.216989])
    parser.add_argument('--inputChannel', type=int, default=1)
    parser.add_argument('--doppiaAngolazioneInput', type=int, choices=[0,1], default=0)
    #parser.add_argument('--keyframeInput', type=bool, default=True)

    parser.add_argument('--model', type=str, default='resnet3d_pretrained')#resnet3d,MVCNN,GVCNN,ViT_B_16,s3d_pretrained,ViT_3D,VideoSwinTransformer,... see utils/models.py

    parser.add_argument('--enable_datiClinici', type=int, choices=[0,1], default=0)
    parser.add_argument('--len_datiClinici', type=int, default=64)
    parser.add_argument('--enable_doppiaAngolazione', type=int, choices=[0,1], default=0)
    parser.add_argument('--enable_keyframe', type=int, choices=[0,1], default=0)
    parser.add_argument('--reduceInChannel', type=int, choices=[0,1], default=1)
    parser.add_argument('--enableGlobalMultiHeadAttention', type=int, choices=[0,1], default=0)
    parser.add_argument('--enableTemporalMultiHeadAttention', type=int, choices=[0,1], default=0)
    parser.add_argument('--enableSpacialTemporalTransformerEncoder', type=int, choices=[0,1], default=0)
    parser.add_argument('--numLayerTransformerEncoder', type=int, default=8)
    parser.add_argument('--numHeadMultiHeadAttention', type=int, default=16)

    parser.add_argument('--gradient_clipping_value', type=int, default=0)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'LBFGS'], default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--enable_scheduler', type=int, choices=[0,1], default=0)
    parser.add_argument('--scheduler_factor', type=float, default=8e-2)
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--scheduler_threshold', type=float, default=1e-2)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--tensorboard_port', type=int, default=6006)
    parser.add_argument('--start_tensorboard_server', type=int, choices=[0,1], default=0)
    parser.add_argument('--ckpt_every', type=int, default=-1)
    parser.add_argument('--resume', default=None) # '.\logs\LOGS1\ckpt\LAST_CKECKPOINT.pth
    parser.add_argument('--save_image_file', type=int, default=0)

    parser.add_argument('--enable_cudaAMP', type=int, choices=[0,1], default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distributed', type=int, choices=[0,1], default=1)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if (args.model == 'i3d') or (args.model == 'i3d_pretrained') or (args.model == 'GVCNN') or (args.model == 'VideoSwinTransformer'):
        if args.resize != [-1,224,224]:
            raise RuntimeError("To use I3D/GVCNN/VideoSwinTransformer model, the dataset resize size must be [-1,224,224].")
    elif args.model == 'ViT_B_16':
        if args.resize != [-1,128,128]:
            raise RuntimeError("To use ViT_B_16 model, the dataset resize size must be [-1,128,128].")
    elif args.model == 'ViT_3D':
        if (args.resize != [-1,64,64]) and (args.pad != [61,-1,-1]) and (args.datasetGridPatches != 16) and (args.datasetGridStride != 4):
            raise RuntimeError("To use ViT_3D model, the dataset resize size must be [-1,64,64] with pad [61,-1,-1] with 16 patched and stride 4.")
    else:
        if args.resize != [-1,256,256]:
            raise RuntimeError("To best performance, the dataset resize size should be [-1,256,256].")
    
    if args.model == 'ViT_3D':
        if (args.dataset2d) or (not args.datasetGrid):
            raise RuntimeError("To use ViT_3D model, the dataset must be 3d into 2d grid.")
    else:
        if args.datasetGrid:
            raise RuntimeError("Don't use datasetGrid.")

    if (args.model == 'MVCNN') or (args.model == 'GVCNN') or (args.model == 'ViT_B_16') or (args.model == 'ViT_3D') or (args.model == 'VideoSwinTransformer'):
        if args.inputChannel != 3:
            raise RuntimeError("MVCNN/GVCNN/ViT_B_16/ViT_3D/VideoSwinTransformer require 3 channel input.")
        if args.reduceInChannel:
            raise RuntimeError("MVCNN/GVCNN/ViT_B_16/ViT_3D/VideoSwinTransformer not implement reduceInChannel.")
    else:
        if args.reduceInChannel and (args.inputChannel != 1):
            raise RuntimeError("ReduceInChannel require 1 channel input.")

    if args.enable_doppiaAngolazione and not args.doppiaAngolazioneInput:
        raise RuntimeError("Multibranch input require multi views dataset.")

    # Convert boolean (as integer) args to boolean type
    if args.dataset3d == 0:
        args.dataset3d = False
    else:
        args.dataset3d = True
    if args.dataset2d == 0:
        args.dataset2d = False
    else:
        args.dataset2d = True
    if args.datasetGrid == 0:
        args.datasetGrid = False
    else:
        args.datasetGrid = True
    if args.doppiaAngolazioneInput == 0:
        args.doppiaAngolazioneInput = False
    else:
        args.doppiaAngolazioneInput = True
    if args.enable_datiClinici == 0:
        args.enable_datiClinici = False
    else:
        args.enable_datiClinici = True
    if args.enable_doppiaAngolazione == 0:
        args.enable_doppiaAngolazione = False
    else:
        args.enable_doppiaAngolazione = True
    if args.enable_keyframe == 0:
        args.enable_keyframe = False
    else:
        args.enable_keyframe = True
    if args.reduceInChannel == 0:
        args.reduceInChannel = False
    else:
        args.reduceInChannel = True
    if args.enableGlobalMultiHeadAttention == 0:
        args.enableGlobalMultiHeadAttention = False
    else:
        args.enableGlobalMultiHeadAttention = True
    if args.enableTemporalMultiHeadAttention == 0:
        args.enableTemporalMultiHeadAttention = False
    else:
        args.enableTemporalMultiHeadAttention = True
    if args.enableSpacialTemporalTransformerEncoder == 0:
        args.enableSpacialTemporalTransformerEncoder = False
    else:
        args.enableSpacialTemporalTransformerEncoder = True
    if args.enable_scheduler == 0:
        args.enable_scheduler = False
    else:
        args.enable_scheduler = True
    if args.start_tensorboard_server == 0:
        args.start_tensorboard_server = False
    else:
        args.start_tensorboard_server = True
    if args.save_image_file == 0:
        args.save_image_file = False
    else:
        args.save_image_file = True
    if args.enable_cudaAMP == 0:
        args.enable_cudaAMP = False
    else:
        args.enable_cudaAMP = True
    if args.distributed == 0:
        args.distributed = False
    else:
        args.distributed = True

    # Generate experiment tags if not defined
    if args.experiment == None:
        args.experiment = args.model
    
    # Create other attributes that for now is not useful
    args.keyframeInput = args.enable_keyframe

    # Define pads automatically
    if args.pad[1] == -1:
        args.pad = [args.pad[0],args.resize[1],args.pad[2]]
    if args.pad[2] == -1:
        args.pad = [args.pad[0],args.pad[1],args.resize[2]]
    
    '''if (args.normalization_datiClinici != None):
        with open(os.path.join(args.normalization_datiClinici)) as fp:
            normalizations = json.load(fp)
        args.means_datiClinici = normalizations["means"]
        args.stds_datiClinici = normalizations["stds"]
    else:
        args.means_datiClinici = None
        args.stds_datiClinici = None'''
    return args


# disable printing when not in master process
import builtins as __builtin__
builtin_print = __builtin__.print
def print_mod(*args, **kwargs):
    force = kwargs.pop('force', False)
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
    else:
        RuntimeError("No RANK found!")
    if (rank==0) or force:
        builtin_print(*args, **kwargs)


def main():
    args = parse()

    # choose device
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            raise RuntimeError("Can't use distributed mode! Check if you don't run correct command: 'python -m torch.distributed.launch --nproc_per_node=num_gpus --use_env train.py'")
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'gloo' # 'nccl'
        print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        device = torch.device(args.gpu)
        # disable printing when not in master process
        __builtin__.print = print_mod
    else:
        if args.device == 'cuda': # choose the most free gpu
            #mem = [(torch.cuda.memory_allocated(i)+torch.cuda.memory_reserved(i)) for i in range(torch.cuda.device_count())]
            mem = [gpu.memoryUtil for gpu in GPUtil.getGPUs()]
            args.device = 'cuda:' + str(mem.index(min(mem)))
        device = torch.device(args.device)
        print('Using device', args.device)

    # Dataset e Loader
    print("Dataset: balanced nested cross-validation use fold (test-set) " + str(args.num_fold) + " and inner_loop (validation-set) " + str(args.inner_loop) + ".")
    loaders, samplers, loss_weights = get_loader(args)

    # Model
    model = get_model(num_classes=2,
                        model_name=args.model,
                        enable_datiClinici=args.enable_datiClinici,
                        in_dim_datiClinici=args.len_datiClinici if args.enable_datiClinici else None,
                        enable_doppiaAngolazione=args.enable_doppiaAngolazione,
                        enable_keyframe=args.enable_keyframe,
                        reduceInChannel=args.reduceInChannel,
                        enableGlobalMultiHeadAttention=args.enableGlobalMultiHeadAttention,
                        enableTemporalMultiHeadAttention=args.enableTemporalMultiHeadAttention,
                        enableSpacialTemporalTransformerEncoder=args.enableSpacialTemporalTransformerEncoder,
                        numLayerTransformerEncoder=args.numLayerTransformerEncoder,
                        numHeadMultiHeadAttention=args.numHeadMultiHeadAttention,
                        loss_weights=loss_weights)
    if args.resume is not None:
        model.load_state_dict(Saver.load_model(args['resume']))
    model.to(device)

    # Enable model distribuited if it is
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:', args.model, '(number of params:', n_parameters, ')')

    # Create optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model_without_ddp.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model_without_ddp.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS(params=model_without_ddp.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Optimizer chosen not implemented!")
    
    # Create scheduler
    if args.enable_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='min',
                                                                factor=args.scheduler_factor,
                                                                patience=args.scheduler_patience,
                                                                threshold=args.scheduler_threshold,
                                                                threshold_mode='rel',
                                                                cooldown=0,
                                                                min_lr=0,
                                                                eps=1e-08,
                                                                verbose=True)

    if args.enable_cudaAMP:
        # Creates GradScaler for CUDA AMP
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Trainer
    class_trainer = Trainer(net=model,
                            class_weights=torch.Tensor(loss_weights).to(device),
                            optim=optimizer,
                            gradient_clipping_value=args.gradient_clipping_value,
                            enable_datiClinici=args.enable_datiClinici,
                            doppiaAngolazioneInput=args.doppiaAngolazioneInput,
                            keyframeInput=args.keyframeInput,
                            scaler=scaler)

    # Saver
    if (not args.distributed) or (args.distributed and (args.rank==0)):
        saver = Saver(Path(args.logdir),
                        vars(args),
                        sub_dirs=list(loaders.keys()),
                        tag=args.experiment)
    else:
        saver = None

    tot_predicted_labels_last = {split:{} for split in loaders}
    if (saver is not None) and (args.ckpt_every <= 0):
        max_validation_accuracy_balanced = 0
        max_test_accuracy_balanced = 0
        save_this_epoch = False
    for epoch in range(args.epochs):
        try:
            for split in loaders:
                if args.distributed:
                    samplers[split].set_epoch(epoch)

                data_loader = loaders[split]

                epoch_metrics = {}
                tot_true_labels = []
                tot_predicted_labels = []
                tot_predicted_scores = []
                tot_image_paths = []
                for batch in tqdm(data_loader, desc=f'{split}, {epoch}/{args.epochs}'):
                    labels, image_paths = batch['label'], batch['image']
                    
                    images_3d = None
                    doppiaAngolazione_3d = None
                    if args.dataset3d:
                        images_3d = batch['image_3d']
                        if args.doppiaAngolazioneInput:
                            doppiaAngolazione_3d = batch['image2_3d']

                    images_2d = None
                    doppiaAngolazione_2d = None
                    if args.dataset2d:
                        images_2d = batch['image_2d']
                        if args.doppiaAngolazioneInput:
                            doppiaAngolazione_2d = batch['image2_2d']
                    
                    datiClinici = None
                    if args.enable_datiClinici:
                        datiClinici = torch.cat((batch['age'], batch['sex']), dim=1)
                    
                    tot_true_labels.extend(labels.tolist())
                    tot_image_paths.extend(image_paths)

                    if args.dataset3d:
                        images_3d = images_3d.to(device)
                        if args.doppiaAngolazioneInput:
                            doppiaAngolazione_3d = doppiaAngolazione_3d.to(device)
                    if args.dataset2d:
                        images_2d = images_2d.to(device)
                        if args.doppiaAngolazioneInput:
                            doppiaAngolazione_2d = doppiaAngolazione_2d.to(device)
                    
                    if args.enable_datiClinici:
                        datiClinici = datiClinici.to(device)

                    labels = labels.to(device)

                    returned_values = class_trainer.forward_batch(images_3d, images_2d, labels, datiClinici, doppiaAngolazione_3d, doppiaAngolazione_2d, split)
                    metrics_dict, (predicted_labels, predicted_scores) = returned_values
                    
                    tot_predicted_labels.extend(predicted_labels.tolist())
                    tot_predicted_scores.extend(predicted_scores.tolist())
                    
                    for k, v in metrics_dict.items():
                        epoch_metrics[k]= epoch_metrics[k] + [v] if k in epoch_metrics else [v]
                
                # Run scheduler
                if args.enable_scheduler and split=="train":
                    scheduler.step(sum(epoch_metrics['loss'])/len(epoch_metrics['loss']))

                # Print metrics
                for k, v in epoch_metrics.items():
                    avg_v = sum(v)/len(v)
                    if args.distributed:
                        torch.distributed.barrier()
                        avg_v_output = [None for _ in range(args.world_size)]
                        torch.distributed.all_gather_object(avg_v_output, avg_v)
                        avg_v = sum(avg_v_output)/len(avg_v_output)
                    if saver is not None:
                        saver.log_scalar("Classifier Epoch/"+k+"_"+split, avg_v, epoch)
                
                if args.distributed:
                    torch.distributed.barrier()

                    tot_true_labels_output = [None for _ in range(args.world_size)]
                    tot_predicted_labels_output = [None for _ in range(args.world_size)]
                    tot_predicted_scores_output = [None for _ in range(args.world_size)]
                    tot_image_paths_output = [None for _ in range(args.world_size)]

                    torch.distributed.all_gather_object(tot_true_labels_output, tot_true_labels)
                    torch.distributed.all_gather_object(tot_predicted_labels_output, tot_predicted_labels)
                    torch.distributed.all_gather_object(tot_predicted_scores_output, tot_predicted_scores)
                    torch.distributed.all_gather_object(tot_image_paths_output, tot_image_paths)

                    tot_true_labels=[]
                    tot_predicted_labels=[]
                    tot_predicted_scores=[]
                    tot_image_paths=[]
                    for i in range(len(tot_true_labels_output)):
                        tot_true_labels.extend(tot_true_labels_output[i])
                        tot_predicted_labels.extend(tot_predicted_labels_output[i])
                        tot_predicted_scores.extend(tot_predicted_scores_output[i])
                        tot_image_paths.extend(tot_image_paths_output[i])
                
                if saver is not None:
                    # Accuracy Balanced classification
                    accuracy_balanced = utils.calc_accuracy_balanced_classification(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch/accuracy_balanced_"+split, accuracy_balanced, epoch)
                    if (saver is not None) and (args.ckpt_every <= 0):
                        if (split == "validation") and (accuracy_balanced >= max_validation_accuracy_balanced):
                            max_validation_accuracy_balanced = accuracy_balanced
                            save_this_epoch = True
                        if (split == "test") and (accuracy_balanced >= max_test_accuracy_balanced):
                            max_test_accuracy_balanced = accuracy_balanced
                            save_this_epoch = True
                    
                    # Accuracy classification
                    accuracy = utils.calc_accuracy_classification(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch "+split+"/accuracy_"+split, accuracy, epoch)

                    # Precision
                    precision = utils.calc_precision(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Precision", precision, epoch)

                    # Recall
                    recall = utils.calc_recall(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Recall", recall, epoch)

                    # Specificity
                    specificity = utils.calc_specificity(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Specificity", specificity, epoch)

                    # F1 Score
                    f1score = utils.calc_f1(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"F1 Score", f1score, epoch)

                    # AUC
                    auc = utils.calc_auc(tot_true_labels, tot_predicted_scores)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"AUC", auc, epoch)

                    # ROC Curve
                    rocCurve_image = utils.calc_rocCurve(tot_true_labels, tot_predicted_scores)
                    saver.log_images("Classifier Epoch "+split+"/"+"ROC Curve", rocCurve_image, epoch, split, "ROCcurve", args.save_image_file)

                    # Precision-Recall Curve
                    precisionRecallCurve_image = utils.calc_precisionRecallCurve(tot_true_labels, tot_predicted_scores)
                    saver.log_images("Classifier Epoch "+split+"/"+"Precision-Recall Curve", precisionRecallCurve_image, epoch, split, "PrecisionRecallCurve", args.save_image_file)
                    
                    # Prediction Agreement Rate: concordanza valutazione stesso campione tra epoca corrente e precedente
                    predictionAgreementRate, tot_predicted_labels_last[split] = utils.calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last[split], tot_image_paths)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Prediction Agreement Rate", predictionAgreementRate, epoch)
        
                    # Confusion Matrix
                    cm_image = utils.plot_confusion_matrix(tot_true_labels, tot_predicted_labels, ['negative', 'positive'], title="Confusion matrix "+split)
                    saver.log_images("Classifier Epoch "+split+"/"+"Confusion Matrix", cm_image, epoch, split, "ConfMat", args.save_image_file)

                    # Save logs of error
                    saver.saveLogsError(tot_true_labels, tot_predicted_labels, tot_predicted_scores, {'image_path':tot_image_paths}, split, epoch)
                
                # Save checkpoint
                if args.distributed:
                    torch.distributed.barrier()
                if saver is not None:
                    if args.ckpt_every > 0:
                        if (split ==  "train") and (epoch % args.ckpt_every == 0):
                            saver.save_model(model_without_ddp, args.experiment, epoch)
                    else: # args.ckpt_every <= 0
                        if save_this_epoch:
                            for filename in glob.glob(str(saver.ckpt_path / (args.experiment+"_best_"+split+"_*"))):
                                os.remove(filename)
                            saver.save_model(model_without_ddp, args.experiment+"_best_"+split, epoch)
                        save_this_epoch = False
        except KeyboardInterrupt:
            print('Caught Keyboard Interrupt: exiting...')
            break

    # Save last checkpoint
    if args.distributed:
        torch.distributed.barrier()
    if saver is not None:
        if args.ckpt_every > 0:
            saver.save_model(model_without_ddp, args.experiment, epoch)
        saver.close()

    if args.start_tensorboard_server:
        print("Finish (Press CTRL+C to close tensorboard and quit)")
    else:
        print("Finish")

    if args.distributed:
        torch.distributed.destroy_process_group()
        

if __name__ == '__main__':
    main()