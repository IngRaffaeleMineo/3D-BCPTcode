import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import argparse
from tqdm import tqdm
from utils.dataset import get_loader
from utils.models.models import get_model
from utils.trainer import Trainer
from utils.saver import Saver
from utils import utils
import torch
import GPUtil

# Graph visualization on browser
import socket
import threading
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['webagg.address'] = '127.0.0.1'
matplotlib.rcParams['webagg.open_in_browser'] = False
matplotlib.rcParams['figure.max_open_warning'] = 0
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from matplotlib import pyplot as plt
import webbrowser

# Explainability M3d-Cam
import scipy
import PIL
import io
from medcam import medcam
from medcam.backends import base as medcam_backends_base
from captum.attr import LayerAttribution

# Explainability pytorch-gradcam-book
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='logs\\MY_FOLDER1')
    parser.add_argument('--cache_rate', type=float, default=0.0)
    parser.add_argument('--start_tensorboard_server', type=bool, default=False)
    parser.add_argument('--tensorboard_port', type=int, default=6006)
    parser.add_argument('--start_tornado_server', type=bool, default=False)
    parser.add_argument('--tornado_port', type=int, default=8800) # matplotlib web interface
    parser.add_argument('--saveLogs', type=bool, default=True)
    parser.add_argument('--enable_explainability', type=bool, default=True)
    parser.add_argument('--explainability_model_multiOutput', type=bool, default=False)
    parser.add_argument('--explainability_mode', type=str, choices=['medcam', 'pytorchgradcambook'], default='pytorchgradcambook')

    parser.add_argument('--enable_cudaAMP', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
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
    # Load configuration
    args = parse()

    # Check logdir if is a dir
    if not os.path.isdir(args.logdir):
        raise EnvironmentError('logdir must be an existing dir.')
    
    # Check/Mod batch_size
    if args.enable_explainability:
        if args.distributed:
            raise RuntimeError("Please not use distribuited mode when explainability enabled for too many ram usage.")
        args.batch_size = 1 # mandatory 1 if explainability
    
    # Load hyperparameters
    args2 = Saver.load_hyperparams(args.logdir)
    args = vars(args)
    for key in args:
        if key in args2:
            del args2[key]
    args.update(args2)
    args = argparse.Namespace(**args)

    # Choose device
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            raise RuntimeError("Can't use distributed mode! Check if you don't run correct command: 'python -m torch.distributed.launch --nproc_per_node=num_gpus --use_env test.py'")
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

    # Print hyperparameters
    if (not args.distributed) or (args.distributed and (args.rank==0)):
        print("---Configs/Hyperparams---")
        for key in vars(args):
            print(key+":", vars(args)[key])
        print("-------------------------")

    # Dataset e Loader
    print("Dataset: balanced nested cross-validation use fold (test-set) " + str(args.num_fold) + " and inner_loop (validation-set) " + str(args.inner_loop) + ".")
    loaders, samplers, loss_weights = get_loader(args)
    del loaders['train']
    del loaders['validation']

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
    checkpoint, epoch = Saver.load_model(args.logdir, return_epoch=True)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

    # Enable explainability on model
    if args.enable_explainability:
        if args.explainability_mode == 'medcam':
            if args.explainability_model_multiOutput:
                # Modify _BaseWrapper.forward() functin in /site-packages/medcam/backends/base.py to work with model's outputs
                def forward_modding(self, batch):
                    """Calls the forward() of the model."""
                    self.model.zero_grad()
                    outputs = self.model.model_forward(batch)
                    self.logits = outputs[0]
                    self._extract_metadata(batch, self.logits)
                    self._set_postprocessor_and_label(self.logits)
                    self.remove_hook(forward=True, backward=False)
                    return outputs
                medcam_backends_base._BaseWrapper.forward = forward_modding
            # Inject model to get attention maps
            #print(medcam.get_layers())
            model = medcam.inject(model, backend='gcampp', save_maps=False, layer='auto') # layer='auto'/'full'
        elif args.explainability_mode == 'pytorchgradcambook':
            #def find_layer_predicate_recursive(model, prefix=''):
            #    for name, layer in model._modules.items():
            #        tmp=prefix+'.'+name
            #        print(tmp)
            #        find_layer_predicate_recursive(layer, tmp)
            #find_layer_predicate_recursive(model)
            cam = GradCAM(model=model, target_layers=[model.avgpool[1].layers[-1]], use_cuda=(args.device!='cpu'), reshape_transform=None)
    
    # Enable model distribuited if it is
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:', args.model, '(number of params:', n_parameters, ')')
    
    if args.enable_cudaAMP:
        # Creates GradScaler for CUDA AMP
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if (not args.distributed) or (args.distributed and (args.rank==0)):
        # TensorBoard Daemon
        if args.start_tensorboard_server:
            tensorboard_port = args.tensorboard_port
            i = 0
            while(True):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    i += 1
                    if s.connect_ex(('localhost', tensorboard_port)) == 0: # check if port is busy
                        tensorboard_port = tensorboard_port + 1
                    else:
                        break
                    if i > 100:
                        raise RuntimeError('Tensorboard: can not find free port at +100 from your chosen port!')
            t = threading.Thread(target=lambda: os.system('tensorboard --logdir=' + str(args.logdir) + ' --port=' + str(tensorboard_port)))
            t.start()
            webbrowser.open('http://localhost:' + str(tensorboard_port) + '/', new=1)

        # Setup server per matplotlib
        if args.start_tornado_server:
            tornado_port = args.tornado_port
            i = 0
            while(True):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    i += 1
                    if s.connect_ex(('localhost', tornado_port)) == 0: # check if port is busy
                        tornado_port = tornado_port + 1
                    else:
                        break
                    if i > 100:
                        raise RuntimeError('Tornado (matplotlib web interface): can not find free port at +100 from your chosen port!')
            matplotlib.rcParams['webagg.port'] = tornado_port

    tot_predicted_labels_last = {split:{} for split in loaders}
    for split in loaders:
        if args.distributed:
            samplers[split].set_epoch(0)

        data_loader = loaders[split]

        if args.enable_explainability:
            if args.dataset3d:
                tot_images_3d = []
                tot_images_3d_gradient = []
            if args.dataset2d:
                tot_images_2d = []
                tot_images_2d_gradient = []
        tot_true_labels = []
        tot_predicted_labels = []
        tot_predicted_scores = []
        tot_image_paths = []
        for batch in tqdm(data_loader, desc=f'{split}'):
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

            if args.enable_explainability:
                if args.dataset3d:
                    tot_images_3d.extend(images_3d.tolist())
                if args.dataset2d:
                    tot_images_2d.extend(images_2d.tolist())
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

            if args.enable_explainability:
                if args.dataset3d:
                    images_3d_clone = images_3d.clone().detach()
                    images_3d_clone = images_3d_clone.to(device)
                if args.dataset2d:
                    images_2d_clone = images_2d.clone().detach()
                    images_2d_clone = images_2d_clone.to(device)

            labels = labels.to(device)
            
            returned_values = Trainer.forward_batch_testing(net=model,
                                                            imgs_3d=images_3d,
                                                            imgs_2d=images_2d,
                                                            datiClinici=datiClinici,
                                                            doppiaAngolazione_3d=doppiaAngolazione_3d,
                                                            doppiaAngolazione_2d=doppiaAngolazione_2d,
                                                            enable_datiClinici=args.enable_datiClinici,
                                                            doppiaAngolazioneInput=args.doppiaAngolazioneInput,
                                                            input3d=args.input3d,
                                                            input2d=args.input2d,
                                                            scaler=scaler)
            predicted_labels, predicted_scores = returned_values
            
            tot_predicted_labels.extend(predicted_labels.tolist())
            tot_predicted_scores.extend(predicted_scores.tolist())
            
            if args.enable_explainability:
                if args.explainability_mode == 'medcam':
                    if args.distributed:
                        upsamp_attr_lgc = LayerAttribution.interpolate(torch.from_numpy(model.module.get_attention_map()), images_3d.shape[2:])
                    else:
                        upsamp_attr_lgc = LayerAttribution.interpolate(torch.from_numpy(model.get_attention_map()), images_3d.shape[2:])
                    upsamp_attr_lgc = upsamp_attr_lgc.cpu().detach().numpy()
                    tot_images_3d_gradient.extend(upsamp_attr_lgc.tolist())
                elif args.explainability_mode == 'pytorchgradcambook':
                    # targets =  specify the target to generate the Class Activation Maps
                    grayscale_cam = cam(input_tensor=images_3d_clone, targets=[ClassifierOutputTarget(1)], aug_smooth=True, eigen_smooth=True)
                    grayscale_cam = grayscale_cam.cpu().detach().numpy()
                    tot_images_3d_gradient.extend(grayscale_cam.tolist())
        
        if args.distributed:
            torch.distributed.barrier()

            if args.enable_explainability:
                if args.dataset3d:
                    tot_images_3d_output = [None for _ in range(args.world_size)]
                    tot_images_3d_gradient_output = [None for _ in range(args.world_size)]
                if args.dataset2d:
                    tot_images_2d_output = [None for _ in range(args.world_size)]
                    tot_images_2d_gradient_output = [None for _ in range(args.world_size)]
            tot_true_labels_output = [None for _ in range(args.world_size)]
            tot_predicted_labels_output = [None for _ in range(args.world_size)]
            tot_predicted_scores_output = [None for _ in range(args.world_size)]
            tot_image_paths_output = [None for _ in range(args.world_size)]

            if args.enable_explainability:
                print("Gathering volumes...")
                if args.dataset3d:
                    torch.distributed.all_gather_object(tot_images_3d_output, tot_images_3d)
                if args.dataset2d:
                    torch.distributed.all_gather_object(tot_images_2d_output, tot_images_2d)
                print("Gathering volume's gradients...")
                if args.dataset3d:
                    torch.distributed.all_gather_object(tot_images_3d_gradient_output, tot_images_3d_gradient)
                if args.dataset2d:
                    torch.distributed.all_gather_object(tot_images_2d_gradient_output, tot_images_2d_gradient)
            torch.distributed.all_gather_object(tot_true_labels_output, tot_true_labels)
            torch.distributed.all_gather_object(tot_predicted_labels_output, tot_predicted_labels)
            torch.distributed.all_gather_object(tot_predicted_scores_output, tot_predicted_scores)
            torch.distributed.all_gather_object(tot_image_paths_output, tot_image_paths)

            if args.enable_explainability:
                if args.dataset3d:
                    tot_images_3d = []
                    tot_images_3d_gradient = []
                if args.dataset2d:
                    tot_images_2d = []
                    tot_images_2d_gradient = []
            tot_true_labels=[]
            tot_predicted_labels=[]
            tot_predicted_scores=[]
            tot_image_paths=[]
            for i in range(len(tot_true_labels_output)):
                if args.enable_explainability:
                    if args.dataset3d:
                        tot_images_3d.extend(tot_images_3d_output[i])
                        tot_images_3d_gradient.extend(tot_images_3d_gradient_output[i])
                    if args.dataset2d:
                        tot_images_2d.extend(tot_images_2d_output[i])
                        tot_images_2d_gradient.extend(tot_images_2d_gradient_output[i])
                tot_true_labels.extend(tot_true_labels_output[i])
                tot_predicted_labels.extend(tot_predicted_labels_output[i])
                tot_predicted_scores.extend(tot_predicted_scores_output[i])
                tot_image_paths.extend(tot_image_paths_output[i])
        
        if (not args.distributed) or (args.distributed and (args.rank==0)):
            # Accuracy Balanced classification
            accuracy_balanced = utils.calc_accuracy_balanced_classification(tot_true_labels, tot_predicted_labels)
            print(split, epoch, "epoch - Accuracy Balanced:", accuracy_balanced)
            
            # Accuracy classification
            accuracy = utils.calc_accuracy_classification(tot_true_labels, tot_predicted_labels)
            print(split, epoch, "epoch - Accuracy:", accuracy)

            # Precision
            precision = utils.calc_precision(tot_true_labels, tot_predicted_labels)
            print(split, epoch, "epoch - Precision:", precision)

            # Recall
            recall = utils.calc_recall(tot_true_labels, tot_predicted_labels)
            print(split, epoch, "epoch - Recall:", recall)

            # Specificity
            specificity = utils.calc_specificity(tot_true_labels, tot_predicted_labels)
            print(split, epoch, "epoch - Specificity:", specificity)

            # F1 Score
            f1score = utils.calc_f1(tot_true_labels, tot_predicted_labels)
            print(split, epoch, "epoch - F1 Score:", f1score)

            # AUC
            auc = utils.calc_auc(tot_true_labels, tot_predicted_scores)
            print(split, epoch, "epoch - AUC:", auc)
            
            # Precision-Recall Score
            prc_score = utils.calc_aps(tot_true_labels, tot_predicted_scores)
            print(split, epoch, "epoch - PRscore:", prc_score)
            
            # Calibration: Brier Score
            brier_score = utils.calc_brierScore(tot_true_labels, tot_predicted_scores)
            print(split, epoch, "epoch - Brier Score:", brier_score)

            # Prediction Agreement Rate: concordanza valutazione stesso campione tra epoca corrente e precedente
            predictionAgreementRate, tot_predicted_labels_last[split] = utils.calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last[split], tot_image_paths)
            print(split, epoch, "epoch - Prediction Agreement Rate", predictionAgreementRate)

            if args.start_tornado_server:
                # Confusion Matrix
                cm_image = utils.plot_confusion_matrix(tot_true_labels, tot_predicted_labels, ['negative', 'positive'], title="Confusion matrix "+split)
                utils.plotImages(split + " " + str(epoch) + " epoch - Confusion Matrix", cm_image)

                # ROC Curve
                rocCurve_image = utils.calc_rocCurve(tot_true_labels, tot_predicted_scores)
                utils.plotImages(split + " " + str(epoch) + " epoch - ROC Curve", rocCurve_image)

                # Precision-Recall Curve
                precisionRecallCurve_image = utils.calc_precisionRecallCurve(tot_true_labels, tot_predicted_scores)
                utils.plotImages(split + " " + str(epoch) + " epoch - Precision-Recall Curve", precisionRecallCurve_image)

            # Print logs of error
            dict_other_info = {'image_path':tot_image_paths}
            Saver.printLogsError(tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
        
            # Save logs
            if args.saveLogs:
                Saver.saveLogs(args.logdir, tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
            
            # Plot GradCAM
            if args.enable_explainability:
                print("Exporting explainability...")
                
                if not os.path.exists(args.logdir + '/export_fold' + str(args.num_fold)):
                    os.makedirs(args.logdir + '/export_fold' + str(args.num_fold))
                
                for i2 in tqdm(range(len(tot_images_3d)), desc='Explainability'):
                    tot_images_3d[i2] = torch.tensor(tot_images_3d[i2])
                    tot_images_3d_gradient[i2] = torch.tensor(tot_images_3d_gradient[i2])
                    imgs=[]
                    plt.clf()
                    for i in range(tot_images_3d[i2].shape[1]):
                        if args.explainability_mode == 'medcam':
                            plt.imshow(tot_images_3d[i2][0,i,:,:].cpu().squeeze().numpy(), cmap='gray')
                            plt.imshow(scipy.ndimage.gaussian_filter(tot_images_3d_gradient[i2][0,i,:,:], sigma=10), interpolation='nearest', alpha=0.25)
                        elif args.explainability_mode == 'pytorchgradcambook':
                            plt.imshow(show_cam_on_image(tot_images_3d[i2][0,i,:,:].cpu().squeeze().numpy(), tot_images_3d_gradient[i2][0,i,:,:], use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5))
                        plt.axis('off')
                        buf = io.BytesIO()
                        plt.savefig(buf, format='jpeg')
                        buf.seek(0)
                        image = PIL.Image.open(buf)
                        imgs.append(image)
                        plt.clf()
                        #plt.show()
                    utils.saveGridImages(args.logdir + '/export_fold' + str(args.num_fold) + '/' + '_'.join(tot_image_paths[i2].replace('\\', '/').split('/')[-3:])[:-4], imgs, n_colonne=8)

    if args.distributed:
        torch.distributed.barrier()
    
    if (not args.distributed) or (args.distributed and (args.rank==0)):
        if args.start_tornado_server:
            # Show graph on browser
            webbrowser.open('http://127.0.0.1:' + str(tornado_port) + '/', new=1)

            # Start Tornado server
            plt.show()

    if args.distributed:
        torch.distributed.destroy_process_group()
        

if __name__ == '__main__':
    main()
