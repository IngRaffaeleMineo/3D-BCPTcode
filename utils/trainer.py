from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils

class Trainer:
    ''' Class to train the classifier '''
    def __init__(self, net, class_weights, optim, gradient_clipping_value, enable_datiClinici, doppiaAngolazioneInput, input3d, input2d, scaler=None):
        self.enable_datiClinici = enable_datiClinici
        self.doppiaAngolazioneInput = doppiaAngolazioneInput
        self.input3d = input3d
        self.input2d = input2d
        # Store model
        self.net = net
        # Store optimizer
        self.optim = optim
        # Create Loss
        self.criterion_label = nn.CrossEntropyLoss(weight = class_weights)
        # Gradient clipping
        self.gradient_clipping_value = gradient_clipping_value
        # CUDA AMP
        self.scaler = scaler

    def forward_batch(self, imgs_3d, imgs_2d, labels, datiClinici, doppiaAngolazione_3d, doppiaAngolazione_2d, split):
        ''' send a batch to net and backpropagate '''
        def forward_batch_part():
            # Set network mode
            if split == 'train':
                self.net.train()
                torch.set_grad_enabled(True)   
            else:
                self.net.eval()
                torch.set_grad_enabled(False)
            
            if self.scaler is None:
                # foward pass
                if self.doppiaAngolazioneInput:
                    if self.enable_datiClinici:
                        if self.input3d and self.input2d:
                            inputs = imgs_3d, doppiaAngolazione_3d, datiClinici, imgs_2d, doppiaAngolazione_2d
                        elif self.input3d:
                            inputs = imgs_3d, doppiaAngolazione_3d, datiClinici
                        elif self.input2d:
                            inputs = imgs_2d, doppiaAngolazione_2d, datiClinici
                        else:
                            raise RuntimeError("No input.")
                    else:
                        if self.input3d and self.input2d:
                            inputs = imgs_3d, doppiaAngolazione_3d, imgs_2d, doppiaAngolazione_2d
                        elif self.input3d:
                            inputs = imgs_3d, doppiaAngolazione_3d
                        elif self.input2d:
                            inputs = imgs_2d, doppiaAngolazione_2d
                        else:
                            raise RuntimeError("No input.")
                else:
                    if self.enable_datiClinici:
                        if self.input3d and self.input2d:
                            inputs = imgs_3d, datiClinici, imgs_2d
                        elif self.input3d:
                            inputs = imgs_3d, datiClinici
                        elif self.input2d:
                            inputs = imgs_2d, datiClinici
                        else:
                            raise RuntimeError("No input.")
                    else:
                        if self.input3d and self.input2d:
                            inputs = imgs_3d, imgs_2d
                        elif self.input3d:
                            inputs = imgs_3d
                        elif self.input2d:
                            inputs = imgs_2d
                        else:
                            raise RuntimeError("No input.")
                
                out = self.net(inputs)
                
                predicted_labels_logits = out

                # compute loss label
                loss_labels = self.criterion_label(predicted_labels_logits, labels)
                loss = loss_labels
            else:
                with torch.cuda.amp.autocast():
                    # foward pass
                    if self.doppiaAngolazioneInput:
                        if self.enable_datiClinici:
                            if self.input3d and self.input2d:
                                inputs = imgs_3d, doppiaAngolazione_3d, datiClinici, imgs_2d, doppiaAngolazione_2d
                            elif self.input3d:
                                inputs = imgs_3d, doppiaAngolazione_3d, datiClinici
                            elif self.input2d:
                                inputs = imgs_2d, doppiaAngolazione_2d, datiClinici
                            else:
                                raise RuntimeError("No input.")
                        else:
                            if self.input3d and self.input2d:
                                inputs = imgs_3d, doppiaAngolazione_3d, imgs_2d, doppiaAngolazione_2d
                            elif self.input3d:
                                inputs = imgs_3d, doppiaAngolazione_3d
                            elif self.input2d:
                                inputs = imgs_2d, doppiaAngolazione_2d
                            else:
                                raise RuntimeError("No input.")
                    else:
                        if self.enable_datiClinici:
                            if self.input3d and self.input2d:
                                inputs = imgs_3d, datiClinici, imgs_2d
                            elif self.input3d:
                                inputs = imgs_3d, datiClinici
                            elif self.input2d:
                                inputs = imgs_2d, datiClinici
                            else:
                                raise RuntimeError("No input.")
                        else:
                            if self.input3d and self.input2d:
                                inputs = imgs_3d, imgs_2d
                            elif self.input3d:
                                inputs = imgs_3d
                            elif self.input2d:
                                inputs = imgs_2d
                            else:
                                raise RuntimeError("No input.")
                    
                    out = self.net(inputs)

                    predicted_labels_logits = out

                    # compute loss label
                    loss_labels = self.criterion_label(predicted_labels_logits, labels)
                    loss = loss_labels
            
            # calculate label predicted and scores
            _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
            predicted_scores = predicted_labels_logits.data.clone().detach().cpu()

            if split == 'train':
                #zero the gradient
                self.optim.zero_grad()

                # backpropagate
                if self.scaler is None:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()
                
                # gradient clipping
                if self.gradient_clipping_value > 0:
                    if self.scaler is None:
                        torch.nn.utils.clip_grad_value_(self.net.parameters(), self.gradient_clipping_value)
                    else:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_value_(self.net.parameters(), self.gradient_clipping_value)
            return loss, predicted_labels, predicted_scores

        loss, predicted_labels, predicted_scores = forward_batch_part()

        if not isinstance(self.optim,torch.optim.LBFGS):
            if split == 'train':
                # update weights (and scaler if exists)
                if self.scaler is None:
                    self.optim.step()
                else:
                    self.scaler.step(self.optim)
                    self.scaler.update()
        else:
            if split == 'train':
                # update weights (and scaler if exists)
                if self.scaler is None:
                    self.optim.step(forward_batch_part)
                else:
                    self.scaler.step(self.optim, forward_batch_part)
                    self.scaler.update()
        
        # metrics
        metrics = {}
        metrics['loss'] = loss.item()
        
        predicted = predicted_labels, predicted_scores

        return metrics, predicted

    def forward_batch_testing(net, imgs_3d, imgs_2d, datiClinici, doppiaAngolazione_3d, doppiaAngolazione_2d, enable_datiClinici, doppiaAngolazioneInput, input3d, input2d, scaler=None):
        ''' send a batch to net and backpropagate '''
        # Set network mode
        net.eval()
        torch.set_grad_enabled(False)
        
        if scaler is None:
            # foward pass
            if doppiaAngolazioneInput:
                if enable_datiClinici:
                    if input3d and input2d:
                        inputs = imgs_3d, doppiaAngolazione_3d, datiClinici, imgs_2d, doppiaAngolazione_2d
                    elif input3d:
                        inputs = imgs_3d, doppiaAngolazione_3d, datiClinici
                    elif input2d:
                        inputs = imgs_2d, doppiaAngolazione_2d, datiClinici
                    else:
                        raise RuntimeError("No input.")
                else:
                    if input3d and input2d:
                        inputs = imgs_3d, doppiaAngolazione_3d, imgs_2d, doppiaAngolazione_2d
                    elif input3d:
                        inputs = imgs_3d, doppiaAngolazione_3d
                    elif input2d:
                        inputs = imgs_2d, doppiaAngolazione_2d
                    else:
                        raise RuntimeError("No input.")
            else:
                if enable_datiClinici:
                    if input3d and input2d:
                        inputs = imgs_3d, datiClinici, imgs_2d
                    elif input3d:
                        inputs = imgs_3d, datiClinici
                    elif input2d:
                        inputs = imgs_2d, datiClinici
                    else:
                        raise RuntimeError("No input.")
                else:
                    if input3d and input2d:
                        inputs = imgs_3d, imgs_2d
                    elif input3d:
                        inputs = imgs_3d
                    elif input2d:
                        inputs = imgs_2d
                    else:
                        raise RuntimeError("No input.")
            
            out = net(inputs)

            predicted_labels_logits = out
        else:
            with torch.cuda.amp.autocast():
                # foward pass
                if doppiaAngolazioneInput:
                    if enable_datiClinici:
                        if input3d and input2d:
                            inputs = imgs_3d, doppiaAngolazione_3d, datiClinici, imgs_2d, doppiaAngolazione_2d
                        elif input3d:
                            inputs = imgs_3d, doppiaAngolazione_3d, datiClinici
                        elif input2d:
                            inputs = imgs_2d, doppiaAngolazione_2d, datiClinici
                        else:
                            raise RuntimeError("No input.")
                    else:
                        if input3d and input2d:
                            inputs = imgs_3d, doppiaAngolazione_3d, imgs_2d, doppiaAngolazione_2d
                        elif input3d:
                            inputs = imgs_3d, doppiaAngolazione_3d
                        elif input2d:
                            inputs = imgs_2d, doppiaAngolazione_2d
                        else:
                            raise RuntimeError("No input.")
                else:
                    if enable_datiClinici:
                        if input3d and input2d:
                            inputs = imgs_3d, datiClinici, imgs_2d
                        elif input3d:
                            inputs = imgs_3d, datiClinici
                        elif input2d:
                            inputs = imgs_2d, datiClinici
                        else:
                            raise RuntimeError("No input.")
                    else:
                        if input3d and input2d:
                            inputs = imgs_3d, imgs_2d
                        elif input3d:
                            inputs = imgs_3d
                        elif input2d:
                            inputs = imgs_2d
                        else:
                            raise RuntimeError("No input.")
                out = net(inputs)
                predicted_labels_logits = out

        # calculate label predicted and scores
        _, predicted_labels = torch.max(predicted_labels_logits.data, 1)
        predicted_scores = predicted_labels_logits.data.clone().detach().cpu()
        
        predicted = predicted_labels, predicted_scores

        return predicted