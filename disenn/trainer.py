import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from disenn.models.disenn import DiSENN
from disenn.models.conceptizers import VaeConceptizer
from disenn.models.parameterizers import ConvParameterizer
from disenn.models.aggregators import SumAggregator
from disenn.models.losses import bvae_loss
from disenn.models.losses import celeba_robustness_loss
from disenn.utils.initialization import init_parameters
from disenn.datasets.get_dataloader import get_dataloader

plt.style.use('seaborn-paper')
RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
LOG_DIR = Path("logs/")
VISUALIZATION_DIR = Path("visualizations/")
BEST_MODEL_FNAME = "best-model.pt"
NUM_VISUALIZE = 5

class DiSENN_Trainer():
    """ Trains a DiSENN model

    A trainer instantiates a model to be trained. It contains logic for training, validating, checkpointing, etc.
    All the parameters that drive the experiment behaviour are specified in a config dictionary.
    """

    def __init__(self, config: dict):
        """ Instantiate a trainer for DiSENN models

        Parameters
        ----------
        config: dict
            dictionary of parameters that drive the training behaviour
        """
        self.config = config

        self.exp_dir = RESULTS / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = CHECKPOINTS / config['exp_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.exp_dir / LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.viz_dir = self.exp_dir / VISUALIZATION_DIR
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        log_name = config["exp_name"]+".log"
        self.logger = logging.getLogger(__name__)
        logfile_handler = logging.FileHandler(filename=self.exp_dir / log_name)
        logfile_handler.setLevel(level = (logging.DEBUG if config["debug"] else logging.INFO))
        logfile_format = logging.Formatter('%(asctime)s - %(levelname)10s - %(funcName)15s : %(message)s')
        logfile_handler.setFormatter(logfile_format)
        self.logger.addHandler(logfile_handler)
        self.logger.setLevel(level = (logging.DEBUG if config["debug"] else logging.INFO))

        print(f"Launched successfully... \nLogs available @ {self.exp_dir / log_name}")
        print("To stop training, press CTRL+C")
        self.logger.info("-"*50)
        self.logger.info(f"EXPERIMENT: {config['exp_name']}")
        self.logger.info("-"*50)

        self.logger.info(f"Setting seed: {config['seed']}")
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.logger.info(f"Loading data {config['data']} ...")
        self.train_dl, self.valid_dl = get_dataloader(config)
        
        # fixed set of validation examples to visualize prediction dynamics
        x, y = next(iter(self.valid_dl))
        self.eval_examples = (x[:NUM_VISUALIZE], y[:NUM_VISUALIZE])

         # get appropriate models from global namespace and instantiate them
        try:
            conceptizer = eval(config['conceptizer'])(config['num_concepts'])
            parameterizer = eval(config['parameterizer'])(config['num_concepts'], config['num_classes'])
            aggregator = eval(config['aggregator'])(config['num_classes'])
        except:
            self.logger.error("Please make sure you specify the correct Conceptizer, Parameterizer and Aggregator classes")
            sys.exit(1)

        self.pred_loss_fn = F.nll_loss
        self.concept_loss_fn = eval(config['concept_loss'])
        self.robust_loss_fn = eval(config['robustness_loss'])

         # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_accuracy = 0.

        self.model = DiSENN(conceptizer, parameterizer, aggregator)
        self.logger.info(f"Using device: {config['device']}")
        self.model.to(config['device'])
        self.opt = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        if config['pretrain_epochs'] > 0:
            try:
                self.logger.info(f"Pre-training Conceptizer for {config['pretrain_epochs']} epochs")
                self.model.conceptizer = self.pretrain_conceptizer()
            except KeyboardInterrupt:
                self.logger.warning("Manual interruption registered. Please wait to finalize...")
                self.finalize()

        if 'load_checkpoint' in config:
            self.load_checkpoint(config['load_checkpoint'])

    def run(self):
        """Run the train-eval loop
        
        If the loop is interrupted manually, finalization will still be executed
        """
        try:
            if self.config['train']:
                self.logger.info(f"Begin training for {self.config['epochs']} epochs")
                self.train()
        except KeyboardInterrupt:
            self.logger.warning("Manual interruption registered. Please wait to finalize...")
            self.finalize()

    def train(self):
        """ Main training loop """
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            for i, (x, labels) in enumerate(self.train_dl):
                self.current_iter += 1
                results = self._batch_iteration(x, labels, training=True)

                self.writer.add_scalar('Accuracy/Train', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Train/Total_loss', results['total_loss'], self.current_iter)
                self.writer.add_scalar('Loss/Train/Concept_loss', results['concept_loss'], self.current_iter)
                self.writer.add_scalar('Loss/Train/Robustness_loss', results['robustness_loss'], self.current_iter)
                self.writer.add_scalar('Loss/Train/Reconstruction_loss', results['reconstruction_loss'], self.current_iter)
                self.writer.add_scalar('Loss/Train/KL_Divergence', results['kl_divergence'], self.current_iter)
                self.writer.add_scalar('Loss/Train/Prediction_loss', results['pred_loss'], self.current_iter)
                self.logger.debug(f"EPOCH:{epoch} STEP:{i}\t Accuracy: {results['accuracy']:.3f} Total Loss: {results['total_loss']:.3f}")

                if i % self.config['valid_freq'] == 0:
                    self.validate()
                if i % self.config['save_freq'] == 0:
                    self.save_checkpoint()

    def validate(self):
        """ Main validation loop """
        self.model.eval()
        losses = []
        pred_losses = []
        concept_losses = []
        kl_divs = []
        recon_losses = []
        robustness_losses = []
        prediction_losses = []
        accuracies = []

        self.logger.debug("Begin evaluation over validation set")
        with torch.no_grad():
            for i, (x, labels) in enumerate(self.valid_dl):
                results = self._batch_iteration(x, labels, training=False)
                losses.append(results['total_loss'])
                pred_losses.append(results['pred_loss'])
                concept_losses.append(results['concept_loss'])
                recon_losses.append(results['reconstruction_loss'])
                kl_divs.append(results['kl_divergence'])
                robustness_losses.append(results['robustness_loss'])
                accuracies.append(results['accuracy'])
            
        mean_accuracy = np.mean(accuracies)
        if mean_accuracy > self.best_accuracy:
            self.best_accuracy = mean_accuracy
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        self.writer.add_scalar('Accuracy/Valid', mean_accuracy, self.current_iter)
        self.writer.add_scalar('Loss/Valid/Total_loss', np.mean(losses), self.current_iter)
        self.writer.add_scalar('Loss/Valid/Concept_loss', np.mean(concept_losses), self.current_iter)
        self.writer.add_scalar('Loss/Valid/Robustness_loss', np.mean(robustness_losses), self.current_iter)
        self.writer.add_scalar('Loss/Valid/Reconstruction_loss', np.mean(recon_losses), self.current_iter)
        self.writer.add_scalar('Loss/Valid/KL_Divergence', np.mean(kl_divs), self.current_iter)
        self.writer.add_scalar('Loss/Valid/Prediction_loss', np.mean(pred_losses), self.current_iter)
        report = (f"[Validation]\t"
                f"Accuracy: {mean_accuracy:.3f} "
                f"Total Loss: {np.mean(losses):.3f}")
        self.logger.info(report)
        self.visualize()

    def _batch_iteration(self, x: torch.Tensor, labels: torch.Tensor, training: bool):
        """ Iterate over one batch """

        x = x.float().to(self.config['device'])
        labels = labels.long().to(self.config['device'])
        
        if training:
            self.model.train()
            self.opt.zero_grad()
            x.requires_grad_(True) # for jacobian calculation
            
            y_pred, (concepts_dist, relevances), x_reconstruct = self.model(x)
            concept_mean, concept_logvar = concepts_dist
            concepts = concept_mean
            pred_loss = self.pred_loss_fn(y_pred.squeeze(-1), labels)
            robustness_loss = self.robust_loss_fn(x, y_pred, concepts, relevances)
            recon_loss, kl_div = self.concept_loss_fn(x, x_reconstruct, concept_mean, concept_logvar)
            concept_loss = recon_loss + self.config['beta'] * kl_div
            total_loss = pred_loss + concept_loss + (self.config['robustness_reg'] * robustness_loss) 
            
            total_loss.backward()
            self.opt.step()

        else:
            self.model.eval()
            with torch.no_grad():
                y_pred, (concepts_dist, relevances), x_reconstruct = self.model(x)
                concept_mean, concept_logvar = concepts_dist
                concepts = concept_mean
                pred_loss = self.pred_loss_fn(y_pred.squeeze(-1), labels)
                robustness_loss = torch.tensor(0.) # jacobian cannot be computed in no_grad mode
                recon_loss, kl_div = self.concept_loss_fn(x, x_reconstruct, concept_mean, concept_logvar)
                concept_loss = recon_loss + self.config['beta'] * kl_div
                total_loss = pred_loss + concept_loss          

        accuracy = self.accuracy(y_pred, labels)
        results = {'accuracy': accuracy, 
                   'total_loss': total_loss.item(),
                   'pred_loss': pred_loss.item(),
                   'robustness_loss': robustness_loss.item(),
                   'concept_loss': concept_loss.item(),
                   'reconstruction_loss': recon_loss.item(),
                   'kl_divergence': kl_div.item()}
        return results

    def accuracy(self, y_pred: torch.Tensor, y: torch.Tensor):
        """Return accuracy of predictions with respect to ground truth.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions of shape (batch_size x ...)

        y : torch.Tensor
            Ground truth of shape (batch_size x ...)

        Returns
        -------
        accuracy: float
            accuracy of predictions
        """
        return (y_pred.argmax(axis=1) == y).float().mean().item()

    def pretrain_conceptizer(self):
        optimizer = optim.Adam(self.model.conceptizer.parameters())
        self.model.conceptizer.train()
        current_iter = 0
        for epoch in range(self.config['pretrain_epochs']):
            for i, (x, _) in enumerate(self.train_dl):
                x = x.to(self.config['device'])
                optimizer.zero_grad()
                concept_mean, concept_logvar, x_reconstruct = self.model.conceptizer(x)
                recon_loss, kl_div = bvae_loss(x, x_reconstruct, concept_mean, concept_logvar)
                loss = recon_loss + self.config['pre-beta'] * kl_div
                loss.backward()
                optimizer.step()
                current_iter += 1
                self.writer.add_scalar('PreTraining/Total_Loss', loss.item(), current_iter)
                self.writer.add_scalar('PreTraining/Reconstruction', recon_loss.item(), current_iter)
                self.writer.add_scalar('PreTraining/KL_Divergence', kl_div.item(), current_iter)
                    
                if i % self.config['save_freq'] == 0:
                    figname = self.viz_dir / f"Pretraining-Epoch[{epoch}]-Step[{current_iter}].png"
                    plt.imsave(fname=figname, arr=x_reconstruct[0].detach().cpu().numpy().transpose(1,2,0))
                    report = (f"[Pre-Training] EPOCH:{epoch} STEP:{i}\t"
                            f"Concept loss: {loss.item():.3f} "
                            f"Recon loss: {recon_loss.item():.3f} "
                            f"KL div: {kl_div.item():e} ")
                    self.logger.debug(report)
                    self.save_checkpoint(file_name="pretrained-conceptizer.pt")

        return self.model.conceptizer
    
    def save_checkpoint(self, file_name: str = None):
        """Save checkpoint in the checkpoint directory.

        Checkpoint directory and checkpoint file need to be specified in the configs.

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = self.checkpoint_dir / file_name
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(state, file_name)
        self.logger.info(f"Checkpoint saved @ {file_name}")

    def load_checkpoint(self, file_name: str):
        """Load the checkpoint with the given file name

        Checkpoint must contain:
            - current epoch
            - current iteration
            - model state
            - best accuracy achieved so far
            - optimizer state

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file
        """
        try:
            file_name = self.checkpoint_dir / file_name
            self.logger.info(f"Loading checkpoint from {file_name}")
            checkpoint = torch.load(file_name, self.config['device'])

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

        except OSError:
            self.logger.error(f"No checkpoint exists @ {self.checkpoint_dir}")
        
    def visualize(self):
        """ Visualize prediction dynamics on a set of fixed examples """
        self.logger.info("Visualizing prediction dynamics")
        
        x, y = self.eval_examples
        self.model.eval()
        fig_size = (18, 5 * self.config['num_concepts']/10) # heuristic works for 64x64 images only
        for i in range(NUM_VISUALIZE):
            figname = self.viz_dir / f"Example[{i}]-Epoch[{self.current_epoch}]-Step[{self.current_iter}].png"
            self.model.explain(x[i].cpu().detach(), save_as=figname, figure_size=fig_size)

    def finalize(self):
        """Finalize all necessary operations before stopping
        
        Saves checkpoint
        """
        self.visualize()
        self.save_checkpoint()
