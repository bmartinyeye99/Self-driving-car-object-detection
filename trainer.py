import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import wandb

from utils import Statistics


class Trainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device, memory_format=torch.channels_last)

        # Create the optimizer
        self.optimizer = torch.optim.NAdam(
            model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        self.loss_fn = cfg.loss_fn

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.max_epochs, eta_min=0)

        self.scaler = GradScaler()

    def setup(self, datamodule):
        self.datamodule = datamodule
        self.datamodule.setup(self.cfg)

    def training_loop(self, run=None):
        for epoch in range(self.cfg.max_epochs):
            print('Epoch ', epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr}\n")

            self.train_epoch(self.model,
                             self.datamodule.dataloader_train, run, epoch)

            print("\n")

            self.validate_epoch(self.model,
                                self.datamodule.dataloader_valid, run, epoch)

            print('-------------------------------------------\n\n')

            self.scheduler.step()  # Update learning rate

    def fit(self):
        if self.cfg.use_wandb:
            with wandb.init(
                project=self.cfg.project_name,
                config=self.cfg,
            ) as run:
                self.training_loop(run)
        else:
            self.training_loop(None)

        self.test(self.model, self.datamodule.dataloader_test)

        print('\n\n')
        print('-----------------------------------------------')
        self.draw(self.model, self.datamodule.dataloader_draw)

    def train_epoch(self, model, dataloader, run, epoch):
        model.train()  # Set model to train mode
        stats = Statistics(self.device, self.cfg)  # Statistic class
        description = "Training"

        with tqdm(dataloader, desc=description) as progress:
            for x, y in progress:
                x = x.to(self.device, memory_format=torch.channels_last)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                with autocast():
                    # Forward pass
                    model_prediction = self.model(x)

                    loss = self.loss_fn(model_prediction, y)

                stats.step(model_prediction, y, loss)

                # Backwards pass
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)

                clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                progress.set_postfix(avg_loss=stats.get_average_loss())

        for metric_name, metric_value in stats.get_metrics():
            print(f"{description} {metric_name}: {metric_value}")

            if run is not None:
                run.log({f"{description} {metric_name}": metric_value}, step=epoch)

    def validate_epoch(self, model, dataloader, run, epoch):
        model.eval()            # Set model to evaluation mode
        stats = Statistics(self.device, self.cfg, True)
        description = "Validation"

        with torch.no_grad():   # No need to compute gradients
            with tqdm(dataloader, desc=description) as progress:
                for x, y in progress:
                    x = x.to(self.device, memory_format=torch.channels_last)
                    y = y.to(self.device)

                    # Forward pass only
                    with autocast():
                        model_prediction = model(x)

                        loss = self.loss_fn(model_prediction, y)

                    stats.step(model_prediction, y, loss)

                    progress.set_postfix(avg_loss=stats.get_average_loss())

        for metric_name, metric_value in stats.get_metrics():
            print(f"{description} {metric_name}: {metric_value}")

            if run is not None:
                run.log({f"{description} {metric_name}": metric_value}, step=epoch)

    def test(self, model, dataloader):
        model.eval()
        stats = Statistics(self.device, self.cfg, True)
        description = "Testing"

        with torch.no_grad():
            with tqdm(dataloader, desc=description) as progress:
                for x, y in progress:
                    x = x.to(self.device, memory_format=torch.channels_last)
                    y = y.to(self.device)

                    with autocast():
                        model_prediction = model(x)

                        loss = self.loss_fn(model_prediction, y)

                    stats.step(model_prediction, y, loss)

                progress.set_postfix(avg_loss=stats.get_average_loss())

        for metric_name, metric_value in stats.get_metrics():
            print(f"{description} {metric_name}: {metric_value}")

    def draw(self, model, dataloader):
        print('Showing predictions on images!')
        model.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device, memory_format=torch.channels_last)
                y = y.to(self.device)

                with autocast():
                    model_prediction = model(x)
