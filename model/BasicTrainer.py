import torch
import math
import os
import time
import copy
import numpy as np
import pynvml
from lib.logger import get_logger
from lib.metrics import All_Metrics
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except pynvml.NVMLError:
    handle = None


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.memory_diversity_weight = float(getattr(args, "memory_diversity_weight", 0.01))
        self.memory_usage_weight = float(getattr(args, "memory_usage_weight", 0.0))
        self.delta_loss_weight = float(getattr(args, "delta_loss_weight", 0.0))
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(args)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info(
            'Regularization settings: weight_decay={}, dropout={}, grad_norm={}, max_grad_norm={}, '
            'memory_diversity_weight={}, memory_usage_weight={}, delta_loss_weight={}'.format(
                getattr(args, "weight_decay", 0.0),
                getattr(args, "dropout", 0.1),
                getattr(args, "grad_norm", False),
                getattr(args, "max_grad_norm", 5.0),
                self.memory_diversity_weight,
                self.memory_usage_weight,
                self.delta_loss_weight,
            )
        )
        self.logger.info(
            'Physical local mask: has_local_mask={}, mask_nonzero={}'.format(
                getattr(model, "has_local_mask", False),
                int(model.local_mask.sum().item()) if hasattr(model, "local_mask") else 0,
            )
        )
        self.logger.info('Prediction head: linear')
        self.logger.info('TPMR retrieval: memory_retrieval={}'.format(getattr(model, "memory_retrieval", "unknown")))
        self.logger.info(
            'TPMR settings: num_prototypes={}, memory_temperature={}, memory_gate_bias={}, memory_scale={}'.format(
                getattr(model, "num_prototypes", "unknown"),
                getattr(model, "memory_temperature", "unknown"),
                getattr(model, "memory_gate_bias", "unknown"),
                getattr(model, "memory_scale", "unknown"),
            )
        )
        self.batches_seen = 0
        self.meminfo = None
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data
                label = target[..., :self.args.output_dim].clone()
                target[...,:self.args.output_dim] = self.scaler.transform(target[...,:self.args.output_dim])
                output = self.model(data, target)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.to(label.device), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, val_loss, time.time() - epoch_time))
        return val_loss

    def test_epoch(self, epoch, test_dataloader):
        self.model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data = data
                label = target[..., :self.args.output_dim].clone()
                target[...,:self.args.output_dim] = self.scaler.transform(target[...,:self.args.output_dim])
                output = self.model(data, target)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output.to(label.device), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
        test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_loss, time.time() - epoch_time))
        return test_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.batches_seen += 1
            data = data
            label = target[..., :self.args.output_dim].clone()  # (..., 1)
            target[..., :self.args.output_dim] = self.scaler.transform(target[..., :self.args.output_dim])
            self.optimizer.zero_grad()

            #data and target shape: B, T, N, F; output shape: B, T, N, F
            pred, memory_attn, A_dyn, fusion_gate = self.model(data, target, self.batches_seen, return_aux=True)
            pred_model_scale = pred.detach()

            if self.args.real_value:
                pred = self.scaler.inverse_transform(pred)
            # ✅ 加这里
            assert pred.shape == label.shape, \
                f"Shape mismatch: pred {pred.shape} vs label {label.shape}"
            base_loss = self.loss(pred.to(label.device), label)
            diversity_loss = self.model.tpmr.prototype_diversity_loss()
            usage_loss = self.model.tpmr.prototype_usage_loss(memory_attn)
            if self.delta_loss_weight > 0.0 and pred.shape[1] > 1:
                delta_pred = pred[:, 1:, ...] - pred[:, :-1, ...]
                delta_label = label[:, 1:, ...] - label[:, :-1, ...]
                delta_loss = torch.nn.functional.l1_loss(delta_pred, delta_label)
            else:
                delta_loss = pred.new_tensor(0.0)
            loss = (
                base_loss
                + self.memory_diversity_weight * diversity_loss
                + self.memory_usage_weight * usage_loss
                + self.delta_loss_weight * delta_loss
            )

            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            # 👇 加在这里
            if batch_idx % 100 == 0:
                print("prototype usage:", memory_attn.mean(dim=(0,1)).detach().cpu().numpy())
                print(
                    "pred model-scale stats:",
                    {
                        "min": round(pred_model_scale.min().detach().cpu().item(), 6),
                        "mean": round(pred_model_scale.mean().detach().cpu().item(), 6),
                        "max": round(pred_model_scale.max().detach().cpu().item(), 6),
                    },
                )
                print(
                    "pred loss-scale stats:",
                    {
                        "min": round(pred.detach().min().cpu().item(), 6),
                        "mean": round(pred.detach().mean().cpu().item(), 6),
                        "max": round(pred.detach().max().cpu().item(), 6),
                    },
                )
                print(
                    "label stats:",
                    {
                        "min": round(label.detach().min().cpu().item(), 6),
                        "mean": round(label.detach().mean().cpu().item(), 6),
                        "max": round(label.detach().max().cpu().item(), 6),
                    },
                )
                debug_stats = getattr(self.model, "debug_stats", {})
                if debug_stats:
                    debug_stats = {k: round(v.detach().cpu().item(), 6) for k, v in debug_stats.items()}
                    print(
                        "quick debug:",
                        {
                            "base": round(base_loss.detach().cpu().item(), 6),
                            "pred_mean": round(pred.detach().mean().cpu().item(), 6),
                            "label_mean": round(label.detach().mean().cpu().item(), 6),
                            "gate_mean": debug_stats.get("gate_mean"),
                            "m_over_h": debug_stats.get("m_over_h"),
                            "attn_max": debug_stats.get("attn_max"),
                        },
                    )
                    print(
                        "loss parts:",
                        {
                            "base": round(base_loss.detach().cpu().item(), 6),
                            "div": round(diversity_loss.detach().cpu().item(), 6),
                            "usage": round(usage_loss.detach().cpu().item(), 6),
                            "delta": round(delta_loss.detach().cpu().item(), 6),
                        },
                    )
                    print("debug stats:", debug_stats)
            #log information
            if (batch_idx+1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx+1, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle) if handle is not None else None
        # test_loss.append(test_epoch_loss)
        # train_time.append(time.time()-start_time)
        # train_M.append((meminfo.used - meminfo1.used) / 1024 ** 3)

        if meminfo is not None and self.meminfo is not None:
            gpu_cost = (meminfo.used - self.meminfo.used) / 1024 ** 3
        else:
            gpu_cost = 0.0
        self.logger.info('********Train Epoch {}: averaged Loss: {:.6f}, GPU cost: {:.2f} GB, train time: {:.2f} s'.format(epoch, train_epoch_loss, gpu_cost,time.time() - epoch_time))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle) if handle is not None else None
        best_model = None
        best_test_model =None
        # start_time = time.time()
        batches_seen = 0
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')
        # train_loss = []
        vaild_loss = []
        test_loss = []
        train_time = []
        train_M = []
        for epoch in range(1, self.args.epochs + 1):

            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # train_loss.append(train_epoch_loss)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            vaild_loss.append(val_epoch_loss)

            # epoch_time = time.time()
            test_epoch_loss = self.test_epoch(epoch, test_dataloader)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            #print('LR:', self.optimizer.param_groups[0]['lr'])
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

            if test_epoch_loss< best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())


        # training_time = time.time() - start_time
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # np.savetxt('./{}_train_time.csv'.format(self.args.dataset), np.array([np.array(train_time),np.array(vaild_loss),np.array(test_loss),np.array(train_M)]).T,delimiter=",")

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best model to " + self.best_test_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(
            self.model,
            self.args,
            self.test_loader,
            self.scaler,
            self.logger,
            plot_name='best_model_forecast.png',
        )

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self.test(
            self.model,
            self.args,
            self.test_loader,
            self.scaler,
            self.logger,
            plot_name='best_test_model_forecast.png',
        )

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None, plot_name='forecast_visualization.png'):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data
                label = target[..., :args.output_dim]
                output = model(data, target)
                y_true.append(label)
                y_pred.append(output)

        #y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = torch.cat(y_true, dim=0)
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
        # np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        # np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%, CORR: {:.4f}".format(
                t + 1, rmse, mae, mape*100,corr))
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%, CORR: {:.4f} ".format(
                    rmse, mae, mape*100,corr))

        if getattr(args, 'plot', False):
            Trainer._plot_forecast_figure(y_true, y_pred, args, logger, plot_name)

        # mae, rmse, mape, _, corr = All_Metrics(y_pred[...,0], y_true[...,0], args.mae_thresh, args.mape_thresh)
        # logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
        #             rmse, mae, mape*100))
        # mae, rmse, mape, _, corr = All_Metrics(y_pred[...,1], y_true[...,1], args.mae_thresh, args.mape_thresh)
        # logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
        #             rmse, mae, mape*100))

    @staticmethod
    def _plot_forecast_figure(y_true, y_pred, args, logger, plot_name):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError as exc:
            message = (
                'matplotlib is required when plot=True. '
                'Install it with: pip install matplotlib'
            )
            logger.error(message)
            raise ImportError(message) from exc

        true = y_true.detach().cpu().numpy()[..., 0]
        pred = y_pred.detach().cpu().numpy()[..., 0]
        abs_err = np.abs(pred - true)

        sample_count, horizon, node_count = true.shape
        horizon_axis = np.arange(1, horizon + 1)
        plot_len = min(sample_count, 288)
        node_error = abs_err.mean(axis=(0, 1))
        node_ids = np.argsort(node_error)
        selected_nodes = [int(node_ids[0]), int(node_ids[len(node_ids) // 2]), int(node_ids[-1])]
        selected_nodes = list(dict.fromkeys(selected_nodes))

        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 9,
            'legend.fontsize': 8,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

        fig = plt.figure(figsize=(13, 8), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0), height_ratios=(1.0, 1.0))

        ax_main = fig.add_subplot(grid[0, 0])
        true_mean = true[:plot_len, 0, :].mean(axis=1)
        pred_mean = pred[:plot_len, 0, :].mean(axis=1)
        ax_main.plot(true_mean, color='#1f2937', linewidth=1.8, label='Ground truth')
        ax_main.plot(pred_mean, color='#d97706', linewidth=1.6, label='Prediction')
        ax_main.fill_between(
            np.arange(plot_len),
            true_mean,
            pred_mean,
            color='#f59e0b',
            alpha=0.18,
            linewidth=0,
        )
        ax_main.set_title('(a) Network-level one-step forecast')
        ax_main.set_xlabel('Test window index')
        ax_main.set_ylabel('Traffic flow')
        ax_main.grid(True, color='#e5e7eb', linewidth=0.7)
        ax_main.legend(frameon=False, loc='upper right')

        ax_h = fig.add_subplot(grid[0, 1])
        mae_by_h = abs_err.mean(axis=(0, 2))
        ax_h.plot(horizon_axis, mae_by_h, color='#2563eb', marker='o', linewidth=1.8, markersize=4)
        ax_h.set_title('(b) MAE across prediction horizons')
        ax_h.set_xlabel('Horizon')
        ax_h.set_ylabel('MAE')
        ax_h.set_xticks(horizon_axis)
        ax_h.grid(True, color='#e5e7eb', linewidth=0.7)

        ax_nodes = fig.add_subplot(grid[1, 0])
        colors = ['#059669', '#7c3aed', '#dc2626']
        labels = ['low-error node', 'median-error node', 'high-error node']
        for idx, node in enumerate(selected_nodes):
            color = colors[idx % len(colors)]
            label = labels[idx] if idx < len(labels) else 'node {}'.format(node)
            ax_nodes.plot(
                true[:plot_len, 0, node],
                color=color,
                linewidth=1.4,
                alpha=0.95,
                label='True {}'.format(label),
            )
            ax_nodes.plot(
                pred[:plot_len, 0, node],
                color=color,
                linewidth=1.2,
                linestyle='--',
                alpha=0.85,
                label='Pred {}'.format(label),
            )
        ax_nodes.set_title('(c) Representative sensor trajectories')
        ax_nodes.set_xlabel('Test window index')
        ax_nodes.set_ylabel('Traffic flow')
        ax_nodes.grid(True, color='#e5e7eb', linewidth=0.7)
        ax_nodes.legend(frameon=False, ncol=2, loc='upper right')

        ax_heat = fig.add_subplot(grid[1, 1])
        node_order = np.argsort(node_error)[-min(40, node_count):]
        heatmap = abs_err[:, :, node_order].mean(axis=0).T
        im = ax_heat.imshow(heatmap, aspect='auto', cmap='magma', interpolation='nearest')
        ax_heat.set_title('(d) Error map for high-error sensors')
        ax_heat.set_xlabel('Horizon')
        ax_heat.set_ylabel('Sensor rank')
        ax_heat.set_xticks(np.arange(horizon))
        ax_heat.set_xticklabels(horizon_axis)
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('MAE')

        fig.suptitle(
            '{} forecast visualization'.format(getattr(args, 'dataset', 'Dataset')),
            fontsize=14,
            fontweight='bold',
        )

        os.makedirs(args.log_dir, exist_ok=True)
        plot_path = os.path.join(args.log_dir, plot_name)
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        logger.info('Forecast visualization saved to {}'.format(plot_path))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
