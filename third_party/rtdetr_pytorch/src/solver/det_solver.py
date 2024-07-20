'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch

from src.misc import dist
from src.misc.tb_logger import TBWriter
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..core import BaseConfig


class DetSolver(BaseSolver):

    @property
    def best_checkpoint_path(self):
        return self.output_dir / 'best_checkpoint.pth'

    def fit(self):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                tb_writer=self.tb_writer, max_norm=args.clip_max_norm, print_freq=args.log_step, ema=self.ema,
                scaler=self.scaler)

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last_checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir,
                tb_writer=self.tb_writer, mode='val', epoch=epoch,
            )

            # TODO
            for k in test_stats.keys():
                if k in best_stat:
                    if test_stats[k][0] > best_stat[k]:
                        best_stat['epoch'] = epoch
                        dist.save_on_master(self.state_dict(epoch), self.best_checkpoint_path)
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
                    dist.save_on_master(self.state_dict(epoch), self.best_checkpoint_path)
            self.tb_writer.add_scalar('best_epoch/map@50-95', best_stat['coco_eval_bbox'], epoch)
            self.tb_writer.add_scalar('best_epoch/epoch', best_stat['epoch'], epoch)
            print('best_stat: ', best_stat)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval_logs').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval_logs" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        self.tb_writer.close()

        return self.best_checkpoint_path


    def val(self):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir, tb_writer=self.tb_writer, mode='test')

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        self.tb_writer.close()
        return
