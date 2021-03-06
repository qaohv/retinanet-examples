import json
from statistics import mean
from math import isfinite 
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel
from .backbones.layers import convert_fixedbn_model

from .data import DataIterator
from .dali import DaliDataIterator
from .utils import ignore_sigint, post_metrics, Profiler
from .infer import infer
from .augmentations import create_augmentations
from .logger import get_root_logger
from .early_stopping import EarlyStopping


def train(model, state, path, annotations, val_path, val_annotations, augs, resize, max_size, jitter, batch_size, iterations, val_iterations, mixed_precision, lr, warmup, milestones, rop_reduce_factor, rop_patience, is_master=True, world=1, use_dali=True, verbose=True, metrics_url=None, logdir=None):
    'Train the model on the given dataset'
    logger = get_root_logger()
    # Prepare model
    nn_model = model
    stride = model.stride

    model = convert_fixedbn_model(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # Setup optimizer and schedule
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9) 

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level = 'O2' if mixed_precision else 'O0',
                                      keep_batchnorm_fp32 = True,
                                      loss_scale = 128.0,
                                      verbosity = is_master)

    if world > 1: 
        model = DistributedDataParallel(model)
    model.train()

    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=rop_reduce_factor, patience=rop_patience, verbose=True)
    logger.info(f"Training with scheduler: {scheduler}")
    # Prepare dataset
    if verbose: logger.info('Preparing dataset...')
    with open(augs) as f:
        augs_cfg = json.load(f)
    transforms = create_augmentations(augs_cfg)
    logger.info("Current augmentations:")
    for t in transforms:
        logger.info(t)
    data_iterator = DataIterator(path, jitter, max_size, batch_size, stride, world, annotations, transforms,
                                 training=True)
    val_iterator = DataIterator(val_path, resize, max_size, batch_size, stride, world, val_annotations, training=True)
    if verbose: logger.info(data_iterator)


    if verbose:
        logger.info('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'))
        logger.info('    batch: {}, precision: {}'.format(batch_size, 'mixed' if mixed_precision else 'full'))
        logger.info('Training model for {} iterations...'.format(iterations))

    # Create TensorBoard writer
    if logdir is not None:
        from tensorboardX import SummaryWriter
        if is_master and verbose:
            logger.info('Writing TensorBoard logs to: {}'.format(logdir))
        writer = SummaryWriter(logdir=logdir)

    profiler = Profiler(['train', 'fw', 'bw'])
    iteration = state.get('iteration', 0)
    es = EarlyStopping(patience=10, mode='min', logger=logger)

    while iteration < iterations and not es.early_stop:
        cls_losses, box_losses = [], []
        for i, (data, target) in enumerate(data_iterator):

            # Forward pass
            profiler.start('fw')

            optimizer.zero_grad()
            cls_loss, box_loss = model([data, target])
            del data
            profiler.stop('fw')

            # Backward pass
            profiler.start('bw')
            with amp.scale_loss(cls_loss + box_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            # Reduce all losses
            cls_loss, box_loss = cls_loss.mean().clone(), box_loss.mean().clone()
            if world > 1:
                torch.distributed.all_reduce(cls_loss)
                torch.distributed.all_reduce(box_loss)
                cls_loss /= world
                box_loss /= world
            if is_master:
                cls_losses.append(cls_loss)
                box_losses.append(box_loss)

            if is_master and not isfinite(cls_loss + box_loss):
                raise RuntimeError('Loss is diverging!\n{}'.format(
                    'Try lowering the learning rate.'))

            del cls_loss, box_loss
            profiler.stop('bw')

            iteration += 1
            profiler.bump('train')
            if is_master and (profiler.totals['train'] > 60 or iteration == iterations):
                focal_loss = torch.stack(list(cls_losses)).mean().item()
                box_loss = torch.stack(list(box_losses)).mean().item()
                learning_rate = optimizer.param_groups[0]['lr']
                if verbose:
                    msg  = '[{:{len}}/{}]'.format(iteration, iterations, len=len(str(iterations)))
                    msg += ' train focal loss: {:.3f}'.format(focal_loss)
                    msg += ', train box loss: {:.3f}'.format(box_loss)
                    msg += ', total train loss: {:.3f}'.format(focal_loss + box_loss)
                    msg += ', {:.3f}s/{}-batch'.format(profiler.means['train'], batch_size)
                    msg += ' (fw: {:.3f}s, bw: {:.3f}s)'.format(profiler.means['fw'], profiler.means['bw'])
                    msg += ', {:.1f} im/s'.format(batch_size / profiler.means['train'])
                    msg += ', lr: {:.2g}'.format(learning_rate)
                    logger.info(msg)

                if logdir is not None:
                    writer.add_scalar('focal_loss', focal_loss,  iteration)
                    writer.add_scalar('box_loss', box_loss, iteration)
                    writer.add_scalar('total_train_loss', focal_loss + box_loss, iteration)
                    writer.add_scalar('learning_rate', learning_rate, iteration)
                    del box_loss, focal_loss

                if metrics_url:
                    post_metrics(metrics_url, {
                        'focal loss': mean(cls_losses),
                        'box loss': mean(box_losses),
                        'total loss': focal_loss + box_loss ,
                        'im_s': batch_size / profiler.means['train'],
                        'lr': learning_rate
                    })

                # Save model weights
                state.update({
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                })

                profiler.reset()
                del cls_losses[:], box_losses[:]

            if val_annotations and (iteration == iterations or iteration % val_iterations == 0):
                # calculate validation loss
                val_cls_losses, val_box_losses = [], []
                for data, target in val_iterator:
                    val_cls_loss, val_box_loss = model([data, target])
                    del data

                    with amp.scale_loss(val_cls_loss + val_box_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                    val_cls_loss, val_box_loss = val_cls_loss.mean().clone(), val_box_loss.mean().clone()
                    if world > 1:
                        torch.distributed.all_reduce(val_cls_loss)
                        torch.distributed.all_reduce(val_box_loss)
                        val_cls_loss /= world
                        val_box_loss /= world
                    if is_master:
                        val_cls_losses.append(val_cls_loss)
                        val_box_losses.append(val_box_loss)

                val_focal_loss = torch.stack(list(val_cls_losses)).mean().item()
                val_bbox_loss = torch.stack(list(val_box_losses)).mean().item()

                msg = '[{:{len}}/{}]'.format(iteration, iterations, len=len(str(iterations)))
                msg += ' val focal loss: {:.3f}'.format(val_focal_loss)
                msg += ', val box loss: {:.3f}'.format(val_bbox_loss)
                msg += ', total val loss: {:.3f}'.format(val_focal_loss + val_bbox_loss)
                logger.info(msg)

                if logdir is not None:
                    writer.add_scalar('val_focal_loss', val_focal_loss,  iteration)
                    writer.add_scalar('val_box_loss', val_bbox_loss, iteration)
                    writer.add_scalar('total_val_loss', val_focal_loss + val_bbox_loss, iteration)

                es(val_bbox_loss + val_bbox_loss)
                scheduler.step(val_focal_loss + val_bbox_loss)

                del val_bbox_loss, val_focal_loss
                del val_cls_loss, val_box_loss
                del val_cls_losses[:], val_box_losses[:]
                # calculate COCO mAP
                infer(model, val_path, None, resize, max_size, batch_size, annotations=val_annotations,
                    mixed_precision=mixed_precision, is_master=is_master, world=world, use_dali=use_dali, is_validation=True, verbose=False)

                with ignore_sigint():
                    nn_model.save(state)

                model.train()

            if iteration == iterations or es.early_stop:
                break

    if logdir is not None:
        writer.close()


