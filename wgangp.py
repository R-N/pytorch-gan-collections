import os
import argparse
import pandas as pd
import json
import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange
from pytorch_gan_metrics import get_inception_score_and_fid

import source.models.wgangp as models
import source.losses as losses
from source.utils import generate_imgs, infiniteloop, set_seed, get_gradients, reduce_grad


net_G_models = {
    'res32': models.ResGenerator32,
    'cnn32': models.Generator32,
}

net_D_models = {
    'res32': models.ResDiscriminator32,
    'cnn32': models.Discriminator32,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}

grad_loss_fns = {
    'mse': losses.mse,
    'mile': losses.mile,
    'mire': losses.mire,
}

ap = argparse.ArgumentParser()

# model and training
ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10'], help="dataset", type=str)
ap.add_argument('--arch', default='res32', choices=net_G_models.keys(), help="architecture", type=str)
ap.add_argument('--total_steps', default=100000, help="total number of training steps", type=int)
ap.add_argument('--batch_size', default=64, help="batch size", type=int)
ap.add_argument('--lr_G', default=2e-4, help="Generator learning rate", type=float)
ap.add_argument('--lr_D', default=2e-4, help="Discriminator learning rate", type=float)
ap.add_argument('--betas', default=[0.0,0.9], help="for Adam", type=list)
ap.add_argument('--n_dis', default=5, help="update Generator every this steps", type=int)
ap.add_argument('--z_dim', default=128, help="latent space dimension", type=int)
ap.add_argument('--alpha', default=10, help="gradient penalty", type=int)
ap.add_argument('--loss', default='was', choices=loss_fns.keys(), help="loss function", type=str)
ap.add_argument('--grad_loss', default='mse', choices=grad_loss_fns.keys(), help="grad loss function", type=str)
ap.add_argument('--seed', default=0, help="random seed", type=int)
# logging
ap.add_argument('--eval_step', default=5000, help="evaluate FID and Inception Score", type=int)
ap.add_argument('--sample_step', default=500, help="sample image every this steps", type=int)
ap.add_argument('--sample_size', default=64, help="sampling size of images", type=int)
ap.add_argument('--logdir', default='./logs/WGANGP_CIFAR10_RES', help='logging folder', type=str)
ap.add_argument('--record', default=False, help="record inception score and FID", type=bool)
ap.add_argument('--fid_cache', default='./stats/cifar10.train.npz', help='FID cache', type=str)
# generate
ap.add_argument('--generate', default=False, help='generate images', type=bool)
ap.add_argument('--pretrain', default=None, help='path to test model', type=bool)
ap.add_argument('--output', default='./outputs', help='path to output dir', type=str)
ap.add_argument('--num_images', default=50000, help='the number of generated images', type=int)\

ap.add_argument('--flagfile', default="./configs/WGANGP_CIFAR10_CNN.txt ", help='flagfile', type=str)

device = torch.device('cuda:0')


def generate(FLAGS):
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain)['net_G'])
    net_G.eval()

    counter = 0
    try:
        os.makedirs(FLAGS.output)
    except FileExistsError:
        pass
    with torch.no_grad():
        for start in trange(
                0, FLAGS.num_images, FLAGS.batch_size, dynamic_ncols=True):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - start)
            z = torch.randn(batch_size, FLAGS.z_dim).to(device)
            x = net_G(z).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(
                    image, os.path.join(FLAGS.output, '%d.png' % counter))
                counter += 1


def cacl_gradient_penalty(net_D, real, fake, grad_loss_fn="mse"):
    t = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.flatten(grad, start_dim=1).norm(2, dim=1)
    loss_gp = grad_loss_fns[grad_loss_fn](
        grad_norm,
        torch.ones(
            grad_norm.shape,
            dtype=grad_norm.dtype,
            device=grad_norm.device,
        )
    )
    return loss_gp

def append_log(log, key, value, extend=False):
    if key not in log:
        log[key] = []

    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "__iter__"):
        log[key].extend(value)
    else:
        log[key].append(value)

def mean(x):
    return sum(x) / len(x)

def train(FLAGS):
    if FLAGS.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if FLAGS.dataset == 'stl10':
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4,
        drop_last=True)

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    loss_fn = loss_fns[FLAGS.loss]()

    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)
    sched_G = optim.lr_scheduler.LambdaLR(
        optim_G, lambda step: 1 - step / FLAGS.total_steps)
    sched_D = optim.lr_scheduler.LambdaLR(
        optim_D, lambda step: 1 - step / FLAGS.total_steps)

    try:
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    except FileExistsError:
        pass
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    s_flags = json.dumps(vars(FLAGS))
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(s_flags)
    writer.add_text(
        "flagfile", s_flags.replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', grid)

    log_history = {}
    log_values = {}

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, desc='Training', ncols=0) as pbar:
        for step in pbar:
            # Discriminator
            gp_grads = []
            d_grads = []
            d_losses = []
            d_gps = []

            for _ in range(FLAGS.n_dis):
                with torch.no_grad():
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    fake = net_G(z).detach()
                real = next(looper).to(device)
                net_D_real = net_D(real)
                net_D_fake = net_D(fake)

                optim_D.zero_grad()

                loss = loss_fn(net_D_real, net_D_fake)
                loss_gp = cacl_gradient_penalty(net_D, real, fake, FLAGS.grad_loss)
                loss_all = loss + FLAGS.alpha * loss_gp

                loss_gp.backward()
                gp_grad = get_gradients(net_D)

                loss.backward()
                grad = get_gradients(net_D)
                d_grad = grad - gp_grad

                gp_grad = reduce_grad(gp_grad)
                d_grad = reduce_grad(d_grad)

                gp_grads.append(gp_grad.item())
                d_grads.append(d_grad.item())
                d_losses.append(loss.item())
                d_gps.append(loss_gp.item())

                #loss_all.backward()

                optim_D.step()

                if FLAGS.loss == 'was':
                    loss = -loss
                pbar.set_postfix(loss='%.4f' % loss)
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_gp', loss_gp, step)

            append_log(log_history, "gp_grad", mean(gp_grads))
            append_log(log_history, "d_grad", mean(d_grads))
            append_log(log_history, "d_loss", mean(d_losses))
            append_log(log_history, "d_gp", mean(d_gps))

            append_log(log_values, "gp_grad", gp_grads)
            append_log(log_values, "d_grad", d_grad)
            append_log(log_values, "d_loss", d_losses)
            append_log(log_values, "d_gp", d_gps)

            # Generator
            for p in net_D.parameters():
                # reduce memory usage
                p.requires_grad_(False)
            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
            loss = loss_fn(net_D(net_G(z)))

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()
            for p in net_D.parameters():
                p.requires_grad_(True)

            sched_G.step()
            sched_D.step()

            if step == 1 or step % FLAGS.sample_step == 0:
                fake = net_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step == 1 or step % FLAGS.eval_step == 0:
                torch.save({
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                }, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.record:
                    imgs = generate_imgs(
                        net_G, device, FLAGS.z_dim,
                        FLAGS.num_images, FLAGS.batch_size)
                    IS, FID = get_inception_score_and_fid(
                        imgs, FLAGS.fid_cache, verbose=True)
                    pbar.write(
                        "%s/%s Inception Score: %.3f(%.5f), "
                        "FID: %6.3f" % (
                            step, FLAGS.total_steps, IS[0], IS[1], FID))
                    writer.add_scalar('Inception_Score', IS[0], step)
                    writer.add_scalar('Inception_Score_std', IS[1], step)
                    writer.add_scalar('FID', FID, step)
    
    df_history = save_log(FLAGS.logdir, log_history, "history")
    df_values = save_log(FLAGS.logdir, log_values, "values")

    df_history["d_loss"] = -df_history["d_loss"]
    df_values["d_loss"] = -df_values["d_loss"]
    df_history[["d_grad", "gp_grad"]].plot()
    df_history[["d_loss", "d_gp"]].plot()
    df_values[["d_grad", "gp_grad"]].hist()
    df_values[["d_loss", "d_gp"]].hist()

    if not FLAGS.record:
        with torch.no_grad():
            imgs = generate_imgs(
                net_G, device, FLAGS.z_dim,
                FLAGS.num_images, FLAGS.batch_size)
            IS, FID = get_inception_score_and_fid(
                imgs, FLAGS.fid_cache, verbose=True)
            print(
                "Inception Score: %.3f(%.5f), "
                "FID: %6.3f" % (
                    IS[0], IS[1], FID))
            counter = 0
            try:
                os.makedirs(FLAGS.output)
            except FileExistsError:
                pass
            for image in imgs:
                image = image.cpu()
                save_image(
                    image, os.path.join(FLAGS.output, '%d.png' % counter))
                counter += 1
    writer.close()

def save_log(log_dir, log, name="log"):
    df = pd.DataFrame.from_dict(log, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(log_dir, f"{name}.csv"))
    return df

def split_args(s=None):
    if isinstance(s, str):
        s = [si.strip() for si in s.replace("\\", "").split(" ")]
        s = [si.strip() for si in s if si]

    if not s:
        s = []

    return s

def parse_args(s=None):
    s = split_args(s)
    FLAGS = ap.parse_args(s)
    if hasattr(FLAGS, "flagfile") and FLAGS.flagfile:
        with open(FLAGS.flagfile) as f:
            sf = f.read()
        sf = split_args(sf)
        FLAGS = ap.parse_args(sf + s)
    return FLAGS


def main(s=None):

    FLAGS = parse_args(s)

    print(FLAGS)
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate(FLAGS)
    else:
        train(FLAGS)


if __name__ == '__main__':
    main()
