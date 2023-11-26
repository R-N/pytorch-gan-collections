import os

import pandas as pd
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

FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'stl10'], "dataset")
flags.DEFINE_enum('arch', 'res32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 1000, "total number of training steps")
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('alpha', 10, "gradient penalty")
flags.DEFINE_enum('loss', 'was', loss_fns.keys(), "loss function")
flags.DEFINE_enum('grad_loss', 'mse', grad_loss_fns.keys(), "grad loss function")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_integer('eval_step', 1000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/WGANGP_CIFAR10_RES', 'logging folder')
flags.DEFINE_bool('record', False, "record inception score and FID")
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', 'FID cache')
# generate
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 1000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
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


def cacl_gradient_penalty(net_D, real, fake):
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
    loss_gp = grad_loss_fns[FLAGS.grad_loss](
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

def train():
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
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

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
                loss_gp = cacl_gradient_penalty(net_D, real, fake)
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
    
    df_history = save_log(log_history, "history")
    df_values = save_log(log_values, "values")

    df_history[["d_grad", "gp_grad"]].plot()
    df_history[["d_loss", "d_gp"]].plot()
    df_values[["d_grad", "gp_grad"]].hist()
    df_values[["d_loss", "d_gp"]].hist()

    if not FLAGS.record:
        imgs = generate_imgs(
            net_G, device, FLAGS.z_dim,
            FLAGS.num_images, FLAGS.batch_size)
        IS, FID = get_inception_score_and_fid(
            imgs, FLAGS.fid_cache, verbose=True)
        print(
            "Inception Score: %.3f(%.5f), "
            "FID: %6.3f" % (
                IS[0], IS[1], FID))
    writer.close()

def save_log(log, name="log"):
    df = pd.DataFrame(log)
    df = pd.DataFrame.from_dict(log, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(FLAGS.logdir, f"name.csv"))
    return df

def main(argv):
    print("grad_loss", FLAGS.grad_loss)
    print("total_steps", FLAGS.total_steps)
    print("eval_step", FLAGS.eval_step)
    print("sample_step", FLAGS.sample_step)
    print("num_images", FLAGS.num_images)
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
