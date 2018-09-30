import torch
import pickle

from torch.autograd import Variable, grad

def wgan_train(netD, netG, optimD, optimG, iter_num, critic_iters,
    data_loader, batch_size, input_dim, gp_lambda, use_gpu,
    plot_freq, plot_func, save_ckpt_list, model_name, output_folder):
    '''
    >>> training of wasserstein gans
    '''

    W_distance_list = []
    for iter_idx in range(iter_num):

        # Update D network
        for p in netD.parameters():
            p.requires_grad = True

        for critic_iter_id in range(critic_iters):
 
            netD.zero_grad()
 
            # Real data
            data_batch = data.next()
            data_batch = torch.Tensor(data_batch)
            if use_gpu:
                data_batch = data_batch.cuda()
            data_batch = Variable(data_batch)
            score_real = netD(data_batch)
            score_real = - score_real.mean()
            score_real.backward()

            # Fake data
            noise = torch.randn(batch_size, input_dim)
            if use_gpu:
                noise = noise.cuda()
            noise = Variable(noise, volatile = True)
            fake_batch = netG(noise).detach()
            score_fake = netD(data_batch)
            score_fake = score_fake.mean()
            score_fake.backward()

            # Gradient punishment
            alpha = torch.randn(batch_size, 1)
            if use_gpu:
                alpha = alpha.cuda()
            point_batch = alpha * data_batch + (1. - alpha) * fake_batch
            point_batch = Variable(point_batch, requires_grad = True)
            point_batch = netD(point_batch)

            basic_grad = torch.ones(point_score.size())
            if use_gpu:
                basic_grad = basic_grad.cuda()
            gradients = grad(outputs = point_score, inputs = point_batch, grad_outputs = basic_grad,
                create_graph = True, retain_graph = True, only_inputs = True)[0]
            grad_penalty = gp_lambda * ((torch.norm(gradients, p = 2, dim = 1) - 1) ** 2).mean()
            grad_penalty.backward()

            # Update
            optimD.step()

        # Update G network
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()

        # Fake data
        noise = torch.randn(batch_size, input_dim)
        if use_gpu:
            noise = noise.cuda()
        noise = Variable(noise, volatile = True)
        fake_batch = netG(noise)
        score_fake = netD(fake_batch)
        score_fake = - score_fake.mean()
        score_fake.backward()

        optimG.step()

        score_ture_data = netD(data_batch).mean()
        score_fake_data = netD(fake_batch).mean()
        W_distance = score_true_data - score_fake_data
        W_distance_list.append(W_distance)

        if plot_freq != None and (iter_idx + 1) % plot_freq == 0:
            netG.eval()
            plot_func(os.path.join(output_folder, '%s_%d.pdf'%(model_name, iter_idx + 1)))
            netG.train()

        if save_ckpt_list != None and (iter_idx + 1) in save_ckpt_list:
            netG.eval()
            torch.save(netD.state_dict(), os.path.join(output_folder, '%s_%d_netD.ckpt'%(model_name, iter_idx + 1)))
            torch.save(netG.state_dict(), os.path.join(output_folder, '%s_%d_netG.ckpt'%(model_name, iter_idx + 1)))
            netG.train()

    # Save model
    torch.save(netD.state_dict(), os.path.join(output_folder, '%s_netD.ckpt'%(model_name)))
    torch.save(netG.state_dict(), os.path.join(output_folder, '%s_netG.ckpt'%(model_name)))

    pickle.dump(W_distance_list, open(os.path.join(output_folder, '%s.pkl'%model_name), 'wb'))

    return W_distance_list