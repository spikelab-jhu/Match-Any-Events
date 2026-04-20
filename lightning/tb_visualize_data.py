import tensorboardX
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def plot_event(event):
    flattened_event = torch.sum(event,dim=1).unsqueeze(1)
    max = torch.abs(flattened_event).max()
    event_plot = flattened_event.numpy().astype(float)/max + 0.5
    return event_plot

def plot_prune(prune, events, t_chans):
    B, T, H_img, W_img = events.shape
    Bs, H_t, W_t, _ = prune.shape

    prune = prune.reshape( B, t_chans, H_t, W_t)
    prune = prune[0].unsqueeze(0)
    events = events[0].unsqueeze(0)

    prune_resized = F.interpolate(prune, size=(H_img, W_img), mode='bilinear',align_corners=False)

    fig, axes = plt.subplots(t_chans, figsize=(5, 5*t_chans))
    axes = axes.flatten()

    for i in range(t_chans):
        img = torch.mean(events[0, i*2:i*2+1], dim = 0)  # (H, W)
        attn_map = prune_resized[0,i]  # (H, W)

        ax = axes[i]
        ax.imshow(img/2*(img.max()+1e-5)+0.5, cmap='gray')
        ax.imshow(attn_map, cmap='jet', alpha=0.2, vmin = 0.0, vmax = 1.0)
        ax.set_title(f"Image {i}")
        ax.axis('off')

    # Hide unused axes
    for j in range(t_chans, len(axes)):
        axes[j].axis('off')

    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)  # Close the figure to free memory
    return image_array[...,1:]


def plot_attn(attn, events, t_chans):

    B, num_images, H_img, W_img = events.shape
    # Only take the first of the batch
    attn = attn[0].unsqueeze(0)
    events = events[0].unsqueeze(0)

    # Step 1: Average over heads → shape becomes [2048, 15]
    attn_avg = attn.mean(dim=-1).squeeze(0).squeeze(1)  # (2048, 15)

    # Step 2: Reshape to (B, 1, H_attn, W_attn)
    attn_maps = attn_avg.T.view(t_chans, 1, int(H_img/14), int(W_img/14)) 

    attn_resized = F.interpolate(attn_maps, size=(H_img, W_img), mode='bilinear',align_corners=False)

    # attn_resized = attn_resized - attn_resized.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    # attn_resized = attn_resized / (attn_resized.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

    fig, axes = plt.subplots(t_chans, figsize=(5, 5*t_chans))
    axes = axes.flatten()

    for i in range(t_chans):
        img = torch.mean(events[0, i*2:i*2+1], dim = 0)  # (H, W)
        attn_map = attn_resized[i, 0]  # (H, W)

        ax = axes[i]
        ax.imshow(img/2*(img.max()+1e-5)+0.5, cmap='gray')
        ax.imshow(attn_map, cmap='jet', alpha=0.2)
        ax.set_title(f"Image {i}")
        ax.axis('off')

    # Hide unused axes
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)  # Close the figure to free memory
    return image_array[...,1:]

def plot_concentrated(con):
    B, _, H_img, W_img = con.shape
    con = np.sum(con, axis=1)[0]
    # plt.figure()
    # plt.imshow(con/2+0.5, cmap='gray')
    # plt.show()
    fig, ax = plt.subplots()
    ax.imshow(con/2+0.5, cmap='gray')
    ax.set_title(f"Concentration")
    ax.axis('off')
    # 
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)  # Close the figure to free memory
    return image_array[...,1:]



def visualize_tb(writer:tensorboardX.SummaryWriter, data:dict, step:int, train = False, scale = 8):
    # Images
    if train:
        mode = 'train'
    else:
        mode = 'test'
    
    event = data['event'].cpu()
    event_plot = plot_event(event)
    image = data['image'].cpu().numpy()
    if image.shape[1]>=1:
        image = plot_event(data['image'].cpu())
    depth_i = data['depth0'].cpu().numpy()
    depth_e = data['depth1'].cpu().numpy()

    if 'con' in data:
        con = data['con'].cpu().numpy()
        con_fig = plot_concentrated(con)
        writer.add_images(mode+'/con', con_fig, step, dataformats='HWC')

    if 'attn' in data:
        attn = data['attn'].cpu()
        attn_fig = plot_attn(attn, event, 8)
        writer.add_images(mode+'/attn', attn_fig, step, dataformats='HWC')

    if 'prune' in data and data['prune'] is not None:
        prune = data['prune'].detached().cpu()
        prune_fig = plot_prune(prune, event, 8)
        writer.add_images(mode+'/prune', prune_fig, step, dataformats='HWC')


    shape = image.shape[-2:]
    if 'spv_b_ids' in data:
        coarse_spv = coarse_ids2points(data['spv_b_ids'].cpu(),data['spv_i_ids'].cpu(),data['spv_j_ids'].cpu(), shape, scale)
        fig_spv = plot_matched_points(image.squeeze(1), event_plot.squeeze(1), coarse_spv.cpu().numpy())
        # fig_spv = plot_matched_points(depth_i, depth_e, coarse_spv.cpu().numpy())
        fine_spv = fine_ids2points(data['spv_w_pt0_i'].cpu(), data['spv_pt1_i'].cpu(), data['spv_b_ids'].cpu(), data['spv_i_ids'].cpu(), data['spv_j_ids'].cpu(), shape, scale)
        fig_fine_spv = plot_matched_points(depth_i, depth_e, fine_spv.cpu().numpy())
        writer.add_images(mode+'/coarse_spv', fig_spv, step, dataformats='HWC')
        writer.add_images(mode+'/fine_spv', fig_fine_spv, step, dataformats='HWC')

    coarse_pred = coarse_ids2points(data['b_ids'].cpu(),data['i_ids'].cpu(),data['j_ids'].cpu(), shape, scale)
    fig_pred = plot_matched_points(image.squeeze(1), event_plot.squeeze(1), coarse_pred.cpu().numpy())

    if 'loss' in data:
    # Scalars
        loss = data['loss']
        writer.add_scalar(mode+'/loss',loss,global_step=step)
        loss_c =  data["loss_scalars"]['loss_c'].item()
        writer.add_scalar(mode+'/loss_c',loss_c,global_step=step)
        loss_f = data["loss_scalars"]['loss_f'].item()
        writer.add_scalar(mode+'/loss_f',loss_f,global_step=step)
        
    fig_depth = plot_depth_map_batch(data)
    if mode == 'test':
        r_err = np.array(data['R_errs']).mean()
        t_err = np.array(data['t_errs']).mean()
    
    writer.add_images(mode+'/image', image, step)
    
    writer.add_images(mode+'/coarse_pred', fig_pred, step, dataformats='HWC')
    writer.add_images(mode+'/event', event_plot, step)
    writer.add_images(mode+'/depth', fig_depth, step, dataformats='HWC')
    # writer.add_scalar('lr',lr,global_step=epoch)
    
    if mode == 'test':
        writer.add_scalar(mode+'/r_err',r_err,global_step=step)
        writer.add_scalar(mode+'/t_err',t_err,global_step=step)

def visualize_tb_lightning(writer:TensorBoardLogger, data:dict, step:int, train = False, scale = 8, num_bins = 8):
    # Images
    if train:
        mode = 'train'
    else:
        mode = 'test'
    
    event = data['event'].cpu()
    event_plot = plot_event(event)
    image = data['image'].cpu().numpy()
    if image.shape[1]>=1:
        image = plot_event(data['image'].cpu())
    depth_i = data['depth0'].cpu().numpy()
    depth_e = data['depth1'].cpu().numpy()

    if 'con' in data:
        con = data['con'].cpu().numpy()
        con_fig = plot_concentrated(con)
        writer.experiment.add_image(mode+'/con', np.transpose(con_fig, (2, 0, 1)), step)

    if 'attn' in data:
        attn = data['attn'].cpu()
        attn_fig = plot_attn(attn, event, num_bins)
        writer.experiment.add_image(mode+'/attn', np.transpose(attn_fig, (2, 0, 1)), step)

    if 'prune' in data and data['prune'] is not None:
        prune = data['prune'].detach().cpu()
        prune_fig = plot_prune(prune, event, num_bins)
        writer.experiment.add_image(mode+'/prune', np.transpose(prune_fig, (2, 0, 1)), step)



    shape = image.shape[-2:]
    if 'spv_b_ids' in data:
        coarse_spv = coarse_ids2points(data['spv_b_ids'].cpu(),data['spv_i_ids'].cpu(),data['spv_j_ids'].cpu(), shape, scale)
        fig_spv = plot_matched_points(image.squeeze(1), event_plot.squeeze(1), coarse_spv.cpu().numpy())
        # fig_spv = plot_matched_points(depth_i, depth_e, coarse_spv.cpu().numpy())
        fine_spv = fine_ids2points(data['spv_w_pt0_i'].cpu(), data['spv_pt1_i'].cpu(), data['spv_b_ids'].cpu(), data['spv_i_ids'].cpu(), data['spv_j_ids'].cpu(), shape, scale)
        fig_fine_spv = plot_matched_points(depth_i, depth_e, fine_spv.cpu().numpy())
        writer.experiment.add_image(mode+'/coarse_spv', np.transpose(fig_spv, (2, 0, 1)), step)
        writer.experiment.add_image(mode+'/fine_spv', np.transpose(fig_fine_spv, (2, 0, 1)), step)

    coarse_pred = coarse_ids2points(data['b_ids'].cpu(),data['i_ids'].cpu(),data['j_ids'].cpu(), shape, scale)
    fig_pred = plot_matched_points(image.squeeze(1), event_plot.squeeze(1), coarse_pred.cpu().numpy())

    if 'loss' in data:
    # Scalars
        loss = data['loss']
        writer.experiment.add_scalar(mode+'/loss',loss,step)
        loss_c = data["loss_scalars"]['loss_c'].item()
        writer.experiment.add_scalar(mode+'/loss_c',loss_c,step)
        # loss_f = data["loss_scalars"]['loss_f'].item()
        # writer.experiment.add_scalar(mode+'/loss_f',loss_f,step)
        # loss_l = data["loss_scalars"]['loss_l'].item()
        # writer.experiment.add_scalar(mode+'/loss_l',loss_l,step)
        # if 'prune' in data and data['prune'] is not None:
        #     loss_t = data["loss_scalars"]['loss_t'].item()
        #     writer.experiment.add_scalar(mode+'/loss_t',loss_t,step)
        
    fig_depth = plot_depth_map_batch(data)
    if mode == 'test':
        r_err = np.array(data['R_errs']).mean()
        t_err = np.array(data['t_errs']).mean()
    
    # writer.experiment.add_image(mode+'/image', image[:,0], step)
    
    writer.experiment.add_image(mode+'/coarse_pred', np.transpose(fig_pred, (2, 0, 1)), step)
    # writer.experiment.add_image(mode+'/event', event_plot[:,0], step)
    writer.experiment.add_image(mode+'/depth', np.transpose(fig_depth, (2, 0, 1)), step)
    # writer.add_scalar('lr',lr,global_step=epoch)
    
    if mode == 'test':
        writer.experiment.add_scalar(mode+'/r_err',r_err,step)
        writer.experiment.add_scalar(mode+'/t_err',t_err,step)

def coarse_ids2points(b_ids ,i_ids, j_ids, shape = [], scale = 8):
    h1 = int(shape[0]//scale)
    w1 = int(shape[1]//scale)
    
    h_i = i_ids // w1
    w_i = i_ids % w1

    h_j = j_ids // w1
    w_j = j_ids % w1

    h0_i = (h_i * scale)
    w0_i = (w_i * scale) 
    h0_j = (h_j * scale) 
    w0_j = (w_j * scale) 

    return torch.stack([b_ids, w0_i, h0_i, w0_j, h0_j], dim = 1)

def fine_ids2points(w_pt0_i ,grid_pt1_i, b_ids ,i_ids, j_ids, shape = [], scale = 8):
    h1 = int(shape[0]//scale)
    w1 = int(shape[1]//scale)
    
    h_i = i_ids // w1
    w_i = i_ids % w1
    h0_i = (h_i * scale)
    w0_i = (w_i * scale) 
    w_pt0_i_valid = w_pt0_i[b_ids, i_ids].round().long()
    all_point = torch.cat([b_ids.unsqueeze(1), w0_i.unsqueeze(1), h0_i.unsqueeze(1), w_pt0_i_valid], dim = 1)
    return all_point


def plot_matched_points(image1, image2, coarse_spv):

    bs = image1.shape[0]
    num_points = len(coarse_spv)  # Number of matched pairs

    # Generate colors using the Jet colormap
    colormap = plt.cm.jet(np.linspace(0, 1, num_points))

    # Create figure with subplots
    fig, axes = plt.subplots(bs, 2, figsize=(10, 20))
    axes = axes.ravel()
    for i in range(bs):
        # Plot Image 1 with matched points
        axes[2*i].imshow(image1[i],'gray')
        axes[2*i].set_title("Image 1")
        axes[2*i].axis("off")
        # for i, (pt, color) in enumerate(zip(points1, colormap)):
        #     axes[2*i].scatter(pt[0], pt[1], color=color, s=80)

        # Plot Image 2 with matched points
        axes[2*i+1].imshow(image2[i],'gray')
        axes[2*i+1].set_title("Image 2")
        axes[2*i+1].axis("off")
    for i, (pt, color) in enumerate(zip(coarse_spv, colormap)):
        axes[2*pt[0]].scatter(pt[1], pt[2], color=color, s=10)
        axes[2*pt[0]+1].scatter(pt[3], pt[4], color=color, s=10)

    # Convert the figure to a NumPy array
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)  # Close the figure to free memory
    return image_array[...,1:]  # Return the NumPy array of the plotted image

def plot_depth_map_batch(data):
    def plot_depth_map(ax, depth_map, cmap="viridis"):
        """
        Plots a depth map on the given axis `ax`, handling NaN values.
        """
        # Plot depth map
        img = ax.imshow(depth_map, cmap=cmap, interpolation='nearest')
        # ax.imshow(np(depth_map), cmap="gray", alpha=0.3)  # Gray overlay for NaNs
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Depth Value")

        # Labels and title
        ax.set_title("Depth Map")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

    depth_e = data['depth1'].cpu()
    depth_i = data['depth0'].cpu()

    bs = depth_e.shape[0]

    fig, axes = plt.subplots(bs, 2, figsize=(10, 20))
    axes = axes.ravel()
    for i in range(bs):
        # Plot Image 1 with matched points
        plot_depth_map(axes[2*i], depth_i[i].numpy())
        plot_depth_map(axes[2*i+1], depth_e[i].numpy())

    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)  # Close the figure to free memory
    return image_array[...,1:]  # Return the NumPy array of the plotted image
    