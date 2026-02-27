import torch

def return_out_mask(batch_data):
    return (batch_data != 0).float()

def vae_loss_fn(recon, x, mu, logvar, mask):
    # TODO: Compute masked reconstruction loss (MSE or BCE) on nonzero entries
    # TODO: Compute KL divergence term: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # TODO: Return total loss = recon_loss + kl_loss
    #       (use weight = 1.0 for the KL term unless you explicitly experiment with different values)
    pass

def train_model(model, dataloader, epochs, lr, device, model_type="ae", noise_fn=None):
    """
    Train AE, DAE, or VAE.

    Args:
        model_type: "ae", "dae", or "vae"
        noise_fn: optional function that adds noise (used for DAE)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_inputs, _ in dataloader:
            batch_inputs = batch_inputs.to(device)
            mask = return_out_mask(batch_inputs).to(device)

            # TODO: If model_type == "dae", apply noise_fn to batch_inputs (do not modify batch_inputs in-place):
            #       noisy_inputs = noise_fn(batch_inputs)
            #       use noisy_inputs as model input, but keep clean batch_inputs for loss

            # TODO: Forward pass
            #       - AE/DAE: recon, z = model(batch_inputs or noisy_inputs)
            #       - VAE: recon, mu, logvar = model(batch_inputs)

            # TODO: Compute loss
            #       - AE/DAE: masked MSE = ((recon - batch_inputs)^2 * mask).sum() / mask.sum()
            #         (consider clamp_min(1.0) to avoid division by zero)
            #       - VAE: vae_loss_fn(recon, batch_inputs, mu, logvar, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss / max(1, num_batches):.4f}")

def evaluate_model(model, data_tensor, device, model_type="ae"):
    model.eval()
    with torch.no_grad():
        data_tensor = data_tensor.to(device)

        if model_type == "vae":
            recon, mu, _ = model(data_tensor)
            latent = mu
        else:
            recon, latent = model(data_tensor)

    return recon.cpu().numpy(), latent.cpu().numpy()
